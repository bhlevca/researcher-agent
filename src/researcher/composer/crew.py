"""ComposerCrew — CrewAI orchestrator for the Music Composer.

Builds agents and tasks for composition, arrangement, harmonization,
and analysis. Follows the same patterns as TutorCrew.
"""

import os
import re
import json
import pathlib
import logging
from pathlib import Path

from crewai import Agent, Crew, Process, Task, LLM
from crewai.agent.planning_config import PlanningConfig

from researcher.config import probe_model_capabilities, TOOL_CAPABLE_MODEL
import researcher.patches as _patches  # noqa: F401 — applies CrewAI compatibility patches
from researcher.composer.tools import COMPOSER_TOOLS
from researcher.composer.musicxml_fix import fix_musicxml
from researcher.composer.musicxml_builder import build_musicxml

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).parent / "config"


def _load_yaml(name: str) -> dict:
    import yaml

    with open(_CONFIG_DIR / name) as f:
        return yaml.safe_load(f)


class ComposerCrew:
    """Music Composer agent orchestrator.

    Dynamically builds tasks from YAML templates with runtime
    variable substitution for genre, key, time signature, etc.
    """

    DEFAULT_LLM_PARAMS: dict = {
        "temperature": 0.7,
        "top_k": 40,
        "top_p": 0.9,
        "min_p": 0.0,
        "seed": 0,
        "num_predict": 16384,  # scores are long; thinking tokens also consume this budget
        "num_ctx": 32768,
        "repeat_penalty": 1.0,  # JSON scores repeat "pitch","duration","staff" constantly
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "max_iter": 12,
        "planning_max_attempts": 2,
        "crew_timeout_sec": 300,
    }

    def __init__(self, model: str | None = None):
        self._model_name = model or os.getenv("MODEL", "ollama/qwen3.5:9b")
        self._llm_params: dict = dict(self.DEFAULT_LLM_PARAMS)
        self._llm = self._make_llm(self._model_name, self._llm_params)
        self._model_caps = probe_model_capabilities(self._model_name)
        self._agents_config = _load_yaml("agents.yaml")
        self._tasks_config = _load_yaml("tasks.yaml")

    def get_llm_params(self) -> dict:
        """Return current LLM parameters."""
        return dict(self._llm_params)

    def update_llm_params(self, params: dict) -> dict:
        """Update LLM parameters and rebuild the LLM instance."""
        for k, v in params.items():
            if k in self.DEFAULT_LLM_PARAMS:
                self._llm_params[k] = v
        self._llm = self._make_llm(self._model_name, self._llm_params)
        return dict(self._llm_params)

    @staticmethod
    def _make_llm(model: str, params: dict) -> LLM:
        is_thinking_model = any(k in model.lower() for k in ("qwen3", "deepseek-r1"))
        seed = int(params.get("seed", 0))
        num_predict = int(params.get("num_predict", 16384))
        ollama_options = {
            "num_ctx": int(params.get("num_ctx", 32768)),
            "num_gpu": 99,
            "num_predict": num_predict,
            "top_k": int(params.get("top_k", 40)),
            "top_p": float(params.get("top_p", 0.9)),
            "min_p": float(params.get("min_p", 0.0)),
            "repeat_penalty": float(params.get("repeat_penalty", 1.0)),
        }
        if seed > 0:
            ollama_options["seed"] = seed
        extra_body: dict = {"options": ollama_options}
        if is_thinking_model:
            extra_body["chat_template_kwargs"] = {"enable_thinking": True}
        return LLM(
            model=model,
            base_url="http://localhost:11434",
            temperature=float(params.get("temperature", 0.7)),
            frequency_penalty=float(params.get("frequency_penalty", 0.0)),
            presence_penalty=float(params.get("presence_penalty", 0.0)),
            extra_body=extra_body,
        )

    @staticmethod
    def _make_planning_llm(model: str) -> LLM:
        """Lightweight LLM for the planning phase.

        Thinking disabled, small context — planning completes fast.
        """
        return LLM(
            model=model,
            base_url="http://localhost:11434",
            temperature=0.2,
            extra_body={
                "options": {
                    "num_ctx": 8192,
                    "num_gpu": 99,
                    "num_predict": 1024,
                },
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )

    def _build_agent(
        self,
        genre: str,
        key_signature: str,
        time_signature: str,
        tempo: int,
        tools: list | None = None,
    ) -> Agent:
        agent_cfg = self._agents_config["music_composer"]

        subs = {
            "{genre}": genre,
            "{key_signature}": key_signature,
            "{time_signature}": time_signature,
            "{tempo}": str(tempo),
        }

        def _sub(text: str) -> str:
            for k, v in subs.items():
                text = text.replace(k, v)
            return text

        # Use the selected model for planning if it supports tools;
        # otherwise fall back to a known tool-capable model.
        if self._model_caps["supports_tools"]:
            planning_model = self._model_name
        else:
            planning_model = TOOL_CAPABLE_MODEL
            logger.info(
                "[COMPOSER] %s lacks tool support — planning with %s",
                self._model_name, planning_model,
            )
        planning_llm = self._make_planning_llm(planning_model)
        planning_llm._allow_fc = True
        planning = PlanningConfig(
            llm=planning_llm,
            max_attempts=int(self._llm_params.get("planning_max_attempts", 2)),
            max_steps=3,
            reasoning_effort="low",
            plan_prompt=(
                "You are {role}.\\n\\n"
                "Task: {description}\\n\\n"
                "Available tools: {tools}\\n\\n"
                "Create a plan with EXACTLY 2 steps:\\n"
                "1. Decide the motif, the phrase plan and the cadence points.\\n"
                "2. Write the complete ABC notation and output it as Final Answer.\\n"
                "Do NOT plan to call any tool: everything needed is already in the "
                "task description.\\n\\n"
                "End your response with EXACTLY this line:\n"
                "READY: I am ready to execute the task.\n"
            ),
        )

        backstory = _sub(agent_cfg["backstory"])

        # For models without native tool support, add explicit ReAct format
        # instructions — they tend to fabricate tool calls in their own format.
        if not self._model_caps["supports_tools"]:
            backstory += (
                "\n\n===== CRITICAL: TOOL CALLING FORMAT =====\n"
                "You MUST use EXACTLY this text format to call tools:\n\n"
                "Thought: I need to call CompositionPrep to get the composition data.\n"
                "Action: CompositionPrep\n"
                'Action Input: {"key": "C major", "style": "jazz"}\n\n'
                "Then WAIT for the Observation (the tool result). "
                "Do NOT invent or fabricate the tool response yourself. "
                "Do NOT use ```tool_code``` or any other format. "
                "The system will execute the tool and provide the Observation.\n"
                "After receiving the Observation, write:\n\n"
                "Thought: I now have the composition data. I will compose the score.\n"
                "Final Answer: [your complete JSON score]\n"
            )

        return Agent(
            role=_sub(agent_cfg["role"]),
            goal=_sub(agent_cfg["goal"]),
            backstory=backstory,
            tools=COMPOSER_TOOLS if tools is None else tools,
            llm=self._llm,
            verbose=True,
            max_iter=int(self._llm_params.get("max_iter", 12)),
            max_retry_limit=2,
            respect_context_window=True,
            planning_config=planning,
            **_patches.executor_kwargs(),
        )

    def _render_task(self, task_key: str, variables: dict) -> tuple[str, str]:
        task_cfg = self._tasks_config[task_key]
        desc = task_cfg["description"]
        exp = task_cfg["expected_output"]
        for k, v in variables.items():
            placeholder = "{" + k + "}"
            desc = desc.replace(placeholder, str(v))
            exp = exp.replace(placeholder, str(v))
        return desc, exp

    # ------------------------------------------------------------------
    # Public build methods
    # ------------------------------------------------------------------

    def build_chat_crew(
        self,
        message: str,
        context: str,
        genre: str,
        key_signature: str,
        time_signature: str,
        tempo: int,
    ) -> Crew:
        agent = self._build_agent(genre, key_signature, time_signature, tempo,
                                  tools=[])  # ABC needs no tools; CompositionPrep emits JSON
        desc, exp = self._render_task("compose_chat_task", {
            "genre": genre,
            "key_signature": key_signature,
            "time_signature": time_signature,
            "tempo": str(tempo),
            "context": context,
            "message": message,
        })
        task = Task(description=desc, expected_output=exp, agent=agent, context=[])
        return Crew(
            agents=[agent], tasks=[task],
            process=Process.sequential, memory=False, verbose=True,
        )

    def build_score_crew(
        self,
        description: str,
        instruments: list[str],
        measures: int,
        style: str,
        genre: str,
        key_signature: str,
        time_signature: str,
        tempo: int,
    ) -> Crew:
        agent = self._build_agent(genre, key_signature, time_signature, tempo,
                                  tools=[])  # ABC needs no tools; CompositionPrep emits JSON
        desc, exp = self._render_task("compose_score_task", {
            "description": description,
            "genre": genre,
            "style": style,
            "key_signature": key_signature,
            "time_signature": time_signature,
            "tempo": str(tempo),
            "instruments": ", ".join(instruments),
            "measures": str(measures),
        })
        task = Task(description=desc, expected_output=exp, agent=agent, context=[])
        return Crew(
            agents=[agent], tasks=[task],
            process=Process.sequential, memory=False, verbose=True,
        )

    def build_harmonize_crew(
        self,
        melody: str,
        style: str,
        genre: str,
        key_signature: str,
        time_signature: str,
        tempo: int,
    ) -> Crew:
        agent = self._build_agent(genre, key_signature, time_signature, tempo,
                                  tools=[])  # ABC needs no tools; CompositionPrep emits JSON
        desc, exp = self._render_task("harmonize_task", {
            "melody": melody,
            "style": style,
            "key_signature": key_signature,
            "time_signature": time_signature,
        })
        task = Task(description=desc, expected_output=exp, agent=agent, context=[])
        return Crew(
            agents=[agent], tasks=[task],
            process=Process.sequential, memory=False, verbose=True,
        )

    def build_analyze_crew(
        self,
        content: str,
        genre: str,
        key_signature: str,
        time_signature: str,
        tempo: int,
    ) -> Crew:
        agent = self._build_agent(genre, key_signature, time_signature, tempo)
        desc, exp = self._render_task("analyze_task", {
            "content": content,
        })
        task = Task(description=desc, expected_output=exp, agent=agent, context=[])
        return Crew(
            agents=[agent], tasks=[task],
            process=Process.sequential, memory=False, verbose=True,
        )


    # ABC key names differ slightly from the UI's "C major" / "A minor" wording.
    _ABC_MODE = {"major": "", "minor": "m", "dorian": "Dor", "mixolydian": "Mix"}

    @staticmethod
    def _abc_key(key_signature: str) -> str:
        """Turn "Bb major" / "A minor" into an ABC key field ("Bb", "Am")."""
        parts = (key_signature or "C major").strip().split()
        tonic = parts[0] if parts else "C"
        mode = parts[1].lower() if len(parts) > 1 else "major"
        return tonic + ComposerCrew._ABC_MODE.get(mode, "")

    def build_abc_crew(
        self,
        description: str,
        genre: str,
        key_signature: str,
        time_signature: str,
        tempo: int,
        measures: int = 16,
        style: str = "",
    ) -> Crew:
        """Build a Crew that composes in ABC notation.

        ABC replaces the note-by-note JSON the composer used to emit. Two
        reasons, and the second matters more than the first:

        * JSON arrived broken often enough to need ~500 lines of regex repair.
          ABC from the same local model parses first time.
        * JSON encodes nothing but pitches and durations, so the model had no
          way to express structure. ABC carries repeats and sections natively,
          is far more compact, and appears in training data as *music* rather
          than as arbitrary data — so the model has heard what good ABC looks
          like.

        Style is injected from genres.yaml plus the interface settings, so the
        request, the genre and the key/metre/tempo pickers all reach the prompt.
        """
        from researcher.composer.knowledge.loader import get_genre_block

        # No tools. CompositionPrep returns a JSON skeleton, which is exactly
        # the format this path replaced — handing it to the agent makes it reach
        # for the skeleton and emit JSON instead of ABC. Everything it needs is
        # already in the prompt.
        agent = self._build_agent(genre, key_signature, time_signature, tempo, tools=[])
        style_guide = get_genre_block(style or genre) or (
            "No specific style guidance available; write idiomatic, singable music."
        )
        # Units per bar with L:1/8 — the model cannot be trusted to derive this,
        # and getting it wrong is what produced 2-beat bars in a 3/4 piece.
        try:
            _num, _den = (time_signature or "4/4").split("/")[:2]
            units = int(int(_num) * 8 / int(_den))
        except Exception:
            units = 8

        desc, exp = self._render_task("compose_abc_task", {
            "units_per_bar": str(units),
            "description": description,
            "genre": genre,
            "style": style or genre,
            "style_guide": style_guide,
            "key_signature": key_signature,
            "key_signature_abc": self._abc_key(key_signature),
            "time_signature": time_signature,
            "tempo": str(tempo),
            "measures": str(measures),
        })
        task = Task(description=desc, expected_output=exp, agent=agent, context=[])
        return Crew(
            agents=[agent], tasks=[task],
            process=Process.sequential, memory=False, verbose=True,
        )


    @staticmethod
    def extract_abc(response: str) -> str | None:
        """Pull the ABC block out of a model response.

        Far simpler than the JSON path it replaces: ABC is line-oriented and
        self-delimiting, so recovering it needs a fence match and a header check
        rather than nineteen regex repairs.
        """
        m = re.search(r"```(?:abc)?\s*(.*?)```", response or "", re.DOTALL)
        abc = (m.group(1) if m else response or "").strip()
        # A valid tune needs at least an index and a key line.
        if not re.search(r"^X:", abc, re.M) or not re.search(r"^K:", abc, re.M):
            return None
        return abc

    @staticmethod
    def abc_to_musicxml(abc: str, tempo: int = 120) -> str | None:
        """Convert ABC to MusicXML via music21.

        music21 validates as it parses, so a malformed tune fails here with a
        real error instead of silently producing a broken score — which is what
        the old JSON path did until the repair layer papered over it.
        """
        try:
            from music21 import converter, tempo as m21tempo

            score = converter.parse(abc, format="abc")
            if tempo and not list(score.recurse().getElementsByClass(m21tempo.MetronomeMark)):
                score.insert(0, m21tempo.MetronomeMark(number=int(tempo)))
            out = score.write("musicxml")
            return pathlib.Path(out).read_text()
        except Exception as e:
            logger.warning("[Composer] ABC -> MusicXML failed: %s", e)
            return None

    @staticmethod
    def describe_abc(abc: str) -> dict:
        """Summarise a tune so the UI can show what was actually written."""
        try:
            from music21 import converter

            score = converter.parse(abc, format="abc")
            notes = list(score.recurse().notes)
            pitches = [n.pitch.midi for n in notes if hasattr(n, "pitch")]
            ks = score.recurse().getElementsByClass("KeySignature")
            ts = score.recurse().getElementsByClass("TimeSignature")
            return {
                "notes": len(notes),
                "measures": len(list(score.recurse().getElementsByClass("Measure"))),
                "key": str(ks[0]) if ks else "",
                "time": ts[0].ratioString if ts else "",
                "range_semitones": (max(pitches) - min(pitches)) if pitches else 0,
                "distinct_pitches": len(set(pitches)),
            }
        except Exception:
            return {}


    @staticmethod
    def validate_abc(abc: str, time_signature: str = "4/4") -> dict:
        """Check a tune for the faults music21 parses without complaint.

        Two failures were found in real output and neither raised anything:

        * Every bar of a 3/4 piece contained only 2 beats. ABC broken rhythm
          ("D>E") is easy to miscount, and a short bar parses fine — it just
          plays wrong.
        * The "melody" was a 12-note unbroken scale run: 90% stepwise in one
          direction. Technically valid, musically empty.

        Returns a report so the caller can retry or warn rather than shipping it.
        """
        report = {"ok": True, "issues": [], "short_bars": 0, "stepwise_pct": 0.0,
                  "longest_run": 0, "measures": 0, "notes": 0}
        try:
            from music21 import converter

            score = converter.parse(abc, format="abc")
        except Exception as e:
            report["ok"] = False
            report["issues"].append(f"does not parse: {e}")
            return report

        try:
            beats = 4.0
            if "/" in time_signature:
                num, den = time_signature.split("/")[:2]
                beats = float(num) * 4.0 / float(den)
        except Exception:
            beats = 4.0

        measures = list(score.recurse().getElementsByClass("Measure"))
        report["measures"] = len(measures)
        # Ignore pickup/final bars, which are legitimately short.
        for m in measures[1:-1] if len(measures) > 2 else measures:
            if abs(float(m.duration.quarterLength) - beats) > 0.01:
                report["short_bars"] += 1
        if report["short_bars"]:
            report["ok"] = False
            report["issues"].append(
                f"{report['short_bars']} bar(s) are not {beats:g} beats — "
                f"the piece will not play in {time_signature}"
            )

        pitches = [n.pitch.midi for n in score.recurse().notes if hasattr(n, "pitch")]
        report["notes"] = len(pitches)
        if len(pitches) > 4:
            steps = [pitches[i] - pitches[i - 1] for i in range(1, len(pitches))]
            stepwise = sum(1 for d in steps if abs(d) in (1, 2))
            report["stepwise_pct"] = round(100 * stepwise / len(steps), 1)
            import itertools

            report["longest_run"] = max(
                len(list(g)) for _, g in itertools.groupby(steps, key=lambda d: d > 0)
            )
            if report["longest_run"] >= 8:
                report["ok"] = False
                report["issues"].append(
                    f"a {report['longest_run']}-note unbroken run in one direction "
                    f"— that is a scale, not a melody"
                )
            if report["stepwise_pct"] >= 95:
                report["ok"] = False
                report["issues"].append(
                    f"{report['stepwise_pct']}% stepwise motion — no leaps at all"
                )
        return report


    # How many corrective passes to attempt. Each is a full generation, so the
    # cost is linear; 2 catches most faults without a long wait.
    COMPOSE_REFINE_ROUNDS = int(os.getenv("COMPOSER_REFINE_ROUNDS", "2"))

    def _units_per_bar(self, time_signature: str) -> int:
        try:
            num, den = (time_signature or "4/4").split("/")[:2]
            return int(int(num) * 8 / int(den))
        except Exception:
            return 8

    def build_abc_fix_crew(self, previous_abc: str, issues: list[str],
                           genre: str, key_signature: str, time_signature: str,
                           tempo: int) -> Crew:
        """Build a Crew that repairs a specific tune against specific faults."""
        agent = self._build_agent(genre, key_signature, time_signature, tempo, tools=[])
        desc, exp = self._render_task("fix_abc_task", {
            "previous_abc": previous_abc,
            "issues": "\n".join(f"- {i}" for i in issues),
            "time_signature": time_signature,
            "units_per_bar": str(self._units_per_bar(time_signature)),
        })
        task = Task(description=desc, expected_output=exp, agent=agent, context=[])
        return Crew(agents=[agent], tasks=[task],
                    process=Process.sequential, memory=False, verbose=True)

    def compose_abc(self, description: str, genre: str, key_signature: str,
                    time_signature: str, tempo: int, measures: int = 16,
                    style: str = "", progress=None) -> dict:
        """Compose, validate, and correct until the tune is musically sound.

        A language model writes valid ABC readily but not reliably GOOD music:
        a prompt that explicitly banned runs longer than five notes still
        produced a twelve-note scale. Instructions alone do not close that gap.

        So the faults are measured rather than hoped away, and fed back
        specifically — the same shape as the image critic. Returns the best
        attempt made, along with its report, rather than failing outright: a
        flawed tune the user can hear beats an error message.
        """
        def _say(msg: str):
            logger.info("[COMPOSER] %s", msg)
            if progress:
                try:
                    progress(msg)
                except Exception:
                    pass

        attempts: list[tuple[dict, str]] = []

        _say("🎼 Composing…")
        crew = self.build_abc_crew(description, genre, key_signature,
                                   time_signature, tempo, measures, style)
        abc = self.extract_abc(self.extract_raw(crew.kickoff()))
        if not abc:
            return {"abc": None, "musicxml": None, "report": None,
                    "error": "the model did not return ABC notation"}

        report = self.validate_abc(abc, time_signature)
        attempts.append((report, abc))

        for round_no in range(1, self.COMPOSE_REFINE_ROUNDS + 1):
            if report["ok"]:
                break
            _say(f"🎼 Fixing: {'; '.join(report['issues'])[:120]}")
            try:
                fix = self.build_abc_fix_crew(abc, report["issues"], genre,
                                              key_signature, time_signature, tempo)
                candidate = self.extract_abc(self.extract_raw(fix.kickoff()))
            except Exception as e:
                _say(f"🎼 Correction pass {round_no} failed ({type(e).__name__})")
                break
            if not candidate:
                _say(f"🎼 Correction pass {round_no} returned no ABC; keeping best so far")
                break
            cand_report = self.validate_abc(candidate, time_signature)
            attempts.append((cand_report, candidate))
            abc, report = candidate, cand_report

        # Keep whichever attempt has the fewest faults, not simply the last:
        # a correction pass can make things worse.
        best_report, best_abc = min(
            attempts, key=lambda ra: (not ra[0]["ok"], len(ra[0]["issues"]),
                                      ra[0]["short_bars"], ra[0]["longest_run"])
        )
        _say("🎼 " + ("Composition passed all checks"
                     if best_report["ok"]
                     else f"Best of {len(attempts)} attempts, remaining: "
                          f"{'; '.join(best_report['issues'])[:120]}"))
        musicxml = self.abc_to_musicxml(best_abc, tempo)
        bars_fixed = 0
        if not best_report["ok"] and best_report["short_bars"]:
            fixed_xml, bars_fixed = self.repair_bar_lengths(best_abc, time_signature)
            if fixed_xml:
                musicxml = fixed_xml
                _say(f"🎼 Re-barred to {bars_fixed} correct {time_signature} bars automatically")

        return {
            "abc": best_abc,
            "musicxml": musicxml,
            "report": best_report,
            "attempts": len(attempts),
            "bars_repaired": bars_fixed,
            "error": None,
        }


    @staticmethod
    def repair_bar_lengths(abc: str, time_signature: str = "4/4") -> tuple[str, int]:
        """Make every bar the correct length, deterministically.

        Three generations in a row produced wrong-length bars even when the
        prompt spelled out the arithmetic and a correction pass named the exact
        fault. Counting durations is mechanical, and the model is bad at it — so
        do it in code rather than asking a fourth time.

        The model's barlines are discarded and the note stream is re-barred at
        the correct metre, with notes that cross a barline split and tied. An
        earlier attempt lengthened the last note of each short bar instead; that
        does not reflow the measure, and left MORE bars wrong than it fixed.

        Returns ``(musicxml, bars)`` — MusicXML rather than ABC, because music21
        has already corrected the stream and re-serialising would risk losing it.
        """
        try:
            from music21 import converter, meter

            score = converter.parse(abc, format="abc")
            part = score.parts[0] if getattr(score, "parts", None) else score
            flat = part.flatten().notesAndRests.stream()
            ts = meter.TimeSignature(time_signature or "4/4")
            # makeMeasures alone leaves notes straddling barlines; makeTies
            # splits them so every bar lands exactly on the metre.
            rebarred = flat.makeMeasures(ts).makeTies()
            bars = len(list(rebarred.getElementsByClass("Measure")))
            out = rebarred.write("musicxml")
            return pathlib.Path(out).read_text(), bars
        except Exception as e:
            logger.warning("[Composer] bar repair failed: %s", e)
            return "", 0


    @staticmethod
    def musicxml_to_midi(musicxml: str, tempo: int = 120) -> bytes | None:
        """Render a MusicXML score to MIDI so it can be played immediately.

        A score you cannot hear is hard to judge. MusicXML needs MuseScore or
        similar to play; MIDI plays in the browser and in every DAW, which
        closes the loop between composing and listening.
        """
        import tempfile as _tempfile

        try:
            from music21 import converter, tempo as m21tempo

            score = converter.parse(musicxml, format="musicxml")
            if tempo and not list(score.recurse().getElementsByClass(m21tempo.MetronomeMark)):
                score.insert(0, m21tempo.MetronomeMark(number=int(tempo)))
            with _tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
                path = tmp.name
            score.write("midi", fp=path)
            data = pathlib.Path(path).read_bytes()
            pathlib.Path(path).unlink(missing_ok=True)
            return data
        except Exception as e:
            logger.warning("[Composer] MIDI render failed: %s", e)
            return None


    @staticmethod
    def midi_to_audio(midi: bytes) -> bytes | None:
        """Render MIDI to WAV so the browser can simply play it.

        Browsers cannot play MIDI natively, and shipping a JavaScript synth plus
        a soundfont would be megabytes. timidity with the system soundfont is
        already present here and gives real instrument sounds rather than beeps.

        Returns None when timidity is unavailable — the caller falls back to
        offering the MIDI file for download.
        """
        import subprocess
        import tempfile as _tempfile

        mid = wav = None
        try:
            with _tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
                f.write(midi)
                mid = f.name
            wav = _tempfile.mktemp(suffix=".wav")
            proc = subprocess.run(
                ["timidity", mid, "-Ow", "-o", wav],
                capture_output=True, timeout=180,
            )
            if proc.returncode != 0 or not pathlib.Path(wav).exists():
                logger.warning("[Composer] timidity failed: %s",
                               proc.stderr.decode(errors="replace")[:200])
                return None
            data = pathlib.Path(wav).read_bytes()
            return data if len(data) > 44 else None      # 44 bytes = empty WAV header
        except FileNotFoundError:
            logger.info("[Composer] timidity not installed; audio preview unavailable")
            return None
        except Exception as e:
            logger.warning("[Composer] audio render failed: %s", e)
            return None
        finally:
            for path in (mid, wav):
                if path:
                    pathlib.Path(path).unlink(missing_ok=True)

    def extract_raw(self, result) -> str:
        return result.raw if hasattr(result, "raw") else str(result)

    def extract_musicxml(self, response: str,
                         key: str = "C major",
                         time_signature: str = "4/4",
                         tempo: int = 120) -> str | None:
        """Extract a score from a model response, whatever format it used.

        Priority:
          1. ABC  → music21, with validation and deterministic re-barring
          2. JSON → build_musicxml()   (legacy)
          3. XML  → fix_musicxml()     (legacy)

        ABC comes first because the agent now writes ABC everywhere. Missing
        this branch is what sent raw ABC to the chat with no playable score
        attached — every path that shows a score must go through here.
        """
        abc = self.extract_abc(response)
        if abc:
            report = self.validate_abc(abc, time_signature)
            # Re-bar when the model miscounted; it reliably does. This is the
            # difference between a score that plays and one that only parses.
            if not report["ok"] and report.get("short_bars"):
                fixed, bars = self.repair_bar_lengths(abc, time_signature)
                if fixed:
                    logger.info("[Composer] re-barred to %d correct bars", bars)
                    return fixed
            xml = self.abc_to_musicxml(abc, tempo)
            if xml:
                return xml

        # Try JSON extraction (legacy path)
        xml = self._extract_json_score(response, key, time_signature, tempo)
        if xml:
            return xml

        # Fallback: try raw XML extraction (legacy)
        # A response may already CONTAIN finished MusicXML — the long-form path
        # assembles the score itself and ships it directly. Without this branch
        # every downstream consumer (Play, MIDI, Download) searched for ABC,
        # found none, and reported that there was no score in a response that
        # was nothing but score.
        m = re.search(r"```(?:musicxml|xml)\s*(.*?)```", response, re.DOTALL)
        if m and "<score-partwise" in m.group(1):
            return m.group(1).strip()
        if "<score-partwise" in (response or ""):
            start = response.index("<?xml") if "<?xml" in response else response.index("<score-partwise")
            end = response.rfind("</score-partwise>")
            if end > start:
                return response[start:end + len("</score-partwise>")]

        m = re.search(r"```xml\s*(.*?)```", response, re.DOTALL)
        if m:
            raw_xml = m.group(1).strip()
            if "<score-partwise" in raw_xml:
                return fix_musicxml(raw_xml)
        return None

    @staticmethod
    def _extract_json_score(response: str, key: str = "C major",
                            time_signature: str = "4/4",
                            tempo: int = 120) -> str | None:
        """Extract JSON score from response — tries fenced JSON, then bare JSON.

        Handles common LLM issues: hallucinated text spliced into JSON,
        truncated output (unclosed braces/brackets), trailing commas.
        """
        raw = None
        # Try ```json ... ``` fences first
        m = re.search(r"```json\s*(.*?)```", response, re.DOTALL)
        if m:
            raw = m.group(1).strip()
        if raw is None:
            # Try bare JSON: find opening { before "parts" key, take to end
            m = re.search(r'\{[^{}]*"parts"\s*:', response)
            if m:
                raw = response[m.start():]
        if not raw:
            return None

        # First attempt: parse as-is
        try:
            score = json.loads(raw)
        except json.JSONDecodeError:
            # Apply repairs and retry
            repaired = ComposerCrew._repair_json(raw)
            try:
                score = json.loads(repaired)
            except json.JSONDecodeError:
                # Last resort: use json_repair library
                try:
                    from json_repair import repair_json
                    repaired2 = repair_json(repaired, return_objects=False)
                    score = json.loads(repaired2)
                except Exception as e:
                    logger.warning("[Composer] Failed to parse JSON score after all repairs: %s", e)
                    return None

        if not isinstance(score, dict) or 'parts' not in score:
            logger.warning("[Composer] JSON score missing 'parts' key")
            return None

        try:
            return build_musicxml(score, key=key,
                                 time_signature=time_signature,
                                 tempo=tempo)
        except Exception as e:
            logger.warning("[Composer] Failed to build MusicXML from JSON: %s", e)
            return None

    @staticmethod
    def _repair_json(raw: str) -> str:
        """Repair common LLM JSON corruption.

        1. Remove hallucinated text lines (no colon, no structural chars).
        2. Remove trailing commas before } or ].
        3. Trim to last completed closing delimiter.
        4. Close any remaining unclosed braces/brackets.
        """
        # ── 1. Strip hallucinated text lines ──
        lines = raw.split('\n')
        cleaned: list[str] = []
        for line in lines:
            s = line.strip()
            if not s:
                continue
            # Keep lines starting with structural JSON chars
            if s[0] in '{}[]':
                cleaned.append(line)
                continue
            # Keep lines containing key:value pairs (colon present)
            if ':' in s:
                cleaned.append(line)
                continue
            # Skip everything else (hallucinated text, stray quotes, etc.)
            logger.debug("[Composer] Stripping hallucinated JSON line: %r", s)
        result = '\n'.join(cleaned)

        # ── 1b. Remove intra-line junk spliced into structural chars ──
        # e.g. "{ piek" → "{", "} blah" → "}"
        # Only on lines that are purely structural (open/close brace/bracket
        # optionally followed by comma) with junk words after or before.
        result = re.sub(
            r'\{\s*[a-zA-Z_][a-zA-Z0-9_]*(?:\s+[a-zA-Z_][a-zA-Z0-9_]*)*\s*\n',
            '{\n',
            result,
        )

        # ── 2. Remove trailing commas before } or ] ──
        result = re.sub(r',\s*([}\]])', r'\1', result)

        # ── 3. Trim back to last completed closing delimiter ──
        last_close = max(result.rfind('}'), result.rfind(']'))
        if last_close >= 0:
            result = result[:last_close + 1]

        # ── 4. Close truncated JSON via delimiter stack ──
        stack: list[str] = []
        in_string = False
        escape = False
        for ch in result:
            if escape:
                escape = False
                continue
            if ch == '\\' and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                stack.append('}')
            elif ch == '[':
                stack.append(']')
            elif ch in '}]' and stack and stack[-1] == ch:
                stack.pop()
        if stack:
            result = result.rstrip().rstrip(',')
            result += ''.join(stack[::-1])

        return result
