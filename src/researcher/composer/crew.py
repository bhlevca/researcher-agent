"""ComposerCrew — CrewAI orchestrator for the Music Composer.

Builds agents and tasks for composition, arrangement, harmonization,
and analysis. Follows the same patterns as TutorCrew.
"""

import os
import re
import json
import logging
from pathlib import Path

from crewai import Agent, Crew, Process, Task, LLM
from crewai.agent.planning_config import PlanningConfig

from researcher.config import probe_model_capabilities, TOOL_CAPABLE_MODEL
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
                "1. Call CompositionPrep tool to get chords, guidelines, and JSON skeleton.\\n"
                "2. Fill the skeleton with composed notes and output as Final Answer (no tool needed).\\n\\n"
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
            tools=COMPOSER_TOOLS,
            llm=self._llm,
            verbose=True,
            max_iter=int(self._llm_params.get("max_iter", 12)),
            max_retry_limit=2,
            respect_context_window=True,
            planning_config=planning,
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
        agent = self._build_agent(genre, key_signature, time_signature, tempo)
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
        agent = self._build_agent(genre, key_signature, time_signature, tempo)
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
        agent = self._build_agent(genre, key_signature, time_signature, tempo)
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

    def extract_raw(self, result) -> str:
        return result.raw if hasattr(result, "raw") else str(result)

    def extract_musicxml(self, response: str,
                         key: str = "C major",
                         time_signature: str = "4/4",
                         tempo: int = 120) -> str | None:
        """Extract score from response — tries JSON builder first, falls back to XML.

        Priority:
          1. ```json block → build_musicxml()
          2. ```xml block → fix_musicxml() (legacy fallback)
        """
        # Try JSON extraction first
        xml = self._extract_json_score(response, key, time_signature, tempo)
        if xml:
            return xml

        # Fallback: try raw XML extraction (legacy)
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
