"""TutorCrew — CrewAI orchestrator for the Language Tutor.

Builds agents and tasks for conversation, lessons, quizzes, and appraisal.
Follows the same patterns as ResearchCrew (crew.py) including the Ollama
monkey-patches already applied at import time by researcher.crew.
"""

import os
import re
import json
import random
import logging
from pathlib import Path

from crewai import Agent, Crew, Process, Task, LLM
from crewai.agent.planning_config import PlanningConfig

from researcher.config import probe_model_capabilities, TOOL_CAPABLE_MODEL
from researcher.tutor.tools import TUTOR_TOOLS

logger = logging.getLogger(__name__)

# Task YAML is loaded manually since we don't use @CrewBase for this class
# (the tutor builds dynamic tasks, not static ones from YAML).
_CONFIG_DIR = Path(__file__).parent / "config"


def _load_yaml(name: str) -> dict:
    import yaml

    with open(_CONFIG_DIR / name) as f:
        return yaml.safe_load(f)


class TutorCrew:
    """Language Tutor agent orchestrator.

    Unlike ResearchCrew which uses ``@CrewBase`` with static YAML tasks,
    TutorCrew dynamically builds tasks from YAML templates with runtime
    variable substitution for the target language, level, etc.
    """

    DEFAULT_LLM_PARAMS: dict = {
        "temperature": 0.6,  # slightly more creative for natural conversation
        "top_k": 40,
        "top_p": 0.9,
        "min_p": 0.0,
        "seed": 0,
        "num_predict": 4096,
        "num_ctx": 32768,
        "repeat_penalty": 1.1,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "max_iter": 15,
        "planning_max_attempts": 3,
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
        """Create an LLM instance with Ollama-compatible options."""
        is_thinking_model = any(k in model.lower() for k in ("qwen3", "deepseek-r1"))
        seed = int(params.get("seed", 0))
        ollama_options = {
            "num_ctx": int(params.get("num_ctx", 32768)),
            "num_gpu": 99,
            "num_predict": int(params.get("num_predict", 4096)),
            "top_k": int(params.get("top_k", 40)),
            "top_p": float(params.get("top_p", 0.9)),
            "min_p": float(params.get("min_p", 0.0)),
            "repeat_penalty": float(params.get("repeat_penalty", 1.1)),
        }
        if seed > 0:
            ollama_options["seed"] = seed
        extra_body: dict = {"options": ollama_options}
        if is_thinking_model:
            extra_body["chat_template_kwargs"] = {"enable_thinking": True}
        return LLM(
            model=model,
            base_url="http://localhost:11434",
            temperature=float(params.get("temperature", 0.6)),
            frequency_penalty=float(params.get("frequency_penalty", 0.0)),
            presence_penalty=float(params.get("presence_penalty", 0.0)),
            extra_body=extra_body,
        )

    @staticmethod
    def _make_planning_llm(model: str) -> LLM:
        """Lightweight LLM for the planning phase."""
        return LLM(
            model=model,
            base_url="http://localhost:11434",
            temperature=0.2,
            extra_body={
                "options": {
                    "num_ctx": 8192,
                    "num_gpu": 99,
                    "num_predict": 2048,
                },
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )

    def _build_agent(
        self,
        target_lang: str,
        native_lang: str,
        level: str,
    ) -> Agent:
        """Build the language tutor agent with language-specific config."""
        agent_cfg = self._agents_config["language_tutor"]

        # Substitute template variables in the YAML config
        subs = {
            "{target_lang}": target_lang,
            "{native_lang}": native_lang,
            "{level}": level,
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
                "[TUTOR] %s lacks tool support — planning with %s",
                self._model_name,
                planning_model,
            )
        planning_llm = self._make_planning_llm(planning_model)
        planning_llm._allow_fc = True  # allow function calling for planning

        planning = PlanningConfig(
            llm=planning_llm,
            max_attempts=int(self._llm_params.get("planning_max_attempts", 3)),
            max_steps=4,
            reasoning_effort="medium",
            plan_prompt=(
                "You are {role}.\n\n"
                "Task: {description}\n\n"
                "Expected output: {expected_output}\n\n"
                "Available tools: {tools}\n\n"
                "Create a CONCISE plan with 1-{max_steps} steps. "
                "For each step write ONE line: the action.\n"
                "Most language tasks can be completed from your own knowledge "
                "— no internet search needed.\n"
                "End with: READY: I am ready to execute the task.\n"
            ),
        )

        backstory = _sub(agent_cfg["backstory"])

        # For models without native tool support, add explicit ReAct format
        if not self._model_caps["supports_tools"]:
            backstory += (
                "\n\n===== CRITICAL: TOOL CALLING FORMAT =====\n"
                "You MUST use EXACTLY this text format to call tools:\n\n"
                "Thought: I need to use a tool.\n"
                "Action: <tool_name>\n"
                'Action Input: {"param": "value"}\n\n'
                "Then WAIT for the Observation (the tool result). "
                "Do NOT invent or fabricate the tool response yourself. "
                "Do NOT use ```tool_code``` or any other format.\n"
                "After receiving the Observation, write:\n\n"
                "Thought: I now have the answer.\n"
                "Final Answer: [your response]\n"
            )

        return Agent(
            role=_sub(agent_cfg["role"]),
            goal=_sub(agent_cfg["goal"]),
            backstory=backstory,
            tools=TUTOR_TOOLS,
            llm=self._llm,
            verbose=True,
            max_iter=int(self._llm_params.get("max_iter", 15)),
            max_retry_limit=2,
            respect_context_window=True,
            planning_config=planning,
        )

    def _render_task(self, task_key: str, variables: dict) -> tuple[str, str]:
        """Render a task template from YAML with variable substitution.

        Returns (description, expected_output).
        """
        task_cfg = self._tasks_config[task_key]
        desc = task_cfg["description"]
        exp = task_cfg["expected_output"]

        for k, v in variables.items():
            placeholder = "{" + k + "}"
            desc = desc.replace(placeholder, str(v))
            exp = exp.replace(placeholder, str(v))

        return desc, exp

    # ------------------------------------------------------------------
    # Public build methods — one per mode
    # ------------------------------------------------------------------

    def build_conversation_crew(
        self,
        message: str,
        context: str,
        target_lang: str,
        native_lang: str,
        level: str,
    ) -> Crew:
        """Build a Crew for a conversation turn."""
        agent = self._build_agent(target_lang, native_lang, level)
        desc, exp = self._render_task(
            "conversation_task",
            {
                "target_lang": target_lang,
                "native_lang": native_lang,
                "level": level,
                "context": context,
                "message": message,
            },
        )

        task = Task(description=desc, expected_output=exp, agent=agent, context=[])
        return Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            memory=False,
            verbose=True,
        )

    def build_lesson_crew(
        self,
        topic: str,
        lesson_type: str,
        target_lang: str,
        native_lang: str,
        level: str,
    ) -> Crew:
        """Build a Crew for generating a lesson plan."""
        agent = self._build_agent(target_lang, native_lang, level)
        desc, exp = self._render_task(
            "lesson_task",
            {
                "topic": topic,
                "lesson_type": lesson_type,
                "target_lang": target_lang,
                "native_lang": native_lang,
                "level": level,
            },
        )

        task = Task(description=desc, expected_output=exp, agent=agent, context=[])
        return Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            memory=False,
            verbose=True,
        )

    # Per quiz-type instructions injected at the top of the quiz prompt so the
    # LLM cannot miss them.  Keys match the values sent from the frontend.
    _QUIZ_TYPE_INSTRUCTIONS: dict[str, str] = {
        "multiple_choice": (
            'You MUST generate ONLY "vocabulary" or "grammar" type questions. '
            'Every question MUST have "options": an array of exactly 4 choices. '
            "Do NOT generate matching, reorder, cloze, translation, or listening questions."
        ),
        "fill_blank": (
            'You MUST generate ONLY "cloze" type questions. '
            "Each question shows the student a {target_lang} sentence with EXACTLY ONE word "
            "replaced by ___ , PLUS a {native_lang} translation of the whole sentence in "
            "parentheses so the student knows exactly what word to write. "
            "The blank MUST test a specific grammar point "
            "(verb conjugation, article, preposition, adjective agreement, etc.). "
            "Format: "
            '"question": "<{target_lang} sentence with ___>  (<{native_lang} full translation with the target word also shown in brackets>)" '
            "Example (French/English): "
            '"question": "Je ___ au cinéma le vendredi.  (I [go] to the cinema on Fridays.)", '
            '"correct_answer": "vais" '
            'The "correct_answer" field MUST contain ONLY the missing word — never the full sentence. '
            '"options": null. '
            "Do NOT generate vocabulary, grammar, matching, reorder, translation, or listening questions."
        ),
        "translation": (
            'You MUST generate ONLY "translation" type questions with "options": null. '
            "Provide a sentence and ask the student to translate it to the other language. "
            "Do NOT generate any other question type."
        ),
        "matching": (
            'You MUST generate ONLY "matching" type questions. '
            'Each question needs "pairs": an array of 4–6 objects '
            'like {"left": "<target_lang word>", "right": "<native_lang translation>"}. '
            '"options": null. '
            '"correct_answer": pairs joined as "word1=trans1; word2=trans2". '
            "Do NOT generate vocabulary, grammar, translation, reorder, cloze, or listening questions."
        ),
        "reorder": (
            'You MUST generate ONLY "reorder" type questions. '
            "Each question asks the student to put shuffled words in the right order. "
            "Fields REQUIRED for every reorder question:\n"
            '  "question": A SHORT HINT in {native_lang} telling the student what to say '
            '(e.g. "Translate: I eat an apple every day"). '
            "Do NOT put the target-language sentence here.\n"
            '  "correct_answer": the COMPLETE, GRAMMATICALLY CORRECT target-language sentence '
            "as a single string. Every word that appears in the sentence MUST be here — "
            "including articles (le/la/les/un/une/du/de la), pronouns (je/tu/il/nous/vous/ils), "
            "prepositions (à/de/en/au/aux), and conjunctions. "
            'Example: "Je mange une pomme chaque jour".\n'
            '  "words": copy correct_answer split into a SHUFFLED array of INDIVIDUAL words. '
            "This array must contain EXACTLY the same words as correct_answer, nothing more, "
            'nothing less. Example: ["pomme","Je","chaque","mange","une","jour"].\n'
            '  "options": null.\n'
            "EXAMPLE (French lesson, native=English):\n"
            '{"type":"reorder","id":0,"question":"Translate: I eat an apple every day",'
            '"words":["pomme","Je","chaque","mange","une","jour"],'
            '"correct_answer":"Je mange une pomme chaque jour","options":null}\n'
            "Do NOT generate any other question type."
        ),
        "listening": (
            'You MUST generate ONLY "listening" type questions. '
            'Each question needs "audio_text": the exact phrase the student will hear via TTS. '
            '"correct_answer" must be identical to "audio_text". '
            '"options": null. '
            "Do NOT generate any other question type."
        ),
        "cloze": (
            'You MUST generate ONLY "cloze" type questions. '
            "Each question is a {target_lang} sentence with EXACTLY ONE word replaced by ___ . "
            "The blank MUST be unambiguous — choose a position where only ONE specific word "
            "is correct (e.g. a specific conjugated verb form, a specific article or "
            "preposition required by grammar rules). "
            "CRITICAL RULES:\n"
            "  1. correct_answer MUST be the grammatically correct word for the SUBJECT shown "
            "in the sentence. Double-check subject/verb agreement before writing the answer.\n"
            "     Examples: 'Je ___ un chat' → correct_answer=\"ai\" (not \"a\", not \"as\")\n"
            "              'Elle ___ un livre' → correct_answer=\"lit\"\n"
            "              'Nous ___ au parc' → correct_answer=\"allons\"\n"
            "  2. Apply elision rules: when the blank answer starts with a vowel and the "
            "preceding pronoun ends in 'e' (Je, Le, La, De, Ne, Que), use the elided form "
            "with apostrophe instead.\n"
            "     Examples: 'Je ai' → write as \"J'___ un chat\"\n"
            "              'Je ai' is NEVER correct — use \"J'\" before vowels\n"
            "              'Je aime' → \"J'___ la musique\"\n"
            "              'Je habite' → \"J'___ à Paris\"\n"
            "  3. When options are provided, make sure correct_answer matches EXACTLY one "
            "of the options, and that the other options are WRONG for the given subject.\n"
            "You MUST include a {native_lang} hint in parentheses after the sentence "
            "showing what the blank is testing. "
            "Format: "
            '"question": "<sentence with ___>  (Hint: <what grammar/word is being tested>)" '
            "Example: "
            '"question": "Elle ___ ses devoirs tous les soirs.  (Hint: conjugate \'faire\' for elle)", '
            '"correct_answer": "fait" '
            'The "correct_answer" MUST be ONLY the missing word — never the full sentence. '
            "Do NOT generate any other question type."
        ),
        "mixed": (
            "Mix question types freely using any combination of: "
            '"vocabulary", "grammar", "translation", "matching", "reorder", '
            '"listening", "cloze". Use at least 3 different types.'
        ),
    }

    # Types allowed per quiz_type key (used for backend filtering)
    _QUIZ_ALLOWED_TYPES: dict[str, set[str]] = {
        "multiple_choice": {"vocabulary", "grammar"},
        "fill_blank": {"cloze"},
        "translation": {"translation"},
        "matching": {"matching"},
        "reorder": {"reorder"},
        "listening": {"listening"},
        "cloze": {"cloze"},
        "mixed": {
            "vocabulary",
            "grammar",
            "translation",
            "matching",
            "reorder",
            "listening",
            "cloze",
        },
    }

    def build_quiz_crew(
        self,
        quiz_type: str,
        num_questions: int,
        target_lang: str,
        native_lang: str,
        level: str,
        vocabulary_context: str = "",
        lesson_context: str = "",
    ) -> Crew:
        """Build a Crew for generating a quiz."""
        quiz_type_instruction = self._QUIZ_TYPE_INSTRUCTIONS.get(
            quiz_type, self._QUIZ_TYPE_INSTRUCTIONS["mixed"]
        )
        agent = self._build_agent(target_lang, native_lang, level)
        desc, exp = self._render_task(
            "quiz_task",
            {
                "quiz_type": quiz_type,
                "quiz_type_instruction": quiz_type_instruction,
                "num_questions": str(num_questions),
                "target_lang": target_lang,
                "native_lang": native_lang,
                "level": level,
                "vocabulary_context": vocabulary_context,
                "lesson_context": lesson_context,
            },
        )

        task = Task(description=desc, expected_output=exp, agent=agent, context=[])
        return Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            memory=False,
            verbose=True,
        )

    def filter_quiz_questions(
        self, questions: list[dict], quiz_type: str
    ) -> list[dict]:
        """Filter and repair questions to match the requested quiz_type.

        1. Keep only questions whose type is in the allowed set.
        2. Apply per-type structural repairs to every kept question.
        3. If nothing survived filtering, coerce all questions to the target type.
        """
        allowed = self._QUIZ_ALLOWED_TYPES.get(quiz_type)
        if allowed is None:
            return questions  # unknown type — pass through

        filtered = [q for q in questions if q.get("type") in allowed]

        # ── Per-type structural repair (runs for ALL questions, including mixed) ──
        for q in filtered:
            qtype = q.get("type")

            if qtype == "cloze":
                if "___" not in q.get("question", ""):
                    ans = q.get("correct_answer", "")
                    sentence = q.get("question", "")
                    if ans and ans in sentence:
                        q["question"] = sentence.replace(ans, "___", 1)
                    elif ans and " " not in ans:
                        # single-word answer — blank the last word of correct_answer
                        pass  # correct_answer already is just the word
                    elif ans:
                        parts = ans.split()
                        # blank the last word
                        q["question"] = " ".join(parts[:-1]) + " ___"
                        q["correct_answer"] = parts[-1]

                # ── French elision repair ──────────────────────────────────────
                # If the question has "Je ___" and the correct_answer starts with
                # a vowel, the sentence should use “J’___” (elision).
                # Also fix options if present.
                _ELISION_PAIRS = [
                    (re.compile(r"\bJe ___"), "J'___"),
                    (re.compile(r"\bje ___"), "j'___"),
                    (re.compile(r"\bLe ___"), "L'___"),
                    (re.compile(r"\ble ___"), "l'___"),
                    (re.compile(r"\bLa ___"), "L'___"),
                    (re.compile(r"\bla ___"), "l'___"),
                    (re.compile(r"\bDe ___"), "D'___"),
                    (re.compile(r"\bde ___"), "d'___"),
                    (re.compile(r"\bNe ___"), "N'___"),
                    (re.compile(r"\bne ___"), "n'___"),
                    (re.compile(r"\bQue ___"), "Qu'___"),
                    (re.compile(r"\bque ___"), "qu'___"),
                ]
                correct = q.get("correct_answer", "")
                if correct and correct[:1].lower() in "aeiouh":
                    question = q.get("question", "")
                    for pat, repl in _ELISION_PAIRS:
                        new_q = pat.sub(repl, question)
                        if new_q != question:
                            q["question"] = new_q
                            break

            elif qtype == "reorder":
                # Always derive words from correct_answer to guarantee all
                # function words (articles, pronouns, prepositions) are present.
                # The LLM-provided words field is unreliable — it often omits
                # words like "Je", "du", "les", "de la", etc.
                src = (q.get("correct_answer") or "").strip()
                if src:
                    parts = src.split()
                    random.shuffle(parts)
                    q["words"] = parts
                else:
                    # Fallback only when correct_answer is empty
                    words = q.get("words")
                    if isinstance(words, str) and words.strip():
                        words = words.split()
                    if words and isinstance(words, list):
                        parts = list(words)
                        random.shuffle(parts)
                        q["words"] = parts

            elif qtype == "matching":
                pairs = q.get("pairs")
                # Try to rebuild pairs from correct_answer: "word1=trans1; word2=trans2"
                if not pairs or not isinstance(pairs, list) or len(pairs) == 0:
                    correct = q.get("correct_answer", "")
                    rebuilt = []
                    if correct and "=" in correct:
                        for item in re.split(r"[;,]\s*", correct):
                            if "=" in item:
                                left, _, right = item.partition("=")
                                if left.strip() and right.strip():
                                    rebuilt.append(
                                        {"left": left.strip(), "right": right.strip()}
                                    )
                    if rebuilt:
                        q["pairs"] = rebuilt
                else:
                    # Normalize all pairs to {left, right} canonical format
                    norm = []
                    for p in pairs:
                        if isinstance(p, dict):
                            left = (
                                p.get("left") or p.get("term") or p.get("word") or ""
                            )
                            right = (
                                p.get("right")
                                or p.get("definition")
                                or p.get("translation")
                                or p.get("meaning")
                                or ""
                            )
                            if not left:
                                vals = list(p.values())
                                left = str(vals[0]) if vals else ""
                                right = str(vals[1]) if len(vals) > 1 else right
                            if left and right:
                                norm.append(
                                    {
                                        "left": str(left).strip(),
                                        "right": str(right).strip(),
                                    }
                                )
                        elif isinstance(p, str):
                            m = re.match(r"^(.+?)\s*[=:\-]\s*(.+)$", p)
                            if m:
                                norm.append(
                                    {
                                        "left": m.group(1).strip(),
                                        "right": m.group(2).strip(),
                                    }
                                )
                    if norm:
                        q["pairs"] = norm
                        # Build correct_answer from normalized pairs when missing
                        if not q.get("correct_answer"):
                            q["correct_answer"] = "; ".join(
                                f"{p['left']}={p['right']}" for p in norm
                            )

        # ── If nothing matched, coerce everything to the target type ──
        if not filtered:
            logger.warning(
                "[TUTOR] quiz filter removed all questions for type=%s — coercing",
                quiz_type,
            )
            work_list = list(questions)
        else:
            work_list = filtered

        # ── Dedicated-mode full coercion ──
        if quiz_type == "reorder":
            for q in work_list:
                q["type"] = "reorder"
                # Always derive from correct_answer for completeness
                src = (q.get("correct_answer") or "").strip()
                if src:
                    parts = src.split()
                    random.shuffle(parts)
                    q["words"] = parts

        elif quiz_type in ("cloze", "fill_blank"):
            for q in work_list:
                q["type"] = "cloze"
                if "___" not in q.get("question", ""):
                    correct = q.get("correct_answer", "")
                    sentence = q.get("question", "")
                    if correct and correct in sentence:
                        q["question"] = sentence.replace(correct, "___", 1)
                    elif correct:
                        parts = correct.split()
                        if len(parts) > 1:
                            q["question"] = " ".join(parts[:-1]) + " ___"
                            q["correct_answer"] = parts[-1]

        return work_list

    def build_appraisal_crew(
        self,
        target_lang: str,
        native_lang: str,
        level: str,
        stats: str,
        recent_messages: str,
    ) -> Crew:
        """Build a Crew for student appraisal."""
        agent = self._build_agent(target_lang, native_lang, level)
        desc, exp = self._render_task(
            "appraisal_task",
            {
                "target_lang": target_lang,
                "native_lang": native_lang,
                "level": level,
                "stats": stats,
                "recent_messages": recent_messages,
            },
        )

        task = Task(description=desc, expected_output=exp, agent=agent, context=[])
        return Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            memory=False,
            verbose=True,
        )

    def extract_raw(self, result) -> str:
        """Extract the raw text from a CrewAI result."""
        return result.raw if hasattr(result, "raw") else str(result)

    @staticmethod
    def extract_vocabulary_from_response(response: str) -> list[dict]:
        """Parse vocabulary tables from a tutor response.

        Looks for markdown tables with columns like Word | Translation | Part of Speech.
        Returns a list of dicts suitable for batch vocabulary insertion.
        """
        entries = []
        # Match markdown table rows (skip header and separator rows)
        table_re = re.compile(
            r"^\|([^|]+)\|([^|]+)\|([^|]+)\|",
            re.MULTILINE,
        )
        for m in table_re.finditer(response):
            word = m.group(1).strip()
            translation = m.group(2).strip()
            pos = m.group(3).strip()
            # Skip header rows and separator rows
            if (
                word.lower() in ("word", "word/phrase", "---", "")
                or set(word) <= {"-", " ", ":"}
                or set(translation) <= {"-", " ", ":"}
            ):
                continue
            entries.append(
                {
                    "word": word,
                    "translation": translation,
                    "part_of_speech": pos,
                    "context": "",
                    "phonetic": "",
                }
            )
        return entries

    @staticmethod
    def extract_quiz_json(response: str) -> list[dict] | None:
        """Extract a JSON quiz array from the tutor's response.

        Looks for ```json fenced blocks first, then tries the whole response.
        Returns the parsed list of question objects, or None on failure.
        """
        # Try fenced JSON blocks
        fenced = re.findall(r"```json\s*([\s\S]*?)```", response)
        for block in fenced:
            try:
                data = json.loads(block.strip())
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                continue

        # Try the whole response as JSON
        try:
            data = json.loads(response.strip())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

        return None

    # ------------------------------------------------------------------
    # Semantic answer grading (Phase 2)
    # ------------------------------------------------------------------

    async def grade_answers(
        self,
        questions: list[dict],
        student_answers: list[dict],
        target_lang: str,
        native_lang: str,
        level: str,
    ) -> list[dict]:
        """Grade quiz answers using LLM semantic evaluation.

        Returns a list of graded dicts with score (0.0–1.0), is_correct, and feedback.
        Falls back to exact-match grading on any LLM/parse error.
        """
        import litellm

        # Build the combined Q&A payload for the prompt
        qa_list = []
        for ans in student_answers:
            qid = ans.get("question_id", -1)
            student_answer = str(ans.get("answer", "")).strip()
            if 0 <= qid < len(questions):
                q = questions[qid]
                qa_list.append(
                    {
                        "question_id": qid,
                        "question": q.get("question", ""),
                        "correct_answer": q.get("correct_answer", ""),
                        "student_answer": student_answer,
                    }
                )

        if not qa_list:
            return []

        qa_json = json.dumps(qa_list, ensure_ascii=False, indent=2)
        desc, _ = self._render_task(
            "grade_task",
            {
                "level": level,
                "target_lang": target_lang,
                "native_lang": native_lang,
                "qa_json": qa_json,
            },
        )

        try:
            response = await litellm.acompletion(
                model=self._model_name,
                base_url="http://localhost:11434",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a language teacher grading student answers. "
                            "Be fair, accept semantically equivalent answers, "
                            "and return only the requested JSON."
                        ),
                    },
                    {"role": "user", "content": desc},
                ],
                temperature=0.0,
                max_tokens=2048,
                extra_body={
                    "options": {"num_ctx": 8192, "num_gpu": 99},
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            raw = (response.choices[0].message.content or "").strip()
            # Strip any <think>…</think> blocks from reasoning models
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        except Exception as exc:
            logger.warning(
                "[TUTOR] semantic grading LLM call failed: %s — falling back", exc
            )
            return self._exact_match_grade(student_answers, questions)

        return self._parse_grading_json(raw, student_answers, questions)

    @staticmethod
    def _parse_grading_json(
        raw: str, student_answers: list[dict], questions: list[dict]
    ) -> list[dict]:
        """Parse grading JSON from LLM output; fall back to exact match on error."""
        # Try ```json fenced block first
        m = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
        text = m.group(1) if m else raw

        # Try bare JSON array if fenced block not found
        if not m:
            arr = re.search(r"\[.*\]", raw, re.DOTALL)
            if arr:
                text = arr.group(0)

        try:
            graded = json.loads(text)
            if isinstance(graded, list):
                result = []
                for g in graded:
                    qid = int(g.get("question_id", -1))
                    score = float(g.get("score", 0.0))
                    score = max(0.0, min(1.0, score))
                    student_ans = next(
                        (
                            a.get("answer", "")
                            for a in student_answers
                            if a.get("question_id") == qid
                        ),
                        "",
                    )
                    correct_ans = (
                        questions[qid].get("correct_answer", "")
                        if 0 <= qid < len(questions)
                        else ""
                    )
                    # Guard against LLM being too strict: if student answer exactly
                    # matches the correct answer (case-insensitive, trailing punct stripped),
                    # force score=1.0 regardless of what the model returned.
                    _trail = re.compile(r"[.,!?;:\s]+$")
                    norm_s = _trail.sub("", str(student_ans).strip()).lower()
                    norm_c = _trail.sub("", str(correct_ans).strip()).lower()
                    if norm_s and norm_s == norm_c:
                        score = 1.0
                    result.append(
                        {
                            "question_id": qid,
                            "student_answer": student_ans,
                            "correct_answer": correct_ans,
                            "score": score,
                            "is_correct": score >= 0.5,
                            "feedback": g.get("feedback", ""),
                        }
                    )
                return result
        except Exception as exc:
            logger.warning("[TUTOR] failed to parse grading JSON: %s", exc)

        # Fallback
        return TutorCrew._exact_match_grade(student_answers, questions)

    @staticmethod
    def _exact_match_grade(
        student_answers: list[dict], questions: list[dict]
    ) -> list[dict]:
        """Simple case-insensitive exact match fallback.

        Strips trailing punctuation before comparing so "I eat an apple." and
        "I eat an apple" are treated as equal.
        """
        import unicodedata

        _punct = re.compile(r"[.,!?;:\s]+$")

        def _norm(s: str) -> str:
            return _punct.sub("", s.strip()).lower()

        def _normalize_text(text: str) -> str:
            """Normalize text for comparison: lowercase, strip punctuation."""
            text = text.lower().strip()
            text = _punct.sub("", text)
            text = re.sub(r"\s+", " ", text)
            return text

        def _strip_accents(s: str) -> str:
            """Remove diacritics/accents for accent-insensitive comparison."""
            return "".join(
                c
                for c in unicodedata.normalize("NFKD", s)
                if not unicodedata.combining(c)
            )

        result = []
        for ans in student_answers:
            qid = ans.get("question_id", -1)
            student_answer = str(ans.get("answer", "")).strip()
            if 0 <= qid < len(questions):
                q = questions[qid]
                correct = q.get("correct_answer", "").strip()
                qtype = q.get("type", "")

                if qtype == "translation":
                    normalized_student = _normalize_text(student_answer)
                    normalized_correct = _normalize_text(correct)
                    is_correct = normalized_student == normalized_correct
                    if is_correct:
                        score = 1.0
                        feedback = "Correct translation!"
                    else:
                        score = 0.0
                        # Do NOT embed correct answer — UI shows it via _showCorrection
                        feedback = "Incorrect."

                elif qtype == "reorder":
                    normalized_student = _normalize_text(student_answer)
                    normalized_correct = _normalize_text(correct)
                    is_correct = normalized_student == normalized_correct
                    if is_correct:
                        score = 1.0
                        feedback = "Correct word order!"
                    else:
                        score = 0.0
                        feedback = "Incorrect."

                elif qtype == "cloze":
                    if student_answer:
                        normalized_student = _normalize_text(student_answer)
                        normalized_correct = _normalize_text(correct)
                        if normalized_student == normalized_correct:
                            score = 1.0
                            is_correct = True
                            feedback = "Correct!"
                        else:
                            score = 0.0
                            is_correct = False
                            feedback = "Incorrect."
                    else:
                        score = 0.0
                        is_correct = False
                        feedback = "Incorrect."

                elif qtype == "listening":
                    # Accept minor accent/diacritic differences (cinema vs cinéma)
                    normalized_student = _strip_accents(_normalize_text(student_answer))
                    normalized_correct = _strip_accents(_normalize_text(correct))
                    is_correct = normalized_student == normalized_correct
                    if is_correct:
                        score = 1.0
                        feedback = "Correct!"
                    else:
                        score = 0.0
                        feedback = "Incorrect."

                else:
                    # Regular questions — case-insensitive exact match
                    is_correct = _norm(student_answer) == _norm(correct)
                    if is_correct:
                        score = 1.0
                        feedback = "Correct!"
                    else:
                        score = 0.0
                        feedback = "Incorrect."

                result.append(
                    {
                        "question_id": qid,
                        "student_answer": student_answer,
                        "correct_answer": correct,
                        "score": score,
                        "is_correct": is_correct,
                        "feedback": feedback,
                    }
                )
        return result
