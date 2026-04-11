"""TutorCrew — CrewAI orchestrator for the Language Tutor.

Builds agents and tasks for conversation, lessons, quizzes, and appraisal.
Follows the same patterns as ResearchCrew (crew.py) including the Ollama
monkey-patches already applied at import time by researcher.crew.
"""

import os
import re
import json
import logging
from pathlib import Path

from crewai import Agent, Crew, Process, Task, LLM
from crewai.agent.planning_config import PlanningConfig

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
        "temperature": 0.6,   # slightly more creative for natural conversation
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

        planning_llm = self._make_planning_llm(self._model_name)
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

        return Agent(
            role=_sub(agent_cfg["role"]),
            goal=_sub(agent_cfg["goal"]),
            backstory=_sub(agent_cfg["backstory"]),
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
        desc, exp = self._render_task("conversation_task", {
            "target_lang": target_lang,
            "native_lang": native_lang,
            "level": level,
            "context": context,
            "message": message,
        })

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
        desc, exp = self._render_task("lesson_task", {
            "topic": topic,
            "lesson_type": lesson_type,
            "target_lang": target_lang,
            "native_lang": native_lang,
            "level": level,
        })

        task = Task(description=desc, expected_output=exp, agent=agent, context=[])
        return Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            memory=False,
            verbose=True,
        )

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
        agent = self._build_agent(target_lang, native_lang, level)
        desc, exp = self._render_task("quiz_task", {
            "quiz_type": quiz_type,
            "num_questions": str(num_questions),
            "target_lang": target_lang,
            "native_lang": native_lang,
            "level": level,
            "vocabulary_context": vocabulary_context,
            "lesson_context": lesson_context,
        })

        task = Task(description=desc, expected_output=exp, agent=agent, context=[])
        return Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            memory=False,
            verbose=True,
        )

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
        desc, exp = self._render_task("appraisal_task", {
            "target_lang": target_lang,
            "native_lang": native_lang,
            "level": level,
            "stats": stats,
            "recent_messages": recent_messages,
        })

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
            entries.append({
                "word": word,
                "translation": translation,
                "part_of_speech": pos,
                "context": "",
                "phonetic": "",
            })
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
