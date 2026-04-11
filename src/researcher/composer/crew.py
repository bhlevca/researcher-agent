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

        planning_llm = self._make_planning_llm(self._model_name)
        planning_llm._allow_fc = True

        planning = PlanningConfig(
            llm=planning_llm,
            max_attempts=int(self._llm_params.get("planning_max_attempts", 2)),
            max_steps=3,
            reasoning_effort="medium",
            plan_prompt=(
                "You are {role}.\n\n"
                "Task: {description}\n\n"
                "Expected output: {expected_output}\n\n"
                "Available tools: {tools}\n\n"
                "Create a CONCISE plan with 1-{max_steps} steps. "
                "For each step write ONE line: the action.\n"
                "Most composition tasks can be completed from your own knowledge "
                "— no internet search needed.\n"
                "End with: READY: I am ready to execute the task.\n"
            ),
        )

        return Agent(
            role=_sub(agent_cfg["role"]),
            goal=_sub(agent_cfg["goal"]),
            backstory=_sub(agent_cfg["backstory"]),
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
        """Extract JSON score from ```json fences and build MusicXML."""
        m = re.search(r"```json\s*(.*?)```", response, re.DOTALL)
        if not m:
            return None
        raw = m.group(1).strip()
        try:
            score = json.loads(raw)
        except json.JSONDecodeError:
            # Try to salvage: fix common JSON issues
            # Remove trailing commas before } or ]
            fixed = re.sub(r',\s*([}\]])', r'\1', raw)
            try:
                score = json.loads(fixed)
            except json.JSONDecodeError as e:
                logger.warning("[Composer] Failed to parse JSON score: %s", e)
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
