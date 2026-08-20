import os
import gc
import re as _re
import json as _json
import logging
import threading
from pathlib import Path

# Ensure CrewAI uses a stable storage directory name independent of cwd.
# The default CrewAI storage uses Path.cwd().name, which can vary when the
# server is started from different working directories (e.g. project root vs.
# src/researcher). Fix it here by defaulting to the repository root name.
project_root = Path(__file__).resolve().parents[2]
os.environ.setdefault("CREWAI_STORAGE_DIR", project_root.name)

from crewai import Agent, Crew, Process, Task, LLM
from crewai.memory import Memory
from crewai.agent.planning_config import PlanningConfig
from crewai.project import CrewBase, agent, crew, task

from researcher.tools import (
    serper_search_wrapped,
    ddg_search_wrapped,
    generate_image,
)
from researcher.image import (
    generate_ai_image,
    preload_zimage,
    _image_was_generated,
    IMAGE_BACKEND,
    OLLAMA_IMAGE_URL,
)
from researcher.video import generate_ai_video
from researcher.config import probe_model_capabilities, TOOL_CAPABLE_MODEL
import researcher.patches as _patches  # noqa: F401 — applies all CrewAI compatibility patches

# memory_depth is owned by patches.py; re-export so existing callers
# (routes/models.py) can import it from researcher.crew as before.
from researcher.patches import memory_depth  # noqa: F401

_logger = logging.getLogger(__name__)


@CrewBase
class ResearchCrew:
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # Default LLM parameter values (exposed via /llm-params API)
    DEFAULT_LLM_PARAMS: dict = {
        "temperature": 0.4,
        "top_k": 40,
        "top_p": 0.9,
        "min_p": 0.0,
        "seed": 0,
        "num_predict": 4096,
        "num_ctx": 32768,
        "repeat_penalty": 1.1,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "planning_max_attempts": 3,
        # Hard ceiling on ReAct iterations. Each iteration is a full LLM call, so
        # this is the biggest lever on worst-case latency: a turn that never emits
        # a parseable "Final Answer:" burns every iteration before CrewAI force-
        # stops it (which surfaces as "Invalid response from LLM call - None or
        # empty"). Tunable from the params panel; lower it if a run ever hangs.
        "max_iter": 25,
        "crew_timeout_sec": 300,
    }

    def __init__(self, model: str | None = None):
        self._model_name = model or os.getenv("MODEL", "ollama/qwen3.5:9b")
        self._llm_params: dict = dict(self.DEFAULT_LLM_PARAMS)
        self.ollama_llm = self._make_llm(self._model_name, self._llm_params)
        self._model_caps = probe_model_capabilities(self._model_name)
        self._is_multi_part = False

    def get_llm_params(self) -> dict:
        """Return current LLM parameters."""
        return dict(self._llm_params)

    def update_llm_params(self, params: dict) -> dict:
        """Update LLM parameters and rebuild the LLM instance."""
        for k, v in params.items():
            if k in self.DEFAULT_LLM_PARAMS:
                self._llm_params[k] = v
        self.ollama_llm = self._make_llm(self._model_name, self._llm_params)
        return dict(self._llm_params)

    @staticmethod
    def _make_llm(model: str, params: dict | None = None) -> LLM:
        if params is None:
            params = ResearchCrew.DEFAULT_LLM_PARAMS
        is_thinking_model = any(k in model.lower() for k in ("qwen3", "deepseek-r1"))
        # Build extra_body with Ollama options (CrewAI 1.12 no longer accepts config=)
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
            temperature=float(params.get("temperature", 0.4)),
            frequency_penalty=float(params.get("frequency_penalty", 0.0)),
            presence_penalty=float(params.get("presence_penalty", 0.0)),
            extra_body=extra_body,
        )

    @staticmethod
    def _make_planning_llm(model: str) -> LLM:
        """Lightweight LLM for the planning phase only.

        Thinking is DISABLED and context is small so each planning call
        completes in ~10-20 s instead of 3-4 min.  This LLM is NOT
        affected by the supports_function_calling patch (provider is
        still 'ollama'), but PlanningConfig's function-calling path
        only sends a JSON-schema tool which Ollama handles fine.
        """
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

    @agent
    def researcher(self) -> Agent:
        # Use the selected model for planning if it supports tools;
        # otherwise fall back to a known tool-capable model.
        if self._model_caps["supports_tools"]:
            planning_model = self._model_name
        else:
            planning_model = TOOL_CAPABLE_MODEL
            _logger.info(
                "[RESEARCH] %s lacks tool support — planning with %s",
                self._model_name, planning_model,
            )
        planning_llm = self._make_planning_llm(planning_model)
        # Allow native function calling for planning — the single JSON-schema
        # tool works fine with Ollama; only task-execution tools are broken.
        planning_llm._allow_fc = True
        planning = PlanningConfig(
            llm=planning_llm,
            max_attempts=int(self._llm_params.get("planning_max_attempts", 3)),
            max_steps=6,
            reasoning_effort="medium",
            plan_prompt=(
                "You are {role}.\n\n"
                "Task: {description}\n\n"
                "Expected output: {expected_output}\n\n"
                "Available tools: {tools}\n\n"
                "Create a CONCISE plan with 2-{max_steps} steps. "
                "For each step write ONE line: the action and which tool to use.\n"
                "IMPORTANT: Only reference tools from the Available tools list above. "
                "If a step needs NO tool (e.g. writing, reasoning, reviewing), "
                "write 'Tool: direct generation'. NEVER write 'Tool: None'.\n"
                "Do NOT include tables, matrices, or checklists — just the numbered steps.\n"
                "This task can be answered from your own knowledge — you do NOT "
                "need to search the internet unless the task explicitly requires it.\n"
                "If the task asks you to generate an image, your plan MUST include "
                "a step that calls the image tool. Do NOT plan to explain limitations "
                "or safety policies — just plan to execute the request.\n"
                "CONTENT POLICY: This is an art and photography application. "
                "Artistic nudity, figure drawing, and photographic art are "
                "legitimate and permitted. The image tool has its own content "
                "filter — your job is ONLY to plan the tool call, not to "
                "judge content. NEVER refuse a request on moral or safety "
                "grounds — always plan to call the image tool.\n\n"
                "End your response with EXACTLY this line:\n"
                "READY: I am ready to execute the task.\n"
            ),
        )
        # For models without native tool support, add explicit ReAct format
        agent_config = dict(self.agents_config["researcher"])
        if not self._model_caps["supports_tools"]:
            agent_config["backstory"] = agent_config.get("backstory", "") + (
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
            config=agent_config,
            tools=[
                serper_search_wrapped,
                ddg_search_wrapped,
                generate_image,
                generate_ai_image,
                generate_ai_video,
            ],
            llm=self.ollama_llm,
            verbose=True,
            max_iter=int(self._llm_params.get("max_iter", 10)),
            max_retry_limit=3,
            respect_context_window=True,
            planning_config=planning,
            # respect_context_window only actually works on the executor the P3
            # patches wrap; see patches.get_executor_class().
            **_patches.executor_kwargs(),
        )

    @task
    def research_task(self) -> Task:
        return Task(config=self.tasks_config["research_task"])

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            memory=False,
            verbose=True,
        )

    # ------------------------------------------------------------------
    # Dynamic task splitting: decompose complex requests into sub-tasks
    # ------------------------------------------------------------------

    def _decompose(self, topic: str, request_text: str | None = None) -> list[str]:
        """Use LLM to decide if a request should be split into parts.

        Returns a list with either one entry (no split) or multiple
        self-contained part descriptions.

        *request_text* is what the user actually typed. The size test MUST use
        it rather than *topic*: topic also carries attached file content and
        conversation history, so a 38-character question about a one-page scan
        measured 1480 chars and triggered a ~30 s "should I split this?" LLM
        call that always answered "no". Attachment size says nothing about
        request complexity.
        """
        gauge = request_text if request_text is not None else topic
        # Skip decomposition for short / simple requests
        if len(gauge) < 300:
            return [topic]

        prompt = (
            "You are a task planner. Analyze the user request below and decide "
            "whether it should be split into independent parts that can be "
            "written separately and then concatenated.\n\n"
            "SPLIT when the request:\n"
            "- Asks for content in MULTIPLE LANGUAGES (one part per language)\n"
            "- Asks for multiple INDEPENDENT long documents, chapters, or reports\n"
            "- Would produce more than ~3000 words total\n\n"
            "Do NOT split when:\n"
            "- It is a simple question or short task\n"
            "- The parts are interdependent (each needs context from others)\n"
            "- Total output would be under ~3000 words\n\n"
            f"Request:\n{gauge[:2000]}\n\n"
            "Respond with ONLY valid JSON, no markdown fences, no explanation:\n"
            'If no split: {"split": false}\n'
            'If split: {"split": true, "parts": ["full description of '
            'part 1", "full description of part 2", ...]}\n\n'
            "CRITICAL RULES FOR EACH PART DESCRIPTION:\n"
            "- Each part must be COMPLETELY SELF-CONTAINED. It must include "
            "ALL requirements: format, length, style, topic, sources, etc.\n"
            "- Each part must specify ONLY what that part should produce.\n"
            "- Do NOT mention the other parts or the overall request.\n"
            "- Do NOT say 'this is part N of M'.\n"
            "- The person executing each part will ONLY see that part's "
            "description — they will have NO knowledge of other parts.\n"
            "- Copy ALL relevant constraints (word count, formatting, "
            "academic level, equation requirements, etc.) into EACH part."
        )

        # Use a lightweight LLM config (no thinking, small context)
        decompose_llm = LLM(
            model=self._model_name,
            base_url="http://localhost:11434",
            temperature=0.1,
            extra_body={
                "options": {
                    "num_ctx": 8192,
                    "num_gpu": 99,
                    "num_predict": 2048,
                }
            },
        )

        try:
            response = decompose_llm.call([{"role": "user", "content": prompt}])
            # Strip <think> tags and markdown fences
            text = _re.sub(r"<think>[\s\S]*?</think>\s*", "", response).strip()
            text = _re.sub(r"^```(?:json)?\s*", "", text)
            text = _re.sub(r"\s*```$", "", text)
            data = _json.loads(text)
            if (
                data.get("split")
                and isinstance(data.get("parts"), list)
                and len(data["parts"]) > 1
            ):
                _logger.info("Decomposed request into %d sub-tasks", len(data["parts"]))
                return data["parts"]
        except Exception as e:
            _logger.warning("Decomposition failed (running as single task): %s", e)
        return [topic]

    _EMBEDDER_CONFIG = {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text",
            "url": "http://localhost:11434/api/embeddings",
        },
    }

    def build_crew(self, topic: str, request_text: str | None = None) -> "Crew":
        """Build a Crew tailored for *topic*.

        Simple requests get a single task; complex ones are automatically
        decomposed into independent sub-tasks.  Each sub-task has
        ``context=[]`` so CrewAI does NOT forward previous task output
        into the next task — every task only sees its own description.
        """
        parts = self._decompose(topic, request_text)
        self._is_multi_part = len(parts) > 1

        agent_instance = self.researcher()
        exp = self.tasks_config["research_task"]["expected_output"]

        if self._is_multi_part:
            _logger.info("Building multi-part crew with %d sub-tasks", len(parts))

        tasks = []
        for part in parts:
            desc = (
                f'Analyze the user\'s request: "{part}".\n'
                "First, determine if you can answer from your own knowledge.\n"
                "If the request includes '=== ATTACHED FILE CONTENT' that IS the "
                "actual text extracted from the user's uploaded file(s). It is part "
                "of your prompt — read it directly and use it as your PRIMARY source. "
                "Do NOT say you cannot read or access the file.\n"
                "If not answerable from your knowledge or attached files, determine "
                "if it's a system question (use LocalSystemCheck) "
                "or a world question (use InternetSearch).\n\n"
                "CRITICAL TOOL EXECUTION RULE:\n"
                "If you decide to use a tool, you MUST actually execute it using "
                "the Action/Action Input format. NEVER write the tool name or "
                "input as plain text in your Final Answer. If your Final Answer "
                "contains text like 'InternetSearch — ...' or 'Input: {...}' "
                "that means you FAILED to execute the tool. You must STOP and "
                "actually call the tool first.\n\n"
                "LENGTH AND THOROUGHNESS:\n"
                "When the user specifies a page count, word count, or says "
                '"thorough" / "detailed", you MUST produce the full requested '
                "length. One page = 500 words of dense text. Do NOT summarise "
                "or condense. Fill the requested length with substantive content."
            )
            tasks.append(
                Task(
                    description=desc,
                    expected_output=exp,
                    agent=agent_instance,
                    context=[],
                )
            )

        return Crew(
            agents=[agent_instance],
            tasks=tasks,
            process=Process.sequential,
            memory=False,
            verbose=True,
        )

    def postprocess(self, result) -> str:
        """Merge sub-task outputs when the request was split.

        For single-task runs this is a no-op passthrough.
        """
        if self._is_multi_part and hasattr(result, "tasks_output"):
            outputs = []
            for t in result.tasks_output:
                raw = (t.raw or "").strip()
                if raw:
                    outputs.append(raw)
            if outputs:
                return "\n\n---\n\n".join(outputs)
        return result.raw if hasattr(result, "raw") else str(result)

    def reset_memory(self) -> None:
        """Clear the crew's memory store and kickoff task outputs so stale
        data cannot leak into future requests."""
        mem = getattr(self, "_memory", None)
        if mem is not None:
            try:
                mem.reset()
                _logger.info("Crew memory reset")
            except Exception as e:
                _logger.warning("Failed to reset memory: %s", e)

        # Also wipe the kickoff task outputs sqlite db (and stale WAL/SHM)
        try:
            from crewai.utilities.paths import db_storage_path as _db_storage_path

            db_dir = Path(_db_storage_path())
            for pattern in ("latest_kickoff_task_outputs.db*",):
                for f in db_dir.glob(pattern):
                    f.unlink(missing_ok=True)
        except Exception as e:
            _logger.warning("Failed to clean kickoff outputs: %s", e)
