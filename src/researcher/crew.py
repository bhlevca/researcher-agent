import os
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task

from crewai.tools import tool
from crewai_tools import SerperDevTool

from langchain_community.tools import DuckDuckGoSearchRun

# Instantiate once, reuse across calls
_ddg_search = DuckDuckGoSearchRun()
_serper_tool = SerperDevTool()

@tool("DuckDuckGoSearch")
def ddg_search_wrapped(query: str):
    """Search the web using DuckDuckGo for general information and fallback data."""
    return _ddg_search.run(query)

@CrewBase
class ResearchCrew():
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self, model: str | None = None):
        self._model_name = model or os.getenv("MODEL", "ollama/qwen3.5:9b")
        self.ollama_llm = self._make_llm(self._model_name)

    @staticmethod
    def _make_llm(model: str) -> LLM:
        return LLM(
            model=model,
            base_url="http://localhost:11434",
            config={
                "options": {
                    "num_ctx": 4096,
                    "temperature": 0.1,
                    "num_gpu": 99,
                    "num_predict": 1024,
                }
            },
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            tools=[_serper_tool, ddg_search_wrapped],
            llm=self.ollama_llm,
            verbose=True,
            max_iter=3,
            memory=False,
        )

    @task
    def research_task(self) -> Task:
        return Task(config=self.tasks_config['research_task'])

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )