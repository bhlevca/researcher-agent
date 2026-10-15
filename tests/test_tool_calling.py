"""How each model is asked to call tools, and what happens when it answers.

Two models, two opposite problems:

- Most Ollama models (qwen3.5, gemma4, mistral) describe a tool call in plain
  prose when sent OpenAI tool schemas. P1 forces them into ReAct text, which
  they do reliably.
- gpt-oss:20b is trained on OpenAI's harmony format and emits a *structured*
  tool call even when no schema was sent at all. Measured directly against
  Ollama with no `tools` parameter: ``content=''`` and a populated
  ``tool_calls``. Forced into ReAct, CrewAI's provider returned that list,
  nothing in the ReAct path consumed it, and the list was stringified into the
  answer the user saw:

      Final Output: [ChatCompletionMessageFunctionToolCall(id='call_ctz5r4tv',
      function=Function(arguments='{"search_query":"..."}', ...

  The run "finished" in about two seconds having done nothing — which also
  made the GPU queue look broken, because two people could not contend when
  neither request lasted long enough to overlap.

So gpt-oss is exempted from P1 and uses the mechanism it was trained for, and
P6 remains as a net for any other model that emits a native call while in
ReAct mode.
"""

import pytest

import researcher.patches as patches


class TestWhichModelsGetNativeToolCalling:
    """Ollama's own "tools" capability decides, not a list of model names.

    Measured on this machine with thinking off: gpt-oss:20b, gemma4:26b,
    gemma4:12b and qwen3.5:9b all emit a correct structured call; phi4:14b
    has no tool support at all. Forcing a tools-capable model into ReAct is
    what broke gpt-oss (structured call stringified into the answer) and
    gemma4 (wrote a plan, never an Action, looped until the step budget ran
    out).
    """

    @pytest.fixture(autouse=True)
    def _no_cache(self):
        """The lookup is cached per model; tests must not leak into each other."""
        patches._fc_capability_cache.clear()
        yield
        patches._fc_capability_cache.clear()

    @staticmethod
    def _answer(monkeypatch, capabilities, fail=False):
        """Stub Ollama's /api/show."""
        class _Resp:
            def raise_for_status(self):
                if fail:
                    raise RuntimeError("not found")

            def json(self):
                return {"capabilities": capabilities}

        import requests
        monkeypatch.setattr(requests, "post",
                            lambda *a, **k: _Resp())

    def test_a_tools_capable_model_gets_native_calling(self, monkeypatch):
        self._answer(monkeypatch, ["completion", "tools", "thinking"])
        assert patches._prefers_native_tool_calling("ollama/gemma4:26b")

    def test_a_model_without_tools_gets_react(self, monkeypatch):
        self._answer(monkeypatch, ["completion"])
        assert not patches._prefers_native_tool_calling("ollama/phi4:14b")

    def test_an_unknown_model_falls_back_to_react(self, monkeypatch):
        """ReAct at least produces readable text, so it is the safe default."""
        self._answer(monkeypatch, [], fail=True)
        assert not patches._prefers_native_tool_calling("ollama/nope:1b")

    def test_ollama_being_unreachable_falls_back_to_react(self, monkeypatch):
        import requests

        def _boom(*a, **k):
            raise requests.ConnectionError("refused")

        monkeypatch.setattr(requests, "post", _boom)
        assert not patches._prefers_native_tool_calling("ollama/gemma4:26b")

    def test_the_answer_is_cached(self, monkeypatch):
        """It is asked on every agent step; one HTTP call per model is enough."""
        calls = []

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"capabilities": ["tools"]}

        import requests
        monkeypatch.setattr(requests, "post",
                            lambda *a, **k: (calls.append(1), _Resp())[1])
        for _ in range(5):
            patches._prefers_native_tool_calling("ollama/gemma4:26b")
        assert len(calls) == 1

    def test_the_registry_tag_is_stripped_before_asking(self, monkeypatch):
        """The model arrives as "ollama/name:tag"; Ollama wants "name:tag"."""
        seen = {}

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"capabilities": ["tools"]}

        import requests
        monkeypatch.setattr(
            requests, "post",
            lambda url, json=None, **k: (seen.update(json or {}), _Resp())[1])
        patches._prefers_native_tool_calling("ollama/gemma4:26b")
        assert seen["model"] == "gemma4:26b"

    def test_a_model_can_be_forced_back_to_react(self, monkeypatch):
        """An escape hatch for a model that advertises tools but answers prose."""
        self._answer(monkeypatch, ["tools"])
        monkeypatch.setattr(patches, "_FORCE_REACT_MODELS", ("gemma4",))
        assert not patches._prefers_native_tool_calling("ollama/gemma4:26b")


class _Fn:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _Call:
    def __init__(self, name="internet_search", arguments='{"search_query": "x"}'):
        self.function = _Fn(name, arguments)


class TestRecognisingAToolCallList:
    def test_a_list_of_call_objects_is_recognised(self):
        assert patches._is_tool_call_list([_Call()])

    def test_the_dict_form_is_recognised(self):
        assert patches._is_tool_call_list(
            [{"function": {"name": "x", "arguments": "{}"}}])

    def test_plain_text_is_not(self):
        """The ordinary case: a text answer must pass straight through."""
        assert not patches._is_tool_call_list("Final Answer: Cancún is warm.")

    def test_an_empty_list_is_not(self):
        assert not patches._is_tool_call_list([])

    def test_none_is_not(self):
        assert not patches._is_tool_call_list(None)

    def test_a_list_of_strings_is_not(self):
        assert not patches._is_tool_call_list(["Action: search"])


class TestTranslationToReact:
    def test_it_produces_text_crewai_can_parse(self):
        """The real check: CrewAI's own parser must return an AgentAction."""
        from crewai.agents.parser import parse

        text = patches._react_text_from_tool_calls([_Call()])
        action = parse(text)
        assert type(action).__name__ == "AgentAction"
        assert action.tool == "internet_search"

    def test_the_arguments_survive(self):
        from crewai.agents.parser import parse
        import json

        call = _Call(arguments='{"search_query":  "tropical  resorts"}')
        action = parse(patches._react_text_from_tool_calls([call]))
        assert json.loads(action.tool_input) == {
            "search_query": "tropical  resorts"}

    def test_only_the_first_call_is_emitted(self):
        """ReAct is one step at a time; the executor feeds back an observation.

        Emitting several would parse as one action with a corrupted input.
        """
        text = patches._react_text_from_tool_calls(
            [_Call(name="internet_search"), _Call(name="local_system_check")])
        assert text.count("Action:") == 1
        assert "local_system_check" not in text

    def test_non_string_arguments_are_serialised(self):
        from crewai.agents.parser import parse
        import json

        call = _Call(arguments={"search_query": "x"})
        action = parse(patches._react_text_from_tool_calls([call]))
        assert json.loads(action.tool_input) == {"search_query": "x"}

    def test_unparseable_arguments_are_passed_through_verbatim(self):
        """Better a repairable input than an exception in the patch."""
        text = patches._react_text_from_tool_calls(
            [_Call(arguments="{not json")])
        assert "{not json" in text

    def test_the_dict_form_translates_too(self):
        from crewai.agents.parser import parse

        text = patches._react_text_from_tool_calls(
            [{"function": {"name": "internet_search",
                           "arguments": '{"search_query": "x"}'}}])
        assert parse(text).tool == "internet_search"


def test_both_patches_are_registered():
    """A patch that is defined but never applied is the worst of both."""
    names = [t[0] for t in patches._PATCH_TARGETS]
    assert "P1" in names and "P6" in names
    src = (patches.__file__)
    with open(src) as f:
        body = f.read()
    assert "_apply_p6_tool_calls_as_react()" in body.split("def apply_all()")[1]
