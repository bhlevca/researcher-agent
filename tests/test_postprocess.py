"""Tests for the post-processing logic in main.py."""

import re
from unittest.mock import patch, MagicMock
from pathlib import Path

from researcher.main import (
    _postprocess,
    _extract_usage,
    _narrated_search_re,
    _narrated_image_re,
    _narrated_json_image_re,
    _clean_assistant_text,
    _is_refusal_response,
    _build_image_prompt,
    _MAX_ASSISTANT_CONTEXT_CHARS,
    STATIC_DIR,
)


# ---------------------------------------------------------------------------
# Regex detection tests — verify patterns match real LLM narration output
# ---------------------------------------------------------------------------


class TestNarratedSearchRegex:
    """Verify _narrated_search_re matches all observed LLM narration patterns."""

    def test_bold_internetsearch(self):
        text = '**InternetSearch** — Google search. Input: {"search_query": "Dacic golden helmet recovered 2026"}'
        m = _narrated_search_re.search(text)
        assert m is not None
        assert m.group(1) == "InternetSearch"
        assert "Dacic golden helmet" in m.group(2)

    def test_plain_internetsearch(self):
        text = 'InternetSearch — Google search. Input: {"search_query": "test query"}'
        m = _narrated_search_re.search(text)
        assert m is not None
        assert m.group(2) == "test query"

    def test_duckduckgo(self):
        text = '**DuckDuckGoSearch** — fallback search. Input: {"query": "some fallback query"}'
        m = _narrated_search_re.search(text)
        assert m is not None
        assert m.group(1) == "DuckDuckGoSearch"
        assert m.group(2) == "some fallback query"

    def test_multiline_narration(self):
        text = (
            '**InternetSearch** — Google search.\n'
            'Input: {"search_query": "multiline query"}'
        )
        m = _narrated_search_re.search(text)
        assert m is not None
        assert m.group(2) == "multiline query"

    def test_no_match_on_normal_text(self):
        text = "The search returned interesting results about the topic."
        assert _narrated_search_re.search(text) is None

    def test_no_match_on_proper_action(self):
        text = 'Action: InternetSearch\nAction Input: {"search_query": "proper call"}'
        assert _narrated_search_re.search(text) is None


class TestNarratedImageRegex:
    """Verify _narrated_image_re matches observed GenerateAIImage narration patterns."""

    def test_bold_generate_ai_image(self):
        text = '**GenerateAIImage** — AI image generation. Input: {"prompt": "a red fox in snow"}'
        m = _narrated_image_re.search(text)
        assert m is not None
        assert m.group(1) == "GenerateAIImage"
        assert m.group(2) == "a red fox in snow"

    def test_plain_generate_ai_image(self):
        text = 'GenerateAIImage — Generate realistic AI image. Input: {"prompt": "landscape painting"}'
        m = _narrated_image_re.search(text)
        assert m is not None
        assert m.group(2) == "landscape painting"

    def test_generate_image_geometric(self):
        text = '**GenerateImage** — geometric drawing. Input: {"instructions": "draw a circle"}'
        m = _narrated_image_re.search(text)
        assert m is not None
        assert m.group(1) == "GenerateImage"
        assert m.group(2) == "draw a circle"

    def test_manet_real_world_narration(self):
        """Match the exact pattern observed in the Manet reproduction failure."""
        text = (
            '**GenerateAIImage** — Generate realistic/artistic AI image. Input: '
            '{"prompt": "Édouard Manet\'s \'Luncheon on the Grass\' masterpiece, '
            'full color reproduction, highly detailed, 8k quality"}'
        )
        m = _narrated_image_re.search(text)
        assert m is not None
        assert "Manet" in m.group(2)

    def test_multiline_narration(self):
        text = (
            '**GenerateAIImage** — AI image.\n'
            'Input: {"prompt": "sunset over mountains"}'
        )
        m = _narrated_image_re.search(text)
        assert m is not None
        assert m.group(2) == "sunset over mountains"

    def test_no_match_on_normal_text(self):
        text = "I generated an image of a beautiful landscape."
        assert _narrated_image_re.search(text) is None

    def test_no_match_on_proper_action(self):
        text = 'Action: GenerateAIImage\nAction Input: {"prompt": "proper call"}'
        assert _narrated_image_re.search(text) is None


class TestNarratedJsonImageRegex:
    """Verify _narrated_json_image_re matches llama-style JSON tool narrations."""

    def test_llama_style_json(self):
        text = '{"name": "generate_ai_image", "parameters": {"prompt": "A color reproduction of Manet painting"}}'
        m = _narrated_json_image_re.search(text)
        assert m is not None
        assert "Manet" in m.group(1)

    def test_llama_with_surrounding_text(self):
        text = (
            'Here is the JSON for a function call:\n\n'
            '{"name": "generate_ai_image", "parameters": {"prompt": "sunset over mountains, 8k"}}\n\n'
            'This should produce a beautiful image.'
        )
        m = _narrated_json_image_re.search(text)
        assert m is not None
        assert "sunset" in m.group(1)

    def test_case_insensitive_name(self):
        text = '{"name": "GenerateAIImage", "parameters": {"prompt": "red fox in snow"}}'
        m = _narrated_json_image_re.search(text)
        assert m is not None
        assert "red fox" in m.group(1)

    def test_extra_whitespace(self):
        text = '{  "name" : "generate_ai_image" , "parameters" : { "prompt" : "a cat" } }'
        m = _narrated_json_image_re.search(text)
        assert m is not None
        assert m.group(1) == "a cat"

    def test_no_match_on_executed_tool(self):
        """Actual tool calls in verbose logs look different."""
        text = 'Tool: generate_ai_image\nArgs: {"prompt": "test"}'
        assert _narrated_json_image_re.search(text) is None

    def test_no_match_on_other_tool(self):
        text = '{"name": "internet_search", "parameters": {"search_query": "test"}}'
        assert _narrated_json_image_re.search(text) is None


class TestJsonNarratedRescue:
    """Verify _postprocess detects JSON-narrated image calls and executes the tool."""

    @patch("researcher.main._generate_ai_image_tool")
    def test_json_narration_in_response(self, mock_tool):
        gen_dir = STATIC_DIR / "generated"
        gen_dir.mkdir(parents=True, exist_ok=True)
        real_img = gen_dir / "json_rescue1.png"
        real_img.write_bytes(b"fake png data")
        try:
            mock_tool.run.return_value = "![generated image](/static/generated/json_rescue1.png)"
            response = (
                'Here is the JSON for a function call:\n\n'
                '{"name": "generate_ai_image", "parameters": {"prompt": '
                '"A color reproduction of Manet Luncheon on the Grass"}}'
            )
            result = _postprocess(response, "")
            assert mock_tool.run.called
            assert "/static/generated/json_rescue1.png" in result
        finally:
            real_img.unlink(missing_ok=True)

    @patch("researcher.main._generate_ai_image_tool")
    def test_json_narration_in_verbose_log(self, mock_tool):
        gen_dir = STATIC_DIR / "generated"
        gen_dir.mkdir(parents=True, exist_ok=True)
        real_img = gen_dir / "json_rescue2.png"
        real_img.write_bytes(b"fake png data")
        try:
            mock_tool.run.return_value = "![generated image](/static/generated/json_rescue2.png)"
            # llama puts the JSON in an earlier task, not the final output
            response = "art pieces from 150 years ago"
            verbose = (
                'Final Answer:\n'
                '{"name": "generate_ai_image", "parameters": {"prompt": '
                '"Manet painting reproduction, oil on canvas, vibrant colors"}}'
            )
            result = _postprocess(response, verbose)
            assert mock_tool.run.called
            assert "/static/generated/json_rescue2.png" in result
        finally:
            real_img.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Narrated tool rescue tests in _postprocess
# ---------------------------------------------------------------------------


class TestNarratedImageRescue:
    """Verify _postprocess detects narrated image calls and executes the tool."""

    @patch("researcher.main._generate_ai_image_tool")
    def test_narrated_ai_image_is_rescued(self, mock_tool):
        # Create a real file on disk so image validation doesn't strip it
        gen_dir = STATIC_DIR / "generated"
        gen_dir.mkdir(parents=True, exist_ok=True)
        real_img = gen_dir / "rescued123.png"
        real_img.write_bytes(b"fake png data")
        try:
            mock_tool.run.return_value = "![generated image](/static/generated/rescued123.png)"
            response = (
                'Here is the image:\n\n'
                '**GenerateAIImage** — AI image generation. Input: '
                '{"prompt": "a beautiful sunset over the ocean, 8k"}\n\n'
                '![generated image](/static/generated/fake_nonexistent.png)\n\n'
                'This is a lovely sunset.'
            )
            result = _postprocess(response, "")
            mock_tool.run.assert_called_once_with("a beautiful sunset over the ocean, 8k")
            assert "/static/generated/rescued123.png" in result
            assert "fake_nonexistent" not in result
        finally:
            real_img.unlink(missing_ok=True)

    @patch("researcher.main._generate_ai_image_tool")
    def test_narrated_ai_image_preserves_text(self, mock_tool):
        gen_dir = STATIC_DIR / "generated"
        gen_dir.mkdir(parents=True, exist_ok=True)
        real_img = gen_dir / "real456.png"
        real_img.write_bytes(b"fake png data")
        try:
            mock_tool.run.return_value = "![generated image](/static/generated/real456.png)"
            response = (
                '**GenerateAIImage** — AI image. Input: {"prompt": "red fox"}\n\n'
                '## Description\n\nA lovely red fox in the snow.'
            )
            result = _postprocess(response, "")
            assert "/static/generated/real456.png" in result
            # Non-narration text should be preserved
            assert "Description" in result
            assert "lovely red fox" in result
        finally:
            real_img.unlink(missing_ok=True)

    @patch("researcher.main._generate_ai_image_tool")
    def test_narrated_image_tool_error_graceful(self, mock_tool):
        mock_tool.run.side_effect = RuntimeError("GPU out of memory")
        response = '**GenerateAIImage** — AI image. Input: {"prompt": "test"}'
        # Should not raise, just pass through
        result = _postprocess(response, "")
        assert isinstance(result, str)


class TestNarratedSearchRescue:
    """Verify _postprocess detects narrated search calls and executes them."""

    @patch("litellm.completion")
    @patch("researcher.crew.serper_search_wrapped")
    def test_narrated_search_is_rescued(self, mock_serper, mock_litellm_completion):
        mock_serper.run.return_value = '{"organic": [{"title": "Test", "snippet": "result"}]}'
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "The answer based on search results."
        mock_litellm_completion.return_value = mock_resp

        response = (
            '**InternetSearch** — Google search. Input: '
            '{"search_query": "Dacic helmet recovery method 2026"}'
        )
        log = 'New request: how were the items recovered?'
        result = _postprocess(response, log)
        mock_serper.run.assert_called_once()
        assert "answer based on search" in result

    @patch("litellm.completion")
    @patch("researcher.crew.ddg_search_wrapped")
    def test_narrated_duckduckgo_is_rescued(self, mock_ddg, mock_litellm_completion):
        mock_ddg.run.return_value = "DuckDuckGo results here"
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "Answer from DuckDuckGo."
        mock_litellm_completion.return_value = mock_resp

        response = '**DuckDuckGoSearch** — fallback. Input: {"query": "test query"}'
        result = _postprocess(response, "New request: test query")
        mock_ddg.run.assert_called_once()


# ---------------------------------------------------------------------------
# Early rescue improvements
# ---------------------------------------------------------------------------


class TestEarlyRescueFollowUp:
    """Verify early rescue triggers for follow-up image corrections."""

    @patch("researcher.main._generate_ai_image_tool")
    def test_correction_triggers_early_rescue(self, mock_tool):
        """When earlier messages generated an image and user gives
        a correction like 'no this is wrong', early rescue should trigger."""
        gen_dir = STATIC_DIR / "generated"
        gen_dir.mkdir(parents=True, exist_ok=True)
        real_img = gen_dir / "fixed123.png"
        real_img.write_bytes(b"fake png data")
        try:
            mock_tool.run.return_value = "![generated image](/static/generated/fixed123.png)"
            response = "## Correction\n\nI apologize for the mistake."
            log = (
                "Assistant: [Generated an image using GenerateAIImage tool — "
                "YOU MUST call the tool again for new images]\n"
                "New request: no, this is manet's portrait not luncheon in the grass"
            )
            result = _postprocess(response, log)
            # The early rescue should have been triggered at least once
            mock_tool.run.assert_called()
            assert "/static/generated/fixed123.png" in result
        finally:
            real_img.unlink(missing_ok=True)

    @patch("researcher.main._generate_ai_image_tool")
    def test_reproduction_keyword_triggers_early_rescue(self, mock_tool):
        gen_dir = STATIC_DIR / "generated"
        gen_dir.mkdir(parents=True, exist_ok=True)
        real_img = gen_dir / "repro123.png"
        real_img.write_bytes(b"fake png data")
        try:
            mock_tool.run.return_value = "![generated image](/static/generated/repro123.png)"
            response = "Some text response without an image."
            log = "New request: create a reproduction of Monet's Water Lilies"
            result = _postprocess(response, log)
            mock_tool.run.assert_called()
            assert "/static/generated/repro123.png" in result
        finally:
            real_img.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Refusal detection tests
# ---------------------------------------------------------------------------


class TestIsRefusalResponse:
    """Verify _is_refusal_response detects LLM self-censorship."""

    def test_manet_refusal_detected(self):
        """The actual Manet refusal response from the user's bug report."""
        text = (
            "## Analysis of Édouard Manet's \"Luncheon on the Grass\"\n\n"
            "### Image Generation Limitations\n\n"
            "I must be transparent about my technical capabilities:\n\n"
            "| Limitation | Explanation |\n"
            "| Artistic Reproduction | I cannot generate exact reproductions "
            "of specific famous paintings |\n"
            "| Style Matching | The GenerateAIImage tool creates original AI images, "
            "not artistic reproductions |\n"
        )
        assert _is_refusal_response(text) is True

    def test_copyright_refusal_detected(self):
        text = (
            "I cannot fulfill this request due to copyright concerns. "
            "I'm not able to generate images that may violate copyright restrictions."
        )
        assert _is_refusal_response(text) is True

    def test_safety_refusal_detected(self):
        text = (
            "I must decline this request. The content is not appropriate "
            "based on our safety guidelines and ethical standards."
        )
        assert _is_refusal_response(text) is True

    def test_normal_response_not_flagged(self):
        text = (
            "Here is the generated image of a sunset over the ocean.\n\n"
            "![generated image](/static/generated/abc123.png)\n\n"
            "The warm colors capture the beauty of the scene."
        )
        assert _is_refusal_response(text) is False

    def test_single_keyword_not_flagged(self):
        """A single refusal-like word shouldn't trigger false positive."""
        text = "I cannot find any results for that specific query."
        assert _is_refusal_response(text) is False

    def test_technical_limitation_essay_detected(self):
        """The pattern where LLM writes a limitations table instead of generating."""
        text = (
            "I cannot create this image. It is against my guidelines to produce "
            "content that may be inappropriate. I must apologize for this limitation."
        )
        assert _is_refusal_response(text) is True


# ---------------------------------------------------------------------------
# Prompt building tests
# ---------------------------------------------------------------------------


class TestBuildImagePrompt:
    """Verify _build_image_prompt converts user requests to good image prompts."""

    def test_painting_reproduction_with_artist(self):
        subj = (
            'create a reproduction of Édouard Manet\'s '
            'Luncheon on the Grass masterpiece'
        )
        prompt = _build_image_prompt(subj)
        assert "Luncheon on the Grass" in prompt
        assert "in the style of" in prompt
        assert "Manet" in prompt
        assert "oil painting" in prompt

    def test_painting_reproduction_quoted_title(self):
        subj = 'reproduction of "Starry Night"'
        prompt = _build_image_prompt(subj)
        assert "Starry Night" in prompt
        assert "oil painting" in prompt

    def test_generic_image_request(self):
        subj = "a cat sitting on a windowsill watching the rain"
        prompt = _build_image_prompt(subj)
        assert "cat" in prompt
        assert "highly detailed" in prompt

    def test_strips_conversational_fluff(self):
        subj = "please I want a beautiful sunset could you make sure it is vivid"
        prompt = _build_image_prompt(subj)
        # Conversational words should be stripped
        assert "please" not in prompt.lower().split(",")[0]
        assert "I want" not in prompt

    def test_strips_copyright_disclaimers(self):
        """Copyright/policy text from user should be removed from the prompt."""
        subj = (
            'create a reproduction of Édouard Manet\'s "Luncheon on the Grass" '
            'masterpiece? please draw in colour and the woman\'s undressing state '
            'needs to be accurate. The art is exempt from copyright since it was '
            'produced in 1863 and art nudity is acceptable in public, so don\'t '
            'pull the explicit card bacause it does not apply'
        )
        prompt = _build_image_prompt(subj)
        assert "copyright" not in prompt.lower()
        assert "exempt" not in prompt.lower()
        assert "explicit" not in prompt.lower()
        assert "Luncheon on the Grass" in prompt
        assert "colour" in prompt  # visual detail preserved

    def test_painting_with_possessive_artist(self):
        subj = "reproduction of Monet's Water Lilies"
        prompt = _build_image_prompt(subj)
        assert "Water Lilies" in prompt
        assert "Monet" in prompt
        assert "oil painting" in prompt


# ---------------------------------------------------------------------------
# Follow-up subject extraction from conversation history
# ---------------------------------------------------------------------------


class TestFollowUpSubjectExtraction:
    """Verify that follow-up complaints mine the original request for the painting subject."""

    @patch("researcher.main._generate_ai_image_tool")
    def test_complaint_uses_earlier_request(self, mock_tool):
        """When user complains about refusal, the earlier request with image
        keywords should be used for the prompt, not the complaint text."""
        gen_dir = STATIC_DIR / "generated"
        gen_dir.mkdir(parents=True, exist_ok=True)
        real_img = gen_dir / "manet_fix.png"
        real_img.write_bytes(b"fake png data")
        try:
            mock_tool.run.return_value = "![generated image](/static/generated/manet_fix.png)"
            refusal_response = (
                "## Analysis of Édouard Manet's \"Luncheon on the Grass\"\n\n"
                "I cannot generate exact reproductions of specific famous paintings. "
                "I must be transparent about my technical capabilities. "
                "I cannot fulfill this request."
            )
            # The verbose log mirrors what CrewAI produces — earlier User: lines
            # contain the original image request with draw/reproduction keywords
            log = (
                'User: can you use the z-image tool to create a reproduction of '
                '\u00c9douard Manet\'s "Luncheon on the Grass" masterpiece? '
                'please draw in colour\n'
                'Assistant: [Generated an image using GenerateAIImage tool — '
                'YOU MUST call the tool again for new images]\n'
                'User: this is wrong, you delivered a cylinder\n'
                'Assistant: [Generated an image using GenerateAIImage tool — '
                'YOU MUST call the tool again for new images]\n'
                'New request: LLM self-censors: qwen3.5 refuses to generate '
                'the image, think again and interpret guidelines properly'
            )
            result = _postprocess(refusal_response, log)
            # Should have called the tool with a prompt containing painting name
            mock_tool.run.assert_called()
            call_prompt = mock_tool.run.call_args[0][0]
            assert "Luncheon on the Grass" in call_prompt or "reproduction" in call_prompt.lower()
            # Refusal essay should be replaced, not prepended
            assert "/static/generated/manet_fix.png" in result
            assert "I cannot generate" not in result
        finally:
            real_img.unlink(missing_ok=True)

    @patch("researcher.main._generate_ai_image_tool")
    def test_direct_image_request_not_affected(self, mock_tool):
        """When user directly asks for an image (not a follow-up),
        the current request should be used."""
        gen_dir = STATIC_DIR / "generated"
        gen_dir.mkdir(parents=True, exist_ok=True)
        real_img = gen_dir / "direct_req.png"
        real_img.write_bytes(b"fake png data")
        try:
            mock_tool.run.return_value = "![generated image](/static/generated/direct_req.png)"
            response = "I apologize but I cannot generate this image."
            log = "New request: draw a beautiful mountain landscape at sunset"
            result = _postprocess(response, log)
            mock_tool.run.assert_called()
            call_prompt = mock_tool.run.call_args[0][0]
            # Should use the current request, containing landscape/sunset
            assert "mountain" in call_prompt.lower() or "sunset" in call_prompt.lower() or "landscape" in call_prompt.lower()
        finally:
            real_img.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# End-to-end refusal rescue test
# ---------------------------------------------------------------------------


class TestRefusalRescueEndToEnd:
    """Test the full flow: refusal detected + early rescue replaces the essay."""

    @patch("researcher.main._generate_ai_image_tool")
    def test_refusal_essay_replaced_by_image(self, mock_tool):
        """When agent writes a refusal essay and early rescue succeeds,
        the essay should be completely replaced by the image."""
        gen_dir = STATIC_DIR / "generated"
        gen_dir.mkdir(parents=True, exist_ok=True)
        real_img = gen_dir / "rescued_art.png"
        real_img.write_bytes(b"fake png data")
        try:
            mock_tool.run.return_value = "![generated image](/static/generated/rescued_art.png)"
            refusal = (
                "I cannot create this image due to copyright concerns. "
                "It is not appropriate to generate such content. "
                "I must decline this request based on safety guidelines."
            )
            log = "New request: create a reproduction of Van Gogh's Starry Night"
            result = _postprocess(refusal, log)
            # Image should be in result
            assert "/static/generated/rescued_art.png" in result
            # Refusal text should be gone (replaced, not prepended)
            assert "cannot create" not in result
            assert "safety guidelines" not in result
        finally:
            real_img.unlink(missing_ok=True)

    @patch("researcher.main._generate_ai_image_tool")
    def test_non_refusal_response_preserved_with_image(self, mock_tool):
        """When agent writes a useful response but forgot to generate,
        the response should be preserved alongside the image."""
        gen_dir = STATIC_DIR / "generated"
        gen_dir.mkdir(parents=True, exist_ok=True)
        real_img = gen_dir / "bonus_img.png"
        real_img.write_bytes(b"fake png data")
        try:
            mock_tool.run.return_value = "![generated image](/static/generated/bonus_img.png)"
            response = (
                "Here is detailed information about the painting's history "
                "and artistic significance."
            )
            log = "New request: draw a portrait of a renaissance nobleman"
            result = _postprocess(response, log)
            # Both image and original text should be present
            assert "/static/generated/bonus_img.png" in result
            assert "artistic significance" in result
        finally:
            real_img.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Conversation context truncation
# ---------------------------------------------------------------------------


class TestCleanAssistantText:
    """Verify assistant text is truncated to prevent context bloat."""

    def test_short_text_unchanged(self):
        text = "The answer is 42."
        assert _clean_assistant_text(text) == text

    def test_long_text_truncated(self):
        text = "A" * (_MAX_ASSISTANT_CONTEXT_CHARS + 500)
        result = _clean_assistant_text(text)
        assert len(result) < len(text)
        assert result.endswith("[... response truncated for context ...]")

    def test_image_text_collapsed(self):
        text = "Here is the image: ![pic](/static/generated/abc123.png)\nSome details."
        result = _clean_assistant_text(text)
        assert "YOU MUST call the tool again" in result
        assert "/static/generated/" not in result

    def test_thinking_tags_stripped(self):
        text = "<think>Internal reasoning here</think>\nFinal answer."
        result = _clean_assistant_text(text)
        assert "<think>" not in result
        assert "Final answer." in result


# ---------------------------------------------------------------------------
# Original postprocess and usage tests
# ---------------------------------------------------------------------------


class TestPostprocess:
    def test_plain_text_passes_through(self):
        result = _postprocess("Hello world", "")
        assert result == "Hello world"

    def test_empty_response_becomes_fallback(self):
        result = _postprocess("", "")
        assert "could not complete" in result.lower() or len(result) > 0

    def test_strips_result_tags(self):
        result = _postprocess("<result>  </result>Some answer", "")
        assert "<result>" not in result
        assert "Some answer" in result

    def test_strips_trailing_thought_chain(self):
        text = "Final answer here.\nThought: I should check\nAction: search\nAction Input: test"
        result = _postprocess(text, "")
        assert "Thought:" not in result
        assert "Final answer here." in result

    def test_image_from_log_when_missing_in_response(self):
        # Create a fake image file
        gen_dir = STATIC_DIR / "generated"
        gen_dir.mkdir(parents=True, exist_ok=True)
        fake_img = gen_dir / "abc123456789.png"
        fake_img.write_bytes(b"fake png")

        try:
            log = "Some output\n![generated image](/static/generated/abc123456789.png)\nMore output"
            result = _postprocess("The agent did something", log)
            assert "/static/generated/abc123456789.png" in result
        finally:
            fake_img.unlink(missing_ok=True)

    def test_invalid_image_reference_removed(self):
        text = "Here is an image: ![pic](/static/generated/nonexistent999.png)"
        result = _postprocess(text, "")
        assert "nonexistent999" not in result


class TestExtractUsage:
    def test_with_token_usage(self):
        mock = type("Result", (), {
            "token_usage": type("TU", (), {
                "total_tokens": 100,
                "prompt_tokens": 40,
                "completion_tokens": 60,
            })()
        })()
        usage = _extract_usage(mock)
        assert usage["total_tokens"] == 100
        assert usage["prompt_tokens"] == 40
        assert usage["completion_tokens"] == 60

    def test_without_token_usage(self):
        mock = type("Result", (), {"token_usage": None})()
        assert _extract_usage(mock) == {}

    def test_no_attribute(self):
        mock = type("Result", (), {})()
        assert _extract_usage(mock) == {}
