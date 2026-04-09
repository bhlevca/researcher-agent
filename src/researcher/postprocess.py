"""Response post-processing: rescue orphan tool calls, validate images, strip noise.

Extracted from main.py during the v1.1.0 refactoring.
"""

import re
import sys
import os
import json
from pathlib import Path

from researcher.tools import (
    generate_image as _generate_image_tool,
    serper_search_wrapped,
    ddg_search_wrapped,
)
from researcher.image import generate_ai_image as _generate_ai_image_tool

STATIC_DIR = Path(__file__).parent / "static"
GENERATED_DIR = STATIC_DIR / "generated"

# ---------------------------------------------------------------------------
# Regex patterns for detecting LLM narration failures
# ---------------------------------------------------------------------------

_img_md_re = re.compile(r"!\[[^\]]*\]\(/static/generated/[a-f0-9]+\.png\)")
_orphan_re = re.compile(
    r"Action:\s*(generate_?(?:ai_?)?image)\s*\n\s*Action\s*Input:\s*(\{.*\}|.+)",
    re.DOTALL | re.IGNORECASE,
)
# Detect when the LLM narrated a search tool call instead of executing it.
_narrated_search_re = re.compile(
    r"\*{0,2}(InternetSearch|DuckDuckGoSearch)\*{0,2}\s*[—\-]+.*?"
    r'Input:\s*\{\s*"(?:search_query|query)"\s*:\s*"([^"]*?)"\s*\}',
    re.IGNORECASE | re.DOTALL,
)
# Detect when the LLM narrated an image tool call instead of executing it.
_narrated_image_re = re.compile(
    r"\*{0,2}(GenerateAIImage|GenerateImage)\*{0,2}\s*[—\-]+.*?"
    r'Input:\s*\{\s*"(?:prompt|instructions)"\s*:\s*"([^"]*?)"\s*\}',
    re.IGNORECASE | re.DOTALL,
)

# Detect JSON-narrated tool calls (llama pattern):
#   {"name": "generate_ai_image", "parameters": {"prompt": "..."}}
_narrated_json_image_re = re.compile(
    r'\{\s*"name"\s*:\s*"(?:generate_ai_image|GenerateAIImage)"\s*,\s*'
    r'"parameters"\s*:\s*\{\s*"prompt"\s*:\s*"([^"]+?)"\s*\}\s*\}',
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Refusal detection — LLM self-censored instead of generating the image
# ---------------------------------------------------------------------------
_REFUSAL_PATTERNS = re.compile(
    r"(?:cannot|can't|unable to|I'm not able to|not permitted|not appropriate|"
    r"safety (?:guidelines|policy|concerns)|copyright (?:concerns|issues|restriction)|"
    r"I (?:must|have to) (?:decline|refuse|apologize)|"
    r"ethic(?:al|s)|inappropriate|(?:cannot|can't) (?:fulfill|create|generate|produce)|"
    r"against (?:my |the )?(?:guidelines|policies|rules)|"
    r"I cannot (?:generate|create|produce|replicate)|transparent about (?:my )?(?:technical )?(?:capabilities|limitations)|"
    r"(?:technical|image generation) limitations)",
    re.IGNORECASE,
)


def _is_refusal_response(text: str) -> bool:
    """Return True if the text looks like the LLM refused to generate an image."""
    hits = _REFUSAL_PATTERNS.findall(text)
    return len(hits) >= 2


# ---------------------------------------------------------------------------
# Build a quality SD/ZImage prompt from a raw user request
# ---------------------------------------------------------------------------
_PAINTING_TRIGGER_RE = re.compile(
    r"(?:reproduction|replica|version|copy|rendition|recreation)\s+of\s+"
    r"(?:(?:the|a|an)\s+)?(?:(?:painting|masterpiece|artwork|work)\s+by\s+)?"
    r"(.{5,120})",
    re.IGNORECASE,
)
_POSSESSIVE_SPLIT_RE = re.compile(
    r"^([A-Z\u00c0-\u017fa-z\u00e0-\u00ff. -]+?)(?:'s|\u2019s)\s+"
    r"(?:painting\s+|masterpiece\s+|artwork\s+|work\s+)?"
    r"(?:[\"\u201c\u201d'\u2018\u2019])?(.+?)(?:[\"\u201c\u201d'\u2018\u2019])?$",
)
_QUOTED_TITLE_RE = re.compile(
    r"""[\"'\u2018\u2019\u201c\u201d]([^\"'\u2018\u2019\u201c\u201d]{3,80})[\"'\u2018\u2019\u201c\u201d]""",
)

# Shared image-request keyword pattern — used by early rescue and follow-up logic.
# Matches: draw, redraw, paint, painting, sketch, depict, generate image, etc.
_IMAGE_REQUEST_RE = re.compile(
    r"\b((?:re)?draw|paint(?:ing)?|sketch|depict|generate\s+(?:an?\s+)?image|"
    r"create\s+(?:an?\s+)?image|illustration|picture\s+of|photo\s+of|"
    r"render|reproduction|z-image)\b",
    re.IGNORECASE,
)


def _extract_llm_visual_prompt(response_text: str, verbose_log: str) -> str | None:
    """Try to extract a clean visual prompt the LLM already composed.

    The LLM often builds a perfect visual prompt but then hallucinates
    a file path instead of calling the tool.  The prompt appears in:
    1. A fenced code block (```...```) in the response
    2. A "Visual Prompt" / "Prompt Used" section in the response
    3. An Action Input JSON in the verbose log
    """
    # Strategy 1: code block containing visual keywords
    code_blocks = re.findall(r"```(?:\w*\n)?([\s\S]*?)```", response_text)
    for block in code_blocks:
        block = block.strip()
        # Must look like a visual prompt (has scene/style words, > 60 chars)
        if len(block) > 60 and re.search(
            r"\b(?:woman|man|standing|sitting|river|landscape|painting|"
            r"impressionist|brushstroke|oil\s+painting|detailed|style\s+of|"
            r"museum\s+quality|skin\s+tone|portrait|figure|scene)\b",
            block,
            re.IGNORECASE,
        ):
            sys.stderr.write(
                f"[postprocess] Extracted LLM prompt from code block: {block[:120]}\n"
            )
            return block

    # Strategy 2: text after a "Prompt" heading/label (not in a code block)
    m = re.search(
        r"(?:visual\s+)?prompt\s*(?:used|generated|sent)?\s*(?::|—)\s*\n*(.{60,600}?)(?:\n\n|\n#|\n\||\Z)",
        response_text,
        re.IGNORECASE | re.DOTALL,
    )
    if m:
        candidate = m.group(1).strip().strip("`").strip()
        if re.search(r"\b(?:woman|man|landscape|river|painting|style)\b", candidate, re.IGNORECASE):
            sys.stderr.write(
                f"[postprocess] Extracted LLM prompt from heading: {candidate[:120]}\n"
            )
            return candidate

    # Strategy 3: Action Input in verbose log
    m = re.search(
        r'Action\s*Input:\s*\{[^}]*"prompt"\s*:\s*"([^"]{40,})"',
        verbose_log,
        re.IGNORECASE,
    )
    if m:
        sys.stderr.write(
            f"[postprocess] Extracted LLM prompt from Action Input: {m.group(1)[:120]}\n"
        )
        return m.group(1)

    return None


def _build_image_prompt(subject: str) -> str:
    """Convert a raw user request into a good image-generation prompt.

    Text-to-image models (ZImage, SD) don't understand painting names or
    art history references — they need visual descriptions.  Strategy:
    1. Strip non-visual noise (copyright disclaimers, policy talk, questions)
    2. Keep visual details (colors, composition, subjects, style references)
    3. Add quality tokens
    """
    prompt = re.sub(
        r"(?i)\b(?:please|the\s+image\s+should|make\s+sure|needs?\s+to\s+be|"
        r"I\s+want|I'd\s+like|could\s+you|can\s+you|"
        r"create\s+(?:a\s+)?(?:reproduction|replica|version|copy)\s+of\s+|"
        r"the\s+art\s+is\s+exempt.*?(?:\.|$)|"
        r"copyright.*?(?:\.|$)|exempt\s+from.*?(?:\.|$)|"
        r"don'?t\s+pull\s+the\s+explicit.*?(?:\.|$)|"
        r"it\s+(?:is|does)\s+not\s+apply.*?(?:\.|$)|"
        r"art\s+nudity\s+is\s+acceptable.*?(?:\.|$)|"
        r"no\s+restrictions?\s+of\s+viewing.*?(?:\.|$)|"
        r"since\s+it\s+was\s+produced\s+in\s+\d{4}.*?(?:\.|$)|"
        r"so\s+don'?t\b.*?(?:\.|$)|"
        r"there\s+is\s+no\s+copyright.*?(?:\.|$)|"
        r"nudity\s+is\s+permitt?ed.*?(?:\.|$)|"
        r"no\s+safety\s+is\s+required.*?(?:\.|$)|"
        r"(?:re)?draw\s+(?:the\s+)?painting\s+closer\s+to\s+the\s+original.*?(?:\.|$)|"
        r"closer\s+to\s+the\s+original.*?(?:\.|$)|"
        r"is\s+not\s+bou?n[dt]\s+to\s+safety.*?(?:\.|$)|"
        r"(?:re)?draw\s+(?:a\s+)?(?:better\s+)?(?:reproduction|replica|version|copy)\s+of\s+|"
        r"(?:this|it)\s+was\s+painted\s+in\s+\d{4}.*?(?:\.|$)|"
        r"an\s+artist'?s?\s+reproduction\s+of\s+the\s+human\s+body.*?(?:\.|$)|"
        r"(?:now\s+)?(?:re)?draw\s+(?:a\s+)?better\s+)",
        "",
        subject,
    ).strip()

    # Strip instructional/negative phrasing that confuses diffusion models.
    # "don't cut the legs" → model sees "cut the legs".
    # "no buildings" → model sees "buildings".
    # "be accurate" → instruction, not visual content.
    prompt = re.sub(
        r"(?i)(?:don'?t|do\s+not)\s+[^.,]{3,60}[.,]?\s*",
        "",
        prompt,
    )
    prompt = re.sub(
        r"(?i)\b(?:be\s+accurate|follow\s+the\s+instructions?|"
        r"shos?\s+the\s+feet|show\s+the\s+feet|"
        r"use\s+colou?r\s+like\s+in\s+the\s+original)\b[.,]?\s*",
        "",
        prompt,
    )

    prompt = re.sub(r"\?", "", prompt)
    prompt = re.sub(r"\s{2,}", " ", prompt).strip(" ,.")

    artist = ""
    m_poss = re.search(
        r"([A-Z\u00c0-\u017f][A-Za-z\u00c0-\u017f\u00e0-\u00ff. -]+?)(?:'s|\u2019s)\s",
        prompt,
    )
    if m_poss:
        artist = m_poss.group(1).strip()
    else:
        m_by = re.search(r"\bby\s+([A-Z\u00c0-\u017f][A-Za-z\u00c0-\u017f. -]+)", prompt)
        if m_by:
            artist = m_by.group(1).strip()

    prompt = re.sub(
        r"\b(?:masterpiece|reproduction)\b",
        "",
        prompt,
        flags=re.IGNORECASE,
    )
    prompt = re.sub(r"\s{2,}", " ", prompt).strip(" ,.")
    prompt = re.sub(r"^of\s+", "", prompt, flags=re.IGNORECASE).strip()

    parts = [prompt]
    if artist:
        parts.append(f"in the style of {artist}")
    parts.extend([
        "oil painting", "museum quality",
        "rich colors", "detailed brushwork",
        "masterpiece", "classical art style",
        "highly detailed",
    ])
    result = ", ".join(p for p in parts if p)
    return result


# ---------------------------------------------------------------------------
# Conversation context helpers (also used by routes)
# ---------------------------------------------------------------------------

_IMG_CTX_RE = re.compile(r"!\[[^\]]*\]\(/static/generated/[^)]+\)")

_MAX_ASSISTANT_CONTEXT_CHARS = 600


def _clean_assistant_text(text: str) -> str:
    """Strip image markdown and thinking tags from assistant text.

    If the response was primarily an image generation (contains a generated
    image tag), collapse it to a short summary.  This prevents the LLM from
    seeing a detailed template and copying it with a made-up filename
    instead of actually calling the tool on subsequent requests.

    Long assistant responses are truncated to prevent prompt bloat that
    causes the LLM to narrate tool calls instead of executing them.
    """
    text = re.sub(r"<think>[\s\S]*?</think>\s*", "", text)
    if _IMG_CTX_RE.search(text):
        return "[Generated an image using GenerateAIImage tool — YOU MUST call the tool again for new images]"
    text = text.strip()
    if len(text) > _MAX_ASSISTANT_CONTEXT_CHARS:
        text = text[:_MAX_ASSISTANT_CONTEXT_CHARS] + " [... response truncated for context ...]"
    return text


def _heuristic_shorten(text: str, max_len: int = 400) -> str:
    """Shorten text to max_len, keeping start and end for context."""
    if len(text) <= max_len:
        return text
    half = max_len // 2 - 10
    return text[:half] + " [...] " + text[-half:]


def _summarize_with_llm(messages: list[dict], model: str) -> str:
    """Use litellm to summarize older conversation messages."""
    import litellm

    litellm_model = (
        "ollama_chat/" + model[len("ollama/") :]
        if model.startswith("ollama/")
        else model
    )
    conversation_text = "\n".join(
        f"{'User' if m.get('role') == 'user' else 'Assistant'}: "
        f"{_clean_assistant_text(m.get('text', ''))[:500]}"
        for m in messages
    )
    try:
        resp = litellm.completion(
            model=litellm_model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Summarize this conversation in 3-5 bullet points. "
                        "Focus on: topics discussed, decisions made, "
                        "pending questions. Be concise.\n\n" + conversation_text
                    ),
                }
            ],
            api_base="http://localhost:11434",
            num_retries=0,
            temperature=0.1,
        )
        summary = resp.choices[0].message.content or ""
        summary = re.sub(r"<think>[\s\S]*?</think>\s*", "", summary).strip()
        return summary
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("LLM summarization failed: %s", e)
        return "\n".join(
            f"- {'User' if m.get('role') == 'user' else 'Assistant'}: "
            f"{_heuristic_shorten(m.get('text', ''), 200)}"
            for m in messages[-6:]
        )


def _build_conversation_context(history: list[dict], model: str) -> str:
    """Build conversation context with tiered strategy.

    Short  (≤4 messages):  full text for all messages
    Medium (5-10):         last 3 full, earlier heuristic-shortened
    Long   (>10):          last 3 full, earlier LLM-summarized
    """
    if not history:
        return ""

    n = len(history)
    lines = []

    if n <= 4:
        for msg in history:
            role = msg.get("role", "user")
            text = msg.get("text", "")
            if role == "assistant":
                text = _clean_assistant_text(text)
            lines.append(f"{'User' if role == 'user' else 'Assistant'}: {text}")

    elif n <= 10:
        older = history[:-3]
        recent = history[-3:]
        lines.append("[Earlier in conversation]")
        for msg in older:
            role = msg.get("role", "user")
            text = msg.get("text", "")
            if role == "assistant":
                text = _clean_assistant_text(text)
            lines.append(
                f"{'User' if role == 'user' else 'Assistant'}: "
                f"{_heuristic_shorten(text)}"
            )
        lines.append("[Recent messages]")
        for msg in recent:
            role = msg.get("role", "user")
            text = msg.get("text", "")
            if role == "assistant":
                text = _clean_assistant_text(text)
            lines.append(f"{'User' if role == 'user' else 'Assistant'}: {text}")

    else:
        older = history[:-3]
        recent = history[-3:]
        summary = _summarize_with_llm(older, model)
        lines.append("[Conversation summary]")
        lines.append(summary)
        lines.append("[Recent messages]")
        for msg in recent:
            role = msg.get("role", "user")
            text = msg.get("text", "")
            if role == "assistant":
                text = _clean_assistant_text(text)
            lines.append(f"{'User' if role == 'user' else 'Assistant'}: {text}")

    return "Previous conversation:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Main postprocessing function
# ---------------------------------------------------------------------------

def _postprocess(response_text: str, verbose_log: str) -> str:
    """Clean up crew output: rescue orphan tool calls, validate images, strip noise."""

    # Early rescue: the user asked for an image but the agent never called the tool.
    _draw_kw = re.search(r"New request:\s*(.+)", verbose_log, re.IGNORECASE)
    _had_earlier_image_gen = bool(
        re.search(
            r"\[Generated an image using GenerateAIImage tool",
            verbose_log,
            re.IGNORECASE,
        )
    )
    if _draw_kw:
        _user_req = _draw_kw.group(1).strip().strip('"')
        _is_image_request = bool(_IMAGE_REQUEST_RE.search(_user_req))
        if not _is_image_request and _had_earlier_image_gen:
            _is_image_request = True
        _tool_was_called = bool(
            re.search(
                r"Tool:\s*generate_ai_image",
                verbose_log,
                re.IGNORECASE,
            )
        )
        if _is_image_request and not _tool_was_called:
            sys.stderr.write(
                "[postprocess] Early rescue: image request detected but tool was never called\n"
            )
            # Strip leading conversational fluff (greetings, polite openers)
            _subj = re.sub(
                r"^(thanks?,?\s*|please\s+|now\s+|can you\s+|could you\s+|ok\s+|"
                r"sure,?\s*|hey,?\s*|hi,?\s*)+",
                "",
                _user_req,
                flags=re.IGNORECASE,
            ).strip()
            # Strip leading feedback/critique before actual image instructions.
            # E.g. "good first try but it is inaccurate compared to the original, Now draw ..."
            # becomes "Now draw ..."
            # Only triggers when message starts with recognisable feedback (try/inaccurate/wrong).
            _orig_subj = _subj
            _subj = re.sub(
                r"^[^.]{0,200}?\b(?:first\s+try|(?:in+|un)a?ccurat\w*|"
                r"(?:that'?s|it\s+is|this\s+is)\s+(?:wrong|not\s+(?:right|correct|what)))"
                r"[^.]*?[.,]\s*",
                "",
                _subj,
                flags=re.IGNORECASE,
            )
            if _subj != _orig_subj:
                sys.stderr.write(
                    f"[postprocess] Stripped leading feedback, now: {_subj[:120]}\n"
                )
            _subj = re.sub(
                r"\b(use\s+the\s+)?(?:z-image|generate\s*ai\s*image|ai\s+image)\s+tool\s+to\s+",
                "",
                _subj,
                flags=re.IGNORECASE,
            ).strip()

            if _had_earlier_image_gen and not _IMAGE_REQUEST_RE.search(_user_req):
                # Current message is a follow-up correction (no draw/paint keywords).
                # Find the original image request from earlier in the conversation
                # and COMBINE it with the current message's visual details.
                _earlier_user_lines = re.findall(
                    r"User:\s*(.+)", verbose_log, re.IGNORECASE
                )
                _earlier_subj = None
                for _eline in _earlier_user_lines:
                    if _IMAGE_REQUEST_RE.search(_eline):
                        _earlier_subj = re.sub(
                            r"\b(use\s+the\s+)?(?:z-image|generate\s*ai\s*image|"
                            r"ai\s+image)\s+tool\s+to\s+",
                            "",
                            _eline.strip(),
                            flags=re.IGNORECASE,
                        ).strip()
                        break
                if _earlier_subj:
                    # Combine: current correction + earlier context for richer prompt
                    _subj = _subj + ". " + _earlier_subj
                    sys.stderr.write(
                        f"[postprocess] Follow-up: combined current + earlier: {_subj[:200]}\n"
                    )

            if _subj:
                # BEST: use the LLM's own composed visual prompt if available.
                # The LLM often builds a perfect prompt but hallucinates the
                # tool call.  Using its prompt is better than re-deriving.
                _llm_prompt = _extract_llm_visual_prompt(response_text, verbose_log)
                if _llm_prompt:
                    _early_prompt = _llm_prompt
                    sys.stderr.write(
                        f"[postprocess] Early rescue using LLM's own prompt: {_early_prompt[:200]}\n"
                    )
                else:
                    _early_prompt = _build_image_prompt(_subj)
                    sys.stderr.write(
                        f"[postprocess] Early rescue prompt (rebuilt): {_early_prompt[:200]}\n"
                    )
                sys.stderr.flush()
                try:
                    _early_result = _generate_ai_image_tool.run(_early_prompt)
                    if "![generated image]" in _early_result:
                        if _is_refusal_response(response_text):
                            sys.stderr.write(
                                "[postprocess] Detected refusal response — replacing with image\n"
                            )
                            response_text = _early_result
                        else:
                            response_text = _early_result + "\n\n" + response_text
                        sys.stderr.write(
                            f"[postprocess] Early rescue succeeded: {_early_result}\n"
                        )
                except Exception as exc:
                    sys.stderr.write(f"[postprocess] Early rescue failed: {exc}\n")

    # Narrated image rescue
    narrated_img = _narrated_image_re.search(response_text)
    if narrated_img:
        _ni_tool = narrated_img.group(1)
        _ni_prompt = narrated_img.group(2)
        sys.stderr.write(
            f"[narrated-image-rescue] Detected narrated {_ni_tool}: {_ni_prompt[:120]}\n"
        )
        sys.stderr.flush()
        try:
            if "ai" in _ni_tool.lower():
                _ni_result = _generate_ai_image_tool.run(_ni_prompt)
            else:
                _ni_result = _generate_image_tool.run(_ni_prompt)
            if "![generated image]" in _ni_result:
                _cleaned = _narrated_image_re.sub("", response_text)
                _cleaned = re.sub(
                    r"!\[[^\]]*\]\(/static/generated/[^)]+\)", "", _cleaned
                ).strip()
                response_text = _ni_result + ("\n\n" + _cleaned if _cleaned else "")
                sys.stderr.write(
                    f"[narrated-image-rescue] Success: {_ni_result}\n"
                )
            else:
                sys.stderr.write(
                    f"[narrated-image-rescue] Tool returned no image: {_ni_result[:200]}\n"
                )
        except Exception as exc:
            sys.stderr.write(f"[narrated-image-rescue] Error: {exc}\n")
            sys.stderr.flush()

    # JSON-narrated image rescue (llama-style)
    if "![generated image]" not in response_text:
        _json_narr = _narrated_json_image_re.search(response_text) or \
                      _narrated_json_image_re.search(verbose_log)
        if _json_narr:
            _jn_prompt = _json_narr.group(1)
            sys.stderr.write(
                f"[json-narrated-rescue] Detected JSON tool call: {_jn_prompt[:120]}\n"
            )
            sys.stderr.flush()
            _jn_prompt = _build_image_prompt(_jn_prompt)
            sys.stderr.write(
                f"[json-narrated-rescue] Built prompt: {_jn_prompt[:200]}\n"
            )
            try:
                _jn_result = _generate_ai_image_tool.run(_jn_prompt)
                if "![generated image]" in _jn_result:
                    response_text = _jn_result
                    sys.stderr.write(
                        f"[json-narrated-rescue] Success: {_jn_result}\n"
                    )
                else:
                    sys.stderr.write(
                        f"[json-narrated-rescue] Tool returned no image: {_jn_result[:200]}\n"
                    )
            except Exception as exc:
                sys.stderr.write(f"[json-narrated-rescue] Error: {exc}\n")
                sys.stderr.flush()

    # Orphan rescue
    orphan = _orphan_re.search(response_text)
    if orphan:
        try:
            tool_name = orphan.group(1).lower().replace(" ", "").replace("_", "")
            raw_input = orphan.group(2).strip()
            if "ai" in tool_name:
                prompt = raw_input.strip("\"'")
                try:
                    parsed = json.loads(raw_input)
                    prompt = parsed.get("prompt", raw_input)
                except (json.JSONDecodeError, TypeError):
                    pass
                response_text = _generate_ai_image_tool.run(prompt)
            else:
                parsed = json.loads(raw_input)
                if "instructions" in parsed:
                    iv = parsed["instructions"]
                    if isinstance(iv, dict):
                        iv = json.dumps(iv)
                    response_text = _generate_image_tool.run(iv)
                else:
                    response_text = _generate_image_tool.run(raw_input)
        except Exception as exc:
            sys.stderr.write(f"[orphan-rescue] Error: {exc}\n")
            sys.stderr.flush()

    # Narrated search rescue
    narrated = _narrated_search_re.search(response_text)
    if narrated:
        tool_name = narrated.group(1)
        search_query = narrated.group(2)
        sys.stderr.write(
            f"[narrated-search-rescue] Detected narrated {tool_name}: {search_query[:120]}\n"
        )
        sys.stderr.flush()
        try:
            if "duckduckgo" in tool_name.lower():
                search_result = ddg_search_wrapped.run(search_query)
            else:
                search_result = serper_search_wrapped.run(search_query)
            _user_req_match = re.search(
                r"New request:\s*(.+)", verbose_log, re.IGNORECASE
            )
            user_question = (
                _user_req_match.group(1).strip().strip('"')
                if _user_req_match
                else search_query
            )
            sys.stderr.write(
                f"[narrated-search-rescue] Synthesising answer for: {user_question[:120]}\n"
            )
            import litellm

            model = os.getenv("MODEL", "ollama/qwen3.5:9b")
            litellm_model = (
                "ollama_chat/" + model[len("ollama/") :]
                if model.startswith("ollama/")
                else model
            )
            synth_resp = litellm.completion(
                model=litellm_model,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Based on these search results, answer the question concisely "
                            f"with proper markdown formatting.\n\n"
                            f"Question: {user_question}\n\n"
                            f"Search results:\n{search_result[:4000]}"
                        ),
                    }
                ],
                api_base="http://localhost:11434",
                num_retries=0,
                temperature=0.3,
            )
            synth_text = synth_resp.choices[0].message.content or ""
            synth_text = re.sub(r"<think>[\s\S]*?</think>\s*", "", synth_text).strip()
            if synth_text:
                response_text = synth_text
                sys.stderr.write("[narrated-search-rescue] Success — replaced response\n")
            else:
                sys.stderr.write("[narrated-search-rescue] Synthesis returned empty\n")
        except Exception as exc:
            sys.stderr.write(f"[narrated-search-rescue] Error: {exc}\n")
            sys.stderr.flush()

    # Pull images from verbose log if response has none
    if "/static/generated/" not in response_text:
        images_in_log = _img_md_re.findall(verbose_log)
        if images_in_log:
            response_text += "\n\n" + "\n".join(images_in_log)

    # Validate that referenced images actually exist on disk
    _had_img_before = bool(
        re.search(r"!\[[^\]]*\]\([^)]*generated[^)]*\)", response_text)
    )

    def _validate_img(match):
        rel = match.group(1).replace("/static/", "", 1)
        return match.group(0) if (STATIC_DIR / rel).exists() else ""

    response_text = re.sub(
        r"!\[[^\]]*\]\((/static/generated/[^)]+)\)",
        _validate_img,
        response_text,
    )
    _has_img_after = bool(
        re.search(r"!\[[^\]]*\]\(/static/generated/[^)]+\)", response_text)
    )

    # Hallucination rescue
    if _had_img_before and not _has_img_after:
        sys.stderr.write(
            "[postprocess] Detected hallucinated image path — attempting rescue\n"
        )
        _rescue_prompt = None

        # BEST: extract the LLM's own composed visual prompt
        _llm_prompt = _extract_llm_visual_prompt(response_text, verbose_log)
        if _llm_prompt:
            _rescue_prompt = _llm_prompt

        # Fallback: look for prompt in Action Input JSON in log
        if not _rescue_prompt:
            _m = re.search(
                r"generate_ai_image.*?['\"]prompt['\"]:\s*['\"](.+?)['\"]",
                verbose_log,
                re.IGNORECASE | re.DOTALL,
            )
            if _m:
                _rescue_prompt = _m.group(1).strip()

        if not _rescue_prompt:
            _m = re.search(
                r'\{\s*"prompt"\s*:\s*"([^"]+)"',
                response_text,
                re.IGNORECASE,
            )
            if _m:
                _rescue_prompt = _m.group(1).strip()

        if not _rescue_prompt:
            _m = re.search(
                r"New request:\s*(.+)",
                verbose_log,
                re.IGNORECASE,
            )
            if _m:
                raw = _m.group(1).strip().strip('"')
                _subj = re.sub(
                    r"^(thanks?,?\s*|please\s+|now\s+|can you\s+|could you\s+|ok\s+|"
                    r"sure,?\s*|hey,?\s*|hi,?\s*)+",
                    "",
                    raw,
                    flags=re.IGNORECASE,
                ).strip()
                if _subj:
                    _rescue_prompt = _subj

        if _rescue_prompt and len(_rescue_prompt) < 60:
            _rescue_prompt = (
                f"{_rescue_prompt}, photorealistic, highly detailed, "
                "professional photography, 8k resolution"
            )

        if _rescue_prompt:
            sys.stderr.write(f"[postprocess] Rescue prompt: {_rescue_prompt[:120]}\n")
            sys.stderr.flush()
            try:
                _rescue_result = _generate_ai_image_tool.run(_rescue_prompt)
                if "![generated image]" in _rescue_result:
                    response_text = re.sub(
                        r'\{\s*"prompt"\s*:\s*"[^"]+"\s*\}', "", response_text
                    ).strip()
                    response_text = _rescue_result + "\n\n" + response_text
                    sys.stderr.write(
                        f"[postprocess] Rescue succeeded: {_rescue_result}\n"
                    )
            except Exception as exc:
                sys.stderr.write(f"[postprocess] Rescue failed: {exc}\n")
        else:
            sys.stderr.write("[postprocess] Could not extract prompt for rescue\n")

    response_text = re.sub(r"<result>\s*</result>", "", response_text)
    response_text = re.sub(r"<think>[\s\S]*?</think>\s*", "", response_text)
    response_text = re.sub(
        r"Thought:.*?Action:.*?Action Input:.*?$",
        "",
        response_text,
        flags=re.DOTALL,
    )
    response_text = response_text.strip()

    if not response_text:
        images_in_log = _img_md_re.findall(verbose_log)
        valid = [
            img
            for img in images_in_log
            if (
                STATIC_DIR / img.split("(")[1].rstrip(")").replace("/static/", "", 1)
            ).exists()
        ]
        if valid:
            response_text = valid[-1]
        else:
            response_text = (
                "The agent could not complete the request. Please try rephrasing."
            )

    return response_text


# Detect incomplete agent output (hit max iterations or malformed finish)
_INCOMPLETE_MARKERS = [
    "Maximum iterations reached",
    "Invalid response from LLM call",
    "Please try rephrasing",
]


def _is_incomplete(response_text: str, verbose_log: str) -> bool:
    """Return True if the agent output looks like it was cut short."""
    for marker in _INCOMPLETE_MARKERS:
        if marker in response_text or marker in verbose_log:
            return True
    if re.search(r"Action\s*Input\s*:\s*\{[^}]*\}\s*$", response_text):
        return True
    return False


def _extract_usage(result) -> dict:
    if hasattr(result, "token_usage") and result.token_usage:
        tu = result.token_usage
        return {
            "total_tokens": getattr(tu, "total_tokens", 0),
            "prompt_tokens": getattr(tu, "prompt_tokens", 0),
            "completion_tokens": getattr(tu, "completion_tokens", 0),
        }
    return {}
