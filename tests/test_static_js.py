"""The browser-side checks pytest can actually make.

These exist because a green suite is not evidence a page works. Two functions
in `tutor-sheets.js` silently overrode `tutor-lessons.js` — it loads last — so
the Lessons tab's Generate button called a blocking, non-streaming function
instead of the streaming one. Every test passed throughout, because none of
them load a page.

This does not replace clicking the thing. It catches the one class of mistake
that is invisible from Python and expensive to find by hand.
"""

import re
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / "src" / "researcher" / "static"
PAGES = sorted(STATIC.glob("*.html"))


def _scripts(html: str) -> list[str]:
    """Script filenames in load order, cache-buster stripped."""
    return [s.split("?")[0] for s in
            re.findall(r'<script src="/static/js/([^"]+)"', html)]


def _declared(src: str) -> set[str]:
    """`function NAME(...)` declarations only.

    Deliberately narrow. These are the ones hoisted into one shared global
    scope, where a later file silently replaces an earlier one with no error
    anywhere — which is exactly how the Lessons tab broke.
    """
    return set(re.findall(r"^(?:async\s+)?function\s+(\w+)", src, re.M))


def _callable_names(html: str) -> set[str]:
    """Everything an onclick could legitimately reach on this page."""
    names: set[str] = set()
    for name in _scripts(html):
        f = STATIC / "js" / name
        if f.is_file():
            src = f.read_text()
            names |= _declared(src)
            names |= set(re.findall(
                r"^\s*(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:function|\()",
                src, re.M))
            names |= set(re.findall(r"window\.(\w+)\s*=", src))
    # Handlers defined in the page's own inline <script> blocks.
    for block in re.findall(r"<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)</script>", html):
        names |= _declared(block)
        names |= set(re.findall(
            r"^\s*(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:function|\()",
            block, re.M))
    return names


def _handlers(html: str) -> set[str]:
    """The function each onclick invokes.

    Only the head of each statement counts: in
    `document.getElementById('x').click()` neither getElementById nor click is
    a handler the page has to define.
    """
    out = set()
    for expr in re.findall(r'onclick="([^"]+)"', html):
        for stmt in re.split(r"[;&|]+", expr):
            m = re.match(r"\s*([a-zA-Z_]\w*)\s*\(", stmt)
            if m:
                out.add(m.group(1))
    return out


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_no_two_scripts_define_the_same_function(page):
    """The later script wins silently. This is the check that earns its keep."""
    seen: dict[str, str] = {}
    clashes = []
    for name in _scripts(page.read_text()):
        f = STATIC / "js" / name
        if not f.is_file():
            continue
        for fn in _declared(f.read_text()):
            if fn in seen and seen[fn] != name:
                clashes.append(f"{fn}(): {seen[fn]} then {name}")
            seen[fn] = name
    assert not clashes, f"{page.name}: {clashes}"


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_every_onclick_handler_is_defined(page):
    html = page.read_text()
    missing = _handlers(html) - _callable_names(html) - {"if", "confirm", "alert"}
    assert not missing, f"{page.name} calls undefined: {sorted(missing)}"


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_scripts_do_not_reach_for_elements_that_never_exist(page):
    """getElementById on a missing id returns null and the handler dies
    silently, so the page just stops responding."""
    html = page.read_text()
    ids = set(re.findall(r'id="([^"]+)"', html))
    missing = []
    for name in _scripts(html):
        f = STATIC / "js" / name
        # Only the scripts belonging to this page; shared ones serve several.
        if not f.is_file() or not name.startswith(page.stem[:5]):
            continue
        src = f.read_text()
        # Elements the script builds itself count as present: either written
        # into an HTML string (id="x") or assigned on a created node
        # (el.id = 'x'), which the first pattern alone missed.
        known = (ids
                 | set(re.findall(r"""id=\\?["']([\w-]+)""", src))
                 | set(re.findall(r"""\.id\s*=\s*["']([\w-]+)["']""", src)))
        for wanted in re.findall(r"getElementById\('([^']+)'\)", src):
            if wanted not in known:
                missing.append(f"{name}: #{wanted}")
    assert not missing, f"{page.name}: {missing}"


def test_html_div_tags_balance():
    """An unbalanced div silently swallows the rest of the layout."""
    for page in PAGES:
        html = page.read_text()
        opens = len(re.findall(r"<div\b", html))
        closes = len(re.findall(r"</div>", html))
        assert opens == closes, f"{page.name}: {opens} <div> vs {closes} </div>"


def test_the_tutor_has_exactly_one_lesson_generator():
    """The streaming one in tutor-lessons.js. A second, blocking generator in
    tutor-sheets.js overrode it and cost several rounds of debugging; printing
    is now an action on a lesson rather than a parallel way to make one."""
    sheets = (STATIC / "js" / "tutor-sheets.js").read_text()
    assert "function generateLesson" not in sheets
    assert "function downloadLesson" not in sheets
    lessons = (STATIC / "js" / "tutor-lessons.js").read_text()
    assert "function generateLesson" in lessons
    assert "processTutorSSE" in lessons, "the lesson generator must stream"


# ---------------------------------------------------------------------------
# Layout rules that are easy to break and invisible from Python.
# ---------------------------------------------------------------------------

def _order_of(src: str, fn_start: str, *markers: str) -> list[int]:
    """Positions of each marker inside the function starting at fn_start."""
    i = src.index(fn_start)
    # Up to the next top-level function, so we only look inside this one.
    j = src.find("\nfunction ", i + 1)
    body = src[i:j if j != -1 else len(src)]
    return [body.index(m) for m in markers]


def test_the_reasoning_box_sits_above_the_answer():
    """It is where the live reasoning box already was.

    Appending the collapsed box below the answer made it jump across the
    answer the moment the result arrived, so the thing you were reading ended
    up somewhere other than where it had been.
    """
    cases = [
        ("js/messages.js", "function addAssistantMessage",
         "reasoning-toggle", "m.className = 'message assistant'"),
        ("js/composer-chat.js", "function addComposerMsg",
         "reasoning-toggle", "m.className = 'message ' + role"),
        ("js/composer-score.js", "function renderScoreResponse",
         "reasoning-toggle", "div.className = 'message assistant'"),
    ]
    for rel, fn, reasoning_marker, answer_marker in cases:
        src = (STATIC / rel).read_text()
        at_reasoning, at_answer = _order_of(src, fn, reasoning_marker, answer_marker)
        assert at_reasoning < at_answer, (
            f"{rel}: the reasoning box is built after the answer")


def test_the_gpu_panel_does_not_float_over_the_input():
    """It was a fixed overlay first, and covered the typing box on every tab.

    In the document flow it becomes the last flex item of <body>, which on
    these pages is a 100vh flex column, so it gets its own line underneath.
    """
    css = (STATIC / "css" / "main.css").read_text()
    block = css[css.index(".gpu-panel {"):css.index(".gpu-panel-head")]
    assert "position: fixed" not in block, "the panel must not overlay the page"
    assert "flex: 0 0 auto" in block, "it must sit in the body's flex column"


def test_the_missing_element_check_still_catches_a_real_mistake():
    """The detection was widened for `el.id = 'x'`; it must not go blind.

    Written because widening a guard is how guards quietly stop guarding.
    """
    html = '<div id="real"></div><script src="/static/js/fake-page.js"></script>'
    src_builds = "const el = document.createElement('div'); el.id = 'built';\n" \
                 "document.getElementById('built');"
    src_missing = "document.getElementById('never-exists');"

    def missing_for(src):
        ids = set(re.findall(r'id="([^"]+)"', html))
        known = (ids
                 | set(re.findall(r"""id=\\?["']([\w-]+)""", src))
                 | set(re.findall(r"""\.id\s*=\s*["']([\w-]+)["']""", src)))
        return [w for w in re.findall(r"getElementById\('([^']+)'\)", src)
                if w not in known]

    assert missing_for(src_builds) == [], "an id the script creates is present"
    assert missing_for(src_missing) == ["never-exists"], "a typo must still fail"
