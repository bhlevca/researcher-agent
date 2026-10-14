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
        # Elements the script builds itself count as present.
        known = ids | set(re.findall(r"""id=\\?["']([\w-]+)""", src))
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
