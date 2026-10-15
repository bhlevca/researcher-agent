"""Every path to the GPU goes through the queue.

A queue that covers some paths is worse than none, because it looks like it is
working. That is exactly how this was found: the tutor queued correctly while
two people ran the Research tab at once, both models loaded, and Ollama spilled
one of them to CPU — `ollama ps` showed `21%/79% CPU/GPU`. The tutor's own
tests were green throughout.

So this is a structural check rather than a behavioural one. Heavy model work
is always launched from async route code via ``run_in_executor``, so that is
the chokepoint: each such call must be lexically inside a ``GPU.reserve`` or
``GPU.exclusive`` block, or carry a ``# gpu-queue-exempt: <reason>`` comment
saying why it does not touch the card.

It cannot prove the queue works — the timing tests in test_gpu_broker.py do
that. It proves nobody added a fifth tab that quietly bypasses it.
"""

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "researcher"
MODULES = sorted(p for p in SRC.rglob("*.py") if "test" not in p.name)

EXEMPT = "gpu-queue-exempt"


def _is_gpu_reservation(node) -> bool:
    """True for ``async with GPU.reserve(...)`` / ``GPU.exclusive(...)``."""
    for item in node.items:
        call = item.context_expr
        if (isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr in {"reserve", "exclusive"}
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "GPU"):
            return True
    return False


def _unguarded(tree, source_lines):
    """Every run_in_executor call with no reservation above it and no waiver."""
    found = []

    def walk(node, guarded):
        if isinstance(node, (ast.AsyncWith, ast.With)) and _is_gpu_reservation(node):
            guarded = True
        for child in ast.iter_child_nodes(node):
            if (isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and child.func.attr == "run_in_executor"
                    and not guarded):
                # A waiver sits on the call or on one of the two lines above.
                window = source_lines[max(0, child.lineno - 4):child.lineno]
                if not any(EXEMPT in ln for ln in window):
                    found.append(child.lineno)
            walk(child, guarded)

    walk(tree, False)
    return found


@pytest.mark.parametrize("module", MODULES, ids=lambda p: p.name)
def test_gpu_work_is_queued(module):
    source = module.read_text()
    lines = source.splitlines()
    bad = _unguarded(ast.parse(source), lines)
    assert not bad, (
        f"{module.relative_to(SRC)}: run_in_executor at line(s) {bad} is not "
        f"inside GPU.reserve/GPU.exclusive. Wrap it, or add a "
        f"'# {EXEMPT}: <why>' comment if it does not use the card.")


def test_every_tab_that_streams_reports_the_queue():
    """A queued request must say so, not just wait silently.

    The four tabs people actually use all stream, so each one's route has to
    emit the shared queue event — otherwise the wait is invisible and
    indistinguishable from a hang.
    """
    for name in ("chat.py", "tutor.py", "composer.py"):
        src = (SRC / "routes" / name).read_text()
        assert "queue_message(" in src, f"{name} never tells anyone they queued"
        assert 'GPU.reserve' in src, f"{name} does not take a place in the queue"


def test_the_browser_shows_it_on_every_page():
    """The note is one shared script; a per-tab copy is how this broke before."""
    static = SRC / "static"
    shared = static / "js" / "gpu-queue.js"
    assert shared.is_file()
    for page in static.glob("*.html"):
        assert "gpu-queue.js" in page.read_text(), f"{page.name} cannot show the queue"
    # chat.js and tutor-chat.js must route the event to the shared renderer.
    for js in ("chat.js", "tutor-chat.js"):
        src = (static / "js" / js).read_text()
        assert "gpuShowQueueNote" in src, f"{js} ignores the queue event"


def test_only_one_script_defines_the_board():
    """The duplicate-function bug, pinned for this specific module."""
    static = SRC / "static" / "js"
    owners = [f.name for f in static.glob("*.js")
              if "function gpuRenderQueueNote" in f.read_text()]
    assert owners == ["gpu-queue.js"], f"board rendered in {owners}"


def test_the_panel_is_permanent_not_only_under_contention():
    """A panel that appears only when two requests collide cannot be trusted.

    When it is absent you cannot tell "queue empty" from "queue broken" — and
    that ambiguity is exactly what made the queue look broken while it was
    working: gpt-oss was failing in two seconds, so nothing ever overlapped
    and no note ever appeared.
    """
    js = (SRC / "static" / "js" / "gpu-queue.js").read_text()
    # Polling starts on page load, not from the SSE handler.
    assert "DOMContentLoaded" in js and "gpuStartPolling" in js
    assert "setInterval(gpuPoll" in js, "the panel must keep polling"
    # The panel renders an explicit idle state rather than hiding itself.
    assert "Nobody is using the card" in js
    assert "'idle'" in js


def test_the_session_log_is_kept():
    """A wait that has passed still explains why something took 90 seconds."""
    js = (SRC / "static" / "js" / "gpu-queue.js").read_text()
    assert "sessionStorage" in js, "history must survive tab navigation"
    assert "gpuLog(" in js
    # Storage can throw in a private window; the panel must still work.
    assert js.count("catch") >= 6


def test_the_panel_needs_no_markup_in_any_page():
    """It is injected, so a new tab cannot forget to include it."""
    js = (SRC / "static" / "js" / "gpu-queue.js").read_text()
    assert "document.body.appendChild" in js
    for page in (SRC / "static").glob("*.html"):
        assert "gpu-panel" not in page.read_text(), (
            f"{page.name} hand-rolls the panel; it is injected")
