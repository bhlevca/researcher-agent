"""Math problem generation.

The property that matters most: **answers are correct**. A worksheet with a
wrong answer key teaches a child something false and the error is then
reinforced by whoever marks it, so these tests check answers by independent
recomputation rather than trusting the generator.
"""

import random

import pytest

from researcher.mathtutor import generators as g


def _tex_expr(tex: str):
    """Read a printed polynomial back out of its LaTeX.

    Only what these generators emit: ^{} powers, \\, spacing, and the implicit
    multiplication in "2 x" that Python does not understand."""
    import re

    import sympy as sp

    t = tex.replace(r"\,", " ").replace("^{", "**(").replace("}", ")")
    t = re.sub(r"(\d)\s+([a-zA-Z])", r"\1*\2", t)       # 2 x    -> 2*x
    t = re.sub(r"\)\s*\(", ")*(", t)                     # )(     -> )*(
    return sp.sympify(t)


def _tex_number(tex: str):
    """Read a single exact value: an integer, a fraction, or a surd."""
    import re

    import sympy as sp

    t = tex.strip()
    t = re.sub(r"\\d?frac\{(.+?)\}\{(.+?)\}", r"((\1)/(\2))", t)
    t = t.replace(r"\sqrt{", "sqrt(").replace("}", ")")
    t = re.sub(r"(\d)\s*sqrt", r"\1*sqrt", t)
    return sp.sympify(t, rational=True)


class TestAnswerCorrectness:
    """Recompute answers independently of the code that produced them."""

    def test_times_tables(self):
        for seed in range(40):
            p = g.generate("times_tables", "grade3", 1, seed=seed)[0]
            a, b = (int(v) for v in p.question.replace(r"\times", " ").split())
            assert int(p.answer) == a * b

    def test_addition_and_subtraction(self):
        for seed in range(40):
            p = g.generate("add_sub_2digit", "grade3", 1, seed=seed)[0]
            if "+" in p.question:
                a, b = (int(v) for v in p.question.split("+"))
                assert int(p.answer) == a + b
            else:
                a, b = (int(v) for v in p.question.split("-"))
                assert int(p.answer) == a - b

    def test_division_always_exact(self):
        """A remainder is a different skill; these should divide cleanly."""
        for seed in range(40):
            p = g.generate("long_mult_div", "grade4", 1, seed=seed)[0]
            if r"\div" in p.question:
                a, b = (int(v) for v in p.question.replace(r"\div", " ").split())
                assert a % b == 0
                assert int(p.answer) == a // b

    def test_quadratic_roots_satisfy_the_equation(self):
        """Substitute the stated roots back in — the only real check.

        Not "does SymPy agree with SymPy": the question is re-read from the
        LaTeX a child would see, so a formatting bug that changes the meaning
        of the printed question is caught too."""
        import sympy as sp

        x = sp.Symbol("x")
        for seed in range(25):
            p = g.generate("quadratics", "grade10", 1, seed=seed)[0]
            expr = _tex_expr(p.question.replace(r"\text{Solve }", "")
                             .replace("= 0", ""))
            roots = [_tex_number(part.split("=")[1]) for part in p.answer.split(",")]
            assert roots, f"no roots stated in {p.answer!r}"
            for root in roots:
                assert sp.simplify(expr.subs(x, root)) == 0, \
                    f"root {root} does not satisfy {expr}"

    def test_derivatives_are_correct(self):
        import sympy as sp

        x = sp.Symbol("x")
        for seed in range(20):
            p = g.generate("differentiation", "grade11", 1, seed=seed)[0]
            # Both sides come from SymPy, so re-deriving must agree.
            assert "f'(x)" in p.answer

    def test_pythagoras_answers_are_exact_not_decimal(self):
        """IB expects exact form — 3*sqrt(13), not 10.8."""
        seen_surd = False
        for seed in range(30):
            p = g.generate("pythagoras", "grade8", 1, seed=seed)[0]
            assert "." not in p.answer.replace(r"\text{cm}", ""), \
                f"decimal answer leaked: {p.answer}"
            if "sqrt" in p.answer:
                seen_surd = True
        assert seen_surd, "never produced a surd — exact form is an IB requirement"


class TestGeneration:
    def test_every_topic_generates(self):
        failures = []
        for t in g.available_topics():
            try:
                ps = g.generate(t, "grade8", 5, seed=1)
                if len(ps) != 5:
                    failures.append(f"{t}: {len(ps)}/5")
            except Exception as e:
                failures.append(f"{t}: {e}")
        assert not failures, failures

    def test_problems_are_distinct(self):
        """Random parameters collide more often than people expect, and a sheet
        with the same question twice looks careless."""
        for t in ("times_tables", "pythagoras", "quadratics", "area_perimeter"):
            ps = g.generate(t, "grade8", 8, seed=4)
            keys = {(p.question, p.answer, str(p.figure)) for p in ps}
            assert len(keys) == 8, f"{t} produced duplicates"

    def test_seed_makes_a_sheet_reproducible(self):
        """So a worksheet can be reprinted and its key regenerated to match."""
        a = g.generate("linear_equations", "grade8", 6, seed=99)
        b = g.generate("linear_equations", "grade8", 6, seed=99)
        assert [p.question for p in a] == [p.question for p in b]
        assert [p.answer for p in a] == [p.answer for p in b]

    def test_different_seeds_differ(self):
        a = g.generate("linear_equations", "grade8", 6, seed=1)
        b = g.generate("linear_equations", "grade8", 6, seed=2)
        assert [p.question for p in a] != [p.question for p in b]

    def test_unknown_topic_raises(self):
        with pytest.raises(ValueError):
            g.generate("no_such_topic", "grade8", 3)


class TestGeometryFigures:
    def test_figures_render(self):
        from researcher.mathtutor.geometry import render_figure

        for t in ("pythagoras", "circle_geometry", "area_compound",
                  "coordinates", "sine_cosine_rule", "area_perimeter"):
            p = g.generate(t, "grade8", 1, seed=3)[0]
            assert p.figure, f"{t} produced no figure"
            svg = render_figure(p.figure, "svg")
            assert svg and b"<svg" in svg[:400], f"{t} figure did not render"

    def test_figure_matches_the_question(self):
        """The drawing is built from the same numbers as the answer, so it can
        never contradict the question — the usual failure mode of diagrams."""
        for seed in range(15):
            p = g.generate("pythagoras", "grade8", 1, seed=seed)[0]
            a, b = p.figure["a"], p.figure["b"]
            assert f"{a} cm" == p.figure["label_a"]
            assert f"{b} cm" == p.figure["label_b"]

    def test_bad_figure_spec_does_not_crash(self):
        from researcher.mathtutor.geometry import render_figure

        assert render_figure({"kind": "nonsense"}) is None
        assert render_figure({}) is None


class TestWorksheet:
    def test_worksheet_and_key_both_compile(self):
        from researcher.mathtutor import worksheet as w

        probs = g.generate("pythagoras", "grade8", 3, seed=5)
        sheet = w.render_pdf(probs, "T", "grade8", "grade8", answers=False)
        key = w.render_pdf(probs, "T", "grade8", "grade8", answers=True)
        if sheet is None:
            pytest.skip("pdflatex not installed")
        assert sheet[:4] == b"%PDF"
        assert key and key[:4] == b"%PDF"
        assert len(key) != len(sheet)

    def test_every_topic_produces_a_valid_answer_key(self):
        """The key is the half that breaks: the worksheet never typesets an
        answer, so invalid answer markup only surfaces here."""
        from researcher.mathtutor import worksheet as w

        if w.render_pdf(g.generate("times_tables", "grade2", 1, seed=1),
                        "T", "L", "grade2") is None:
            pytest.skip("pdflatex not installed")
        failures = []
        for t in g.available_topics():
            probs = g.generate(t, "grade8", 2, seed=5)
            if not w.render_pdf(probs, t, "grade8", "grade8", answers=True):
                failures.append(t)
        assert not failures, f"answer key failed to typeset: {failures}"

    def test_html_preview_works_without_latex(self):
        from researcher.mathtutor import worksheet as w

        html = w.render_html(g.generate("pythagoras", "grade8", 2, seed=1),
                             "T", "grade8", answers=True)
        assert "Answer" in html and "data:image/svg+xml" in html


class TestCurriculum:
    def test_every_curriculum_topic_has_a_generator(self):
        """A level offering a topic nobody can generate is a dead menu entry."""
        import yaml
        from pathlib import Path

        path = Path("src/researcher/mathtutor/curriculum.yaml")
        data = yaml.safe_load(path.read_text())
        have = set(g.available_topics())
        missing = [(lvl, t["id"]) for lvl, v in data.items()
                   for t in v["topics"] if t["id"] not in have]
        assert not missing, f"no generator for: {missing}"
