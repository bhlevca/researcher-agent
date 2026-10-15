"""Problem generation with computed answers.

The governing rule: **an LLM never produces an answer**. Every answer here is
computed by SymPy from the same expression that generated the question, so the
answer key is correct by construction rather than by hope.

This matters more than it does anywhere else in this project. A wrong note in a
composition is a bad listen; a wrong answer on a worksheet teaches a child
something false, and the mistake is then reinforced by the person marking it.
Language models are fluent at arithmetic and unreliable at it — they will
confidently produce `17 x 23 = 389` — so they are kept away from the answer
entirely. Their role is wording and context, applied to problems that are
already correct.

Each generator returns `Problem` objects carrying LaTeX for the question, the
answer, and worked steps where showing the method matters.
"""

from __future__ import annotations

import re
import random
import logging
from dataclasses import dataclass, field

from researcher.mathtutor.i18n import T

logger = logging.getLogger(__name__)


@dataclass
class Problem:
    """One problem, with its answer already computed."""

    question: str                      # LaTeX, no surrounding $
    answer: str                        # LaTeX
    steps: list[str] = field(default_factory=list)   # LaTeX working
    topic: str = ""
    level: str = ""
    # Geometry problems carry a figure spec; worksheet.py renders it.
    figure: dict | None = None
    # Multiple choice options when the format calls for it.
    options: list[str] = field(default_factory=list)
    # On a bilingual sheet, the same question in the second language. Generators
    # never set these: they call T() once per sentence and generate() splits the
    # result, so a generator stays unaware of how many languages are in play.
    question_alt: str = ""
    answer_alt: str = ""
    steps_alt: list[str] = field(default_factory=list)
    lang: str = "en"

    def as_dict(self) -> dict:
        return {
            "question": self.question, "answer": self.answer,
            "steps": self.steps, "topic": self.topic, "level": self.level,
            "figure": self.figure, "options": self.options,
            "question_alt": self.question_alt, "answer_alt": self.answer_alt,
            "steps_alt": self.steps_alt, "lang": self.lang,
        }


def _sp():
    import sympy
    return sympy


_TRIG_CALL = re.compile(
    r"\\(sin|cos|tan|sec|csc|cot|arcsin|arccos|arctan|log|ln)"
    r"(\^\{[^{}]*\})?\{\\left\((.+?) \\right\)\}")


def _latex(expr) -> str:
    """LaTeX in the notation an IB examiner writes.

    SymPy is correct but verbose: it emits \\sin{\\left(x \\right)} where a
    mark scheme has \\sin x, and \\operatorname{atan} for \\arctan. On a
    worksheet the difference is the gap between "typeset by a person" and
    "dumped by a computer algebra system"."""
    from sympy import latex

    out = latex(expr, inv_trig_style="full")

    def _tidy(m):
        fn, sup, arg = m.group(1), m.group(2) or "", m.group(3).strip()
        # A bare symbol needs no brackets; anything else keeps them.
        return (rf"\{fn}{sup} {arg}" if re.fullmatch(r"[a-zA-Z]\w*", arg)
                else rf"\{fn}{sup}\left({arg}\right)")

    return _TRIG_CALL.sub(_tidy, out)


def _grade(level: str) -> int:
    """Grade number out of a level id like "grade7".

    An unrecognised level reads as 8, the middle of the range, so a generator
    scales to something sensible rather than crashing the sheet."""
    m = re.search(r"(\d+)", level or "")
    return int(m.group(1)) if m else 8


# ---------------------------------------------------------------------------
# Primary (PYP)
# ---------------------------------------------------------------------------


def gen_add_sub_2digit(rng: random.Random, level: str, lang: str = "en") -> Problem:
    a, b = rng.randint(120, 899), rng.randint(110, 450)
    if rng.random() < 0.5:
        return Problem(question=f"{a} + {b}", answer=str(a + b),
                       topic="add_sub_2digit", level=level)
    a, b = max(a, b), min(a, b)
    return Problem(question=f"{a} - {b}", answer=str(a - b),
                   topic="add_sub_2digit", level=level)


def gen_times_tables(rng: random.Random, level: str, lang: str = "en") -> Problem:
    a, b = rng.randint(3, 12), rng.randint(3, 12)
    return Problem(question=rf"{a} \times {b}", answer=str(a * b),
                   topic="times_tables", level=level)


def gen_long_mult_div(rng: random.Random, level: str, lang: str = "en") -> Problem:
    if rng.random() < 0.5:
        a, b = rng.randint(23, 99), rng.randint(12, 49)
        return Problem(question=rf"{a} \times {b}", answer=str(a * b),
                       steps=[rf"{a} \times {b} = {a*b}"],
                       topic="long_mult_div", level=level)
    # Division is constructed from its answer so it always divides exactly —
    # a remainder here would be a different skill, taught separately.
    b, q = rng.randint(3, 19), rng.randint(12, 89)
    return Problem(question=rf"{b*q} \div {b}", answer=str(q),
                   topic="long_mult_div", level=level)


def gen_fractions_unit(rng: random.Random, level: str, lang: str = "en") -> Problem:
    sp = _sp()
    den = rng.choice([2, 3, 4, 5, 6, 8, 10])
    whole = den * rng.randint(2, 9)
    ans = sp.Rational(whole, den)
    return Problem(
        question=T("fraction_of_amount", lang,
                   frac=rf"\frac{{1}}{{{den}}}", n=whole),
        answer=_latex(ans), topic="fractions_unit", level=level)


def gen_fractions_ops(rng: random.Random, level: str, lang: str = "en") -> Problem:
    sp = _sp()
    a = sp.Rational(rng.randint(1, 5), rng.choice([2, 3, 4, 5, 6, 8]))
    b = sp.Rational(rng.randint(1, 5), rng.choice([2, 3, 4, 5, 6, 8]))
    op, res = (r"+", a + b) if rng.random() < 0.5 else (r"-", a - b)
    return Problem(
        question=rf"{_latex(a)} {op} {_latex(b)}",
        answer=_latex(res),
        steps=[T("step_common_denominator", lang, n=sp.ilcm(a.q, b.q))],
        topic="fractions_ops", level=level)


def gen_decimals(rng: random.Random, level: str, lang: str = "en") -> Problem:
    sp = _sp()
    a = sp.Rational(rng.randint(105, 999), 100)
    b = sp.Rational(rng.randint(105, 999), 100)
    res = a + b if rng.random() < 0.5 else a - b
    op = "+" if res == a + b else "-"
    return Problem(question=rf"{float(a)} {op} {float(b)}",
                   answer=f"{float(res):g}", topic="decimals", level=level)


def gen_percentages(rng: random.Random, level: str, lang: str = "en") -> Problem:
    sp = _sp()
    pct = rng.choice([5, 10, 12.5, 15, 20, 25, 40, 60, 75])
    amount = rng.choice([40, 60, 80, 120, 150, 200, 240, 360])
    ans = sp.Rational(int(pct * 10), 1000) * amount
    return Problem(question=T("percent_of_amount", lang,
                              pct=rf"{pct:g}\%", n=amount),
                   answer=_latex(sp.nsimplify(ans)),
                   topic="percentages", level=level)


# ---------------------------------------------------------------------------
# Lower secondary (MYP)
# ---------------------------------------------------------------------------


def gen_integers(rng: random.Random, level: str, lang: str = "en") -> Problem:
    a, b = rng.randint(-15, -2), rng.randint(2, 15)
    style = rng.randint(0, 2)
    if style == 0:
        return Problem(question=f"({a}) + ({-b})", answer=str(a - b),
                       topic="integers", level=level)
    if style == 1:
        return Problem(question=rf"({a}) \times ({-b})", answer=str(a * -b),
                       topic="integers", level=level)
    return Problem(question=f"({a}) - ({b})", answer=str(a - b),
                   topic="integers", level=level)


def gen_ratio_proportion(rng: random.Random, level: str, lang: str = "en") -> Problem:
    sp = _sp()
    a, b = rng.randint(2, 9), rng.randint(2, 9)
    k = rng.randint(3, 12)
    total = (a + b) * k
    return Problem(
        question=T("share_in_ratio", lang, total=total, ratio=f"{a}:{b}"),
        answer=T("pair_and", lang, a=a * k, b=b * k),
        steps=[T("step_total_parts", lang, expr=f"{a}+{b} = {a+b}"),
               T("step_one_part", lang, expr=rf"{total} \div {a+b} = {k}")],
        topic="ratio_proportion", level=level)


def gen_linear_simple(rng: random.Random, level: str, lang: str = "en") -> Problem:
    sp = _sp()
    x = sp.Symbol("x")
    a, sol = rng.randint(2, 9), rng.randint(-9, 9)
    b = rng.randint(-15, 15)
    eq = sp.Eq(a * x + b, a * sol + b)
    ans = sp.solve(eq, x)[0]
    return Problem(
        question=T("solve", lang, eq=f"{_latex(eq.lhs)} = {_latex(eq.rhs)}"),
        answer=rf"x = {_latex(ans)}",
        steps=[rf"{_latex(a*x)} = {_latex(eq.rhs - b)}", rf"x = {_latex(ans)}"],
        topic="linear_simple", level=level)


def gen_linear_equations(rng: random.Random, level: str, lang: str = "en") -> Problem:
    sp = _sp()
    x = sp.Symbol("x")
    sol = sp.Rational(rng.randint(-12, 12), rng.choice([1, 1, 1, 2]))
    a, b, c = rng.randint(2, 9), rng.randint(-9, 9), rng.randint(2, 7)
    lhs = a * x + b
    rhs = c * x + (lhs.subs(x, sol) - c * sol)
    eq = sp.Eq(lhs, sp.nsimplify(rhs))
    ans = sp.solve(eq, x)[0]
    return Problem(
        question=T("solve", lang, eq=f"{_latex(eq.lhs)} = {_latex(eq.rhs)}"),
        answer=rf"x = {_latex(ans)}",
        steps=[T("step_collect_x_terms", lang, x="x"), rf"x = {_latex(ans)}"],
        topic="linear_equations", level=level)


def gen_simultaneous(rng: random.Random, level: str, lang: str = "en") -> Problem:
    sp = _sp()
    x, y = sp.symbols("x y")
    sx, sy = rng.randint(-8, 8), rng.randint(-8, 8)
    a, b = rng.randint(1, 6), rng.randint(1, 6)
    c, d = rng.randint(1, 6), rng.randint(-6, -1)
    e1 = sp.Eq(a * x + b * y, a * sx + b * sy)
    e2 = sp.Eq(c * x + d * y, c * sx + d * sy)
    sol = sp.solve([e1, e2], [x, y])
    return Problem(
        question=T("solve_simultaneous", lang,
                   e1=f"{_latex(e1.lhs)} = {_latex(e1.rhs)}",
                   e2=f"{_latex(e2.lhs)} = {_latex(e2.rhs)}"),
        answer=rf"x = {_latex(sol[x])},\ y = {_latex(sol[y])}",
        topic="simultaneous", level=level)


def gen_indices(rng: random.Random, level: str, lang: str = "en") -> Problem:
    sp = _sp()
    x = sp.Symbol("x")
    m, n = rng.randint(2, 7), rng.randint(2, 6)
    if rng.random() < 0.5:
        expr = x**m * x**n
        return Problem(question=T("simplify", lang, expr=_latex(expr)),
                       answer=_latex(sp.powsimp(expr, force=True)),
                       topic="indices", level=level)
    expr = (x**m) ** n
    return Problem(question=T("simplify", lang, expr=_latex(expr)),
                   answer=_latex(sp.powsimp(expr, force=True)),
                   topic="indices", level=level)


def gen_quadratics(rng: random.Random, level: str, lang: str = "en") -> Problem:
    sp = _sp()
    x = sp.Symbol("x")
    r1, r2 = rng.randint(-9, 9), rng.randint(-9, 9)
    a = rng.choice([1, 1, 1, 2])
    expr = sp.expand(a * (x - r1) * (x - r2))
    roots = sorted(sp.solve(sp.Eq(expr, 0), x), key=lambda r: float(r))
    return Problem(
        question=T("solve", lang, eq=f"{_latex(expr)} = 0"),
        answer=", ".join(rf"x = {_latex(r)}" for r in roots),
        steps=[rf"{_latex(sp.factor(expr))} = 0"],
        topic="quadratics", level=level)


def gen_surds(rng: random.Random, level: str, lang: str = "en") -> Problem:
    sp = _sp()
    k = rng.choice([2, 3, 5, 6, 7, 10])
    sq = rng.choice([4, 9, 16, 25, 36])
    expr = sp.sqrt(k * sq)
    return Problem(question=T("simplify", lang, expr=rf"\sqrt{{{k*sq}}}"),
                   answer=_latex(sp.sqrt(k * sq)),
                   topic="surds", level=level)


def gen_sequences(rng: random.Random, level: str, lang: str = "en") -> Problem:
    sp = _sp()
    if rng.random() < 0.5:
        a1, d = rng.randint(-10, 15), rng.randint(2, 9)
        n = rng.randint(8, 25)
        return Problem(
            question=T("arithmetic_sequence", lang, u1=f"u_1 = {a1}",
                       d=f"d = {d}", un=f"u_{{{n}}}"),
            answer=str(a1 + (n - 1) * d),
            steps=[rf"u_n = u_1 + (n-1)d = {a1} + {n-1}\times{d}"],
            topic="sequences", level=level)
    a1, r = rng.randint(1, 6), rng.choice([2, 3])
    n = rng.randint(5, 10)
    return Problem(
        question=T("geometric_sequence", lang, u1=f"u_1 = {a1}",
                   r=f"r = {r}", un=f"u_{{{n}}}"),
        answer=str(a1 * r ** (n - 1)),
        steps=[rf"u_n = u_1 r^{{n-1}} = {a1}\times{r}^{{{n-1}}}"],
        topic="sequences", level=level)


def gen_probability(rng: random.Random, level: str, lang: str = "en") -> Problem:
    sp = _sp()
    red, blue = rng.randint(3, 9), rng.randint(3, 9)
    total = red + blue
    # Without replacement — the case students get wrong.
    p = sp.Rational(red, total) * sp.Rational(red - 1, total - 1)
    return Problem(
        question=T("bag_two_without_replacement", lang, red=red, blue=blue),
        answer=_latex(p),
        steps=[rf"\frac{{{red}}}{{{total}}} \times \frac{{{red-1}}}{{{total-1}}}"],
        topic="probability", level=level)


# ---------------------------------------------------------------------------
# Diploma Programme
# ---------------------------------------------------------------------------


def gen_functions(rng: random.Random, level: str, lang: str = "en") -> Problem:
    sp = _sp()
    x = sp.Symbol("x")
    a, b, c = rng.randint(1, 5), rng.randint(-8, 8), rng.randint(-6, 6)
    f = a * x + b
    g = x**2 + c
    comp = sp.expand(g.subs(x, f))
    return Problem(
        question=T("composite_function", lang, f=f"f(x) = {_latex(f)}",
                   g=f"g(x) = {_latex(g)}", comp=r"(g \circ f)(x)"),
        answer=_latex(comp),
        steps=[rf"g(f(x)) = ({_latex(f)})^2 + {c}"],
        topic="functions", level=level)


def gen_differentiation(rng: random.Random, level: str, lang: str = "en") -> Problem:
    sp = _sp()
    x = sp.Symbol("x")
    a, b, c = rng.randint(1, 6), rng.randint(-8, 8), rng.randint(-9, 9)
    n = rng.randint(2, 4)
    expr = a * x**n + b * x + c
    d = sp.diff(expr, x)
    return Problem(
        question=T("differentiate", lang, f=f"f(x) = {_latex(expr)}"),
        answer=rf"f'(x) = {_latex(d)}",
        topic="differentiation", level=level)


def gen_integration(rng: random.Random, level: str, lang: str = "en") -> Problem:
    sp = _sp()
    x = sp.Symbol("x")
    a, n = rng.randint(1, 6), rng.randint(1, 4)
    b = rng.randint(-6, 6)
    expr = a * x**n + b
    lo, hi = rng.randint(0, 2), rng.randint(3, 6)
    val = sp.integrate(expr, (x, lo, hi))
    return Problem(
        question=T("evaluate", lang,
                   expr=rf"\int_{{{lo}}}^{{{hi}}} \left({_latex(expr)}\right)\,dx"),
        answer=_latex(val),
        steps=[rf"\left[{_latex(sp.integrate(expr, x))}\right]_{{{lo}}}^{{{hi}}}"],
        topic="integration", level=level)


def gen_logs_exp(rng: random.Random, level: str, lang: str = "en") -> Problem:
    sp = _sp()
    x = sp.Symbol("x", positive=True)
    base = rng.choice([2, 3, 5, 10])
    k = rng.randint(2, 5)
    eq = sp.Eq(base**x, base**k)
    return Problem(
        question=T("solve", lang, eq=rf"{base}^{{x}} = {base**k}"),
        answer=rf"x = {k}",
        steps=[rf"x = \log_{{{base}}} {base**k} = {k}"],
        topic="logs_exp", level=level)


def gen_binomial(rng: random.Random, level: str, lang: str = "en") -> Problem:
    sp = _sp()
    x = sp.Symbol("x")
    n = rng.randint(4, 7)
    term = rng.randint(2, n - 1)
    expr = sp.expand((1 + x) ** n)
    coeff = sp.binomial(n, term)
    return Problem(
        question=T("binomial_coefficient", lang, term=f"x^{{{term}}}",
                   expr=f"(1+x)^{{{n}}}"),
        answer=str(coeff),
        steps=[rf"\binom{{{n}}}{{{term}}} = {coeff}"],
        topic="binomial", level=level)


def gen_complex_numbers(rng: random.Random, level: str, lang: str = "en") -> Problem:
    sp = _sp()
    a, b = rng.randint(-5, 5), rng.randint(1, 5)
    c, d = rng.randint(-5, 5), rng.randint(1, 5)
    z1 = a + b * sp.I
    z2 = c + d * sp.I
    prod = sp.expand(z1 * z2)
    return Problem(
        question=T("complex_product", lang, z1=f"z_1 = {_latex(z1)}",
                   z2=f"z_2 = {_latex(z2)}", prod="z_1 z_2"),
        answer=_latex(prod),
        topic="complex_numbers", level=level)


def gen_integration_parts(rng: random.Random, level: str, lang: str = "en") -> Problem:
    sp = _sp()
    x = sp.Symbol("x")
    a = rng.randint(1, 4)
    expr = rng.choice([x * sp.exp(a * x), x * sp.sin(a * x), x * sp.cos(a * x)])
    val = sp.integrate(expr, x)
    return Problem(
        question=T("find_expr", lang, expr=rf"\int {_latex(expr)}\,dx"),
        answer=rf"{_latex(sp.simplify(val))} + c",
        steps=[T("step_integration_by_parts", lang,
                 formula=r"\int u\,dv = uv - \int v\,du")],
        topic="integration_parts", level=level)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Grade 1-2
# ---------------------------------------------------------------------------


def gen_number_bonds(rng: random.Random, level: str, lang: str = "en") -> Problem:
    """Addition and subtraction within 20 (Grade 1) or 100 (Grade 2)."""
    top = 20 if _grade(level) <= 1 else 100
    a = rng.randint(2, top - 2)
    b = rng.randint(1, top - a)
    roll = rng.random()
    if roll < 0.35:
        # A missing addend, which builds number sense rather than giving the
        # child one more sum to grind through.
        return Problem(question=rf"{a} + \square = {a + b}", answer=str(b),
                       steps=[rf"{a + b} - {a} = {b}"],
                       topic="number_bonds", level=level)
    if roll < 0.7:
        return Problem(question=f"{a} + {b}", answer=str(a + b),
                       topic="number_bonds", level=level)
    return Problem(question=f"{a + b} - {b}", answer=str(a),
                   topic="number_bonds", level=level)


# ---------------------------------------------------------------------------
# Grade 11-12 (DP Analysis & Approaches)
# ---------------------------------------------------------------------------


def gen_quadratic_theory(rng: random.Random, level: str, lang: str = "en") -> Problem:
    """Discriminant: its value, the nature of the roots, and solving for k."""
    sp = _sp()
    x, k = sp.symbols("x k")
    mode = rng.choice(["discriminant", "equal_roots", "two_distinct"])

    if mode == "discriminant":
        a = rng.randint(1, 3)
        b = rng.randint(-9, 9)
        c = rng.randint(-6, 6)
        disc = b * b - 4 * a * c
        nature_key = ("nature_two_distinct" if disc > 0 else
                      "nature_repeated" if disc == 0 else "nature_none")
        return Problem(
            question=T("discriminant_and_nature", lang,
                       eq=f"{_latex(a * x**2 + b * x + c)} = 0"),
            answer=rf"\Delta = {disc}, " + T(nature_key, lang),
            steps=[rf"\Delta = b^{{2}} - 4ac = ({b})^{{2}} - 4({a})({c}) = {disc}"],
            topic="quadratic_theory", level=level)

    c = rng.choice([4, 9, 16, 25, 36, 49, 64])
    if mode == "equal_roots":
        roots = sp.solve(sp.Eq(k**2 - 4 * c, 0), k)
        return Problem(
            question=T("equal_roots_find_k", lang,
                       eq=rf"x^{{2}} + kx + {c} = 0", k="k"),
            answer=", \\ ".join(f"k = {_latex(r)}" for r in roots),
            steps=[rf"\Delta = k^{{2}} - 4({c}) = 0", rf"k^{{2}} = {4 * c}"],
            topic="quadratic_theory", level=level)

    sol = sp.solve_univariate_inequality(k**2 - 4 * c > 0, k, relational=False)
    return Problem(
        question=T("two_distinct_find_k", lang, k="k",
                   eq=rf"x^{{2}} + kx + {c} = 0"),
        answer=rf"k \in {_latex(sol)}",
        steps=[rf"\Delta = k^{{2}} - 4({c}) > 0"],
        topic="quadratic_theory", level=level)


# Identities worth simplifying, each with the result it must reduce to. The
# pair is checked with SymPy before the problem is handed out, so a template
# typed in wrongly here fails loudly instead of printing a false answer.
def _trig_simplify_table(sp, x):
    return [
        ((1 - sp.cos(x)**2) / sp.sin(x), sp.sin(x)),
        (sp.sin(x) / sp.tan(x), sp.cos(x)),
        (sp.cos(x)**2 - 1, -sp.sin(x)**2),
        (1 / sp.cos(x)**2 - sp.tan(x)**2, sp.Integer(1)),
        (sp.sin(2 * x) / (2 * sp.sin(x)), sp.cos(x)),
        ((1 - sp.cos(2 * x)) / 2, sp.sin(x)**2),
        (sp.tan(x) * sp.cos(x), sp.sin(x)),
    ]


def gen_trig_identities(rng: random.Random, level: str, lang: str = "en") -> Problem:
    sp = _sp()
    x, th = sp.Symbol("x"), sp.Symbol("theta")
    mode = rng.choice(["exact_values", "solve_basic", "simplify"])

    if mode == "exact_values":
        # Pythagorean pairs, so the companion ratio is exact and rational.
        num, den = rng.choice([(3, 5), (4, 5), (5, 13), (12, 13),
                               (8, 17), (15, 17), (7, 25), (24, 25)])
        val = sp.Rational(num, den)
        other = sp.sqrt(1 - val**2)
        given, found = rng.choice([("sin", "cos"), ("cos", "sin")])
        tan = sp.simplify(val / other) if given == "sin" else sp.simplify(other / val)
        return Problem(
            question=T("trig_given_acute", lang,
                       given=rf"\{given}\theta = {_latex(val)}",
                       theta=r"\theta",
                       find1=rf"\{found}\theta", find2=r"\tan\theta"),
            answer=(rf"\{found}\theta = {_latex(other)}, \ "
                    rf"\tan\theta = {_latex(tan)}"),
            steps=[r"\sin^{2}\theta + \cos^{2}\theta = 1",
                   rf"\{found}\theta = \sqrt{{1 - \left({_latex(val)}\right)^{{2}}}}"],
            topic="trig_identities", level=level)

    if mode == "solve_basic":
        fn, val = rng.choice([
            ("sin", sp.Rational(1, 2)), ("sin", sp.sqrt(2) / 2),
            ("sin", sp.sqrt(3) / 2), ("cos", sp.Rational(1, 2)),
            ("cos", sp.sqrt(2) / 2), ("cos", sp.sqrt(3) / 2),
            ("tan", sp.Integer(1)), ("tan", sp.sqrt(3)),
        ])
        if fn == "sin":
            first = sp.deg(sp.asin(val))
            second = 180 - first
        elif fn == "cos":
            first = sp.deg(sp.acos(val))
            second = 360 - first
        else:
            first = sp.deg(sp.atan(val))
            second = first + 180
        return Problem(
            question=T("solve_trig_in_range", lang,
                       eq=rf"\{fn} x = {_latex(val)}",
                       range=r"0^\circ \le x \le 360^\circ"),
            answer=rf"x = {_latex(first)}^\circ, \ x = {_latex(second)}^\circ",
            steps=[T("step_principal_value", lang,
                     v=rf"x = {_latex(first)}^\circ"),
                   T("step_second_solution_symmetry", lang, fn=rf"\{fn}")],
            topic="trig_identities", level=level)

    expr, expected = rng.choice(_trig_simplify_table(sp, x))
    if sp.simplify(sp.expand_trig(expr) - expected) != 0:
        raise ValueError("identity table entry failed verification")
    return Problem(
        question=T("simplify", lang, expr=_latex(expr)),
        answer=_latex(expected),
        topic="trig_identities", level=level)


def gen_proof_induction(rng: random.Random, level: str, lang: str = "en") -> Problem:
    r"""Prove by induction that \sum f(r) equals a closed form.

    The proof itself is prose, but every piece of algebra a mark scheme wants
    is computable: the base case, the inductive step, and the fact that
    S_k + u_{k+1} really does collapse to the closed form at k+1. That last
    identity is verified here, so a mistyped template cannot reach a child.
    """
    sp = _sp()
    n, k, r = sp.symbols("n k r", positive=True, integer=True)
    term, closed = rng.choice([
        (r, n * (n + 1) / 2),
        (2 * r - 1, n**2),
        (r**2, n * (n + 1) * (2 * n + 1) / 6),
        (r**3, n**2 * (n + 1)**2 / 4),
        (r * (r + 1), n * (n + 1) * (n + 2) / 3),
        (r * (r + 2), n * (n + 1) * (2 * n + 7) / 6),
        (1 / (r * (r + 1)), n / (n + 1)),
        (r * (r + 3), n * (n + 1) * (n + 5) / 3),
        (r * 2**r, (n - 1) * 2**(n + 1) + 2),
        (1 / ((2 * r - 1) * (2 * r + 1)), n / (2 * n + 1)),
        (3 * r**2 - r, n**2 * (n + 1)),
        (r * (r + 1) * (r + 2), n * (n + 1) * (n + 2) * (n + 3) / 4),
    ])
    # The statement must be true before it is set as an exercise.
    if sp.simplify(sp.summation(term, (r, 1, n)) - closed) != 0:
        raise ValueError("induction statement failed verification")

    base = closed.subs(n, 1)
    step = sp.simplify(closed.subs(n, k) + term.subs(r, k + 1))
    target = sp.simplify(closed.subs(n, k + 1))
    if sp.simplify(step - target) != 0:
        raise ValueError("inductive step failed verification")

    return Problem(
        question=T("prove_by_induction", lang,
                   statement=(rf"\sum_{{r=1}}^{{n}} {_latex(term)} "
                              rf"= {_latex(closed)}"),
                   domain=r"n \in \mathbb{Z}^{+}"),
        answer=T("answer_induction", lang, n1="n=1", base=_latex(base),
                 step=_latex(closed.subs(n, k + 1))),
        steps=[
            T("step_base_case", lang, n1="n = 1",
              lhs=_latex(sp.simplify(term.subs(r, 1))), rhs=_latex(base)),
            T("step_assume_true", lang, nk="n = k",
              statement=(rf"\sum_{{r=1}}^{{k}} {_latex(term)} "
                         rf"= {_latex(closed.subs(n, k))}")),
            (rf"\sum_{{r=1}}^{{k+1}} = {_latex(closed.subs(n, k))} + "
             rf"{_latex(term.subs(r, k + 1))} = {_latex(closed.subs(n, k + 1))}"),
            T("step_induction_conclusion", lang, nk1="n = k+1",
              domain=r"n \ge 1"),
        ],
        topic="proof_induction", level=level)


def gen_calculus_advanced(rng: random.Random, level: str, lang: str = "en") -> Problem:
    """Implicit differentiation, via SymPy's idiff."""
    sp = _sp()
    x, y = sp.symbols("x y")
    a, b = rng.randint(1, 4), rng.randint(1, 4)
    c = rng.randint(2, 12)
    # Both sides are carried separately: reconstructing the right-hand side by
    # adding a constant back onto a rearranged equation produced nonsense like
    # "x^3 + y^3 + 10 = 10" for the templates whose constant was not on it.
    lhs, rhs = rng.choice([
        (a * x**2 + b * y**2, sp.Integer(c)),
        (x**3 + y**3, c * x * y),
        (x**2 * y + y**3, sp.Integer(c)),
        (a * x * y**2 - y, sp.Integer(c)),
        (x**2 + x * y + b * y**2, sp.Integer(c)),
    ])
    dydx = sp.simplify(sp.idiff(lhs - rhs, y, x))
    return Problem(
        question=T("implicit_differentiation", lang,
                   eq=f"{_latex(lhs)} = {_latex(rhs)}",
                   dydx=r"\dfrac{dy}{dx}", vars="x, y"),
        answer=rf"\dfrac{{dy}}{{dx}} = {_latex(dydx)}",
        steps=[T("step_differentiate_both_sides", lang, x="x", y="y"),
               T("step_collect_dydx", lang, dydx=r"\frac{dy}{dx}")],
        topic="calculus_advanced", level=level)


def gen_differential_eq(rng: random.Random, level: str, lang: str = "en") -> Problem:
    """First-order separable equation with an initial condition.

    The initial condition matters: without it the answer carries an arbitrary
    constant, and a child cannot check their own work against it.
    """
    sp = _sp()
    x = sp.Symbol("x")
    y = sp.Function("y")
    k = rng.randint(2, 5)
    y0 = rng.randint(1, 6)
    # Written over a plain symbol so the question reads "dy/dx = 3xy" rather
    # than SymPy's "y{\left(x \right)}"; substituted back for dsolve.
    Y = sp.Symbol("y")
    shown = rng.choice([
        k * Y, k * x * Y, x / Y, k * x**2 * Y, Y / (x + 1), k * Y / (x + 1),
    ])
    rhs = shown.subs(Y, y(x))
    ode = sp.Eq(y(x).diff(x), rhs)
    sol = sp.dsolve(ode, y(x), ics={y(0): y0})
    if isinstance(sol, (list, tuple)):
        sol = next(s for s in sol if sp.simplify(s.rhs.subs(x, 0) - y0) == 0)
    # Verify: the solution must actually satisfy the equation and the condition.
    check = sp.simplify(sol.rhs.diff(x) - rhs.subs(y(x), sol.rhs))
    if sp.simplify(check) != 0 or sp.simplify(sol.rhs.subs(x, 0) - y0) != 0:
        raise ValueError("dsolve result failed verification")
    return Problem(
        question=T("solve_ode_with_condition", lang,
                   eq=rf"\dfrac{{dy}}{{dx}} = {_latex(shown)}",
                   ic=rf"y(0) = {y0}"),
        answer=rf"y = {_latex(sol.rhs)}",
        steps=[T("step_separate_variables", lang),
               T("step_apply_condition", lang, ic=rf"y(0) = {y0}")],
        topic="differential_eq", level=level)


def gen_vectors_3d(rng: random.Random, level: str, lang: str = "en") -> Problem:
    """Scalar and vector products, magnitude, angle, and lines in space."""
    sp = _sp()

    def vec():
        return sp.Matrix([rng.randint(-5, 5) for _ in range(3)])

    a, b = vec(), vec()
    if a.norm() == 0 or b.norm() == 0:
        raise ValueError("zero vector")
    av, bv = _latex(a.T).replace("\\left[", "\\begin{pmatrix}"), None
    # Column vectors read better than SymPy's default row-matrix output.
    def col(m):
        return (r"\begin{pmatrix}" +
                r"\\".join(_latex(v) for v in m) + r"\end{pmatrix}")

    mode = rng.choice(["dot", "cross", "magnitude", "angle",
                       "perpendicular", "line"])

    if mode == "dot":
        return Problem(
            question=T("vector_find_where", lang,
                       expr=r"\mathbf{a} \cdot \mathbf{b}",
                       a=rf"\mathbf{{a}} = {col(a)}",
                       b=rf"\mathbf{{b}} = {col(b)}"),
            answer=_latex(a.dot(b)), topic="vectors_3d", level=level)

    if mode == "cross":
        return Problem(
            question=T("vector_find_where", lang,
                       expr=r"\mathbf{a} \times \mathbf{b}",
                       a=rf"\mathbf{{a}} = {col(a)}",
                       b=rf"\mathbf{{b}} = {col(b)}"),
            answer=col(a.cross(b)), topic="vectors_3d", level=level)

    if mode == "magnitude":
        return Problem(
            question=T("vector_magnitude_exact", lang,
                       expr=r"|\mathbf{a}|", a=rf"\mathbf{{a}} = {col(a)}"),
            answer=_latex(sp.sqrt(sum(v**2 for v in a))),
            steps=[rf"|\mathbf{{a}}| = \sqrt{{"
                   + " + ".join(f"({v})^{{2}}" for v in a) + "}"],
            topic="vectors_3d", level=level)

    if mode == "angle":
        cos = sp.simplify(a.dot(b) / (a.norm() * b.norm()))
        if abs(cos) > 1:
            raise ValueError("degenerate")
        deg = sp.N(sp.deg(sp.acos(cos)), 4)
        return Problem(
            question=T("vector_angle_between", lang,
                       a=rf"\mathbf{{a}} = {col(a)}",
                       b=rf"\mathbf{{b}} = {col(b)}"),
            answer=rf"\cos\theta = {_latex(cos)}, \ \theta = {deg}^\circ",
            steps=[r"\cos\theta = \dfrac{\mathbf{a}\cdot\mathbf{b}}"
                   r"{|\mathbf{a}||\mathbf{b}|}"],
            topic="vectors_3d", level=level)

    if mode == "perpendicular":
        t = sp.Symbol("t")
        c = sp.Matrix([a[0], a[1], t])
        sol = sp.solve(sp.Eq(c.dot(b), 0), t)
        if not sol or not b[2]:
            raise ValueError("no unique t")
        return Problem(
            question=T("vector_find_t_perpendicular", lang, t="t",
                       c=col(c), b=col(b)),
            answer=rf"t = {_latex(sol[0])}",
            steps=[T("step_perpendicular_dot_zero", lang)],
            topic="vectors_3d", level=level)

    if a == b:
        raise ValueError("coincident points")
    d = b - a
    return Problem(
        question=T("vector_line_through", lang, A=f"A{tuple(a)}",
                   B=f"B{tuple(b)}"),
        answer=rf"\mathbf{{r}} = {col(a)} + \lambda {col(d)}",
        steps=[rf"\overrightarrow{{AB}} = \mathbf{{b}} - \mathbf{{a}} = {col(d)}"],
        topic="vectors_3d", level=level)


def _ascending(expr, x) -> str:
    """LaTeX a polynomial in ascending powers, as a series is conventionally
    written. SymPy's default ordering puts the highest power first."""
    sp = _sp()
    poly = sp.Poly(sp.expand(expr), x)
    parts = []
    for i, coeff in enumerate(poly.all_coeffs()[::-1]):
        if coeff == 0:
            continue
        parts.append(_latex(coeff * x**i) if i else _latex(coeff))
    return " + ".join(parts).replace("+ -", "-")


def gen_maclaurin(rng: random.Random, level: str, lang: str = "en") -> Problem:
    sp = _sp()
    x = sp.Symbol("x")
    a = rng.randint(1, 3)
    f = rng.choice([
        sp.exp(a * x), sp.sin(a * x), sp.cos(a * x),
        sp.log(1 + a * x), sp.sqrt(1 + a * x),
        1 / (1 + a * x), sp.exp(x) * sp.cos(x),
        sp.atan(a * x), (1 + a * x) * sp.exp(x),
    ])
    n = rng.choice([3, 4])
    ser = sp.series(f, x, 0, n + 1).removeO()
    if ser == 0:
        raise ValueError("series vanished")
    return Problem(
        question=T("maclaurin_series", lang, f=f"f(x) = {_latex(f)}",
                   term=f"x^{{{n}}}"),
        answer=_ascending(ser, x),
        steps=[rf"f(x) = \sum_{{k=0}}^{{{n}}} "
               rf"\dfrac{{f^{{(k)}}(0)}}{{k!}} x^{{k}}"],
        topic="maclaurin", level=level)


GENERATORS = {
    "add_sub_2digit": gen_add_sub_2digit,
    "times_tables": gen_times_tables,
    "long_mult_div": gen_long_mult_div,
    "fractions_unit": gen_fractions_unit,
    "fractions_ops": gen_fractions_ops,
    "decimals": gen_decimals,
    "percentages": gen_percentages,
    "integers": gen_integers,
    "ratio_proportion": gen_ratio_proportion,
    "linear_simple": gen_linear_simple,
    "linear_equations": gen_linear_equations,
    "simultaneous": gen_simultaneous,
    "indices": gen_indices,
    "quadratics": gen_quadratics,
    "surds": gen_surds,
    "sequences": gen_sequences,
    "probability": gen_probability,
    "functions": gen_functions,
    "differentiation": gen_differentiation,
    "integration": gen_integration,
    "logs_exp": gen_logs_exp,
    "binomial": gen_binomial,
    "complex_numbers": gen_complex_numbers,
    "integration_parts": gen_integration_parts,
    "number_bonds": gen_number_bonds,
    "quadratic_theory": gen_quadratic_theory,
    "trig_identities": gen_trig_identities,
    "proof_induction": gen_proof_induction,
    "calculus_advanced": gen_calculus_advanced,
    "differential_eq": gen_differential_eq,
    "vectors_3d": gen_vectors_3d,
    "maclaurin": gen_maclaurin,
}


def available_topics() -> list[str]:
    """Topic ids that have a working generator."""
    from researcher.mathtutor import geometry

    return sorted(set(GENERATORS) | set(geometry.GEOMETRY_GENERATORS))


def generate(topic: str, level: str = "grade8", count: int = 10,
             seed: int | None = None, lang: str = "en") -> list[Problem]:
    """Generate *count* problems for *topic*.

    A seed makes a worksheet reproducible, so the same sheet can be reprinted
    and an answer key regenerated to match.

    Duplicates are rejected: random parameters collide more often than people
    expect, and a worksheet with the same question twice looks careless.
    """
    from researcher.mathtutor import geometry

    fn = GENERATORS.get(topic) or geometry.GEOMETRY_GENERATORS.get(topic)
    if fn is None:
        raise ValueError(f"No generator for topic '{topic}'")

    from researcher.mathtutor.i18n import split_alt

    rng = random.Random(seed)
    out: list[Problem] = []
    seen: set[str] = set()
    attempts = 0
    while len(out) < count and attempts < count * 40:
        attempts += 1
        try:
            p = fn(rng, level, lang)
        except Exception as e:                      # a generator bug must not
            logger.warning("[math] %s failed: %s", topic, e)   # kill the sheet
            continue
        # Separate the two renderings a bilingual language spec produced.
        p.lang = lang
        p.question, p.question_alt = split_alt(p.question)
        p.answer, p.answer_alt = split_alt(p.answer)
        if p.steps:
            pairs = [split_alt(st) for st in p.steps]
            p.steps = [a for a, _ in pairs]
            p.steps_alt = [b for _, b in pairs if b]
        # Geometry problems share one question text ("Find the hypotenuse") and
        # differ only in the FIGURE, so keying the duplicate check on text alone
        # discarded almost every one of them. The answer is included too: two
        # figures can coincide while the quantity asked for differs.
        key = f"{p.question}|{p.answer}|{sorted((p.figure or {}).items(), key=str)}"
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out
