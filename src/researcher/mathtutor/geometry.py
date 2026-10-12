"""Geometry problems with drawn figures.

A geometry question without a diagram is a reading comprehension exercise. The
figure is generated from the *same* numbers that produced the answer, so the
drawing can never disagree with the question — which is the usual failure when
a diagram is drawn separately or, worse, asked of a language model.

Figures render to SVG for the browser and PDF for printing. matplotlib is used
rather than TikZ because it needs no LaTeX round-trip to preview, and the output
embeds cleanly in both targets.
"""

from __future__ import annotations

import io
import math
import random
import logging

logger = logging.getLogger(__name__)

# A deliberately plain look: black on white, thin lines, no grid unless the
# problem is about coordinates. Worksheets are photocopied, and colour or
# shading turns into grey mush.
_LINE = {"color": "black", "linewidth": 1.4, "solid_capstyle": "round"}
_LABEL = {"fontsize": 11, "color": "black"}


def _new_axes(figsize=(3.2, 2.6)):
    import matplotlib
    matplotlib.use("Agg")           # no display on a server
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def _render(fig, fmt: str = "svg") -> bytes:
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight", pad_inches=0.08,
                transparent=False, facecolor="white")
    plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Figure builders. Each takes the problem's own numbers.
# ---------------------------------------------------------------------------


def draw_right_triangle(a: float, b: float, label_hyp: str = "?",
                        label_a: str | None = None,
                        label_b: str | None = None) -> dict:
    return {"kind": "right_triangle", "a": a, "b": b,
            "label_hyp": label_hyp, "label_a": label_a, "label_b": label_b}


def draw_rectangle(w: float, h: float, label_w=None, label_h=None) -> dict:
    return {"kind": "rectangle", "w": w, "h": h,
            "label_w": label_w, "label_h": label_h}


def draw_circle(r: float, show: str = "radius") -> dict:
    return {"kind": "circle", "r": r, "show": show}


def draw_compound(parts: list[dict]) -> dict:
    return {"kind": "compound", "parts": parts}


def draw_triangle_sides(a: float, b: float, C_deg: float) -> dict:
    return {"kind": "triangle_sas", "a": a, "b": b, "C": C_deg}


def draw_coordinates(points: list[tuple], line: tuple | None = None) -> dict:
    return {"kind": "coordinates", "points": points, "line": line}


def draw_polygon(n: int, rotate: float = 0.0, marks: bool = False) -> dict:
    """A regular n-gon, optionally with its vertices dotted for counting."""
    return {"kind": "polygon", "n": int(n), "rotate": float(rotate),
            "marks": bool(marks)}


def render_figure(spec: dict, fmt: str = "svg") -> bytes | None:
    """Turn a figure spec into image bytes."""
    if not spec:
        return None
    try:
        kind = spec.get("kind")
        if kind == "right_triangle":
            return _fig_right_triangle(spec, fmt)
        if kind == "rectangle":
            return _fig_rectangle(spec, fmt)
        if kind == "circle":
            return _fig_circle(spec, fmt)
        if kind == "compound":
            return _fig_compound(spec, fmt)
        if kind == "triangle_sas":
            return _fig_triangle_sas(spec, fmt)
        if kind == "coordinates":
            return _fig_coordinates(spec, fmt)
        if kind == "polygon":
            return _fig_polygon(spec, fmt)
    except Exception as e:
        logger.warning("[geometry] could not render %s: %s", spec.get("kind"), e)
    return None


def _fig_polygon(s: dict, fmt: str) -> bytes:
    import math

    n = max(3, int(s["n"]))
    rot = math.radians(float(s.get("rotate", 0.0)))
    # Start at the top so the shape sits the way a child would draw it.
    pts = [(math.cos(rot + math.pi / 2 + 2 * math.pi * i / n),
            math.sin(rot + math.pi / 2 + 2 * math.pi * i / n)) for i in range(n)]
    fig, ax = _new_axes()
    xs = [p[0] for p in pts] + [pts[0][0]]
    ys = [p[1] for p in pts] + [pts[0][1]]
    ax.plot(xs, ys, **_LINE)
    if s.get("marks"):
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o",
                color="black", markersize=4)
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_aspect("equal")
    return _render(fig, fmt)


def _fig_right_triangle(s: dict, fmt: str) -> bytes:
    a, b = float(s["a"]), float(s["b"])
    fig, ax = _new_axes()
    ax.plot([0, b, 0, 0], [0, 0, a, 0], **_LINE)
    # The right-angle square, drawn proportionally so it reads at any scale.
    m = 0.08 * max(a, b)
    ax.plot([m, m, 0], [0, m, m], color="black", linewidth=1.0)
    ax.text(b / 2, -0.12 * a, s.get("label_b") or f"{b:g}",
            ha="center", va="top", **_LABEL)
    ax.text(-0.06 * b, a / 2, s.get("label_a") or f"{a:g}",
            ha="right", va="center", **_LABEL)
    ax.text(b / 2 + 0.06 * b, a / 2 + 0.06 * a, s.get("label_hyp", "?"),
            ha="left", va="bottom", **_LABEL)
    ax.set_xlim(-0.25 * b, 1.15 * b)
    ax.set_ylim(-0.25 * a, 1.15 * a)
    return _render(fig, fmt)


def _fig_rectangle(s: dict, fmt: str) -> bytes:
    w, h = float(s["w"]), float(s["h"])
    fig, ax = _new_axes()
    ax.plot([0, w, w, 0, 0], [0, 0, h, h, 0], **_LINE)
    ax.text(w / 2, -0.12 * h, s.get("label_w") or f"{w:g}",
            ha="center", va="top", **_LABEL)
    ax.text(-0.04 * w, h / 2, s.get("label_h") or f"{h:g}",
            ha="right", va="center", **_LABEL)
    ax.set_xlim(-0.2 * w, 1.1 * w)
    ax.set_ylim(-0.25 * h, 1.1 * h)
    return _render(fig, fmt)


def _fig_circle(s: dict, fmt: str) -> bytes:
    import matplotlib.patches as patches

    r = float(s["r"])
    fig, ax = _new_axes((2.6, 2.6))
    ax.add_patch(patches.Circle((0, 0), r, fill=False, **{
        k: v for k, v in _LINE.items() if k != "solid_capstyle"}))
    if s.get("show") == "diameter":
        ax.plot([-r, r], [0, 0], **_LINE)
        ax.text(0, 0.05 * r, f"d = {2*r:g}", ha="center", va="bottom", **_LABEL)
    else:
        ax.plot([0, r], [0, 0], **_LINE)
        ax.text(r / 2, 0.05 * r, f"r = {r:g}", ha="center", va="bottom", **_LABEL)
    ax.plot([0], [0], marker="o", color="black", markersize=3)
    ax.set_xlim(-1.2 * r, 1.2 * r)
    ax.set_ylim(-1.2 * r, 1.2 * r)
    return _render(fig, fmt)


def _fig_compound(s: dict, fmt: str) -> bytes:
    fig, ax = _new_axes((3.4, 2.8))
    xs, ys = [], []
    for part in s["parts"]:
        x, y = float(part["x"]), float(part["y"])
        w, h = float(part["w"]), float(part["h"])
        ax.plot([x, x + w, x + w, x, x], [y, y, y + h, y + h, y], **_LINE)
        ax.text(x + w / 2, y + h / 2, f"{w:g} x {h:g}",
                ha="center", va="center", fontsize=9)
        xs += [x, x + w]; ys += [y, y + h]
    ax.set_xlim(min(xs) - 1, max(xs) + 1)
    ax.set_ylim(min(ys) - 1, max(ys) + 1)
    return _render(fig, fmt)


def _fig_triangle_sas(s: dict, fmt: str) -> bytes:
    a, b, C = float(s["a"]), float(s["b"]), math.radians(float(s["C"]))
    # Place the included angle at the origin so the drawing matches the data.
    A = (0.0, 0.0)
    B = (b, 0.0)
    Cpt = (a * math.cos(C), a * math.sin(C))
    fig, ax = _new_axes((3.2, 2.6))
    ax.plot([A[0], B[0], Cpt[0], A[0]], [A[1], B[1], Cpt[1], A[1]], **_LINE)
    ax.text(b / 2, -0.08 * max(a, b), f"{b:g}", ha="center", va="top", **_LABEL)
    ax.text(Cpt[0] / 2 - 0.06 * b, Cpt[1] / 2, f"{a:g}",
            ha="right", va="center", **_LABEL)
    ax.text(0.12 * b, 0.05 * b, f"{s['C']:g}°", ha="left", va="bottom", fontsize=10)
    ax.set_xlim(min(0, Cpt[0]) - 0.2 * b, b * 1.2)
    ax.set_ylim(-0.2 * max(a, b), max(a, Cpt[1]) * 1.25)
    return _render(fig, fmt)


def _fig_coordinates(s: dict, fmt: str) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = [(float(x), float(y)) for x, y in s["points"]]
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    ax.set_aspect("equal")
    lim = max(6, max(abs(v) for p in pts for v in p) + 2)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.grid(True, linewidth=0.4, color="#bbbbbb")
    ax.axhline(0, color="black", linewidth=1.0)
    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_xticks(range(-int(lim), int(lim) + 1, max(1, int(lim) // 5)))
    ax.set_yticks(range(-int(lim), int(lim) + 1, max(1, int(lim) // 5)))
    ax.tick_params(labelsize=7)
    for i, (x, y) in enumerate(pts):
        ax.plot(x, y, "o", color="black", markersize=4)
        ax.annotate(f"{chr(65+i)}({x:g}, {y:g})", (x, y),
                    textcoords="offset points", xytext=(5, 4), fontsize=8)
    if s.get("line"):
        m, c = s["line"]
        xx = [-lim, lim]
        ax.plot(xx, [m * v + c for v in xx], color="black", linewidth=1.2)
    return _render(fig, fmt)


# ---------------------------------------------------------------------------
# Geometry problem generators — answers computed, figures drawn from the same
# numbers so the two can never disagree.
# ---------------------------------------------------------------------------


def gen_shapes_2d(rng: random.Random, level: str):
    from researcher.mathtutor.generators import Problem

    from researcher.mathtutor.generators import _grade

    shapes = [("triangle", 3), ("square", 4), ("pentagon", 5), ("hexagon", 6),
              ("heptagon", 7), ("octagon", 8)]
    name, sides = rng.choice(shapes)
    grade = _grade(level)
    article = "an" if name[0] in "aeiou" else "a"

    if grade <= 3:
        # Young children count the picture; the name is what they are learning,
        # so it must not be the thing the question withholds.
        ask = rng.choice(["sides", "corners"])
        return Problem(
            question=rf"\text{{This shape is {article} {name}. "
                     rf"How many {ask} does it have?}}",
            answer=str(sides),
            figure=draw_polygon(sides, rotate=rng.choice([0, 15, 30]),
                                marks=(ask == "corners")),
            topic="shapes_2d", level=level)

    ask = rng.choice(["sides", "vertices", "lines of symmetry"])
    answer = sides          # a regular n-gon has n of each
    return Problem(
        question=rf"\text{{How many {ask} does a regular {name} have?}}",
        answer=str(answer),
        figure=draw_polygon(sides, rotate=rng.choice([0, 15, 30])),
        topic="shapes_2d", level=level)


def gen_area_perimeter(rng: random.Random, level: str):
    from researcher.mathtutor.generators import Problem

    w, h = rng.randint(3, 15), rng.randint(3, 12)
    want = rng.choice(["area", "perimeter"])
    ans = w * h if want == "area" else 2 * (w + h)
    # Keep the exponent in MATH mode: \text{cm^2} is invalid LaTeX.
    unit = r"\text{cm}^2" if want == "area" else r"\text{cm}"
    return Problem(
        question=rf"\text{{Find the {want} of this rectangle.}}",
        answer=rf"{ans}\ {unit}",
        steps=[rf"{w} \times {h} = {w*h}" if want == "area"
               else rf"2({w} + {h}) = {2*(w+h)}"],
        topic="area_perimeter", level=level,
        figure=draw_rectangle(w, h, f"{w} cm", f"{h} cm"))


def gen_angles_basic(rng: random.Random, level: str):
    from researcher.mathtutor.generators import Problem

    known = rng.randint(25, 150)
    kind = rng.choice(["straight", "triangle"])
    if kind == "straight":
        return Problem(
            question=(rf"\text{{Angles on a straight line. One angle is }}{known}°."
                      rf"\text{{ Find the other.}}"),
            answer=rf"{180-known}°",
            steps=[rf"180° - {known}° = {180-known}°"],
            topic="angles_basic", level=level)
    a = rng.randint(30, 80)
    b = rng.randint(30, 180 - a - 20)
    return Problem(
        question=(rf"\text{{A triangle has angles }}{a}° \text{{ and }} {b}°."
                  rf"\text{{ Find the third angle.}}"),
        answer=rf"{180-a-b}°",
        steps=[rf"180° - {a}° - {b}° = {180-a-b}°"],
        topic="angles_basic", level=level)


def gen_area_compound(rng: random.Random, level: str):
    from researcher.mathtutor.generators import Problem

    w1, h1 = rng.randint(4, 10), rng.randint(3, 8)
    w2, h2 = rng.randint(3, 8), rng.randint(2, 6)
    total = w1 * h1 + w2 * h2
    parts = [{"x": 0, "y": 0, "w": w1, "h": h1},
             {"x": w1, "y": 0, "w": w2, "h": h2}]
    return Problem(
        question=r"\text{Find the total area of this compound shape.}",
        answer=rf"{total}\ \text{{cm}}^2",
        steps=[rf"({w1}\times{h1}) + ({w2}\times{h2}) = {w1*h1} + {w2*h2} = {total}"],
        topic="area_compound", level=level, figure=draw_compound(parts))


def gen_coordinates(rng: random.Random, level: str):
    from researcher.mathtutor.generators import Problem
    import sympy as sp

    x1, y1 = rng.randint(-6, 6), rng.randint(-6, 6)
    x2, y2 = rng.randint(-6, 6), rng.randint(-6, 6)
    while (x2, y2) == (x1, y1):
        x2, y2 = rng.randint(-6, 6), rng.randint(-6, 6)
    mid = (sp.Rational(x1 + x2, 2), sp.Rational(y1 + y2, 2))
    return Problem(
        question=(rf"\text{{Find the midpoint of }} A({x1}, {y1}) "
                  rf"\text{{ and }} B({x2}, {y2})"),
        answer=rf"\left({sp.latex(mid[0])}, {sp.latex(mid[1])}\right)",
        topic="coordinates", level=level,
        figure=draw_coordinates([(x1, y1), (x2, y2)]))


def gen_pythagoras(rng: random.Random, level: str):
    from researcher.mathtutor.generators import Problem
    import sympy as sp

    # Half the time use a Pythagorean triple so the answer is a whole number;
    # the rest give a surd, which is the IB expectation for exact form.
    if rng.random() < 0.5:
        a, b, c = rng.choice([(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25),
                              (6, 8, 10), (9, 12, 15)])
        ans = str(c)
    else:
        a, b = rng.randint(2, 11), rng.randint(2, 11)
        ans = sp.latex(sp.sqrt(a**2 + b**2))
    return Problem(
        question=r"\text{Find the length of the hypotenuse, in exact form.}",
        answer=rf"{ans}\ \text{{cm}}",
        steps=[rf"c^2 = {a}^2 + {b}^2 = {a**2 + b**2}"],
        topic="pythagoras", level=level,
        figure=draw_right_triangle(a, b, "?", f"{a} cm", f"{b} cm"))


def gen_circle_geometry(rng: random.Random, level: str):
    from researcher.mathtutor.generators import Problem
    import sympy as sp

    r = rng.randint(2, 12)
    want = rng.choice(["area", "circumference"])
    ans = sp.pi * r**2 if want == "area" else 2 * sp.pi * r
    # Keep the exponent in MATH mode: \text{cm^2} is invalid LaTeX.
    unit = r"\text{cm}^2" if want == "area" else r"\text{cm}"
    return Problem(
        question=rf"\text{{Find the {want} of this circle, in terms of }} \pi.",
        answer=rf"{sp.latex(ans)}\ {unit}",
        steps=[rf"A = \pi r^2 = \pi \times {r}^2" if want == "area"
               else rf"C = 2\pi r = 2\pi \times {r}"],
        topic="circle_geometry", level=level, figure=draw_circle(r))


def gen_straight_line(rng: random.Random, level: str):
    from researcher.mathtutor.generators import Problem
    import sympy as sp

    x1, y1 = rng.randint(-6, 6), rng.randint(-6, 6)
    x2 = x1 + rng.choice([-4, -3, -2, 2, 3, 4])
    y2 = rng.randint(-6, 6)
    m = sp.Rational(y2 - y1, x2 - x1)
    c = sp.nsimplify(y1 - m * x1)
    return Problem(
        question=(rf"\text{{Find the equation of the line through }} "
                  rf"({x1}, {y1}) \text{{ and }} ({x2}, {y2})"),
        answer=rf"y = {sp.latex(m)}x + {sp.latex(c)}" if c >= 0
               else rf"y = {sp.latex(m)}x - {sp.latex(abs(c))}",
        steps=[rf"m = \frac{{{y2}-{y1}}}{{{x2}-{x1}}} = {sp.latex(m)}"],
        topic="straight_line", level=level,
        figure=draw_coordinates([(x1, y1), (x2, y2)], line=(float(m), float(c))))


def gen_trigonometry(rng: random.Random, level: str):
    from researcher.mathtutor.generators import Problem
    import sympy as sp

    adj = rng.randint(3, 14)
    angle = rng.choice([25, 30, 35, 40, 45, 50, 55, 60, 65])
    opp = sp.tan(sp.rad(angle)) * adj
    return Problem(
        question=(rf"\text{{Find the length marked }} x, "
                  rf"\text{{ to 3 significant figures.}}"),
        answer=rf"{float(opp):.3g}\ \text{{cm}}",
        steps=[rf"\tan {angle}° = \frac{{x}}{{{adj}}}",
               rf"x = {adj}\tan {angle}°"],
        topic="trigonometry", level=level,
        figure=draw_right_triangle(float(opp), adj, "x", "x", f"{adj} cm"))


def gen_sine_cosine_rule(rng: random.Random, level: str):
    from researcher.mathtutor.generators import Problem
    import sympy as sp

    a, b = rng.randint(5, 15), rng.randint(5, 15)
    C = rng.choice([30, 40, 50, 60, 70, 80, 100, 110, 120])
    c2 = a**2 + b**2 - 2 * a * b * sp.cos(sp.rad(C))
    c = sp.sqrt(c2)
    return Problem(
        question=(rf"\text{{Use the cosine rule to find the third side, }}"
                  rf"\text{{to 3 significant figures.}}"),
        answer=rf"{float(c):.3g}\ \text{{cm}}",
        steps=[rf"c^2 = a^2 + b^2 - 2ab\cos C",
               rf"c^2 = {a}^2 + {b}^2 - 2({a})({b})\cos {C}°"],
        topic="sine_cosine_rule", level=level,
        figure=draw_triangle_sides(a, b, C))


def gen_vectors_2d(rng: random.Random, level: str):
    from researcher.mathtutor.generators import Problem
    import sympy as sp

    a1, a2 = rng.randint(-6, 6), rng.randint(-6, 6)
    b1, b2 = rng.randint(-6, 6), rng.randint(-6, 6)
    mag = sp.sqrt((a1 + b1) ** 2 + (a2 + b2) ** 2)
    return Problem(
        question=(rf"\text{{Given }} \mathbf{{a}} = \binom{{{a1}}}{{{a2}}} "
                  rf"\text{{ and }} \mathbf{{b}} = \binom{{{b1}}}{{{b2}}}, "
                  rf"\text{{ find }} |\mathbf{{a}} + \mathbf{{b}}| \text{{ in exact form.}}"),
        answer=sp.latex(sp.sqrt(sp.Integer((a1 + b1) ** 2 + (a2 + b2) ** 2))),
        steps=[rf"\mathbf{{a}}+\mathbf{{b}} = \binom{{{a1+b1}}}{{{a2+b2}}}"],
        topic="vectors_2d", level=level)


def gen_transformations(rng: random.Random, level: str):
    from researcher.mathtutor.generators import Problem

    x, y = rng.randint(-6, 6), rng.randint(-6, 6)
    kind = rng.choice(["x-axis", "y-axis", "origin"])
    img = {"x-axis": (x, -y), "y-axis": (-x, y), "origin": (-x, -y)}[kind]
    return Problem(
        question=(rf"\text{{Reflect the point }} ({x}, {y}) "
                  rf"\text{{ in the {kind}.}}"),
        answer=rf"({img[0]}, {img[1]})",
        topic="transformations", level=level,
        figure=draw_coordinates([(x, y), img]))


def gen_measure_length(rng: random.Random, level: str):
    from researcher.mathtutor.generators import Problem

    from researcher.mathtutor.generators import _grade

    grade = _grade(level)

    if grade <= 2:
        # Comparison and single-unit arithmetic. Metric conversion needs place
        # value the child has not met yet, so it is not asked here.
        #
        # The noun and the comparative travel with the unit: a word problem that
        # says "a ribbon is 9 L long" teaches a child that litres measure
        # length, which is worse than asking nothing at all.
        unit, noun, verb, longer, total = rng.choice([
            ("cm", "ribbon", "long", "longer", "long are they together"),
            ("m", "rope", "long", "longer", "long are they together"),
            ("kg", "parcel", "heavy", "heavier", "heavy are they together"),
            ("L", "jug", "full", "more", "much do they hold together"),
        ])
        a, b = rng.sample(range(2, 20), 2)
        if rng.random() < 0.5:
            return Problem(
                question=rf"\text{{Which is {longer}: }} {a}\ \text{{{unit}}} "
                         rf"\text{{ or }} {b}\ \text{{{unit}}}?",
                answer=rf"{max(a, b)}\ \text{{{unit}}}",
                topic="measure_length", level=level)
        return Problem(
            question=rf"\text{{One {noun} holds }} {a}\ \text{{{unit}}}"
                     rf"\text{{. Another holds }} {b}\ \text{{{unit}}}"
                     rf"\text{{. How {total}?}}"
            if unit == "L" else
            rf"\text{{One {noun} is }} {a}\ \text{{{unit}}} "
            rf"\text{{{verb}. Another is }} {b}\ \text{{{unit}}}"
            rf"\text{{. How {total}?}}",
            answer=rf"{a + b}\ \text{{{unit}}}",
            topic="measure_length", level=level)

    pairs = [("m", "cm", 100), ("cm", "mm", 10), ("km", "m", 1000),
             ("kg", "g", 1000), ("L", "mL", 1000)]
    frm, to, factor = rng.choice(pairs)
    if grade <= 4:
        val = rng.randint(2, 20)
        return Problem(
            question=rf"\text{{Convert }} {val}\ \text{{{frm} to {to}}}",
            answer=rf"{val * factor}\ \text{{{to}}}",
            topic="measure_length", level=level)

    # From Grade 5 the conversion runs both ways and uses decimals, which is
    # where the place-value understanding is actually tested.
    if rng.random() < 0.5:
        val = round(rng.uniform(1.5, 9.5), 1)
        return Problem(
            question=rf"\text{{Convert }} {val}\ \text{{{frm} to {to}}}",
            answer=rf"{val * factor:g}\ \text{{{to}}}",
            topic="measure_length", level=level)
    big = rng.randint(2, 95) * factor // rng.choice([1, 2, 4])
    return Problem(
        question=rf"\text{{Convert }} {big}\ \text{{{to} to {frm}}}",
        answer=rf"{big / factor:g}\ \text{{{frm}}}",
        topic="measure_length", level=level)


GEOMETRY_GENERATORS = {
    "shapes_2d": gen_shapes_2d,
    "area_perimeter": gen_area_perimeter,
    "angles_basic": gen_angles_basic,
    "area_compound": gen_area_compound,
    "coordinates": gen_coordinates,
    "pythagoras": gen_pythagoras,
    "circle_geometry": gen_circle_geometry,
    "straight_line": gen_straight_line,
    "trigonometry": gen_trigonometry,
    "sine_cosine_rule": gen_sine_cosine_rule,
    "vectors_2d": gen_vectors_2d,
    "transformations": gen_transformations,
    "measure_length": gen_measure_length,
}
