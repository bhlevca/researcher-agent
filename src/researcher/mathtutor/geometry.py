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
        if kind == "circle_theorem":
            return _fig_circle_theorem(spec, fmt)
        if kind == "inscribed":
            return _fig_inscribed(spec, fmt)
        if kind == "sector":
            return _fig_sector(spec, fmt)
        if kind == "similar":
            return _fig_similar(spec, fmt)
        if kind == "parallel":
            return _fig_parallel(spec, fmt)
        if kind == "cuboid":
            return _fig_cuboid(spec, fmt)
        if kind == "solid":
            return _fig_solid(spec, fmt)
        if kind == "analytic":
            return _fig_analytic(spec, fmt)
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


def draw_circle_theorem(variant: str, pts: dict, labels: dict | None = None) -> dict:
    """A circle with the chords, radii or tangent a theorem question needs.

    *pts* holds each named point as an angle in degrees around the circle, so
    the drawing is built from the same geometry the answer is. *labels* maps a
    point name to the angle marked there — drawn just inside the circle beside
    that vertex, because an angle written anywhere else is unreadable.
    """
    return {"kind": "circle_theorem", "variant": variant, "pts": pts,
            "labels": labels or {}}


def draw_inscribed(variant: str, size: float) -> dict:
    """A shape inside a circle, or a circle inside a shape, shaded between."""
    return {"kind": "inscribed", "variant": variant, "size": float(size)}


def draw_sector(r: float, theta: float, segment: bool = False) -> dict:
    return {"kind": "sector", "r": float(r), "theta": float(theta),
            "segment": bool(segment)}


def draw_similar(a: float, b: float, k: float, labels: dict) -> dict:
    """Two similar triangles side by side, the second scaled by *k*."""
    return {"kind": "similar", "a": float(a), "b": float(b), "k": float(k),
            "labels": labels}


def draw_parallel_lines(angle: float, known: str, unknown: str = "x",
                        mark: str = "alternate") -> dict:
    return {"kind": "parallel", "angle": float(angle), "known": known,
            "unknown": unknown, "mark": mark}


def draw_cuboid(a: float, b: float, c: float, diagonal: bool = False,
                labels: tuple | None = None) -> dict:
    return {"kind": "cuboid", "a": float(a), "b": float(b), "c": float(c),
            "diagonal": bool(diagonal), "labels": labels}


def draw_solid(kind: str, dims: dict) -> dict:
    """A cylinder, cone, sphere or square-based pyramid in outline."""
    return {"kind": "solid", "solid": kind, "dims": dims}


def draw_analytic(points: list[tuple], line=None, circle=None,
                  segment: bool = False) -> dict:
    """Coordinate axes carrying points, a line and/or a circle."""
    return {"kind": "analytic", "points": points, "line": line,
            "circle": circle, "segment": bool(segment)}


def _fig_circle_theorem(s: dict, fmt: str) -> bytes:
    import matplotlib.patches as patches

    R = 1.0
    pts = {name: (R * math.cos(math.radians(a)), R * math.sin(math.radians(a)))
           for name, a in s["pts"].items()}
    fig, ax = _new_axes((3.0, 3.0))
    ax.add_patch(patches.Circle((0, 0), R, fill=False, color="black",
                                linewidth=1.4))
    variant = s["variant"]

    def seg(p, q, **kw):
        ax.plot([p[0], q[0]], [p[1], q[1]], **{**_LINE, **kw})

    O = (0.0, 0.0)
    if variant in ("centre_angle", "semicircle"):
        A, B, P = pts["A"], pts["B"], pts["P"]
        seg(A, P); seg(B, P)
        if variant == "centre_angle":
            seg(O, A); seg(O, B)
            ax.plot(*O, marker="o", color="black", markersize=3)
            ax.annotate("O", O, textcoords="offset points", xytext=(4, -10),
                        fontsize=9)
        else:
            seg(A, B)
    elif variant == "same_segment":
        A, B, P, Q = pts["A"], pts["B"], pts["P"], pts["Q"]
        seg(A, P); seg(B, P); seg(A, Q); seg(B, Q)
    elif variant == "cyclic_quad":
        order = [pts[k] for k in ("A", "B", "C", "D")]
        for i in range(4):
            seg(order[i], order[(i + 1) % 4])
    elif variant == "tangent":
        A = pts["A"]
        seg(O, A)
        ax.plot(*O, marker="o", color="black", markersize=3)
        # The tangent at A is perpendicular to OA.
        tx, ty = -A[1], A[0]
        n = math.hypot(tx, ty)
        tx, ty = tx / n * 0.9, ty / n * 0.9
        ax.plot([A[0] - tx, A[0] + tx], [A[1] - ty, A[1] + ty], **_LINE)
        ax.annotate("O", O, textcoords="offset points", xytext=(4, -10),
                    fontsize=9)

    for name, p in pts.items():
        ax.plot(*p, marker="o", color="black", markersize=3.5)
        off = (10 * p[0], 10 * p[1])
        ax.annotate(name, p, textcoords="offset points", xytext=off, fontsize=9,
                    ha="center", va="center")
    # Each marked angle sits just inside the circle at its own vertex.
    for name, text in (s.get("labels") or {}).items():
        if name == "O":
            # Between the two radii, so it reads as the angle at the centre.
            a = math.radians(s["pts"]["A"])
            b = math.radians(s["pts"]["B"])
            mid = math.atan2(math.sin(a) + math.sin(b), math.cos(a) + math.cos(b))
            ax.text(0.3 * math.cos(mid), 0.3 * math.sin(mid), text,
                    ha="center", va="center", fontsize=9)
        elif name in s["pts"]:
            ang = math.radians(s["pts"][name])
            ax.text(0.76 * math.cos(ang), 0.76 * math.sin(ang), text,
                    ha="center", va="center", fontsize=9)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    return _render(fig, fmt)


def _fig_inscribed(s: dict, fmt: str) -> bytes:
    import matplotlib.patches as patches

    v, size = s["variant"], s["size"]
    fig, ax = _new_axes((2.9, 2.9))
    shade = {"facecolor": "#d9d9d9", "edgecolor": "black", "linewidth": 1.4}

    if v == "circle_in_square":
        # Shade the square, then punch the circle out in white.
        ax.add_patch(patches.Rectangle((0, 0), size, size, **shade))
        ax.add_patch(patches.Circle((size / 2, size / 2), size / 2,
                                    facecolor="white", edgecolor="black",
                                    linewidth=1.4))
        ax.text(size / 2, -0.1 * size, f"{size:g} cm", ha="center", va="top",
                **_LABEL)
        lim = size
    elif v == "square_in_circle":
        r = size
        ax.add_patch(patches.Circle((0, 0), r, **shade))
        h = r / math.sqrt(2)
        ax.add_patch(patches.Rectangle((-h, -h), 2 * h, 2 * h,
                                       facecolor="white", edgecolor="black",
                                       linewidth=1.4))
        ax.text(0, -r * 1.1, f"radius {r:g} cm", ha="center", va="top", **_LABEL)
        lim = r * 1.18
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim * 1.15, lim)
        return _render(fig, fmt)
    elif v == "quarter_circles":
        ax.add_patch(patches.Rectangle((0, 0), size, size, **shade))
        for cx, cy, t1, t2 in ((0, 0, 0, 90), (size, 0, 90, 180),
                               (size, size, 180, 270), (0, size, 270, 360)):
            ax.add_patch(patches.Wedge((cx, cy), size / 2, t1, t2,
                                       facecolor="white", edgecolor="black",
                                       linewidth=1.2))
        ax.text(size / 2, -0.1 * size, f"{size:g} cm", ha="center", va="top",
                **_LABEL)
        lim = size
    else:   # circle_in_triangle — equilateral, incircle
        sdim = size
        h = sdim * math.sqrt(3) / 2
        tri = [(0, 0), (sdim, 0), (sdim / 2, h)]
        ax.add_patch(patches.Polygon(tri, **shade))
        ax.add_patch(patches.Circle((sdim / 2, h / 3), sdim / (2 * math.sqrt(3)),
                                    facecolor="white", edgecolor="black",
                                    linewidth=1.4))
        ax.text(sdim / 2, -0.1 * sdim, f"{sdim:g} cm", ha="center", va="top",
                **_LABEL)
        ax.set_xlim(-0.12 * sdim, 1.12 * sdim)
        ax.set_ylim(-0.25 * sdim, h * 1.1)
        return _render(fig, fmt)

    ax.set_xlim(-0.12 * lim, 1.12 * lim)
    ax.set_ylim(-0.25 * lim, 1.12 * lim)
    return _render(fig, fmt)


def _fig_sector(s: dict, fmt: str) -> bytes:
    import matplotlib.patches as patches

    r, th = s["r"], s["theta"]
    fig, ax = _new_axes((2.9, 2.9))
    ax.add_patch(patches.Wedge((0, 0), r, 0, th, facecolor="#e8e8e8",
                               edgecolor="black", linewidth=1.4))
    if s.get("segment"):
        # The chord, so the segment between it and the arc is visible.
        p = (r * math.cos(math.radians(th)), r * math.sin(math.radians(th)))
        ax.plot([r, p[0]], [0, p[1]], color="black", linewidth=1.4)
    ax.text(r * 0.55, -0.06 * r, f"{r:g} cm", ha="center", va="top", **_LABEL)
    ax.text(0.17 * r, 0.09 * r, f"{th:g}" + "°", fontsize=9)
    ax.set_xlim(-1.15 * r, 1.15 * r)
    ax.set_ylim(-1.15 * r, 1.15 * r)
    return _render(fig, fmt)


def _fig_similar(s: dict, fmt: str) -> bytes:
    a, b, k = s["a"], s["b"], s["k"]
    lab = s.get("labels", {})
    fig, ax = _new_axes((3.6, 2.4))
    ax.plot([0, b, 0, 0], [0, 0, a, 0], **_LINE)
    gap = b * 1.45
    ax.plot([gap, gap + b * k, gap, gap], [0, 0, a * k, 0], **_LINE)
    ax.text(b / 2, -0.1 * a, lab.get("b", f"{b:g}"), ha="center", va="top", **_LABEL)
    ax.text(-0.05 * b, a / 2, lab.get("a", f"{a:g}"), ha="right", va="center", **_LABEL)
    ax.text(gap + b * k / 2, -0.1 * a, lab.get("b2", f"{b*k:g}"),
            ha="center", va="top", **_LABEL)
    ax.text(gap - 0.05 * b, a * k / 2, lab.get("a2", f"{a*k:g}"),
            ha="right", va="center", **_LABEL)
    top = max(a, a * k)
    ax.set_xlim(-0.25 * b, gap + b * k + 0.25 * b)
    ax.set_ylim(-0.3 * top, 1.15 * top)
    return _render(fig, fmt)


def _fig_parallel(s: dict, fmt: str) -> bytes:
    ang = math.radians(s["angle"])
    fig, ax = _new_axes((3.4, 2.4))
    # Two horizontal parallels crossed by one transversal.
    for y in (0.0, 1.0):
        ax.plot([-1.4, 1.4], [y, y], **_LINE)
        for xa in (-1.0, 0.9):
            ax.annotate("", xy=(xa + 0.14, y), xytext=(xa, y),
                        arrowprops=dict(arrowstyle="->", lw=1.0, color="black"))
    t = 1.9 / math.tan(ang) if abs(math.tan(ang)) > 1e-6 else 0.0
    ax.plot([-t / 2 - 0.45, t / 2 + 0.45], [-0.45, 1.45], **_LINE)
    ax.text(0.14, 0.08, s.get("known", ""), fontsize=9)
    ax.text(0.14 - t / 2, 1.08, s.get("unknown", "x"), fontsize=9)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-0.6, 1.6)
    return _render(fig, fmt)


def _fig_cuboid(s: dict, fmt: str) -> bytes:
    a, b, c = s["a"], s["b"], s["c"]       # width, depth, height
    dx, dy = 0.42 * b, 0.32 * b            # isometric offset for the depth
    fig, ax = _new_axes((3.2, 2.6))
    front = [(0, 0), (a, 0), (a, c), (0, c)]
    back = [(x + dx, y + dy) for x, y in front]

    def poly(pts, **kw):
        xs = [p[0] for p in pts] + [pts[0][0]]
        ys = [p[1] for p in pts] + [pts[0][1]]
        ax.plot(xs, ys, **{**_LINE, **kw})

    poly(front)
    # Hidden edges dashed, as they are drawn on paper.
    ax.plot([back[0][0], back[1][0]], [back[0][1], back[1][1]],
            color="black", linewidth=1.0, linestyle=(0, (3, 3)))
    ax.plot([back[0][0], back[3][0]], [back[0][1], back[3][1]],
            color="black", linewidth=1.0, linestyle=(0, (3, 3)))
    ax.plot([front[0][0], back[0][0]], [front[0][1], back[0][1]],
            color="black", linewidth=1.0, linestyle=(0, (3, 3)))
    for i in (1, 2, 3):
        ax.plot([front[i][0], back[i][0]], [front[i][1], back[i][1]], **_LINE)
    poly([back[1], back[2], front[2], front[1]])
    poly([back[3], back[2], front[2], front[3]])

    lab = s.get("labels") or (f"{a:g}", f"{b:g}", f"{c:g}")
    ax.text(a / 2, -0.1 * c, lab[0], ha="center", va="top", **_LABEL)
    ax.text(a + dx * 0.55, dy * 0.4, lab[1], ha="left", va="top", fontsize=10)
    ax.text(-0.04 * a, c / 2, lab[2], ha="right", va="center", **_LABEL)
    if s.get("diagonal"):
        ax.plot([front[0][0], back[2][0]], [front[0][1], back[2][1]],
                color="black", linewidth=1.2, linestyle=(0, (5, 2)))
    ax.set_xlim(-0.22 * a, a + dx + 0.18 * a)
    ax.set_ylim(-0.28 * c, c + dy + 0.12 * c)
    return _render(fig, fmt)


def _fig_solid(s: dict, fmt: str) -> bytes:
    import matplotlib.patches as patches

    kind, d = s["solid"], s["dims"]
    fig, ax = _new_axes((2.7, 2.9))
    if kind == "cylinder":
        r, h = d["r"], d["h"]
        ax.plot([-r, -r], [0, h], **_LINE)
        ax.plot([r, r], [0, h], **_LINE)
        ax.add_patch(patches.Ellipse((0, h), 2 * r, 0.45 * r, fill=False,
                                     edgecolor="black", linewidth=1.4))
        # The near half of the base rim is solid, the far half dashed.
        ax.add_patch(patches.Arc((0, 0), 2 * r, 0.45 * r, theta1=180, theta2=360,
                                 edgecolor="black", linewidth=1.4))
        ax.add_patch(patches.Arc((0, 0), 2 * r, 0.45 * r, theta1=0, theta2=180,
                                 edgecolor="black", linewidth=1.0,
                                 linestyle=(0, (3, 3))))
        ax.text(0, h + 0.3 * r, f"r = {r:g}", ha="center", fontsize=10)
        ax.text(r * 1.15, h / 2, f"h = {h:g}", ha="left", va="center", fontsize=10)
        ax.set_xlim(-1.7 * r, 2.2 * r); ax.set_ylim(-0.5 * r, h + 0.8 * r)
    elif kind == "cone":
        r, h = d["r"], d["h"]
        ax.plot([-r, 0, r], [0, h, 0], **_LINE)
        ax.add_patch(patches.Arc((0, 0), 2 * r, 0.45 * r, theta1=180, theta2=360,
                                 edgecolor="black", linewidth=1.4))
        ax.add_patch(patches.Arc((0, 0), 2 * r, 0.45 * r, theta1=0, theta2=180,
                                 edgecolor="black", linewidth=1.0,
                                 linestyle=(0, (3, 3))))
        ax.plot([0, 0], [0, h], color="black", linewidth=0.9,
                linestyle=(0, (3, 3)))
        ax.text(0.06 * r, h / 2, f"h = {h:g}", ha="left", va="center", fontsize=10)
        ax.text(r / 2, -0.22 * r, f"r = {r:g}", ha="center", fontsize=10)
        ax.set_xlim(-1.5 * r, 1.5 * r); ax.set_ylim(-0.6 * r, h + 0.3 * r)
    elif kind == "sphere":
        r = d["r"]
        ax.add_patch(patches.Circle((0, 0), r, fill=False, edgecolor="black",
                                    linewidth=1.4))
        ax.add_patch(patches.Arc((0, 0), 2 * r, 0.5 * r, theta1=180, theta2=360,
                                 edgecolor="black", linewidth=1.0))
        ax.add_patch(patches.Arc((0, 0), 2 * r, 0.5 * r, theta1=0, theta2=180,
                                 edgecolor="black", linewidth=0.9,
                                 linestyle=(0, (3, 3))))
        ax.plot([0, r], [0, 0], **_LINE)
        ax.text(r / 2, 0.05 * r, f"r = {r:g}", ha="center", va="bottom", fontsize=10)
        ax.set_xlim(-1.3 * r, 1.3 * r); ax.set_ylim(-1.3 * r, 1.3 * r)
    else:   # square-based pyramid
        b, h = d["b"], d["h"]
        fl, fr = (-b / 2, 0), (b / 2, 0)
        bl, br = (-b / 2 + 0.3 * b, 0.22 * b), (b / 2 + 0.3 * b, 0.22 * b)
        apex = (0.15 * b, h)
        for p in (fl, fr, br):
            ax.plot([p[0], apex[0]], [p[1], apex[1]], **_LINE)
        ax.plot([bl[0], apex[0]], [bl[1], apex[1]], color="black",
                linewidth=1.0, linestyle=(0, (3, 3)))
        ax.plot([fl[0], fr[0], br[0]], [fl[1], fr[1], br[1]], **_LINE)
        ax.plot([fl[0], bl[0], br[0]], [fl[1], bl[1], br[1]], color="black",
                linewidth=1.0, linestyle=(0, (3, 3)))
        ax.text(0, -0.12 * b, f"base {b:g}", ha="center", va="top", fontsize=10)
        ax.text(apex[0] + 0.06 * b, h / 2, f"h = {h:g}", ha="left", fontsize=10)
        ax.set_xlim(-0.9 * b, 1.1 * b); ax.set_ylim(-0.35 * b, h * 1.12)
    return _render(fig, fmt)


def _fig_analytic(s: dict, fmt: str) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    pts = [(float(x), float(y)) for x, y in (s.get("points") or [])]
    circ = s.get("circle")
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    ax.set_aspect("equal")

    span = [abs(v) for p in pts for v in p]
    if circ:
        cx, cy, r = circ
        span += [abs(cx) + r, abs(cy) + r]
    lim = max(6, (max(span) if span else 5) + 1)
    lim = int(math.ceil(lim))
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.grid(True, linewidth=0.4, color="#bbbbbb")
    ax.axhline(0, color="black", linewidth=1.0)
    ax.axvline(0, color="black", linewidth=1.0)
    step = max(1, lim // 5)
    ax.set_xticks(range(-lim, lim + 1, step))
    ax.set_yticks(range(-lim, lim + 1, step))
    ax.tick_params(labelsize=7)

    if circ:
        cx, cy, r = circ
        ax.add_patch(patches.Circle((cx, cy), r, fill=False, color="black",
                                    linewidth=1.4))
        ax.plot(cx, cy, "o", color="black", markersize=3)
    if s.get("line"):
        m, c = s["line"]
        xx = [-lim, lim]
        ax.plot(xx, [m * v + c for v in xx], color="black", linewidth=1.3)
    if s.get("segment") and len(pts) >= 2:
        ax.plot([pts[0][0], pts[1][0]], [pts[0][1], pts[1][1]],
                color="black", linewidth=1.3)
    for i, (x, y) in enumerate(pts):
        ax.plot(x, y, "o", color="black", markersize=4)
        ax.annotate(f"{chr(65+i)}({x:g}, {y:g})", (x, y),
                    textcoords="offset points", xytext=(5, 4), fontsize=8)
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
            question=(rf"\text{{Angles on a straight line. One angle is }}{known}^\circ."
                      rf"\text{{ Find the other.}}"),
            answer=rf"{180-known}^\circ",
            steps=[rf"180^\circ - {known}^\circ = {180-known}^\circ"],
            topic="angles_basic", level=level)
    a = rng.randint(30, 80)
    b = rng.randint(30, 180 - a - 20)
    return Problem(
        question=(rf"\text{{A triangle has angles }}{a}^\circ \text{{ and }} {b}^\circ."
                  rf"\text{{ Find the third angle.}}"),
        answer=rf"{180-a-b}^\circ",
        steps=[rf"180^\circ - {a}^\circ - {b}^\circ = {180-a-b}^\circ"],
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
        steps=[rf"\tan {angle}^\circ = \frac{{x}}{{{adj}}}",
               rf"x = {adj}\tan {angle}^\circ"],
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
               rf"c^2 = {a}^2 + {b}^2 - 2({a})({b})\cos {C}^\circ"],
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


def _g():
    """SymPy, plus the LaTeX writer that uses examiner notation."""
    from researcher.mathtutor.generators import _sp, _latex

    return _sp(), _latex


def _exact(expr) -> str:
    r"""LaTeX an exact value with the positive term first.

    SymPy's canonical order writes the segment area as "-36 + 60\pi". No
    examiner writes that; the convention is "60\pi - 36". None of SymPy's
    `order=` options get this right in every case — `lex` fixes this one but
    turns "100 - 25\pi" into "-25\pi + 100" — so the rule is applied here.
    """
    from researcher.mathtutor.generators import _sp, _latex

    sp = _sp()
    if isinstance(expr, sp.Add):
        # Positive terms first, then the negative ones as subtractions. Done by
        # partition rather than by inspecting arg order, because SymPy's
        # canonical order puts the negative term first only some of the time.
        pos = [t for t in expr.args if not t.could_extract_minus_sign()]
        neg = [t for t in expr.args if t.could_extract_minus_sign()]
        if pos and neg:
            out = " + ".join(_latex(t) for t in pos)
            return out + "".join(f" - {_latex(-t)}" for t in neg)
    return _latex(expr)


def _line_eq(m, c) -> str:
    """y = mx + c, written in that order.

    A line is conventionally written with the gradient term first, which is the
    opposite of what _exact wants, so it gets its own helper rather than a flag.
    """
    from researcher.mathtutor.generators import _sp, _latex

    sp = _sp()
    x = sp.Symbol("x")
    if m == 0:
        return f"y = {_latex(c)}"
    term = "x" if m == 1 else ("-x" if m == -1 else f"{_latex(m)} x")
    if c == 0:
        return f"y = {term}"
    sign = "-" if c.could_extract_minus_sign() else "+"
    return f"y = {term} {sign} {_latex(abs(c))}"


def gen_circle_theorems(rng: random.Random, level: str):
    """The circle theorems, each with the configuration drawn.

    These are the backbone of a real geometry course and are almost absent from
    the North American sequence. Every one is stated as a reason, because an
    answer without the theorem that justifies it earns no marks.
    """
    from researcher.mathtutor.generators import Problem

    variant = rng.choice(["centre_angle", "semicircle", "same_segment",
                          "cyclic_quad", "tangent_radius"])

    if variant == "centre_angle":
        x = rng.randrange(20, 76)              # angle at the circumference
        centre = 2 * x
        ask_centre = rng.random() < 0.5
        pts = {"A": 200, "B": 340, "P": 80}
        if ask_centre:
            q = (r"\text{O is the centre. }\angle APB = " rf"{x}^\circ"
                 r"\text{. Find }\angle AOB\text{, giving a reason.}")
            ans = rf"\angle AOB = {centre}^\circ"
            labels = {"P": f"{x}°", "O": "?"}
        else:
            q = (r"\text{O is the centre. }\angle AOB = " rf"{centre}^\circ"
                 r"\text{. Find }\angle APB\text{, giving a reason.}")
            ans = rf"\angle APB = {x}^\circ"
            labels = {"O": f"{centre}°", "P": "?"}
        return Problem(
            question=q, answer=ans,
            steps=[r"\text{The angle at the centre is twice the angle at the "
                   r"circumference on the same arc.}"],
            topic="circle_theorems", level=level,
            figure=draw_circle_theorem("centre_angle", pts, labels))

    if variant == "semicircle":
        a = rng.randrange(20, 71)
        other = 90 - a
        return Problem(
            question=(r"\text{AB is a diameter. }\angle BAP = " rf"{a}^\circ"
                      r"\text{. Find }\angle ABP\text{, giving a reason.}"),
            answer=rf"\angle ABP = {other}^\circ",
            steps=[r"\text{The angle in a semicircle is } 90^\circ "
                   r"\text{, so } \angle APB = 90^\circ",
                   rf"180^\circ - 90^\circ - {a}^\circ = {other}^\circ"],
            topic="circle_theorems", level=level,
            figure=draw_circle_theorem("semicircle", {"A": 180, "B": 0, "P": 70},
                                       {"A": f"{a}°", "P": "90°", "B": "?"}))

    if variant == "same_segment":
        a = rng.randrange(22, 76)
        return Problem(
            question=(r"\text{P and Q lie on the same arc. }\angle APB = "
                      rf"{a}^\circ" r"\text{. Find }\angle AQB"
                      r"\text{, giving a reason.}"),
            answer=rf"\angle AQB = {a}^\circ",
            steps=[r"\text{Angles in the same segment are equal.}"],
            topic="circle_theorems", level=level,
            figure=draw_circle_theorem("same_segment",
                                       {"A": 200, "B": 340, "P": 80, "Q": 130},
                                       {"P": f"{a}°", "Q": "?"}))

    if variant == "cyclic_quad":
        a = rng.randrange(65, 126)
        return Problem(
            question=(r"\text{ABCD is a cyclic quadrilateral. }\angle DAB = "
                      rf"{a}^\circ" r"\text{. Find }\angle BCD"
                      r"\text{, giving a reason.}"),
            answer=rf"\angle BCD = {180 - a}^\circ",
            steps=[r"\text{Opposite angles of a cyclic quadrilateral sum to }"
                   r"180^\circ",
                   rf"180^\circ - {a}^\circ = {180 - a}^\circ"],
            topic="circle_theorems", level=level,
            figure=draw_circle_theorem("cyclic_quad",
                                       {"A": 30, "B": 120, "C": 210, "D": 310},
                                       {"A": f"{a}°", "C": "?"}))

    # tangent_radius
    a = rng.randrange(25, 66)
    return Problem(
        question=(r"\text{TA is a tangent to the circle, centre O. The angle "
                  r"between TA and the chord AO is }" rf"{90 - a}^\circ"
                  r"\text{. Find }\angle AOT\text{ in triangle OAT, giving a "
                  r"reason.}"),
        answer=rf"\angle AOT = {a}^\circ",
        steps=[r"\text{A tangent meets a radius at } 90^\circ",
               rf"180^\circ - 90^\circ - {90 - a}^\circ = {a}^\circ"],
        topic="circle_theorems", level=level,
        figure=draw_circle_theorem("tangent", {"A": 40}, {"A": "90°"}))


def gen_inscribed_shapes(rng: random.Random, level: str):
    r"""Shaded-area problems: a circle inside a shape, or a shape inside a circle.

    The classic exercise of the Eastern European syllabus, and the one that
    actually tests whether a child can decompose a region. Answers stay exact
    and in terms of \pi.
    """
    from researcher.mathtutor.generators import Problem, _grade

    sp, latex = _g()
    grade = _grade(level)
    choices = ["circle_in_square", "square_in_circle", "quarter_circles"]
    if grade >= 10:
        choices.append("circle_in_triangle")
    variant = rng.choice(choices)

    if variant == "circle_in_square":
        s = rng.choice([4, 6, 8, 10, 12, 14])
        shaded = sp.nsimplify(s**2 - sp.pi * (sp.Rational(s, 2))**2)
        return Problem(
            question=(rf"\text{{A circle is inscribed in a square of side }} {s}"
                      r"\text{ cm. Find the shaded area, in terms of }\pi."),
            answer=rf"{_exact(sp.simplify(shaded))}\ \text{{cm}}^{{2}}",
            steps=[rf"\text{{Square}} = {s}^{{2}} = {s**2}",
                   rf"\text{{Circle}} = \pi \left(\frac{{{s}}}{{2}}\right)^{{2}}"
                   rf" = {latex(sp.pi * sp.Rational(s, 2)**2)}"],
            topic="inscribed_shapes", level=level,
            figure=draw_inscribed("circle_in_square", s))

    if variant == "square_in_circle":
        r = rng.choice([3, 4, 5, 6, 8, 10])
        # The diagonal of the square is the diameter, so the side is r*sqrt(2).
        shaded = sp.simplify(sp.pi * r**2 - 2 * r**2)
        return Problem(
            question=(rf"\text{{A square is inscribed in a circle of radius }} {r}"
                      r"\text{ cm. Find the shaded area, in terms of }\pi."),
            answer=rf"{_exact(shaded)}\ \text{{cm}}^{{2}}",
            steps=[r"\text{The diagonal of the square is the diameter, so the "
                   rf"side is }} {r}\sqrt{{2}}",
                   rf"\text{{Square}} = \left({r}\sqrt{{2}}\right)^{{2}} = {2*r**2}",
                   rf"\text{{Circle}} = {latex(sp.pi * r**2)}"],
            topic="inscribed_shapes", level=level,
            figure=draw_inscribed("square_in_circle", r))

    if variant == "quarter_circles":
        s = rng.choice([4, 6, 8, 10, 12])
        # Four quarter-circles of radius s/2 make exactly one full circle.
        shaded = sp.simplify(s**2 - sp.pi * sp.Rational(s, 2)**2)
        return Problem(
            question=(r"\text{A quarter circle is drawn at each corner of a "
                      rf"square of side }} {s} \text{{ cm, each of radius }} "
                      rf"{s // 2} \text{{ cm. Find the shaded area, in "
                      r"terms of }\pi."),
            answer=rf"{_exact(shaded)}\ \text{{cm}}^{{2}}",
            steps=[r"\text{Four quarter circles make one whole circle of radius }"
                   rf"{s // 2}",
                   rf"{s**2} - {latex(sp.pi * sp.Rational(s, 2)**2)}"],
            topic="inscribed_shapes", level=level,
            figure=draw_inscribed("quarter_circles", s))

    s = rng.choice([6, 8, 10, 12])
    # Equilateral triangle: area = s^2*sqrt(3)/4, inradius = s/(2*sqrt(3)).
    tri = sp.simplify(sp.sqrt(3) * s**2 / 4)
    inr = sp.simplify(s / (2 * sp.sqrt(3)))
    shaded = sp.simplify(tri - sp.pi * inr**2)
    return Problem(
        question=(r"\text{A circle is inscribed in an equilateral triangle of "
                  rf"side }} {s} \text{{ cm. Find the shaded area, in exact form.}}"),
        answer=rf"{_exact(shaded)}\ \text{{cm}}^{{2}}",
        steps=[rf"\text{{Triangle}} = \frac{{\sqrt{{3}}}}{{4}}({s})^{{2}} = {latex(tri)}",
               rf"\text{{Inradius}} = \frac{{{s}}}{{2\sqrt{{3}}}} = {latex(inr)}",
               rf"\text{{Circle}} = \pi r^{{2}} = {latex(sp.simplify(sp.pi * inr**2))}"],
        topic="inscribed_shapes", level=level,
        figure=draw_inscribed("circle_in_triangle", s))


def gen_sectors_arcs(rng: random.Random, level: str):
    r"""Arc length, sector area, perimeter and (higher up) segment area."""
    from researcher.mathtutor.generators import Problem, _grade

    sp, latex = _g()
    r = rng.choice([3, 4, 5, 6, 7, 8, 9, 10, 12])
    theta = rng.choice([30, 36, 40, 45, 60, 72, 90, 120, 135, 150, 240, 270])
    frac = sp.Rational(theta, 360)

    kinds = ["arc", "area", "perimeter"]
    if _grade(level) >= 11:
        kinds.append("segment")
    kind = rng.choice(kinds)

    if kind == "arc":
        val = sp.simplify(frac * 2 * sp.pi * r)
        return Problem(
            question=(rf"\text{{A sector has radius }} {r} \text{{ cm and angle }} "
                      rf"{theta}^\circ" r"\text{. Find the arc length, in terms "
                      r"of }\pi."),
            answer=rf"{_exact(val)}\ \text{{cm}}",
            steps=[rf"\frac{{{theta}}}{{360}} \times 2\pi({r})"],
            topic="sectors_arcs", level=level,
            figure=draw_sector(r, theta))

    if kind == "area":
        val = sp.simplify(frac * sp.pi * r**2)
        return Problem(
            question=(rf"\text{{A sector has radius }} {r} \text{{ cm and angle }} "
                      rf"{theta}^\circ" r"\text{. Find its area, in terms of }\pi."),
            answer=rf"{latex(val)}\ \text{{cm}}^{{2}}",
            steps=[rf"\frac{{{theta}}}{{360}} \times \pi({r})^{{2}}"],
            topic="sectors_arcs", level=level,
            figure=draw_sector(r, theta))

    if kind == "perimeter":
        val = sp.simplify(frac * 2 * sp.pi * r + 2 * r)
        return Problem(
            question=(rf"\text{{A sector has radius }} {r} \text{{ cm and angle }} "
                      rf"{theta}^\circ" r"\text{. Find its perimeter, in terms "
                      r"of }\pi."),
            answer=rf"{_exact(val)}\ \text{{cm}}",
            steps=[r"\text{Perimeter} = \text{arc} + 2r",
                   rf"\frac{{{theta}}}{{360}} \times 2\pi({r}) + 2({r})"],
            topic="sectors_arcs", level=level,
            figure=draw_sector(r, theta))

    # Segment = sector - triangle. Exact, so the angle is one with an exact sine.
    theta = rng.choice([30, 45, 60, 90, 120, 135, 150])
    frac = sp.Rational(theta, 360)
    sector = sp.simplify(frac * sp.pi * r**2)
    tri = sp.simplify(sp.Rational(1, 2) * r**2 * sp.sin(sp.rad(theta)))
    return Problem(
        question=(rf"\text{{A chord subtends an angle of }} {theta}^\circ "
                  rf"\text{{ at the centre of a circle of radius }} {r} "
                  r"\text{ cm. Find the area of the minor segment, in exact form.}"),
        answer=rf"{_exact(sp.simplify(sector - tri))}\ \text{{cm}}^{{2}}",
        steps=[rf"\text{{Sector}} = \frac{{{theta}}}{{360}} \times \pi({r})^{{2}}"
               rf" = {latex(sector)}",
               rf"\text{{Triangle}} = \tfrac{{1}}{{2}}({r})^{{2}}\sin {theta}^\circ"
               rf" = {latex(tri)}"],
        topic="sectors_arcs", level=level,
        figure=draw_sector(r, theta, segment=True))


def gen_similar_triangles(rng: random.Random, level: str):
    """Similar figures: missing sides, and the k^2 / k^3 area and volume ratios.

    The area ratio is where almost everyone goes wrong — doubling the lengths
    quadruples the area — so it is asked directly rather than hidden.
    """
    from researcher.mathtutor.generators import Problem, _grade

    sp, latex = _g()
    kind = rng.choice(["missing_side", "area_ratio", "volume_ratio"]
                      if _grade(level) >= 10 else ["missing_side", "area_ratio"])

    a, b = rng.choice([(3, 4), (5, 12), (6, 8), (8, 15), (4, 7), (5, 9)])
    k = rng.choice([sp.Rational(3, 2), sp.Integer(2), sp.Rational(5, 2),
                    sp.Integer(3), sp.Rational(7, 4)])

    if kind == "missing_side":
        a2, b2 = sp.simplify(a * k), sp.simplify(b * k)
        return Problem(
            question=(r"\text{The two triangles are similar. The shorter sides "
                      rf"are }} {a} \text{{ cm and }} {latex(a2)} "
                      rf"\text{{ cm. The longer side of the first is }} {b} "
                      r"\text{ cm. Find the longer side of the second.}"),
            answer=rf"{latex(b2)}\ \text{{cm}}",
            steps=[rf"\text{{Scale factor}} = \frac{{{latex(a2)}}}{{{a}}} = {latex(k)}",
                   rf"{b} \times {latex(k)} = {latex(b2)}"],
            topic="similar_triangles", level=level,
            figure=draw_similar(a, b, float(k),
                                {"a": f"{a}", "b": f"{b}",
                                 "a2": f"{sp.nsimplify(a2)}", "b2": f"{sp.nsimplify(b2)}"}))

    if kind == "area_ratio":
        area1 = rng.choice([12, 18, 20, 24, 32, 40])
        area2 = sp.simplify(area1 * k**2)
        return Problem(
            question=(r"\text{Two similar triangles have corresponding sides in "
                      rf"the ratio }} 1 : {latex(k)} \text{{. The smaller has "
                      rf"area }} {area1} \text{{ cm}}^{{2}} "
                      r"\text{. Find the area of the larger.}"),
            answer=rf"{latex(area2)}\ \text{{cm}}^{{2}}",
            steps=[rf"\text{{Area scale factor}} = \left({latex(k)}\right)^{{2}}"
                   rf" = {latex(sp.simplify(k**2))}",
                   rf"{area1} \times {latex(sp.simplify(k**2))} = {latex(area2)}"],
            topic="similar_triangles", level=level,
            figure=draw_similar(a, b, float(k), {}))

    vol1 = rng.choice([16, 24, 40, 54, 80])
    vol2 = sp.simplify(vol1 * k**3)
    return Problem(
        question=(r"\text{Two similar solids have corresponding lengths in the "
                  rf"ratio }} 1 : {latex(k)} \text{{. The smaller has volume }} "
                  rf"{vol1} \text{{ cm}}^{{3}}"
                  r"\text{. Find the volume of the larger.}"),
        answer=rf"{latex(vol2)}\ \text{{cm}}^{{3}}",
        steps=[rf"\text{{Volume scale factor}} = \left({latex(k)}\right)^{{3}}"
               rf" = {latex(sp.simplify(k**3))}",
               rf"{vol1} \times {latex(sp.simplify(k**3))} = {latex(vol2)}"],
        topic="similar_triangles", level=level)


def gen_angle_rules(rng: random.Random, level: str):
    """Angles in parallel lines and polygons, including algebraic forms."""
    from researcher.mathtutor.generators import Problem, _grade

    sp, latex = _g()
    grade = _grade(level)
    kinds = ["parallel", "polygon_sum", "regular_polygon", "exterior"]
    if grade >= 8:
        kinds.append("algebraic")
    kind = rng.choice(kinds)

    if kind == "parallel":
        a = rng.randrange(35, 146)
        rel = rng.choice(["alternate", "corresponding", "co-interior"])
        if rel == "co-interior":
            ans, reason = 180 - a, ("Co-interior angles between parallel lines "
                                    "sum to } 180^\\circ")
        else:
            ans, reason = a, (f"{rel.capitalize()} angles between parallel "
                              "lines are equal.")
        return Problem(
            question=(rf"\text{{The lines are parallel and one angle is }} {a}^\circ"
                      r"\text{. Find } x \text{, giving a reason.}"),
            answer=rf"x = {ans}^\circ",
            steps=[rf"\text{{{reason}}}"],
            topic="angle_rules", level=level,
            figure=draw_parallel_lines(a if rel != "co-interior" else 180 - a,
                                       f"{a}°"))

    if kind == "polygon_sum":
        n = rng.randrange(5, 13)
        return Problem(
            question=(rf"\text{{Find the sum of the interior angles of a polygon "
                      rf"with }} {n} \text{{ sides.}}"),
            answer=rf"{(n - 2) * 180}^\circ",
            steps=[rf"(n-2) \times 180^\circ = ({n}-2) \times 180^\circ"],
            topic="angle_rules", level=level,
            figure=draw_polygon(n))

    if kind == "regular_polygon":
        n = rng.choice([5, 6, 8, 9, 10, 12])
        interior = sp.Rational((n - 2) * 180, n)
        return Problem(
            question=(rf"\text{{Find the size of each interior angle of a regular "
                      rf"{n}-sided polygon.}}"),
            answer=rf"{latex(interior)}^\circ",
            steps=[rf"\frac{{({n}-2) \times 180^\circ}}{{{n}}}"],
            topic="angle_rules", level=level,
            figure=draw_polygon(n))

    if kind == "exterior":
        n = rng.choice([5, 6, 8, 9, 10, 12, 15, 18])
        ext = sp.Rational(360, n)
        if rng.random() < 0.5:
            return Problem(
                question=(rf"\text{{Find the exterior angle of a regular }} {n}"
                          r"\text{-sided polygon.}"),
                answer=rf"{latex(ext)}^\circ",
                steps=[rf"\frac{{360^\circ}}{{{n}}}"],
                topic="angle_rules", level=level, figure=draw_polygon(n))
        return Problem(
            question=(r"\text{A regular polygon has an exterior angle of }"
                      rf"{latex(ext)}^\circ" r"\text{. How many sides has it?}"),
            answer=str(n),
            steps=[rf"n = \frac{{360^\circ}}{{{latex(ext)}^\circ}} = {n}"],
            topic="angle_rules", level=level, figure=draw_polygon(n))

    # Algebraic angles — the form that actually requires an equation.
    x = sp.Symbol("x")
    coeffs = rng.choice([(1, 2, 3), (1, 1, 2), (2, 3, 4), (1, 3, 5), (2, 2, 5)])
    total = sum(coeffs)
    val = sp.Rational(180, total)
    angles = [sp.simplify(c * val) for c in coeffs]
    shown = ", ".join(f"{c}x" if c > 1 else "x" for c in coeffs)
    return Problem(
        question=(rf"\text{{The angles of a triangle are }} {shown} "
                  r"\text{. Find } x \text{ and the three angles.}"),
        answer=(rf"x = {latex(val)}^\circ, \ "
                + ", ".join(f"{latex(a)}^\\circ" for a in angles)),
        steps=[rf"{total}x = 180^\circ", rf"x = {latex(val)}^\circ"],
        topic="angle_rules", level=level)


def gen_analytic_geometry(rng: random.Random, level: str):
    """Coordinate geometry: distance, midpoint, gradient, perpendicular bisector.

    Lengths come out as exact surds, which is the whole point of doing it
    analytically rather than measuring off the page.
    """
    from researcher.mathtutor.generators import Problem, _grade

    sp, latex = _g()

    def pt():
        return (rng.randrange(-7, 8), rng.randrange(-7, 8))

    A, B = pt(), pt()
    if A == B:
        raise ValueError("coincident points")
    dx, dy = B[0] - A[0], B[1] - A[1]

    kinds = ["distance", "midpoint", "gradient", "line_equation"]
    if _grade(level) >= 10:
        kinds += ["perpendicular", "perp_bisector"]
    kind = rng.choice(kinds)

    if kind == "distance":
        d = sp.sqrt(dx**2 + dy**2)
        return Problem(
            question=(rf"\text{{Find the exact distance between }} A{A} "
                      rf"\text{{ and }} B{B}."),
            answer=latex(sp.simplify(d)),
            steps=[rf"\sqrt{{({dx})^{{2}} + ({dy})^{{2}}}} = \sqrt{{{dx**2 + dy**2}}}"],
            topic="analytic_geometry", level=level,
            figure=draw_analytic([A, B], segment=True))

    if kind == "midpoint":
        m = (sp.Rational(A[0] + B[0], 2), sp.Rational(A[1] + B[1], 2))
        return Problem(
            question=rf"\text{{Find the midpoint of }} A{A} \text{{ and }} B{B}.",
            answer=rf"\left({latex(m[0])}, {latex(m[1])}\right)",
            steps=[r"\left(\frac{x_1+x_2}{2}, \frac{y_1+y_2}{2}\right)"],
            topic="analytic_geometry", level=level,
            figure=draw_analytic([A, B], segment=True))

    if dx == 0:
        raise ValueError("vertical line — gradient undefined")
    m = sp.Rational(dy, dx)

    if kind == "gradient":
        return Problem(
            question=rf"\text{{Find the gradient of the line through }} A{A} \text{{ and }} B{B}.",
            answer=latex(m),
            steps=[rf"\frac{{{B[1]} - ({A[1]})}}{{{B[0]} - ({A[0]})}} = {latex(m)}"],
            topic="analytic_geometry", level=level,
            figure=draw_analytic([A, B], line=(float(m), float(A[1] - m * A[0]))))

    c = sp.simplify(A[1] - m * A[0])
    if kind == "line_equation":
        return Problem(
            question=(rf"\text{{Find the equation of the line through }} A{A} "
                      rf"\text{{ and }} B{B} \text{{, in the form }} y = mx + c."),
            answer=_line_eq(m, c),
            steps=[rf"m = {latex(m)}", rf"c = {latex(c)}"],
            topic="analytic_geometry", level=level,
            figure=draw_analytic([A, B], line=(float(m), float(c))))

    if m == 0:
        raise ValueError("horizontal line — no perpendicular gradient")
    mp = sp.simplify(-1 / m)

    if kind == "perpendicular":
        return Problem(
            question=(rf"\text{{A line has gradient }} {latex(m)} "
                      r"\text{. Find the gradient of a line perpendicular to it.}"),
            answer=latex(mp),
            steps=[rf"m_1 m_2 = -1 \implies m_2 = \frac{{-1}}{{{latex(m)}}}"],
            topic="analytic_geometry", level=level)

    mid = (sp.Rational(A[0] + B[0], 2), sp.Rational(A[1] + B[1], 2))
    cp = sp.simplify(mid[1] - mp * mid[0])
    return Problem(
        question=(rf"\text{{Find the equation of the perpendicular bisector of }} "
                  rf"A{A} \text{{ and }} B{B}."),
        answer=_line_eq(mp, cp),
        steps=[rf"\text{{Midpoint}} = \left({latex(mid[0])}, {latex(mid[1])}\right)",
               rf"\text{{Gradient of AB}} = {latex(m)}"
               rf"\text{{, so perpendicular gradient}} = {latex(mp)}"],
        topic="analytic_geometry", level=level,
        figure=draw_analytic([A, B], segment=True))


def gen_circle_equations(rng: random.Random, level: str):
    """The equation of a circle: reading it off, and completing the square."""
    from researcher.mathtutor.generators import Problem

    sp, latex = _g()
    x, y = sp.symbols("x y")
    a, b = rng.randrange(-6, 7), rng.randrange(-6, 7)
    r = rng.choice([2, 3, 4, 5, 6, 7])
    kind = rng.choice(["read_off", "complete_square", "from_centre", "from_diameter"])

    if kind == "read_off":
        lhs = (x - a)**2 + (y - b)**2
        return Problem(
            question=(rf"\text{{Find the centre and radius of the circle }} "
                      rf"{latex(lhs)} = {r**2}."),
            answer=rf"\text{{Centre }}({a}, {b}), \ \text{{radius }} {r}",
            topic="circle_equations", level=level,
            figure=draw_analytic([], circle=(a, b, r)))

    if kind == "complete_square":
        expanded = sp.expand((x - a)**2 + (y - b)**2 - r**2)
        return Problem(
            question=(rf"\text{{By completing the square, find the centre and "
                      rf"radius of }} {latex(expanded)} = 0."),
            answer=rf"\text{{Centre }}({a}, {b}), \ \text{{radius }} {r}",
            steps=[rf"(x {'-' if a >= 0 else '+'} {abs(a)})^{{2}} "
                   rf"+ (y {'-' if b >= 0 else '+'} {abs(b)})^{{2}} = {r**2}"],
            topic="circle_equations", level=level,
            figure=draw_analytic([], circle=(a, b, r)))

    if kind == "from_centre":
        return Problem(
            question=(rf"\text{{Find the equation of the circle with centre }} "
                      rf"({a}, {b}) \text{{ and radius }} {r}."),
            answer=rf"{latex((x - a)**2 + (y - b)**2)} = {r**2}",
            topic="circle_equations", level=level,
            figure=draw_analytic([], circle=(a, b, r)))

    # Endpoints of a diameter: centre is the midpoint, radius is half the length.
    P = (a - r, b)
    Q = (a + r, b)
    return Problem(
        question=(rf"\text{{P}}{P} \text{{ and Q}}{Q} \text{{ are the ends of a "
                  r"diameter. Find the equation of the circle.}"),
        answer=rf"{latex((x - a)**2 + (y - b)**2)} = {r**2}",
        steps=[rf"\text{{Centre is the midpoint}} = ({a}, {b})",
               rf"\text{{Radius}} = \tfrac{{1}}{{2}} \times {2*r} = {r}"],
        topic="circle_equations", level=level,
        figure=draw_analytic([P, Q], circle=(a, b, r)))


def gen_geometry_3d(rng: random.Random, level: str):
    """Solids: volume, surface area, the space diagonal, and angles in 3D.

    The space diagonal and the angle a diagonal makes with the base are the
    problems that separate someone who can picture a solid from someone who has
    only memorised formulae.
    """
    from researcher.mathtutor.generators import Problem, _grade

    sp, latex = _g()
    grade = _grade(level)
    kinds = ["cuboid", "cylinder", "cone", "sphere", "pyramid"]
    if grade >= 9:
        kinds.append("space_diagonal")
    if grade >= 10:
        kinds.append("angle_to_base")
    kind = rng.choice(kinds)

    if kind == "cuboid":
        a, b, c = (rng.randrange(2, 13) for _ in range(3))
        want = rng.choice(["volume", "surface area"])
        val = a * b * c if want == "volume" else 2 * (a*b + b*c + a*c)
        unit = "3" if want == "volume" else "2"
        return Problem(
            question=(rf"\text{{Find the {want} of a cuboid measuring }} "
                      rf"{a} \times {b} \times {c} \text{{ cm.}}"),
            answer=rf"{val}\ \text{{cm}}^{{{unit}}}",
            steps=[rf"V = {a} \times {b} \times {c}" if want == "volume"
                   else rf"2({a}\cdot{b} + {b}\cdot{c} + {a}\cdot{c})"],
            topic="geometry_3d", level=level,
            figure=draw_cuboid(a, b, c))

    if kind == "cylinder":
        r, h = rng.randrange(2, 10), rng.randrange(3, 16)
        want = rng.choice(["volume", "curved surface area"])
        val = sp.pi * r**2 * h if want == "volume" else 2 * sp.pi * r * h
        unit = "3" if want == "volume" else "2"
        return Problem(
            question=(rf"\text{{A cylinder has radius }} {r} \text{{ cm and "
                      rf"height }} {h} \text{{ cm. Find its {want}, in terms "
                      r"of }\pi."),
            answer=rf"{latex(val)}\ \text{{cm}}^{{{unit}}}",
            steps=[rf"\pi r^{{2}} h = \pi({r})^{{2}}({h})" if want == "volume"
                   else rf"2\pi r h = 2\pi({r})({h})"],
            topic="geometry_3d", level=level,
            figure=draw_solid("cylinder", {"r": r, "h": h}))

    if kind == "cone":
        r, h = rng.choice([(3, 4), (6, 8), (5, 12), (9, 12), (8, 15)])
        want = rng.choice(["volume", "slant height"])
        if want == "volume":
            val = sp.simplify(sp.Rational(1, 3) * sp.pi * r**2 * h)
            return Problem(
                question=(rf"\text{{A cone has radius }} {r} \text{{ cm and "
                          rf"height }} {h} \text{{ cm. Find its volume, in "
                          r"terms of }\pi."),
                answer=rf"{latex(val)}\ \text{{cm}}^{{3}}",
                steps=[rf"\tfrac{{1}}{{3}}\pi r^{{2}} h = "
                       rf"\tfrac{{1}}{{3}}\pi({r})^{{2}}({h})"],
                topic="geometry_3d", level=level,
                figure=draw_solid("cone", {"r": r, "h": h}))
        l = sp.sqrt(r**2 + h**2)
        return Problem(
            question=(rf"\text{{A cone has radius }} {r} \text{{ cm and height }} "
                      rf"{h} \text{{ cm. Find its slant height, in exact form.}}"),
            answer=rf"{latex(sp.simplify(l))}\ \text{{cm}}",
            steps=[rf"l = \sqrt{{{r}^{{2}} + {h}^{{2}}}} = \sqrt{{{r**2 + h**2}}}"],
            topic="geometry_3d", level=level,
            figure=draw_solid("cone", {"r": r, "h": h}))

    if kind == "sphere":
        r = rng.randrange(2, 13)
        want = rng.choice(["volume", "surface area"])
        val = (sp.simplify(sp.Rational(4, 3) * sp.pi * r**3) if want == "volume"
               else sp.simplify(4 * sp.pi * r**2))
        unit = "3" if want == "volume" else "2"
        return Problem(
            question=(rf"\text{{Find the {want} of a sphere of radius }} {r} "
                      r"\text{ cm, in terms of }\pi."),
            answer=rf"{latex(val)}\ \text{{cm}}^{{{unit}}}",
            steps=[rf"\tfrac{{4}}{{3}}\pi r^{{3}}" if want == "volume"
                   else rf"4\pi r^{{2}}"],
            topic="geometry_3d", level=level,
            figure=draw_solid("sphere", {"r": r}))

    if kind == "pyramid":
        b, h = rng.randrange(3, 13), rng.randrange(4, 16)
        val = sp.simplify(sp.Rational(1, 3) * b**2 * h)
        return Problem(
            question=(rf"\text{{A pyramid has a square base of side }} {b} "
                      rf"\text{{ cm and height }} {h} \text{{ cm. Find its volume.}}"),
            answer=rf"{latex(val)}\ \text{{cm}}^{{3}}",
            steps=[rf"\tfrac{{1}}{{3}} \times {b}^{{2}} \times {h}"],
            topic="geometry_3d", level=level,
            figure=draw_solid("pyramid", {"b": b, "h": h}))

    a, b, c = (rng.randrange(2, 13) for _ in range(3))
    if kind == "space_diagonal":
        d = sp.sqrt(a**2 + b**2 + c**2)
        return Problem(
            question=(rf"\text{{Find the length of the space diagonal of a cuboid "
                      rf"measuring }} {a} \times {b} \times {c} "
                      r"\text{ cm, in exact form.}"),
            answer=rf"{latex(sp.simplify(d))}\ \text{{cm}}",
            steps=[rf"d = \sqrt{{{a}^{{2}} + {b}^{{2}} + {c}^{{2}}}} "
                   rf"= \sqrt{{{a**2 + b**2 + c**2}}}"],
            topic="geometry_3d", level=level,
            figure=draw_cuboid(a, b, c, diagonal=True))

    # The angle the space diagonal makes with the base.
    base = sp.sqrt(a**2 + b**2)
    ang = sp.N(sp.deg(sp.atan(c / base)), 4)
    return Problem(
        question=(rf"\text{{A cuboid measures }} {a} \times {b} \times {c} "
                  r"\text{ cm. Find the angle between the space diagonal and "
                  r"the base, correct to 3 significant figures.}"),
        answer=rf"{ang}^\circ",
        steps=[rf"\text{{Base diagonal}} = \sqrt{{{a}^{{2}} + {b}^{{2}}}} "
               rf"= {latex(sp.simplify(base))}",
               rf"\tan\theta = \frac{{{c}}}{{{latex(sp.simplify(base))}}}"],
        topic="geometry_3d", level=level,
        figure=draw_cuboid(a, b, c, diagonal=True))


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
    "circle_theorems": gen_circle_theorems,
    "inscribed_shapes": gen_inscribed_shapes,
    "sectors_arcs": gen_sectors_arcs,
    "similar_triangles": gen_similar_triangles,
    "angle_rules": gen_angle_rules,
    "analytic_geometry": gen_analytic_geometry,
    "circle_equations": gen_circle_equations,
    "geometry_3d": gen_geometry_3d,
}
