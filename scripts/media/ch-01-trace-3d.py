"""Generate ch-01 animated figure: 3-D index -> flat offset, looping multi-sample trace.

For shape [2, 3, 4] (strides [12, 4, 1]) we cycle through six [d, r, c] positions.
In each frame:
  * the active cell inside one of the two 3x4 slabs lights up amber,
  * an arrow drops down to the matching slot in the 24-cell flat buffer,
  * that buffer cell also lights up,
  * a formula line displays:  offset = d x 12 + r x 4 + c x 1 = R,
    with each index colour-matched to its axis.

Pure-Python SVG generator with SMIL animations (no external deps). Renders to a
single deterministic file: docs/assets/ch-01/trace-3d.svg.

Like the 2-D trace, we use calcMode="discrete" so highlights snap cleanly between
frames rather than cross-fading.
"""

from pathlib import Path

# ---- shared palette ----
PAPER    = "#faf7f0"
INK      = "#1d1d1b"
SLATE    = "#64748b"
PURPLE   = "#7c3aed"   # axis 0 (slab, stride 12)
PURPLE_L = "#ede9fe"
TEAL     = "#0f766e"   # axis 1 (row, stride 4)
TEAL_L   = "#d8f0eb"
AMBER    = "#b45309"   # axis 2 (col, stride 1) AND highlight colour
AMBER_L  = "#fef3c7"

W, H = 1040, 680
OUT = Path(__file__).resolve().parents[2] / "docs/assets/ch-01/trace-3d.svg"

# Shape [2, 3, 4]. Row-major strides [12, 4, 1].
D, R, C = 2, 3, 4
S0, S1, S2 = 12, 4, 1

# Six [d, r, c] samples that visit both slabs and several rows/cols.
FRAMES = [(0, 0, 0), (0, 1, 2), (0, 2, 3), (1, 0, 0), (1, 1, 2), (1, 2, 3)]
N = len(FRAMES)
FRAME_SEC = 1.8                       # a hair slower than 2-D trace -- more to read
DUR = FRAME_SEC * N                   # 10.8s total
DUR_STR = f"{DUR:g}s"

# keyTimes for an N-step discrete loop with closing value at t=1.
KEY_TIMES = ";".join(f"{i / N:.4f}" for i in range(N)) + ";1"


def values_for(active_value, default_value, is_active_in_frame):
    vs = [active_value if is_active_in_frame(f) else default_value for f in range(N)]
    vs.append(vs[0])
    return ";".join(vs)


def animate(attr, vals):
    return (f'<animate attributeName="{attr}" values="{vals}" '
            f'keyTimes="{KEY_TIMES}" dur="{DUR_STR}" '
            f'repeatCount="indefinite" calcMode="discrete"/>')


def txt(x, y, s, anchor="middle", size=13, fill=INK, weight="normal",
        family="ui-sans-serif, -apple-system, Helvetica, Arial"):
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
            f'font-family="{family}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}">{s}</text>')


def rect(x, y, w, h, fill=PAPER, stroke=INK, sw=1.3, rx=4):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def build():
    out = []
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
               f'width="{W}" height="{H}" '
               f'font-family="ui-sans-serif, -apple-system, Helvetica, Arial">')
    out.append('<defs>'
               '<marker id="arr3d" viewBox="0 0 10 10" refX="9" refY="5" '
               'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
               '<path d="M0,0 L10,5 L0,10 z" fill="' + AMBER + '"/></marker>'
               '</defs>')
    out.append(f'<rect width="{W}" height="{H}" fill="{PAPER}"/>')

    # ---- Title ----
    out.append(txt(W / 2, 30,
                   "Tracing  index \u2192 flat offset  for shape [2, 3, 4]",
                   size=18, weight="700"))
    out.append(txt(W / 2, 50,
                   "every 1.8 seconds we pick a new [d, r, c] and watch it light up in all three views",
                   size=12, fill=SLATE))

    # ---- Two slabs (top half) ----
    cw, ch = 46, 38
    slab_w = C * cw
    slab_h = R * ch
    slab_gap = 90
    total_w = slab_w * 2 + slab_gap
    slab0_x = (W - total_w) / 2
    slab1_x = slab0_x + slab_w + slab_gap
    slab_y = 110

    # axis-2 (col) header above each slab
    for d, sx in enumerate((slab0_x, slab1_x)):
        out.append(txt(sx + slab_w / 2, slab_y - 30,
                       "axis 2 (cols, stride 1)  \u2192",
                       size=11, fill=AMBER, weight="600"))
        for j in range(C):
            out.append(txt(sx + j * cw + cw / 2, slab_y - 12, str(j),
                           size=11, fill=AMBER))
        # axis-0 (slab) label below
        out.append(txt(sx + slab_w / 2, slab_y + slab_h + 28,
                       f"axis 0 = {d}  (slab, stride 12)",
                       size=12, fill=PURPLE, weight="700"))
        # axis-1 (row) labels on left of first slab
        if d == 0:
            out.append(txt(sx - 44, slab_y + slab_h / 2 - 6, "axis 1",
                           size=11, fill=TEAL, weight="600"))
            out.append(txt(sx - 44, slab_y + slab_h / 2 + 8, "(rows,",
                           size=10, fill=TEAL))
            out.append(txt(sx - 44, slab_y + slab_h / 2 + 20, "stride 4)",
                           size=10, fill=TEAL))
        for i in range(R):
            out.append(txt(sx - 10, slab_y + i * ch + ch / 2 + 4, str(i),
                           size=11, fill=TEAL, anchor="end"))

    # cells in each slab (animated)
    for d, sx in enumerate((slab0_x, slab1_x)):
        for i in range(R):
            for j in range(C):
                offset = d * S0 + i * S1 + j * S2
                cx = sx + j * cw
                cy = slab_y + i * ch
                fill_vals = values_for(
                    AMBER_L, PAPER,
                    lambda f, dd=d, ii=i, jj=j: FRAMES[f] == (dd, ii, jj))
                stroke_vals = values_for(
                    AMBER, INK,
                    lambda f, dd=d, ii=i, jj=j: FRAMES[f] == (dd, ii, jj))
                sw_vals = values_for(
                    "2.4", "1.3",
                    lambda f, dd=d, ii=i, jj=j: FRAMES[f] == (dd, ii, jj))
                out.append(
                    f'<rect x="{cx}" y="{cy}" width="{cw}" height="{ch}" rx="4" '
                    f'fill="{PAPER}" stroke="{INK}" stroke-width="1.3">'
                    + animate("fill", fill_vals)
                    + animate("stroke", stroke_vals)
                    + animate("stroke-width", sw_vals)
                    + '</rect>'
                )
                color_vals = values_for(
                    AMBER, INK,
                    lambda f, dd=d, ii=i, jj=j: FRAMES[f] == (dd, ii, jj))
                weight_vals = values_for(
                    "700", "500",
                    lambda f, dd=d, ii=i, jj=j: FRAMES[f] == (dd, ii, jj))
                out.append(
                    f'<text x="{cx + cw / 2}" y="{cy + ch / 2 + 4}" text-anchor="middle" '
                    f'font-size="13" font-weight="500" fill="{INK}">{offset}'
                    + animate("fill", color_vals)
                    + animate("font-weight", weight_vals)
                    + '</text>'
                )

    out.append(txt(W / 2, slab_y + slab_h + 56,
                   "(each cell shows its own flat-buffer offset)",
                   size=11, fill=SLATE))

    # ---- Flat buffer (bottom) ----
    bcw, bch = 34, 44
    buf_w = D * R * C * bcw           # 24 cells
    buf_x = (W - buf_w) / 2
    buf_y = slab_y + slab_h + 110

    out.append(txt(W / 2, buf_y - 14,
                   "Flat buffer  \u2014  Float64Array(24)",
                   size=13, fill=INK, weight="600"))

    for k in range(D * R * C):
        bx = buf_x + k * bcw
        fill_vals = values_for(
            AMBER_L, PAPER,
            lambda f, kk=k: FRAMES[f][0] * S0 + FRAMES[f][1] * S1 + FRAMES[f][2] * S2 == kk)
        stroke_vals = values_for(
            AMBER, INK,
            lambda f, kk=k: FRAMES[f][0] * S0 + FRAMES[f][1] * S1 + FRAMES[f][2] * S2 == kk)
        sw_vals = values_for(
            "2.4", "1.2",
            lambda f, kk=k: FRAMES[f][0] * S0 + FRAMES[f][1] * S1 + FRAMES[f][2] * S2 == kk)
        out.append(
            f'<rect x="{bx}" y="{buf_y}" width="{bcw}" height="{bch}" rx="3" '
            f'fill="{PAPER}" stroke="{INK}" stroke-width="1.2">'
            + animate("fill", fill_vals)
            + animate("stroke", stroke_vals)
            + animate("stroke-width", sw_vals)
            + '</rect>'
        )
        color_vals = values_for(
            AMBER, INK,
            lambda f, kk=k: FRAMES[f][0] * S0 + FRAMES[f][1] * S1 + FRAMES[f][2] * S2 == kk)
        weight_vals = values_for(
            "700", "500",
            lambda f, kk=k: FRAMES[f][0] * S0 + FRAMES[f][1] * S1 + FRAMES[f][2] * S2 == kk)
        out.append(
            f'<text x="{bx + bcw / 2}" y="{buf_y + bch / 2 + 4}" text-anchor="middle" '
            f'font-size="11" font-weight="500" fill="{INK}">{k}'
            + animate("fill", color_vals)
            + animate("font-weight", weight_vals)
            + '</text>'
        )

    # ---- Arrow from active slab cell to active buffer cell ----
    def slab_center(d, i, j):
        sx = slab0_x if d == 0 else slab1_x
        return (sx + j * cw + cw / 2, slab_y + i * ch + ch)        # bottom of cell

    def buf_center(k):
        return (buf_x + k * bcw + bcw / 2, buf_y)                   # top of cell

    x1s, y1s, x2s, y2s = [], [], [], []
    for (d, i, j) in FRAMES:
        k = d * S0 + i * S1 + j * S2
        gx, gy = slab_center(d, i, j)
        bx, by = buf_center(k)
        x1s.append(f"{gx:.1f}"); y1s.append(f"{gy:.1f}")
        x2s.append(f"{bx:.1f}"); y2s.append(f"{by - 4:.1f}")
    x1s.append(x1s[0]); y1s.append(y1s[0])
    x2s.append(x2s[0]); y2s.append(y2s[0])

    out.append(
        f'<line x1="{x1s[0]}" y1="{y1s[0]}" x2="{x2s[0]}" y2="{y2s[0]}" '
        f'stroke="{AMBER}" stroke-width="2.2" marker-end="url(#arr3d)">'
        + animate("x1", ";".join(x1s))
        + animate("y1", ";".join(y1s))
        + animate("x2", ";".join(x2s))
        + animate("y2", ";".join(y2s))
        + '</line>'
    )

    # ---- Formula line (one <text> per frame, opacity-toggled) ----
    # The formula gets a tinted, bordered "answer box" so it stands out on scroll.
    formula_y = buf_y + bch + 60
    out.append(txt(W / 2, formula_y - 38,
                   "offset  =  d \u00d7 stride[0]  +  r \u00d7 stride[1]  +  c \u00d7 stride[2]",
                   size=12, fill=SLATE))

    box_w, box_h = 760, 52
    box_x = (W - box_w) / 2
    box_y = formula_y - 32
    out.append(rect(box_x, box_y, box_w, box_h,
                    fill=AMBER_L, stroke=AMBER, sw=1.5, rx=10))

    for f, (d, i, j) in enumerate(FRAMES):
        k = d * S0 + i * S1 + j * S2
        opa_vals = ";".join("1" if g == f else "0" for g in range(N)) + (
            ";1" if f == 0 else ";0"
        )
        out.append(
            f'<text x="{W / 2}" y="{box_y + box_h / 2 + 7}" text-anchor="middle" '
            f'font-size="20" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" '
            f'opacity="{1 if f == 0 else 0}">'
            f'<tspan fill="{INK}">[ </tspan>'
            f'<tspan fill="{PURPLE}" font-weight="700">{d}</tspan>'
            f'<tspan fill="{INK}">, </tspan>'
            f'<tspan fill="{TEAL}" font-weight="700">{i}</tspan>'
            f'<tspan fill="{INK}">, </tspan>'
            f'<tspan fill="{AMBER}" font-weight="700">{j}</tspan>'
            f'<tspan fill="{INK}"> ]  \u2192  </tspan>'
            f'<tspan fill="{PURPLE}" font-weight="700">{d}</tspan>'
            f'<tspan fill="{INK}">\u00d712 + </tspan>'
            f'<tspan fill="{TEAL}" font-weight="700">{i}</tspan>'
            f'<tspan fill="{INK}">\u00d74 + </tspan>'
            f'<tspan fill="{AMBER}" font-weight="700">{j}</tspan>'
            f'<tspan fill="{INK}">\u00d71 = </tspan>'
            f'<tspan fill="{AMBER}" font-weight="800">{k}</tspan>'
            + animate("opacity", opa_vals)
            + '</text>'
        )

    out.append('</svg>')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    build()
