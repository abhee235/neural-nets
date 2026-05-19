"""Generate ch-01 animated figure: 2-D index -> flat offset, looping multi-sample trace.

For shape [3, 4] (strides [4, 1]) we cycle through six (row, col) positions. In each
frame:
  * the active cell in the 3x4 grid lights up amber,
  * an arrow drops down to the matching slot in the 12-cell flat buffer,
  * that buffer cell also lights up,
  * a formula line displays:  offset = row x 4 + col x 1 = R.

Pure-Python SVG generator with SMIL animations (no external deps). Renders to a
single deterministic file: docs/assets/ch-01/trace-2d.svg.

We use calcMode="discrete" so each highlight snaps cleanly between frames instead of
cross-fading -- the goal is to make the mapping easy to read, not to look smooth.
"""

from pathlib import Path

# ---- shared palette ----
PAPER   = "#faf7f0"
INK     = "#1d1d1b"
SLATE   = "#64748b"
TEAL    = "#0f766e"   # axis 1 (cols inside a row, stride 1) -- here doubles as "row" axis colour
TEAL_L  = "#d8f0eb"
AMBER   = "#b45309"   # highlight colour
AMBER_L = "#fef3c7"
PURPLE  = "#7c3aed"   # axis 0 (rows, stride 4)
PURPLE_L = "#ede9fe"

W, H = 760, 580
OUT = Path(__file__).resolve().parents[2] / "docs/assets/ch-01/trace-2d.svg"

# Shape [3, 4]: 3 rows, 4 cols. Row-major strides [4, 1].
ROWS, COLS = 3, 4
STRIDE0, STRIDE1 = 4, 1

# The six (row, col) samples we cycle through. Each chosen to land in a different
# part of the grid so the viewer sees the full pattern.
FRAMES = [(0, 0), (0, 3), (1, 0), (1, 2), (2, 1), (2, 3)]
N = len(FRAMES)
# Tune FRAME_SEC up if the animation feels too fast to follow (e.g. 2.5 or 3.0).
# The companion filmstrip in trace-2d-frames.svg lets the reader dwell on any
# single frame for as long as they want, so we keep this brisk.
FRAME_SEC = 1.6                       # seconds each highlight stays on
DUR = FRAME_SEC * N                   # total loop length, e.g. 9.6s
DUR_STR = f"{DUR:g}s"

# keyTimes for a discrete N-step loop. Each frame f starts at f/N and the loop
# closes by repeating the first value at time 1 (required by SMIL).
KEY_TIMES = ";".join(f"{i / N:.4f}" for i in range(N)) + ";1"


def values_for(active_value, default_value, is_active_in_frame):
    """Build a SMIL `values="..."` string for an attribute that should equal
    active_value during frames where is_active_in_frame(f) is True, and
    default_value otherwise. Wraps back to frame 0's value at t=1 so the loop
    closes cleanly.
    """
    vs = [active_value if is_active_in_frame(f) else default_value for f in range(N)]
    vs.append(vs[0])
    return ";".join(vs)


def animate(attr, vals):
    return (f'<animate attributeName="{attr}" values="{vals}" '
            f'keyTimes="{KEY_TIMES}" dur="{DUR_STR}" '
            f'repeatCount="indefinite" calcMode="discrete"/>')


def rect(x, y, w, h, fill=PAPER, stroke=INK, sw=1.3, rx=4):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def txt(x, y, s, anchor="middle", size=13, fill=INK, weight="normal",
        family="ui-sans-serif, -apple-system, Helvetica, Arial"):
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
            f'font-family="{family}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}">{s}</text>')


def build():
    out = []
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
               f'width="{W}" height="{H}" '
               f'font-family="ui-sans-serif, -apple-system, Helvetica, Arial">')
    out.append('<defs>'
               '<marker id="arr2d" viewBox="0 0 10 10" refX="9" refY="5" '
               'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
               '<path d="M0,0 L10,5 L0,10 z" fill="' + AMBER + '"/></marker>'
               '</defs>')
    out.append(f'<rect width="{W}" height="{H}" fill="{PAPER}"/>')

    # ---- Title ----
    out.append(txt(W / 2, 30, "Tracing  index \u2192 flat offset  for shape [3, 4]",
                   size=18, weight="700"))
    out.append(txt(W / 2, 50,
                   "every 1.6 seconds we pick a new cell and watch it light up in BOTH views",
                   size=12, fill=SLATE))

    # ---- Grid (top half) ----
    cw, ch = 64, 56
    grid_w = COLS * cw
    grid_h = ROWS * ch
    grid_x = (W - grid_w) / 2
    grid_y = 100

    # axis labels
    out.append(txt(grid_x + grid_w / 2, grid_y - 30,
                   "axis 1 (cols, stride 1)  \u2192", size=12, fill=TEAL, weight="600"))
    for j in range(COLS):
        out.append(txt(grid_x + j * cw + cw / 2, grid_y - 10, str(j),
                       size=11, fill=TEAL))
    out.append(txt(grid_x - 50, grid_y + grid_h / 2 - 8,
                   "axis 0", size=12, fill=PURPLE, weight="600"))
    out.append(txt(grid_x - 50, grid_y + grid_h / 2 + 8,
                   "(rows,", size=11, fill=PURPLE))
    out.append(txt(grid_x - 50, grid_y + grid_h / 2 + 22,
                   "stride 4)", size=11, fill=PURPLE))
    for i in range(ROWS):
        out.append(txt(grid_x - 10, grid_y + i * ch + ch / 2 + 4, str(i),
                       size=11, fill=PURPLE, anchor="end"))

    # grid cells (each animated)
    for i in range(ROWS):
        for j in range(COLS):
            offset = i * STRIDE0 + j * STRIDE1
            cx = grid_x + j * cw
            cy = grid_y + i * ch
            fill_vals = values_for(AMBER_L, PAPER,
                                   lambda f, ii=i, jj=j: FRAMES[f] == (ii, jj))
            stroke_vals = values_for(AMBER, INK,
                                     lambda f, ii=i, jj=j: FRAMES[f] == (ii, jj))
            sw_vals = values_for("2.4", "1.3",
                                 lambda f, ii=i, jj=j: FRAMES[f] == (ii, jj))
            out.append(
                f'<rect x="{cx}" y="{cy}" width="{cw}" height="{ch}" rx="4" '
                f'fill="{PAPER}" stroke="{INK}" stroke-width="1.3">'
                + animate("fill", fill_vals)
                + animate("stroke", stroke_vals)
                + animate("stroke-width", sw_vals)
                + '</rect>'
            )
            # cell label: shows the offset that lives here
            text_color_vals = values_for(AMBER, INK,
                                         lambda f, ii=i, jj=j: FRAMES[f] == (ii, jj))
            weight_vals = values_for("700", "500",
                                     lambda f, ii=i, jj=j: FRAMES[f] == (ii, jj))
            out.append(
                f'<text x="{cx + cw / 2}" y="{cy + ch / 2 + 4}" text-anchor="middle" '
                f'font-size="14" font-weight="500" fill="{INK}">{offset}'
                + animate("fill", text_color_vals)
                + animate("font-weight", weight_vals)
                + '</text>'
            )

    # tiny legend under grid
    out.append(txt(W / 2, grid_y + grid_h + 24,
                   "(each cell shows its own flat-buffer offset)",
                   size=11, fill=SLATE))

    # ---- Flat buffer (bottom) ----
    bcw, bch = 44, 50
    buf_w = COLS * ROWS * bcw     # 12 cells
    buf_x = (W - buf_w) / 2
    buf_y = grid_y + grid_h + 130

    out.append(txt(W / 2, buf_y - 18,
                   "Flat buffer  \u2014  Float64Array(12)",
                   size=13, fill=INK, weight="600"))

    for k in range(ROWS * COLS):
        bx = buf_x + k * bcw
        fill_vals = values_for(AMBER_L, PAPER,
                               lambda f, kk=k: FRAMES[f][0] * STRIDE0
                                              + FRAMES[f][1] * STRIDE1 == kk)
        stroke_vals = values_for(AMBER, INK,
                                 lambda f, kk=k: FRAMES[f][0] * STRIDE0
                                                + FRAMES[f][1] * STRIDE1 == kk)
        sw_vals = values_for("2.4", "1.2",
                             lambda f, kk=k: FRAMES[f][0] * STRIDE0
                                            + FRAMES[f][1] * STRIDE1 == kk)
        out.append(
            f'<rect x="{bx}" y="{buf_y}" width="{bcw}" height="{bch}" rx="3" '
            f'fill="{PAPER}" stroke="{INK}" stroke-width="1.2">'
            + animate("fill", fill_vals)
            + animate("stroke", stroke_vals)
            + animate("stroke-width", sw_vals)
            + '</rect>'
        )
        weight_vals = values_for("700", "500",
                                 lambda f, kk=k: FRAMES[f][0] * STRIDE0
                                                + FRAMES[f][1] * STRIDE1 == kk)
        color_vals = values_for(AMBER, INK,
                                lambda f, kk=k: FRAMES[f][0] * STRIDE0
                                               + FRAMES[f][1] * STRIDE1 == kk)
        out.append(
            f'<text x="{bx + bcw / 2}" y="{buf_y + bch / 2 + 4}" text-anchor="middle" '
            f'font-size="13" font-weight="500" fill="{INK}">{k}'
            + animate("fill", color_vals)
            + animate("font-weight", weight_vals)
            + '</text>'
        )

    # ---- Arrow from active grid cell to active buffer cell ----
    # We animate the endpoint coordinates with calcMode=discrete so the arrow
    # snaps to a new position each frame.
    def grid_center(i, j):
        return (grid_x + j * cw + cw / 2, grid_y + i * ch + ch)   # bottom of cell

    def buf_center(k):
        return (buf_x + k * bcw + bcw / 2, buf_y)                  # top of cell

    x1_vals, y1_vals, x2_vals, y2_vals = [], [], [], []
    for (i, j) in FRAMES:
        k = i * STRIDE0 + j * STRIDE1
        gx, gy = grid_center(i, j)
        bx, by = buf_center(k)
        x1_vals.append(f"{gx:.1f}")
        y1_vals.append(f"{gy:.1f}")
        x2_vals.append(f"{bx:.1f}")
        y2_vals.append(f"{by - 4:.1f}")   # stop 4px short so marker doesn't overlap cell
    # close the loop
    x1_vals.append(x1_vals[0]); y1_vals.append(y1_vals[0])
    x2_vals.append(x2_vals[0]); y2_vals.append(y2_vals[0])

    out.append(
        f'<line x1="{x1_vals[0]}" y1="{y1_vals[0]}" x2="{x2_vals[0]}" y2="{y2_vals[0]}" '
        f'stroke="{AMBER}" stroke-width="2.2" marker-end="url(#arr2d)">'
        + animate("x1", ";".join(x1_vals))
        + animate("y1", ";".join(y1_vals))
        + animate("x2", ";".join(x2_vals))
        + animate("y2", ";".join(y2_vals))
        + '</line>'
    )

    # ---- Formula line (animated text content via stacked <text>s with opacity) ----
    # The formula gets a tinted, bordered "answer box" so it stands out when the
    # reader scrolls past, the way a textbook sets off a key equation.
    formula_y = buf_y + bch + 60
    out.append(txt(W / 2, formula_y - 38,
                   "offset  =  row \u00d7 stride[0]  +  col \u00d7 stride[1]",
                   size=12, fill=SLATE))

    box_w, box_h = 540, 52
    box_x = (W - box_w) / 2
    box_y = formula_y - 32
    out.append(rect(box_x, box_y, box_w, box_h,
                    fill=AMBER_L, stroke=AMBER, sw=1.5, rx=10))

    for f, (i, j) in enumerate(FRAMES):
        k = i * STRIDE0 + j * STRIDE1
        s = (f'<tspan fill="{PURPLE}" font-weight="700">{i}</tspan>'
             f'<tspan fill="{INK}"> \u00d7 4  +  </tspan>'
             f'<tspan fill="{TEAL}" font-weight="700">{j}</tspan>'
             f'<tspan fill="{INK}"> \u00d7 1  =  </tspan>'
             f'<tspan fill="{AMBER}" font-weight="800">{k}</tspan>')
        opa_vals = ";".join("1" if g == f else "0" for g in range(N)) + (
            ";1" if f == 0 else ";0"
        )
        out.append(
            f'<text x="{W / 2}" y="{box_y + box_h / 2 + 8}" text-anchor="middle" '
            f'font-size="22" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" '
            f'opacity="{1 if f == 0 else 0}">'
            f'<tspan fill="{INK}">[ </tspan>'
            f'<tspan fill="{PURPLE}" font-weight="700">{i}</tspan>'
            f'<tspan fill="{INK}">, </tspan>'
            f'<tspan fill="{TEAL}" font-weight="700">{j}</tspan>'
            f'<tspan fill="{INK}"> ]   \u2192   {s}</tspan>'
            + animate("opacity", opa_vals)
            + '</text>'
        )

    out.append('</svg>')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    build()
