"""Generate ch-01 figure: static filmstrip of the 2-D trace.

The animated companion (trace-2d.svg) loops automatically and cannot be paused
when embedded via Markdown's ![](...) syntax (no browser UI, no JS allowed in
GitHub-rendered Markdown). This filmstrip is the "pause button": each of the six
[row, col] samples is rendered as its own static thumbnail, so the reader can dwell
on any frame for as long as they want and compare two frames side-by-side.

Layout: 2 rows x 3 cols of tiles. Each tile contains a mini 3x4 grid, a mini 12-cell
flat buffer, and the formula for that specific frame -- the same information the
animation shows in motion, but frozen.
"""

from pathlib import Path

# ---- shared palette (must match trace-2d.py) ----
PAPER    = "#faf7f0"
INK      = "#1d1d1b"
SLATE    = "#64748b"
TEAL     = "#0f766e"
TEAL_L   = "#d8f0eb"
AMBER    = "#b45309"
AMBER_L  = "#fef3c7"
PURPLE   = "#7c3aed"

# Same FRAMES as trace-2d.py -- this is intentional: the strip lists exactly what
# the animation cycles through.
FRAMES = [(0, 0), (0, 3), (1, 0), (1, 2), (2, 1), (2, 3)]
ROWS, COLS = 3, 4
STRIDE0, STRIDE1 = 4, 1

# Tile geometry.
TILE_W = 296
TILE_H = 232                          # extra room for the boxed formula at the bottom
GAP_X  = 18
GAP_Y  = 30
COLS_OF_TILES = 3
ROWS_OF_TILES = 2

W = COLS_OF_TILES * TILE_W + (COLS_OF_TILES + 1) * GAP_X
H = 70 + ROWS_OF_TILES * TILE_H + (ROWS_OF_TILES - 1) * GAP_Y + 20
OUT = Path(__file__).resolve().parents[2] / "docs/assets/ch-01/trace-2d-frames.svg"


def rect(x, y, w, h, fill=PAPER, stroke=INK, sw=1.0, rx=3):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def txt(x, y, s, anchor="middle", size=12, fill=INK, weight="normal",
        family="ui-sans-serif, -apple-system, Helvetica, Arial"):
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
            f'font-family="{family}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}">{s}</text>')


def tile(out, f_idx, target, x0, y0):
    """Render one filmstrip tile at the top-left corner (x0, y0)."""
    i, j = target
    k = i * STRIDE0 + j * STRIDE1

    # Tile background (subtle so adjacent tiles read as separate units).
    out.append(rect(x0, y0, TILE_W, TILE_H, fill=PAPER, stroke="#e6e1d2",
                    sw=1.0, rx=8))

    # Tile heading
    out.append(txt(x0 + 12, y0 + 18,
                   f"Frame {f_idx + 1}",
                   anchor="start", size=11, fill=SLATE, weight="700"))
    out.append(txt(x0 + TILE_W - 12, y0 + 18,
                   f"[{i}, {j}]  \u2192  offset {k}",
                   anchor="end", size=11, fill=AMBER, weight="700",
                   family="ui-monospace, SFMono-Regular, Menlo, monospace"))

    # ---- mini grid ----
    cw, ch = 30, 22
    g_w = COLS * cw
    g_h = ROWS * ch
    g_x = x0 + (TILE_W - g_w) / 2
    g_y = y0 + 32
    for ii in range(ROWS):
        for jj in range(COLS):
            offset = ii * STRIDE0 + jj * STRIDE1
            is_target = (ii, jj) == (i, j)
            out.append(rect(g_x + jj * cw, g_y + ii * ch, cw, ch,
                            fill=AMBER_L if is_target else PAPER,
                            stroke=AMBER if is_target else INK,
                            sw=1.8 if is_target else 1.0))
            out.append(txt(g_x + jj * cw + cw / 2, g_y + ii * ch + ch / 2 + 4,
                           str(offset), size=10,
                           fill=AMBER if is_target else INK,
                           weight="700" if is_target else "500"))

    # ---- arrow connector ----
    arr_x_top = g_x + j * cw + cw / 2
    arr_y_top = g_y + g_h + 1
    bcw, bch = 22, 22
    b_w = ROWS * COLS * bcw
    b_x = x0 + (TILE_W - b_w) / 2
    b_y = g_y + g_h + 22
    arr_x_bot = b_x + k * bcw + bcw / 2
    arr_y_bot = b_y - 3
    out.append(f'<line x1="{arr_x_top}" y1="{arr_y_top}" '
               f'x2="{arr_x_bot}" y2="{arr_y_bot}" '
               f'stroke="{AMBER}" stroke-width="1.6" '
               f'marker-end="url(#arrSm)"/>')

    # ---- mini buffer ----
    for kk in range(ROWS * COLS):
        is_target = kk == k
        out.append(rect(b_x + kk * bcw, b_y, bcw, bch,
                        fill=AMBER_L if is_target else PAPER,
                        stroke=AMBER if is_target else INK,
                        sw=1.8 if is_target else 1.0, rx=2))
        out.append(txt(b_x + kk * bcw + bcw / 2, b_y + bch / 2 + 3,
                       str(kk), size=9,
                       fill=AMBER if is_target else INK,
                       weight="700" if is_target else "500"))

    # ---- formula (boxed, textbook-style "key result" callout) ----
    #
    # A tinted background + amber border makes the equation easy to spot when
    # the reader scrolls past the page. Larger font and generous padding fight
    # the "vertically squashed" feeling of the older inline formula.
    box_w = TILE_W - 28
    box_h = 40
    box_x = x0 + (TILE_W - box_w) / 2
    box_y = b_y + bch + 12
    out.append(rect(box_x, box_y, box_w, box_h,
                    fill=AMBER_L, stroke=AMBER, sw=1.4, rx=8))
    out.append(
        f'<text x="{x0 + TILE_W / 2}" y="{box_y + box_h / 2 + 6}" '
        f'text-anchor="middle" '
        f'font-size="17" font-family="ui-monospace, SFMono-Regular, Menlo, monospace">'
        f'<tspan fill="{PURPLE}" font-weight="700">{i}</tspan>'
        f'<tspan fill="{INK}">\u00d74 + </tspan>'
        f'<tspan fill="{TEAL}" font-weight="700">{j}</tspan>'
        f'<tspan fill="{INK}">\u00d71 = </tspan>'
        f'<tspan fill="{AMBER}" font-weight="800">{k}</tspan>'
        f'</text>'
    )


def build():
    out = []
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
               f'width="{W}" height="{H}" '
               f'font-family="ui-sans-serif, -apple-system, Helvetica, Arial">')
    out.append('<defs>'
               '<marker id="arrSm" viewBox="0 0 10 10" refX="9" refY="5" '
               'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
               '<path d="M0,0 L10,5 L0,10 z" fill="' + AMBER + '"/></marker>'
               '</defs>')
    out.append(f'<rect width="{W}" height="{H}" fill="{PAPER}"/>')

    # Title
    out.append(txt(W / 2, 28,
                   "Filmstrip  \u2014  same six samples as the animation, frozen",
                   size=15, weight="700"))
    out.append(txt(W / 2, 48,
                   "study any tile for as long as you want; compare two side-by-side",
                   size=11, fill=SLATE))

    # Render tiles in row-major order.
    for idx, (i, j) in enumerate(FRAMES):
        row = idx // COLS_OF_TILES
        col = idx % COLS_OF_TILES
        x0 = GAP_X + col * (TILE_W + GAP_X)
        y0 = 70 + row * (TILE_H + GAP_Y)
        tile(out, idx, (i, j), x0, y0)

    out.append('</svg>')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    build()
