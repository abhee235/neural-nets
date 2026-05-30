"""Generate ch-01 figure: static filmstrip of the 3-D trace.

Static companion to trace-3d.svg. Same six [d, r, c] samples, but each frozen as
its own tile so the reader can dwell on / compare specific frames. See
ch-01-trace-2d-frames.py for the rationale ("we cannot add a pause button to an
SMIL-animated SVG loaded via Markdown image syntax").

Layout: 2 rows x 3 cols of tiles. Each tile contains two mini 3x4 slabs, a mini
24-cell flat buffer, and the formula for that specific frame.
"""

from pathlib import Path

# ---- shared palette (must match trace-3d.py) ----
PAPER    = "#faf7f0"
INK      = "#1d1d1b"
SLATE    = "#64748b"
PURPLE   = "#7c3aed"
TEAL     = "#0f766e"
AMBER    = "#b45309"
AMBER_L  = "#fef3c7"

# Same FRAMES as trace-3d.py.
FRAMES = [(0, 0, 0), (0, 1, 2), (0, 2, 3), (1, 0, 0), (1, 1, 2), (1, 2, 3)]
D, R, C = 2, 3, 4
S0, S1, S2 = 12, 4, 1

# Tile geometry.
TILE_W = 380
TILE_H = 264                          # extra room for the boxed formula at the bottom
GAP_X  = 18
GAP_Y  = 30
COLS_OF_TILES = 3
ROWS_OF_TILES = 2

W = COLS_OF_TILES * TILE_W + (COLS_OF_TILES + 1) * GAP_X
H = 70 + ROWS_OF_TILES * TILE_H + (ROWS_OF_TILES - 1) * GAP_Y + 20
OUT = Path(__file__).resolve().parents[2] / "docs/assets/ch-01/trace-3d-frames.svg"


def rect(x, y, w, h, fill=PAPER, stroke=INK, sw=1.0, rx=3):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def txt(x, y, s, anchor="middle", size=12, fill=INK, weight="normal",
        family="ui-sans-serif, -apple-system, Helvetica, Arial"):
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
            f'font-family="{family}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}">{s}</text>')


def tile(out, f_idx, target, x0, y0):
    d, i, j = target
    k = d * S0 + i * S1 + j * S2

    # Tile background.
    out.append(rect(x0, y0, TILE_W, TILE_H, fill=PAPER, stroke="#e6e1d2",
                    sw=1.0, rx=8))

    # Heading
    out.append(txt(x0 + 12, y0 + 18,
                   f"Frame {f_idx + 1}",
                   anchor="start", size=11, fill=SLATE, weight="700"))
    out.append(txt(x0 + TILE_W - 12, y0 + 18,
                   f"[{d}, {i}, {j}]  \u2192  offset {k}",
                   anchor="end", size=11, fill=AMBER, weight="700",
                   family="ui-monospace, SFMono-Regular, Menlo, monospace"))

    # ---- two mini slabs ----
    cw, ch = 22, 18
    slab_w = C * cw
    slab_h = R * ch
    slab_gap = 22
    total = 2 * slab_w + slab_gap
    sx0 = x0 + (TILE_W - total) / 2
    sx1 = sx0 + slab_w + slab_gap
    sy  = y0 + 36

    for dd, sx in enumerate((sx0, sx1)):
        # tiny slab label
        out.append(txt(sx + slab_w / 2, sy + slab_h + 12,
                       f"d={dd}", size=9, fill=PURPLE, weight="700"))
        for ii in range(R):
            for jj in range(C):
                offset = dd * S0 + ii * S1 + jj * S2
                is_target = (dd, ii, jj) == (d, i, j)
                out.append(rect(sx + jj * cw, sy + ii * ch, cw, ch,
                                fill=AMBER_L if is_target else PAPER,
                                stroke=AMBER if is_target else INK,
                                sw=1.6 if is_target else 0.9))
                out.append(txt(sx + jj * cw + cw / 2, sy + ii * ch + ch / 2 + 3,
                               str(offset), size=8,
                               fill=AMBER if is_target else INK,
                               weight="700" if is_target else "500"))

    # ---- mini buffer ----
    bcw, bch = 14, 20
    b_w = D * R * C * bcw
    b_x = x0 + (TILE_W - b_w) / 2
    b_y = sy + slab_h + 30
    for kk in range(D * R * C):
        is_target = kk == k
        out.append(rect(b_x + kk * bcw, b_y, bcw, bch,
                        fill=AMBER_L if is_target else PAPER,
                        stroke=AMBER if is_target else INK,
                        sw=1.6 if is_target else 0.9, rx=2))
        out.append(txt(b_x + kk * bcw + bcw / 2, b_y + bch / 2 + 3,
                       str(kk), size=7,
                       fill=AMBER if is_target else INK,
                       weight="700" if is_target else "500"))

    # ---- arrow from active slab cell down to active buffer cell ----
    active_sx = sx0 if d == 0 else sx1
    arr_x_top = active_sx + j * cw + cw / 2
    arr_y_top = sy + i * ch + ch / 2
    # Start the line from BELOW the slabs so the arrow shows direction clearly.
    arr_y_top_clear = sy + slab_h + 1
    arr_x_bot = b_x + k * bcw + bcw / 2
    arr_y_bot = b_y - 3
    # Two segments so we don't draw through slab cells: vertical short stub +
    # diagonal to buffer.
    out.append(f'<line x1="{arr_x_top}" y1="{arr_y_top_clear}" '
               f'x2="{arr_x_bot}" y2="{arr_y_bot}" '
               f'stroke="{AMBER}" stroke-width="1.4" '
               f'marker-end="url(#arrSm3d)"/>')

    # ---- formula (boxed, textbook-style "key result" callout) ----
    box_w = TILE_W - 28
    box_h = 44
    box_x = x0 + (TILE_W - box_w) / 2
    box_y = b_y + bch + 14
    out.append(rect(box_x, box_y, box_w, box_h,
                    fill=AMBER_L, stroke=AMBER, sw=1.4, rx=8))
    out.append(
        f'<text x="{x0 + TILE_W / 2}" y="{box_y + box_h / 2 + 6}" '
        f'text-anchor="middle" '
        f'font-size="17" font-family="ui-monospace, SFMono-Regular, Menlo, monospace">'
        f'<tspan fill="{PURPLE}" font-weight="700">{d}</tspan>'
        f'<tspan fill="{INK}">\u00d712 + </tspan>'
        f'<tspan fill="{TEAL}" font-weight="700">{i}</tspan>'
        f'<tspan fill="{INK}">\u00d74 + </tspan>'
        f'<tspan fill="{AMBER}" font-weight="700">{j}</tspan>'
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
               '<marker id="arrSm3d" viewBox="0 0 10 10" refX="9" refY="5" '
               'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
               '<path d="M0,0 L10,5 L0,10 z" fill="' + AMBER + '"/></marker>'
               '</defs>')
    out.append(f'<rect width="{W}" height="{H}" fill="{PAPER}"/>')

    # Title
    out.append(txt(W / 2, 28,
                   "Filmstrip  \u2014  same six [d, r, c] samples as the animation, frozen",
                   size=15, weight="700"))
    out.append(txt(W / 2, 48,
                   "study any tile for as long as you want; compare two side-by-side",
                   size=11, fill=SLATE))

    for idx, target in enumerate(FRAMES):
        row = idx // COLS_OF_TILES
        col = idx % COLS_OF_TILES
        x0 = GAP_X + col * (TILE_W + GAP_X)
        y0 = 70 + row * (TILE_H + GAP_Y)
        tile(out, idx, target, x0, y0)

    out.append('</svg>')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    build()
