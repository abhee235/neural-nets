"""
Chapter: 01 — Tensor Type System
Figure:  tensor-ranks.svg
Concept: Scalars, vectors, matrices, and 3-D tensors as visual shapes.
Output:  docs/assets/ch-01/tensor-ranks.svg

Teaches: A tensor is one flat buffer plus a shape. The four panels show how
         the same idea (a box of numbers) scales from rank 0 to rank 3.

Run:
    python3 scripts/media/ch-01-tensor-ranks.py
"""

import os

OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "assets", "ch-01")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "tensor-ranks.svg")

# ── palette ────────────────────────────────────────────────────────────────────
INK     = "#1d1d1b"
PAPER   = "#faf7f0"
TEAL    = "#0f766e"
TEAL_S  = "#d8f0eb"
TEAL_M  = "#a7dcd2"
AMBER   = "#b45309"
AMBER_S = "#fef3c7"
SLATE   = "#64748b"
GRID_S  = "#e2dfd8"
WHITE   = "#ffffff"
FONT    = "ui-monospace, SFMono-Regular, Menlo, monospace"
LABEL_F = "Georgia, 'Iowan Old Style', serif"

# ── helpers ────────────────────────────────────────────────────────────────────
def rect(x, y, w, h, fill=WHITE, stroke=GRID_S, sw=1.2, rx=6, opacity=1):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" '
            f'opacity="{opacity}"/>')

def txt(x, y, s, anchor="middle", size=11, fill=INK, weight="normal",
        family=FONT, rotate=None):
    extra = f' transform="rotate({rotate[0]}, {rotate[1]}, {rotate[2]})"' if rotate else ""
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
            f'font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}"{extra}>{s}</text>')

def arrow(x1, y1, x2, y2, color=SLATE, sw=1.2):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{color}" stroke-width="{sw}" marker-end="url(#arr)"/>')

# ── panel frame ───────────────────────────────────────────────────────────────
def panel_frame(px, py, pw, ph, title, shape_text):
    out = [
        rect(px, py, pw, ph, fill=WHITE, stroke=GRID_S, sw=1.2, rx=14),
        txt(px + pw / 2, py + 26, title, size=14, weight="bold",
            fill=INK, family=LABEL_F),
        rect(px + pw/2 - 50, py + 36, 100, 20, fill=TEAL_S, stroke=TEAL,
             sw=0.8, rx=10),
        txt(px + pw / 2, py + 50, shape_text, size=10.5, fill=TEAL, family=FONT),
    ]
    return out

def panel_footer(px, py, pw, ph, text):
    return [txt(px + pw/2, py + ph - 16, text, size=10, fill=SLATE,
                family=LABEL_F)]

# ── panel: scalar ─────────────────────────────────────────────────────────────
def scalar_panel(px, py):
    pw, ph, cw, ch = 150, 260, 56, 50
    out = panel_frame(px, py, pw, ph, "Scalar", "shape: []")
    cx = px + pw/2 - cw/2
    cy = py + 120
    out.append(rect(cx, cy, cw, ch, fill=AMBER_S, stroke=AMBER, sw=1.4, rx=10))
    out.append(txt(cx + cw/2, cy + ch/2 + 6, "4.0",
                   size=18, weight="bold", fill=AMBER))
    out.append(txt(px + pw/2, cy + ch + 24, "(no axes)",
                   size=10, fill=SLATE, family=LABEL_F))
    out += panel_footer(px, py, pw, ph, "rank 0  ·  1 value")
    return "\n".join(out)

# ── panel: vector ─────────────────────────────────────────────────────────────
def vector_panel(px, py):
    pw, ph = 230, 260
    n  = 4
    cw, ch, gap = 40, 38, 5
    total = n * cw + (n - 1) * gap
    ox = px + (pw - total) / 2
    oy = py + 130

    out = panel_frame(px, py, pw, ph, "Vector", "shape: [4]")

    # axis arrow ABOVE the row of cells
    ay = oy - 18
    out.append(arrow(ox, ay, ox + total, ay, color=SLATE, sw=1.3))
    out.append(txt(ox + total + 6, ay + 4, "axis 0", anchor="start",
                   size=10, fill=SLATE, family=LABEL_F))

    # cells
    for i in range(n):
        cx = ox + i * (cw + gap)
        out.append(rect(cx, oy, cw, ch, fill=TEAL_S, stroke=TEAL, sw=1.3, rx=6))
        out.append(txt(cx + cw/2, oy + ch/2 + 5, f"{i+1}.0",
                       size=13, weight="bold", fill=TEAL))
        out.append(txt(cx + cw/2, oy + ch + 18, f"[{i}]",
                       size=9.5, fill=SLATE))

    out += panel_footer(px, py, pw, ph, "rank 1  ·  4 values")
    return "\n".join(out)

# ── panel: matrix ─────────────────────────────────────────────────────────────
def matrix_panel(px, py):
    pw, ph = 260, 260
    rows, cols = 3, 4
    cw, ch, gap = 40, 32, 4
    total_w = cols * cw + (cols - 1) * gap
    total_h = rows * ch + (rows - 1) * gap
    ox = px + (pw - total_w) / 2 + 8     # nudge right to make room for left axis
    oy = py + 110

    out = panel_frame(px, py, pw, ph, "Matrix", "shape: [3, 4]")

    # axis 1 — horizontal arrow ABOVE the grid
    ay = oy - 18
    out.append(arrow(ox, ay, ox + total_w, ay, color=SLATE, sw=1.3))
    out.append(txt(ox + total_w + 6, ay + 4, "axis 1 (cols)", anchor="start",
                   size=10, fill=SLATE, family=LABEL_F))

    # axis 0 — vertical arrow LEFT of grid
    axx = ox - 18
    out.append(arrow(axx, oy, axx, oy + total_h, color=SLATE, sw=1.3))
    out.append(txt(axx - 6, oy + total_h / 2, "axis 0 (rows)",
                   anchor="middle", size=10, fill=SLATE, family=LABEL_F,
                   rotate=(-90, axx - 6, oy + total_h / 2)))

    # cells with running values 1..12
    for r in range(rows):
        for c in range(cols):
            cx = ox + c * (cw + gap)
            cy = oy + r * (ch + gap)
            fill = TEAL_S if (r + c) % 2 == 0 else WHITE
            out.append(rect(cx, cy, cw, ch, fill=fill, stroke=TEAL, sw=1.3, rx=5))
            out.append(txt(cx + cw/2, cy + ch/2 + 4,
                           str(r * cols + c + 1),
                           size=12, weight="bold", fill=TEAL))

    out += panel_footer(px, py, pw, ph, "rank 2  ·  12 values")
    return "\n".join(out)

# ── panel: 3-D tensor (simple isometric stack of two clean slabs) ────────────
def tensor3d_panel(px, py):
    pw, ph = 260, 260
    out = panel_frame(px, py, pw, ph, "3-D Tensor", "shape: [2, 3, 4]")

    cw, ch, gap = 22, 18, 2
    cols, rows = 4, 3
    slab_w = cols * cw + (cols - 1) * gap
    slab_h = rows * ch + (rows - 1) * gap

    # iso offset per depth (back slab is up-right of front slab)
    dxs, dys = 22, -18

    # centre the whole construction in the panel
    center_x = px + pw / 2
    center_y = py + 150

    base_x = center_x - slab_w / 2 - dxs / 2
    base_y = center_y - slab_h / 2 - dys / 2

    # draw back slab first (depth=1), then front slab (depth=0)
    for d in (1, 0):
        ox = base_x + d * dxs
        oy = base_y + d * dys
        # diagonal connector lines between back and front corners
        # (only draw before front slab so they sit behind it)
        if d == 1:
            # back corners
            corners_back = [
                (ox, oy),
                (ox + slab_w, oy),
                (ox, oy + slab_h),
                (ox + slab_w, oy + slab_h),
            ]
            for (bx, by) in corners_back:
                fx = bx - dxs
                fy = by - dys
                out.append(
                    f'<line x1="{bx}" y1="{by}" x2="{fx}" y2="{fy}" '
                    f'stroke="{TEAL_M}" stroke-width="1" '
                    f'stroke-dasharray="3 3" opacity="0.6"/>')

        for r in range(rows):
            for c in range(cols):
                cx = ox + c * (cw + gap)
                cy = oy + r * (ch + gap)
                # front slab is solid teal, back slab is dimmer
                fill = TEAL_S if d == 0 else TEAL_M
                op   = 1.0 if d == 0 else 0.55
                out.append(rect(cx, cy, cw, ch, fill=fill, stroke=TEAL,
                                sw=1.1, rx=3, opacity=op))

    # axis labels — placed OUTSIDE so they never cross cells
    front_x = base_x
    front_y = base_y

    # axis 0 (depth) — diagonal arrow OUTSIDE bottom-left
    a0_x1 = front_x - 24
    a0_y1 = front_y + slab_h + 18
    a0_x2 = a0_x1 + dxs
    a0_y2 = a0_y1 + dys
    out.append(arrow(a0_x1, a0_y1, a0_x2, a0_y2, color=SLATE, sw=1.3))
    out.append(txt(a0_x1 - 10, a0_y1 + 4, "axis 0", anchor="end",
                   size=9.5, fill=SLATE, family=LABEL_F))

    # axis 1 (rows) — vertical arrow LEFT of front slab
    a1_x = front_x - 12
    out.append(arrow(a1_x, front_y, a1_x, front_y + slab_h,
                     color=SLATE, sw=1.3))
    out.append(txt(a1_x - 4, front_y + slab_h / 2, "axis 1",
                   anchor="middle", size=9.5, fill=SLATE, family=LABEL_F,
                   rotate=(-90, a1_x - 4, front_y + slab_h / 2)))

    # axis 2 (cols) — horizontal arrow ABOVE front slab (under the back slab visually)
    a2_y = front_y - 12
    out.append(arrow(front_x, a2_y, front_x + slab_w, a2_y,
                     color=SLATE, sw=1.3))
    out.append(txt(front_x + slab_w + 6, a2_y + 4, "axis 2",
                   anchor="start", size=9.5, fill=SLATE, family=LABEL_F))

    out += panel_footer(px, py, pw, ph, "rank 3  ·  24 values")
    return "\n".join(out)

# ── assemble SVG ──────────────────────────────────────────────────────────────
PAD = 30
GAP = 22
WIDTHS = [150, 230, 260, 260]
PANEL_H = 260

TOTAL_W = PAD + sum(WIDTHS) + GAP * (len(WIDTHS) - 1) + PAD
TOTAL_H = PAD + 60 + PANEL_H + PAD

panel_y = PAD + 60

# x offsets per panel
xs = []
cur = PAD
for w in WIDTHS:
    xs.append(cur)
    cur += w + GAP

defs = f'''<defs>
  <marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
    <path d="M0,0 L0,6 L7,3 z" fill="{SLATE}"/>
  </marker>
</defs>'''

title = txt(TOTAL_W/2, PAD + 18, "Tensor Ranks — from scalar to 3-D",
            size=16, weight="bold", fill=INK, family=LABEL_F)
subtitle = txt(TOTAL_W/2, PAD + 40,
               "Same idea at every rank: a flat buffer + a shape label",
               size=11, fill=SLATE, family=LABEL_F)

svg = [
    f'<svg xmlns="http://www.w3.org/2000/svg" '
    f'width="{TOTAL_W}" height="{TOTAL_H}" viewBox="0 0 {TOTAL_W} {TOTAL_H}">',
    rect(0, 0, TOTAL_W, TOTAL_H, fill=PAPER, stroke=PAPER, sw=0, rx=0),
    defs,
    title,
    subtitle,
    scalar_panel(xs[0], panel_y),
    vector_panel(xs[1], panel_y),
    matrix_panel(xs[2], panel_y),
    tensor3d_panel(xs[3], panel_y),
    "</svg>",
]

with open(OUT_FILE, "w") as f:
    f.write("\n".join(svg))

print(f"wrote {OUT_FILE}")
