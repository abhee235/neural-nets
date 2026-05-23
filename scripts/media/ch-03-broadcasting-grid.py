"""
Chapter: 03 — Elementwise Ops & Broadcasting
Figure:  broadcasting-grid.svg
Concept: Right-aligned broadcasting for [3, 1] + [1, 4] -> [3, 4].
Output:  docs/assets/ch-03/broadcasting-grid.svg

Run:
    python3 scripts/media/ch-03-broadcasting-grid.py
"""

import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "assets", "ch-03")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "broadcasting-grid.svg")

INK = "#1d1d1b"
PAPER = "#faf7f0"
SLATE = "#64748b"
GRID_S = "#e2dfd8"
PURPLE = "#7c3aed"
PURPLE_L = "#ede9fe"
TEAL = "#0f766e"
TEAL_L = "#d8f0eb"
AMBER = "#b45309"
AMBER_L = "#fef3c7"
WHITE = "#ffffff"
FONT = "ui-monospace, SFMono-Regular, Menlo, monospace"
SERIF = "Georgia, 'Iowan Old Style', serif"

W, H = 960, 520
svg = [
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="{SERIF}">',
    f'<rect width="{W}" height="{H}" fill="{PAPER}"/>',
]

svg.append(
    f'<text x="{W/2}" y="40" text-anchor="middle" font-size="20" font-weight="bold" fill="{INK}">'
    'broadcasting  ·  right-align shapes, then stretch the size-1 axes'
    '</text>'
)
svg.append(
    f'<text x="{W/2}" y="62" text-anchor="middle" font-size="12" fill="{SLATE}">'
    '[3, 1] + [1, 4]  →  [3, 4]  ·  no new math, just reuse along axes of size 1'
    '</text>'
)

CELL = 52
GAP = 4

def draw_grid(x, y, rows, cols, values, fill, stroke, text_fill, title, subtitle):
    total_w = cols * CELL + (cols - 1) * GAP
    svg.append(
        f'<text x="{x + total_w/2}" y="{y - 26}" text-anchor="middle" font-size="14" '
        f'font-family="{FONT}" font-weight="bold" fill="{INK}">{title}</text>'
    )
    svg.append(
        f'<text x="{x + total_w/2}" y="{y - 8}" text-anchor="middle" font-size="11" '
        f'font-family="{FONT}" fill="{SLATE}">{subtitle}</text>'
    )
    for i in range(rows):
        for j in range(cols):
            cx = x + j * (CELL + GAP)
            cy = y + i * (CELL + GAP)
            v = values[i][j]
            svg.append(
                f'<rect x="{cx}" y="{cy}" width="{CELL}" height="{CELL}" rx="6" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>'
            )
            svg.append(
                f'<text x="{cx + CELL/2}" y="{cy + CELL/2 + 7}" text-anchor="middle" '
                f'font-size="18" font-family="{FONT}" fill="{text_fill}" font-weight="bold">{v}</text>'
            )


# Left operand: column vector [3, 1]
left_vals = [[1], [2], [3]]
draw_grid(90, 140, 3, 1, left_vals, PURPLE_L, PURPLE, PURPLE, 'A = [[1], [2], [3]]', 'shape [3, 1]  ·  one column')

# Right operand: row vector [1, 4]
right_vals = [[10, 20, 30, 40]]
draw_grid(320, 192, 1, 4, right_vals, TEAL_L, TEAL, TEAL, 'B = [[10, 20, 30, 40]]', 'shape [1, 4]  ·  one row')

# Result: outer-style sum [3, 4]
res_vals = [
    [11, 21, 31, 41],
    [12, 22, 32, 42],
    [13, 23, 33, 43],
]
draw_grid(610, 140, 3, 4, res_vals, AMBER_L, AMBER, AMBER, 'A + B', 'shape [3, 4]  ·  each axis takes max(size)')

# Arrows
svg.append('<defs>'
           f'<marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
           f'<path d="M0,0 L10,5 L0,10 Z" fill="{SLATE}"/></marker>'
           '</defs>')
svg.append(f'<path d="M160 310 C 230 310, 250 250, 305 250" fill="none" stroke="{SLATE}" stroke-width="2" marker-end="url(#arr)"/>')
svg.append(f'<path d="M530 250 C 575 250, 585 250, 600 250" fill="none" stroke="{SLATE}" stroke-width="2" marker-end="url(#arr)"/>')

# Rule table
box_x = 80
box_y = 390
box_w = 800
box_h = 96
svg.append(
    f'<rect x="{box_x}" y="{box_y}" width="{box_w}" height="{box_h}" rx="10" '
    f'fill="{WHITE}" stroke="{GRID_S}" stroke-width="1.2"/>'
)
svg.append(
    f'<text x="{box_x + 20}" y="{box_y + 28}" font-size="13" font-family="{FONT}" font-weight="bold" fill="{INK}">'
    'Broadcasting memory rule'
    '</text>'
)
rule_lines = [
    ('1.', 'Line shapes up from the right.'),
    ('2.', 'At each axis, sizes must match or one of them must be 1.'),
    ('3.', 'The result takes the larger size at each axis.'),
]
for idx, (num, text) in enumerate(rule_lines):
    y = box_y + 52 + idx * 18
    svg.append(f'<text x="{box_x + 24}" y="{y}" font-size="12" font-family="{FONT}" fill="{AMBER}">{num}</text>')
    svg.append(f'<text x="{box_x + 52}" y="{y}" font-size="12" font-family="{FONT}" fill="{INK}">{text}</text>')

# Answer box
ans_x = 610
ans_y = 96
ans_w = 250
ans_h = 32
svg.append(
    f'<rect x="{ans_x}" y="{ans_y}" width="{ans_w}" height="{ans_h}" rx="8" '
    f'fill="{AMBER_L}" stroke="{AMBER}" stroke-width="1.5"/>'
)
svg.append(
    f'<text x="{ans_x + ans_w/2}" y="{ans_y + 21}" text-anchor="middle" '
    f'font-size="17" font-family="{FONT}" fill="{AMBER}">'
    'broadcastShapes([3,1], [1,4]) = [3,4]'
    '</text>'
)

svg.append('</svg>')

with open(OUT_FILE, 'w') as f:
    f.write('\n'.join(svg))

print(f'wrote {OUT_FILE}')