"""
Chapter: 02 — Tensor Creation
Figure:  identity-matrix.svg
Concept: A 4x4 identity matrix with the diagonal cells highlighted, plus the
         flat row-major layout below it.
Output:  docs/assets/ch-02/identity-matrix.svg

Run:
    python3 scripts/media/ch-02-identity-matrix.py
"""

import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "assets", "ch-02")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "identity-matrix.svg")

INK     = "#1d1d1b"
PAPER   = "#faf7f0"
SLATE   = "#64748b"
GRID_S  = "#e2dfd8"
TEAL    = "#0f766e"
TEAL_L  = "#d8f0eb"
PURPLE  = "#7c3aed"
PURPLE_L = "#ede9fe"
AMBER   = "#b45309"
AMBER_L = "#fef3c7"
WHITE   = "#ffffff"
FONT    = "ui-monospace, SFMono-Regular, Menlo, monospace"
SERIF   = "Georgia, 'Iowan Old Style', serif"

W, H = 880, 460
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
       f'font-family="{SERIF}">',
       f'<rect width="{W}" height="{H}" fill="{PAPER}"/>']

svg.append(f'<text x="{W/2}" y="42" text-anchor="middle" font-size="20" '
           f'font-weight="bold" fill="{INK}">eye(4)  ·  the 4 × 4 identity matrix</text>')
svg.append(f'<text x="{W/2}" y="62" text-anchor="middle" font-size="12" '
           f'fill="{SLATE}">I[i, j] = 1 if i = j, else 0  ·  shape [4, 4]  ·  size 16</text>')

# ── matrix grid (left side) ─────────────────────────────────────────────────
N = 4
CELL = 56
GAP = 4
GRID_X = 90
GRID_Y = 110
total = N * CELL + (N - 1) * GAP

# axis labels
svg.append(f'<text x="{GRID_X + total/2}" y="{GRID_Y - 22}" '
           f'text-anchor="middle" font-size="12" font-family="{FONT}" '
           f'fill="{TEAL}">column j</text>')
for j in range(N):
    cx = GRID_X + j * (CELL + GAP) + CELL/2
    svg.append(f'<text x="{cx}" y="{GRID_Y - 6}" text-anchor="middle" '
               f'font-size="10.5" font-family="{FONT}" fill="{TEAL}">{j}</text>')

svg.append(f'<text x="{GRID_X - 36}" y="{GRID_Y + total/2}" text-anchor="middle" '
           f'font-size="12" font-family="{FONT}" fill="{PURPLE}" '
           f'transform="rotate(-90 {GRID_X - 36} {GRID_Y + total/2})">row i</text>')
for i in range(N):
    cy = GRID_Y + i * (CELL + GAP) + CELL/2 + 4
    svg.append(f'<text x="{GRID_X - 12}" y="{cy}" text-anchor="end" '
               f'font-size="10.5" font-family="{FONT}" fill="{PURPLE}">{i}</text>')

# cells
for i in range(N):
    for j in range(N):
        x = GRID_X + j * (CELL + GAP)
        y = GRID_Y + i * (CELL + GAP)
        is_diag = (i == j)
        fill = AMBER_L if is_diag else WHITE
        stroke = AMBER if is_diag else GRID_S
        sw = 1.6 if is_diag else 1.0
        svg.append(f'<rect x="{x}" y="{y}" width="{CELL}" height="{CELL}" '
                   f'rx="6" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        val = "1" if is_diag else "0"
        color = AMBER if is_diag else SLATE
        weight = "bold" if is_diag else "normal"
        svg.append(f'<text x="{x + CELL/2}" y="{y + CELL/2 + 8}" '
                   f'text-anchor="middle" font-size="22" font-family="{FONT}" '
                   f'fill="{color}" font-weight="{weight}">{val}</text>')

# diagonal pointer arrow (annotation)
arrow_x0 = GRID_X + total + 14
arrow_y0 = GRID_Y + total + 14
arrow_x1 = GRID_X + (N-1)*(CELL+GAP) + CELL/2 + 18
arrow_y1 = GRID_Y + (N-1)*(CELL+GAP) + CELL/2 + 18
svg.append('<defs>'
           f'<marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" '
           f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
           f'<path d="M0,0 L10,5 L0,10 Z" fill="{AMBER}"/></marker>'
           '</defs>')

# ── flat row-major view (right side) ───────────────────────────────────────
FLAT_X = GRID_X + total + 90
FLAT_Y = GRID_Y + 30
SW = 38       # small cell width
SH = 38

svg.append(f'<text x="{FLAT_X}" y="{FLAT_Y - 30}" font-size="13" '
           f'font-weight="bold" font-family="{FONT}" fill="{INK}">'
           f'flat row-major buffer (Ch 01 layout)</text>')
svg.append(f'<text x="{FLAT_X}" y="{FLAT_Y - 12}" font-size="11" '
           f'font-family="{FONT}" fill="{SLATE}">'
           f'data[i*4 + j]  ·  diagonal lives at offsets 0, 5, 10, 15</text>')

for k in range(16):
    i, j = divmod(k, N)
    is_diag = (i == j)
    cx = FLAT_X + (k % 8) * (SW + 2)
    cy = FLAT_Y + (k // 8) * (SH + 30)
    fill = AMBER_L if is_diag else WHITE
    stroke = AMBER if is_diag else GRID_S
    sw = 1.5 if is_diag else 1.0
    svg.append(f'<rect x="{cx}" y="{cy}" width="{SW}" height="{SH}" rx="5" '
               f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    val = "1" if is_diag else "0"
    color = AMBER if is_diag else SLATE
    weight = "bold" if is_diag else "normal"
    svg.append(f'<text x="{cx + SW/2}" y="{cy + SH/2 + 6}" text-anchor="middle" '
               f'font-size="15" font-family="{FONT}" fill="{color}" '
               f'font-weight="{weight}">{val}</text>')
    # offset label below
    svg.append(f'<text x="{cx + SW/2}" y="{cy + SH + 14}" text-anchor="middle" '
               f'font-size="9" font-family="{FONT}" fill="{SLATE}">{k}</text>')

# ── answer box: the formula ─────────────────────────────────────────────────
BOX_X = 90
BOX_Y = H - 60
BOX_W = W - 180
svg.append(f'<rect x="{BOX_X}" y="{BOX_Y}" width="{BOX_W}" height="42" '
           f'rx="8" fill="{AMBER_L}" stroke="{AMBER}" stroke-width="1.5"/>')
svg.append(f'<text x="{BOX_X + BOX_W/2}" y="{BOX_Y + 27}" '
           f'text-anchor="middle" font-size="17" font-family="{FONT}" '
           f'fill="{AMBER}">'
           f'for i in 0..n-1:  data[i * n + i] = 1   ←  one write per diagonal cell</text>')

svg.append('</svg>')
with open(OUT_FILE, "w") as f:
    f.write("\n".join(svg))
print(f"wrote {OUT_FILE}")
