"""
Chapter: 01 — Tensor Type System
Figure:  flat-memory-layout.svg
Concept: A 2×3 logical grid and the one flat buffer that stores the same data.
Output:  docs/assets/ch-01/flat-memory-layout.svg

Teaches: Computers store memory in one straight line. Shape is just a label
         that tells us how to interpret that line as rows and columns.

Run:
    python3 scripts/media/ch-01-flat-memory.py
"""

import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "assets", "ch-01")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "flat-memory-layout.svg")

# ── palette ───────────────────────────────────────────────────────────────────
INK     = "#1d1d1b"
PAPER   = "#faf7f0"
TEAL    = "#0f766e"
TEAL_S  = "#d8f0eb"
AMBER   = "#b45309"
AMBER_S = "#fef3c7"
SLATE   = "#64748b"
GRID    = "#e2dfd8"
WHITE   = "#ffffff"
FONT    = "ui-monospace, SFMono-Regular, Menlo, monospace"
LABEL   = "Georgia, 'Iowan Old Style', serif"

# ── data ─────────────────────────────────────────────────────────────────────
DATA   = [10, 20, 30, 40, 50, 60]
ROWS, COLS = 2, 3

# ── canvas ────────────────────────────────────────────────────────────────────
W, H = 920, 380

def r(x, y, w, h, fill=WHITE, stroke=GRID, sw=1.3, rx=8, dash=None):
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{extra}/>')

def t(x, y, s, anchor="middle", size=11, fill=INK, weight="normal",
      family=FONT, rotate=None):
    extra = f' transform="rotate({rotate[0]}, {rotate[1]}, {rotate[2]})"' if rotate else ""
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
            f'font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}"{extra}>{s}</text>')

parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" '
    f'width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
    f'<defs>'
    f'<marker id="arr_teal"  markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">'
    f'<path d="M0,0 L0,6 L7,3 z" fill="{TEAL}"/></marker>'
    f'<marker id="arr_amber" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">'
    f'<path d="M0,0 L0,6 L7,3 z" fill="{AMBER}"/></marker>'
    f'</defs>',
    r(0, 0, W, H, fill=PAPER, stroke=PAPER, sw=0, rx=0),
]

# ── title ────────────────────────────────────────────────────────────────────
parts.append(t(W/2, 32, "Same data — two ways to see it",
               size=16, weight="bold", fill=INK, family=LABEL))
parts.append(t(W/2, 54,
               "The Float64Array never changes. Only the shape label changes.",
               size=11, fill=SLATE, family=LABEL))

# ─────────────────────────────────────────────────────
# LEFT: logical 2×3 grid
# ─────────────────────────────────────────────────────
CW, CH, GP = 70, 56, 6
GRID_W = COLS * CW + (COLS - 1) * GP
GRID_H = ROWS * CH + (ROWS - 1) * GP

GX = 90
GY = 140

# Column header (above grid, well clear of arrows)
parts.append(t(GX + GRID_W / 2, GY - 60,
               "Logical view",
               size=13, fill=TEAL, weight="bold", family=LABEL))
parts.append(t(GX + GRID_W / 2, GY - 44,
               "shape [2, 3]",
               size=11, fill=TEAL, family=FONT))

# col axis arrow (above grid, separate from heading)
col_y = GY - 22
parts.append(
    f'<line x1="{GX}" y1="{col_y}" x2="{GX + GRID_W}" y2="{col_y}" '
    f'stroke="{TEAL}" stroke-width="1.4" marker-end="url(#arr_teal)"/>')
parts.append(t(GX + GRID_W + 8, col_y + 4, "col", anchor="start",
               size=10, fill=TEAL, family=LABEL))

# row axis arrow (left of grid)
row_x = GX - 22
parts.append(
    f'<line x1="{row_x}" y1="{GY}" x2="{row_x}" y2="{GY + GRID_H}" '
    f'stroke="{TEAL}" stroke-width="1.4" marker-end="url(#arr_teal)"/>')
parts.append(t(row_x - 6, GY + GRID_H / 2, "row",
               size=10, fill=TEAL, family=LABEL,
               rotate=(-90, row_x - 6, GY + GRID_H / 2)))

# cells
for row in range(ROWS):
    for col in range(COLS):
        idx = row * COLS + col
        cx = GX + col * (CW + GP)
        cy = GY + row * (CH + GP)
        highlight = (row == 1 and col == 2)
        fill = AMBER_S if highlight else TEAL_S
        stroke = AMBER if highlight else TEAL
        parts.append(r(cx, cy, CW, CH, fill=fill, stroke=stroke, sw=1.4, rx=10))
        parts.append(t(cx + CW/2, cy + CH/2 - 2, str(DATA[idx]),
                       size=17, fill=stroke, weight="bold"))
        parts.append(t(cx + CW/2, cy + CH/2 + 18,
                       f"[{row},{col}]", size=9.5, fill=SLATE))

# size annotation under grid
parts.append(t(GX + GRID_W / 2, GY + GRID_H + 26,
               f"{ROWS} rows × {COLS} cols = {ROWS * COLS} elements",
               size=10.5, fill=SLATE, family=LABEL))

# ─────────────────────────────────────────────────────
# CENTER: curved arrow with "row-major" label
# ─────────────────────────────────────────────────────
MX_left  = GX + GRID_W + 16
MX_right = MX_left + 70
MY = GY + GRID_H / 2

parts.append(t((MX_left + MX_right) / 2, MY - 38,
               "row-major", size=12, weight="bold",
               fill=AMBER, family=LABEL))
parts.append(t((MX_left + MX_right) / 2, MY - 22,
               "order", size=10, fill=AMBER, family=LABEL))
parts.append(
    f'<path d="M{MX_left},{MY} '
    f'C{MX_left + 25},{MY - 8} '
    f'{MX_right - 25},{MY + 8} '
    f'{MX_right},{MY}" '
    f'stroke="{AMBER}" stroke-width="2" fill="none" '
    f'marker-end="url(#arr_amber)"/>')

# ─────────────────────────────────────────────────────
# RIGHT: flat buffer
# ─────────────────────────────────────────────────────
BX = MX_right + 20
BW, BH, BGP = 58, 50, 5
BUF_W = len(DATA) * BW + (len(DATA) - 1) * BGP

BY = GY + 4

# Header (above buffer)
parts.append(t(BX + BUF_W / 2, GY - 60,
               "Physical memory",
               size=13, fill=AMBER, weight="bold", family=LABEL))
parts.append(t(BX + BUF_W / 2, GY - 44,
               "Float64Array(6)",
               size=11, fill=AMBER, family=FONT))

# offset axis above the buffer
off_y = GY - 22
parts.append(
    f'<line x1="{BX}" y1="{off_y}" x2="{BX + BUF_W}" y2="{off_y}" '
    f'stroke="{AMBER}" stroke-width="1.4" marker-end="url(#arr_amber)"/>')
parts.append(t(BX + BUF_W + 8, off_y + 4, "offset", anchor="start",
               size=10, fill=AMBER, family=LABEL))

# buffer cells
for i, v in enumerate(DATA):
    bx = BX + i * (BW + BGP)
    parts.append(r(bx, BY, BW, BH, fill=AMBER_S, stroke=AMBER, sw=1.4, rx=10))
    parts.append(t(bx + BW/2, BY + BH/2 - 2, str(v),
                   size=17, fill=AMBER, weight="bold"))
    parts.append(t(bx + BW/2, BY + BH/2 + 18, f"[{i}]",
                   size=9.5, fill=SLATE))

# bold ring around the highlighted offset (index 5)
i_hi = 5
bx_hi = BX + i_hi * (BW + BGP)
parts.append(r(bx_hi - 4, BY - 4, BW + 8, BH + 8, fill="none",
               stroke=INK, sw=2.2, rx=12, dash="4 3"))

# ─────────────────────────────────────────────────────
# BOTTOM: worked example bar
# ─────────────────────────────────────────────────────
EX0 = 60
EY0 = H - 90
EW  = W - 120
EH  = 60

parts.append(r(EX0, EY0, EW, EH, fill="rgba(180,83,9,0.06)",
               stroke=AMBER, sw=1.3, rx=14, dash="6 4"))
parts.append(t(EX0 + 22, EY0 + 24,
               "Worked example", anchor="start",
               size=11, weight="bold", fill=AMBER, family=LABEL))
parts.append(t(EX0 + EW / 2, EY0 + 24,
               "data[ row × cols + col ]",
               size=13, fill=AMBER, family=FONT, weight="bold"))
parts.append(t(EX0 + EW / 2, EY0 + 46,
               "for [row=1, col=2]:   1 × 3 + 2 = 5   →   data[5] = 60",
               size=12, fill=INK, family=FONT))

parts.append("</svg>")

with open(OUT_FILE, "w") as f:
    f.write("\n".join(parts))

print(f"wrote {OUT_FILE}")
