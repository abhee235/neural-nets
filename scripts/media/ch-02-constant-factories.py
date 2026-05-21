"""
Chapter: 02 — Tensor Creation
Figure:  constant-factories.svg
Concept: Three identically-shaped 2x3 grids filled with 0, 1, and 7 to show
         that constant factories share one filling rule (every cell = c) and
         differ only in how c is supplied. Plus a flat row-major strip below
         each grid making clear that one constant fills every memory slot.
Output:  docs/assets/ch-02/constant-factories.svg

Run:
    python3 scripts/media/ch-02-constant-factories.py
"""

import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "assets", "ch-02")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "constant-factories.svg")

INK     = "#1d1d1b"
PAPER   = "#faf7f0"
SLATE   = "#64748b"
GRID_S  = "#e2dfd8"
TEAL    = "#0f766e"
PURPLE  = "#7c3aed"
AMBER   = "#b45309"
AMBER_L = "#fef3c7"
WHITE   = "#ffffff"
FONT    = "ui-monospace, SFMono-Regular, Menlo, monospace"
SERIF   = "Georgia, 'Iowan Old Style', serif"

W, H = 880, 460
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
       f'font-family="{SERIF}">',
       f'<rect width="{W}" height="{H}" fill="{PAPER}"/>']

# ── title ───────────────────────────────────────────────────────────────────
svg.append(f'<text x="{W/2}" y="42" text-anchor="middle" font-size="20" '
           f'font-weight="bold" fill="{INK}">constant factories  ·  same shape, different c</text>')
svg.append(f'<text x="{W/2}" y="62" text-anchor="middle" font-size="12" '
           f'fill="{SLATE}">T[i, j] = c for every (i, j)  ·  shape [2, 3]  ·  size 6</text>')

# ── three panels ────────────────────────────────────────────────────────────
ROWS, COLS = 2, 3
CELL = 56
GAP = 4
PANEL_W = COLS * CELL + (COLS - 1) * GAP
PANEL_H = ROWS * CELL + (ROWS - 1) * GAP
GRID_Y = 130

# evenly distribute three panels across width
panels = [
    {"label": "zeros([2, 3])",   "sub": "c = 0   (built-in)",     "value": "0", "x": 70},
    {"label": "ones([2, 3])",    "sub": "c = 1   (built-in)",     "value": "1", "x": 70 + (PANEL_W + 90)},
    {"label": "fill([2, 3], 7)", "sub": "c = 7   (caller picks)", "value": "7", "x": 70 + 2*(PANEL_W + 90)},
]

for p in panels:
    px = p["x"]
    # panel title
    svg.append(f'<text x="{px + PANEL_W/2}" y="{GRID_Y - 28}" text-anchor="middle" '
               f'font-size="14" font-weight="bold" font-family="{FONT}" fill="{INK}">{p["label"]}</text>')
    svg.append(f'<text x="{px + PANEL_W/2}" y="{GRID_Y - 10}" text-anchor="middle" '
               f'font-size="11" font-family="{FONT}" fill="{AMBER}">{p["sub"]}</text>')

    # cells — every cell highlighted because every cell holds the constant
    for i in range(ROWS):
        for j in range(COLS):
            x = px + j * (CELL + GAP)
            y = GRID_Y + i * (CELL + GAP)
            svg.append(f'<rect x="{x}" y="{y}" width="{CELL}" height="{CELL}" '
                       f'rx="6" fill="{AMBER_L}" stroke="{AMBER}" stroke-width="1.5"/>')
            svg.append(f'<text x="{x + CELL/2}" y="{y + CELL/2 + 8}" '
                       f'text-anchor="middle" font-size="22" font-family="{FONT}" '
                       f'fill="{AMBER}" font-weight="bold">{p["value"]}</text>')

    # flat row-major strip below
    FLAT_Y = GRID_Y + PANEL_H + 24
    SW = 28
    svg.append(f'<text x="{px}" y="{FLAT_Y - 6}" font-size="10.5" '
               f'font-family="{FONT}" fill="{SLATE}">flat buffer (size 6)</text>')
    for k in range(ROWS * COLS):
        cx = px + k * (SW + 2)
        svg.append(f'<rect x="{cx}" y="{FLAT_Y}" width="{SW}" height="{SW}" rx="4" '
                   f'fill="{AMBER_L}" stroke="{AMBER}" stroke-width="1.2"/>')
        svg.append(f'<text x="{cx + SW/2}" y="{FLAT_Y + SW/2 + 5}" text-anchor="middle" '
                   f'font-size="12" font-family="{FONT}" fill="{AMBER}" '
                   f'font-weight="bold">{p["value"]}</text>')
        svg.append(f'<text x="{cx + SW/2}" y="{FLAT_Y + SW + 12}" text-anchor="middle" '
                   f'font-size="9" font-family="{FONT}" fill="{SLATE}">{k}</text>')

# ── answer box: one rule that covers all three ──────────────────────────────
BOX_X = 90
BOX_Y = H - 60
BOX_W = W - 180
svg.append(f'<rect x="{BOX_X}" y="{BOX_Y}" width="{BOX_W}" height="42" '
           f'rx="8" fill="{AMBER_L}" stroke="{AMBER}" stroke-width="1.5"/>')
svg.append(f'<text x="{BOX_X + BOX_W/2}" y="{BOX_Y + 27}" '
           f'text-anchor="middle" font-size="17" font-family="{FONT}" '
           f'fill="{AMBER}">'
           f'for k in 0..size-1:  data[k] = c   ←  one write per slot, same c every time</text>')

svg.append('</svg>')
with open(OUT_FILE, "w") as f:
    f.write("\n".join(svg))
print(f"wrote {OUT_FILE}")
