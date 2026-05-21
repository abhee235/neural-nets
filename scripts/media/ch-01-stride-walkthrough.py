"""
Chapter: 01 — Tensor Type System
Figure:  stride-walkthrough.svg
Concept: How strides connect logical axes to flat memory offsets.
Output:  docs/assets/ch-01/stride-walkthrough.svg

Layout: LEFT half has the 2×4 logical grid (top) and the flat buffer (bottom);
RIGHT half is a self-contained formula card. Stride annotations live OUTSIDE
the grid so they never overlap cell content.

Run:
    python3 scripts/media/ch-01-stride-walkthrough.py
"""

import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "assets", "ch-01")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "stride-walkthrough.svg")

# ── palette ───────────────────────────────────────────────────────────────────
INK    = "#1d1d1b"
PAPER  = "#faf7f0"
TEAL   = "#0f766e"
TEAL_S = "#d8f0eb"
AMBER  = "#b45309"
AMBER_S= "#fef3c7"
PURPLE = "#7c3aed"
SLATE  = "#64748b"
GRID   = "#e2dfd8"
WHITE  = "#ffffff"
FONT   = "ui-monospace, SFMono-Regular, Menlo, monospace"
LABEL  = "Georgia, 'Iowan Old Style', serif"

# ── data ─────────────────────────────────────────────────────────────────────
ROWS, COLS = 2, 4
DATA = list(range(ROWS * COLS))   # 0..7

# ── canvas ────────────────────────────────────────────────────────────────────
W, H = 980, 520

# ── helpers ───────────────────────────────────────────────────────────────────
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

def arrow_marker(color):
    cid = color.replace("#", "")
    return (f'<marker id="arr_{cid}" markerWidth="8" markerHeight="8" '
            f'refX="6" refY="3" orient="auto">'
            f'<path d="M0,0 L0,6 L7,3 z" fill="{color}"/></marker>')

def line(x1, y1, x2, y2, color=SLATE, sw=1.4):
    cid = color.replace("#", "")
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{color}" stroke-width="{sw}" '
            f'marker-end="url(#arr_{cid})"/>')

# ── start svg ─────────────────────────────────────────────────────────────────
parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" '
    f'width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
    f'<defs>'
    f'{arrow_marker(TEAL)}{arrow_marker(AMBER)}'
    f'{arrow_marker(PURPLE)}{arrow_marker(SLATE)}'
    f'</defs>',
    r(0, 0, W, H, fill=PAPER, stroke=PAPER, sw=0, rx=0),
]

# ── title ────────────────────────────────────────────────────────────────────
parts.append(t(W/2, 32, "Strides — how each axis maps to memory steps",
               size=16, weight="bold", fill=INK, family=LABEL))
parts.append(t(W/2, 54,
               "shape [2, 4]   →   strides [4, 1]",
               size=12, fill=SLATE, family=FONT))

# ── LEFT half — logical grid (top) and flat buffer (bottom) ──────────────────
LEFT_W = 580   # everything to the left of this lives in the left half

# ── logical grid ─────────────────────────────────────────────────────────────
CW, CH, GP = 80, 54, 6
GRID_W = COLS * CW + (COLS - 1) * GP
GRID_H = ROWS * CH + (ROWS - 1) * GP

GX = 90
GY = 130

parts.append(t(GX + GRID_W / 2, GY - 22,
               "Logical view  ·  shape [2, 4]",
               size=12, fill=TEAL, weight="bold", family=LABEL))

# stride[1] = 1  — amber arrow ABOVE the grid between cells [0,0] and [0,1]
ax1_y = GY - 6  # just above first row
sx1_x1 = GX + CW * 0.6
sx1_x2 = GX + CW + GP + CW * 0.4
# Wait — this puts it INSIDE row 0. Move higher.
ax1_y = GY - 38
# Put a clearly above-grid horizontal arrow then a label
parts.append(line(GX + CW * 0.5, ax1_y, GX + CW + GP + CW * 0.5,
                  ax1_y, color=AMBER, sw=1.6))
parts.append(t(GX + CW + GP / 2, ax1_y - 8,
               "stride[1] = 1", size=10, weight="bold",
               fill=AMBER, family=LABEL))

# stride[0] = 4 — purple vertical arrow LEFT of the grid between rows
sx0_x = GX - 32
sx0_y1 = GY + CH * 0.5
sx0_y2 = GY + CH + GP + CH * 0.5
parts.append(line(sx0_x, sx0_y1, sx0_x, sx0_y2, color=PURPLE, sw=1.6))
parts.append(t(sx0_x - 8, (sx0_y1 + sx0_y2) / 2, "stride[0] = 4",
               size=10, weight="bold", fill=PURPLE, family=LABEL,
               rotate=(-90, sx0_x - 8, (sx0_y1 + sx0_y2) / 2)))

# grid cells
for row in range(ROWS):
    for col in range(COLS):
        flat = row * COLS + col
        cx = GX + col * (CW + GP)
        cy = GY + row * (CH + GP)
        parts.append(r(cx, cy, CW, CH, fill=TEAL_S, stroke=TEAL, sw=1.4, rx=10))
        parts.append(t(cx + CW/2, cy + CH/2 - 2, str(flat),
                       size=18, weight="bold", fill=TEAL))
        parts.append(t(cx + CW/2, cy + CH/2 + 18,
                       f"[{row},{col}]", size=9.5, fill=SLATE))

# ── flat buffer (below the grid) ─────────────────────────────────────────────
BX = 90
BY = GY + GRID_H + 110

BW, BH, BGP = 60, 46, 5
BUF_W = len(DATA) * BW + (len(DATA) - 1) * BGP

parts.append(t(BX + BUF_W / 2, BY - 60,
               "Physical memory  ·  Float64Array(8)",
               size=12, fill=AMBER, weight="bold", family=LABEL))

# Purple arc OVER the buffer showing the stride[0]=4 jump from [0] → [4]
# (place arc ABOVE the buffer, between buffer and label)
ax = BX + BW / 2
bx_dst = BX + 4 * (BW + BGP) + BW / 2
arc_top_y = BY - 26
parts.append(
    f'<path d="M{ax},{BY - 2} '
    f'C{ax},{arc_top_y - 6} '
    f'{bx_dst},{arc_top_y - 6} '
    f'{bx_dst},{BY - 2}" '
    f'stroke="{PURPLE}" stroke-width="2" fill="none" '
    f'marker-end="url(#arr_{PURPLE.replace(chr(35), "")})"/>')
parts.append(t((ax + bx_dst) / 2, arc_top_y - 12,
               "+4 offsets to next row  (stride[0])",
               size=10, weight="bold", fill=PURPLE, family=LABEL))

# buffer cells
for i, v in enumerate(DATA):
    bx = BX + i * (BW + BGP)
    row = i // COLS
    fill = AMBER_S if row == 0 else "rgba(254,243,199,0.55)"
    parts.append(r(bx, BY, BW, BH, fill=fill, stroke=AMBER, sw=1.4, rx=10))
    parts.append(t(bx + BW/2, BY + BH/2 - 2, str(v),
                   size=18, weight="bold", fill=AMBER))
    parts.append(t(bx + BW/2, BY + BH/2 + 18, f"[{i}]",
                   size=9.5, fill=SLATE))

# Amber tick BELOW the buffer between [0] and [1] — stride[1]=1
sa_x = BX + BW / 2
sb_x = BX + BW + BGP + BW / 2
tick_y = BY + BH + 18
parts.append(line(sa_x, tick_y, sb_x, tick_y, color=AMBER, sw=1.6))
parts.append(t((sa_x + sb_x) / 2, tick_y + 16,
               "+1 offset to next col  (stride[1])",
               size=10, weight="bold", fill=AMBER, family=LABEL))

# row braces under buffer (below the amber tick area)
brace_y = BY + BH + 50
for row in range(ROWS):
    bx0 = BX + row * COLS * (BW + BGP)
    bx1 = bx0 + COLS * (BW + BGP) - BGP
    parts.append(f'<line x1="{bx0}" y1="{brace_y}" x2="{bx1}" y2="{brace_y}" '
                 f'stroke="{SLATE}" stroke-width="1.1"/>')
    parts.append(f'<line x1="{bx0}" y1="{brace_y}" x2="{bx0}" y2="{brace_y - 5}" '
                 f'stroke="{SLATE}" stroke-width="1.1"/>')
    parts.append(f'<line x1="{bx1}" y1="{brace_y}" x2="{bx1}" y2="{brace_y - 5}" '
                 f'stroke="{SLATE}" stroke-width="1.1"/>')
    parts.append(t((bx0 + bx1) / 2, brace_y + 14,
                   f"row {row}  (offsets {row*COLS}..{(row+1)*COLS - 1})",
                   size=9.5, fill=SLATE, family=LABEL))

# ── RIGHT half — formula card (self-contained, no overlaps) ──────────────────
FX = LEFT_W + 50
FY = 130
FW = W - FX - 40
FH = 340

parts.append(r(FX, FY, FW, FH, fill=WHITE, stroke=TEAL, sw=1.3,
               rx=16, dash="6 4"))
parts.append(t(FX + FW / 2, FY + 30, "General rule",
               size=13, fill=TEAL, weight="bold", family=LABEL))

# clean two-column layout
lines = [
    ("shape",           "[d_0, d_1, ..., d_n]",     INK,    "normal"),
    ("stride[i]",       "d_{i+1} × ... × d_n",     INK,    "normal"),
    ("",                "",                         INK,    "normal"),
    ("for shape [2, 4]",":",                        SLATE,  "normal"),
    ("  stride[0]",     "= 4   ← row stride",       PURPLE, "bold"),
    ("  stride[1]",     "= 1   ← col stride",       AMBER,  "bold"),
    ("",                "",                         INK,    "normal"),
    ("flatIndex([i,j])","= i × stride[0] + j × stride[1]", INK, "normal"),
    ("",                "= i × 4 + j × 1",          INK,    "normal"),
    ("",                "",                         INK,    "normal"),
    ("for [i=1, j=2]",  ":",                        SLATE,  "normal"),
    ("",                "1 × 4 + 2 = 6   →   data[6]", AMBER, "bold"),
]
line_h = 22
left_col_x = FX + 24
right_col_x = FX + 168
for k, (lhs, rhs, color, weight) in enumerate(lines):
    y = FY + 70 + k * line_h
    if lhs:
        parts.append(t(left_col_x, y, lhs, anchor="start", size=11,
                       fill=color, weight=weight, family=FONT))
    if rhs:
        parts.append(t(right_col_x, y, rhs, anchor="start", size=11,
                       fill=color, weight=weight, family=FONT))

# Bottom insight bar (full width)
EX0 = 60
EY0 = H - 50
EW  = W - 120
EH  = 36
parts.append(r(EX0, EY0, EW, EH, fill="rgba(124,58,237,0.05)",
               stroke=PURPLE, sw=1.2, rx=12, dash="6 4"))
parts.append(t(EX0 + EW / 2, EY0 + 22,
               "stride[i]  =  how many flat-buffer cells you skip when you advance one step along axis i",
               size=11, fill=PURPLE, family=LABEL, weight="bold"))

parts.append("</svg>")

with open(OUT_FILE, "w") as f:
    f.write("\n".join(parts))

print(f"wrote {OUT_FILE}")
