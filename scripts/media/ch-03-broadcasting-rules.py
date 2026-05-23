"""
Chapter: 03 — Elementwise Ops & Broadcasting
Figure:  broadcasting-rules.svg  (the three outcomes side by side)
Concept: Visual answer to "why are 2 and 4 incompatible if broadcasting can stretch?"

Three panels, left to right:
  1. ALLOWED — size 1 stretches to N (one value reused).
  2. ALLOWED — equal sizes pair 1:1.
  3. REJECTED — 2 vs 4: no single source value to reuse, and 2 distinct slots
     cannot decide which one feeds each of 4 output slots.

Output: docs/assets/ch-03/broadcasting-rules.svg
Run:    python3 scripts/media/ch-03-broadcasting-rules.py
"""

from pathlib import Path

# ── shared palette ────────────────────────────────────────────────────────────
PAPER    = "#faf7f0"
INK      = "#1d1d1b"
SLATE    = "#64748b"
PURPLE   = "#7c3aed"
PURPLE_L = "#ede9fe"
TEAL     = "#0f766e"
TEAL_L   = "#d8f0eb"
AMBER    = "#b45309"
AMBER_L  = "#fef3c7"
GREEN    = "#15803d"
GREEN_L  = "#dcfce7"
RED      = "#b91c1c"
RED_L    = "#fee2e2"
WHITE    = "#ffffff"

FONT_MONO = "ui-monospace, SFMono-Regular, Menlo, monospace"
FONT_SANS = "ui-sans-serif, -apple-system, Helvetica, Arial, sans-serif"


def rect(x, y, w, h, fill, stroke, sw=1.4, rx=6):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def txt(x, y, s, anchor="middle", size=12, fill=INK, family=FONT_SANS, weight="normal"):
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{family}" '
            f'font-size="{size}" font-weight="{weight}" fill="{fill}">{s}</text>')


def cell(cx, cy, value, fill, stroke, color, w=46, h=46, vsize=18, weight="bold"):
    return (rect(cx, cy, w, h, fill, stroke, sw=1.4, rx=6)
            + txt(cx + w / 2, cy + h / 2 + 6, str(value),
                  size=vsize, fill=color, family=FONT_MONO, weight=weight))


def arrow_marker(color, mid):
    return (f'<marker id="arr-{mid}" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{color}"/></marker>')


def badge(cx, cy, w, h, text, color_bg, color_border, color_text):
    return (rect(cx, cy, w, h, color_bg, color_border, sw=1.2, rx=12)
            + txt(cx + w / 2, cy + h / 2 + 5, text,
                  size=12, fill=color_text, family=FONT_MONO, weight="bold"))


# ── panel builders ───────────────────────────────────────────────────────────
TILE_W = 360
TILE_H = 380


def panel_reuse(x0, y0):
    """Panel 1: shape [1] stretches to shape [4]. Allowed."""
    out = []
    # heading
    out.append(txt(x0 + TILE_W / 2, y0 + 28, "1 → 4   (allowed)",
                   size=15, fill=INK, family=FONT_MONO, weight="bold"))
    out.append(txt(x0 + TILE_W / 2, y0 + 48,
                   "one value, reused everywhere",
                   size=11, fill=SLATE))

    # source: single cell with value 7
    src_cx = x0 + TILE_W / 2 - 23
    src_cy = y0 + 75
    out.append(txt(x0 + TILE_W / 2, y0 + 70, "shape [1]",
                   size=10, fill=SLATE, family=FONT_MONO))
    out.append(cell(src_cx, src_cy + 6, 7, PURPLE_L, PURPLE, PURPLE))

    # target: 4 cells with the same value
    tgt_y = y0 + 220
    out.append(txt(x0 + TILE_W / 2, tgt_y - 14, "shape [4]",
                   size=10, fill=SLATE, family=FONT_MONO))
    gap = 6
    cell_w = 46
    grid_w = 4 * cell_w + 3 * gap
    gx = x0 + (TILE_W - grid_w) / 2
    for j in range(4):
        cx = gx + j * (cell_w + gap)
        out.append(cell(cx, tgt_y, 7, PURPLE_L, PURPLE, PURPLE))

    # arrows fanning out from source to each target cell
    sx = src_cx + 23
    sy = src_cy + 6 + 46
    for j in range(4):
        tx = gx + j * (cell_w + gap) + 23
        out.append(f'<path d="M {sx} {sy} L {tx} {tgt_y}" '
                   f'stroke="{PURPLE}" stroke-width="1.2" '
                   f'stroke-dasharray="3 3" fill="none" opacity="0.7" '
                   f'marker-end="url(#arr-purple)"/>')

    # green allowed badge at the bottom
    out.append(badge(x0 + (TILE_W - 220) / 2, y0 + TILE_H - 58, 220, 36,
                     "✓ reuse one slot 4 times", GREEN_L, GREEN, GREEN))
    return out


def panel_pair(x0, y0):
    """Panel 2: shape [4] paired with shape [4]. Allowed."""
    out = []
    out.append(txt(x0 + TILE_W / 2, y0 + 28, "4 ↔ 4   (allowed)",
                   size=15, fill=INK, family=FONT_MONO, weight="bold"))
    out.append(txt(x0 + TILE_W / 2, y0 + 48,
                   "matching sizes, pair 1:1",
                   size=11, fill=SLATE))

    gap = 6
    cell_w = 46
    grid_w = 4 * cell_w + 3 * gap
    gx = x0 + (TILE_W - grid_w) / 2

    # top row (teal)
    top_y = y0 + 82
    out.append(txt(x0 + TILE_W / 2, top_y - 6, "shape [4]",
                   size=10, fill=SLATE, family=FONT_MONO))
    top_vals = [1, 2, 3, 4]
    for j in range(4):
        cx = gx + j * (cell_w + gap)
        out.append(cell(cx, top_y, top_vals[j], TEAL_L, TEAL, TEAL))

    # bottom row (purple)
    bot_y = y0 + 220
    out.append(txt(x0 + TILE_W / 2, bot_y - 6, "shape [4]",
                   size=10, fill=SLATE, family=FONT_MONO))
    bot_vals = [10, 20, 30, 40]
    for j in range(4):
        cx = gx + j * (cell_w + gap)
        out.append(cell(cx, bot_y, bot_vals[j], PURPLE_L, PURPLE, PURPLE))

    # straight 1:1 arrows
    for j in range(4):
        cx = gx + j * (cell_w + gap) + 23
        out.append(f'<path d="M {cx} {top_y + 46 + 2} L {cx} {bot_y - 2}" '
                   f'stroke="{INK}" stroke-width="1.2" '
                   f'fill="none" opacity="0.7" '
                   f'marker-end="url(#arr-ink)"/>')

    out.append(badge(x0 + (TILE_W - 220) / 2, y0 + TILE_H - 58, 220, 36,
                     "✓ same shape, pair positions", GREEN_L, GREEN, GREEN))
    return out


def panel_reject(x0, y0):
    """Panel 3: shape [2] vs shape [4]. Rejected. Show why with question marks."""
    out = []
    out.append(txt(x0 + TILE_W / 2, y0 + 28, "2 vs 4   (rejected)",
                   size=15, fill=RED, family=FONT_MONO, weight="bold"))
    out.append(txt(x0 + TILE_W / 2, y0 + 48,
                   "no single source to reuse",
                   size=11, fill=SLATE))

    gap = 6
    cell_w = 46

    # top: shape [2] (two cells)
    top_grid_w = 2 * cell_w + gap
    top_gx = x0 + (TILE_W - top_grid_w) / 2
    top_y = y0 + 82
    out.append(txt(x0 + TILE_W / 2, top_y - 6, "shape [2]",
                   size=10, fill=SLATE, family=FONT_MONO))
    top_vals = ["a", "b"]
    for j in range(2):
        cx = top_gx + j * (cell_w + gap)
        out.append(cell(cx, top_y, top_vals[j], AMBER_L, AMBER, AMBER))

    # bottom: shape [4] (four cells, all "?")
    bot_grid_w = 4 * cell_w + 3 * gap
    bot_gx = x0 + (TILE_W - bot_grid_w) / 2
    bot_y = y0 + 220
    out.append(txt(x0 + TILE_W / 2, bot_y - 6, "shape [4]",
                   size=10, fill=SLATE, family=FONT_MONO))
    for j in range(4):
        cx = bot_gx + j * (cell_w + gap)
        out.append(cell(cx, bot_y, "?", RED_L, RED, RED, weight="bold"))

    # ambiguous arrows: from each "a" or "b" to two target cells, all dashed red
    # source 'a' (j=0) tries to feed targets 0 and 1
    # source 'b' (j=1) tries to feed targets 2 and 3
    # then crossing arrows showing the conflict
    src0_x = top_gx + 23
    src1_x = top_gx + cell_w + gap + 23
    src_y = top_y + 46 + 2
    tgt_y = bot_y - 2

    # each target gets two competing arrows -> visual ambiguity
    pairs = [
        (src0_x, 0), (src1_x, 0),  # target 0 fed by both
        (src0_x, 1), (src1_x, 1),  # target 1 fed by both
        (src0_x, 2), (src1_x, 2),  # target 2 fed by both
        (src0_x, 3), (src1_x, 3),  # target 3 fed by both
    ]
    for sx, tj in pairs:
        tx = bot_gx + tj * (cell_w + gap) + 23
        out.append(f'<path d="M {sx} {src_y} L {tx} {tgt_y}" '
                   f'stroke="{RED}" stroke-width="1.0" '
                   f'stroke-dasharray="2 4" fill="none" opacity="0.45"/>')

    # big "no rule" tag floating in the middle
    tag_x = x0 + (TILE_W - 150) / 2
    tag_y = y0 + 150
    out.append(rect(tag_x, tag_y, 150, 28, RED_L, RED, sw=1.2, rx=14))
    out.append(txt(tag_x + 75, tag_y + 19, "no rule for 2 → 4",
                   size=11, fill=RED, family=FONT_MONO, weight="bold"))

    out.append(badge(x0 + (TILE_W - 240) / 2, y0 + TILE_H - 58, 240, 36,
                     "✗ broadcasting cannot invent", RED_L, RED, RED))
    return out


# ── canvas ───────────────────────────────────────────────────────────────────
def build_svg():
    GAP = 18
    W = 3 * TILE_W + 4 * GAP
    H = 90 + TILE_H + 80

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="{FONT_SANS}">',
           f'<rect width="{W}" height="{H}" fill="{PAPER}"/>',
           '<defs>',
           arrow_marker(PURPLE, "purple"),
           arrow_marker(INK,    "ink"),
           arrow_marker(RED,    "red"),
           '</defs>']

    # Title + subtitle
    out.append(txt(W / 2, 36, "When does broadcasting say yes, and when does it say no?",
                   size=20, fill=INK, weight="bold"))
    out.append(txt(W / 2, 60,
                   "the rule per axis: equal sizes OK · size 1 OK (stretch) · anything else is rejected",
                   size=12, fill=SLATE))

    # Three panels
    panels = [panel_reuse, panel_pair, panel_reject]
    for idx, p in enumerate(panels):
        x0 = GAP + idx * (TILE_W + GAP)
        y0 = 90
        # tile background
        border = GREEN if idx < 2 else RED
        out.append(rect(x0, y0, TILE_W, TILE_H, WHITE, "#e6e1d2", sw=1.0, rx=12))
        out.extend(p(x0, y0))

    # Footer takeaway
    foot_y = H - 28
    out.append(txt(W / 2, foot_y,
                   "memory rule:  broadcasting can REUSE a size-1 axis · "
                   "broadcasting cannot INVENT new positions",
                   size=13, fill=INK, family=FONT_MONO, weight="bold"))

    out.append('</svg>')
    return '\n'.join(out)


def main():
    out_dir = Path(__file__).resolve().parents[2] / "docs/assets/ch-03"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "broadcasting-rules.svg"
    out_path.write_text(build_svg())
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
