"""
Chapter: 03 — Elementwise Ops & Broadcasting
Figure:  broadcasting-merge-frames.svg  (static filmstrip)
Concept: Step-by-step visual showing how [3,1] and [1,4] merge into [3,4].
Output:  docs/assets/ch-03/broadcasting-merge-frames.svg

Run:
    python3 scripts/media/ch-03-broadcasting-merge.py

Companion: broadcasting-merge.svg (animated SMIL version, generated below).
The filmstrip is the always-readable "paused" version. The animation cycles
through the same four frames for readers who want motion.
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
GHOST    = "#f3f1ea"
WHITE    = "#ffffff"

FONT_MONO = "ui-monospace, SFMono-Regular, Menlo, monospace"
FONT_SANS = "ui-sans-serif, -apple-system, Helvetica, Arial, sans-serif"

# Source data
A_COL = [1, 2, 3]            # shape [3, 1]
B_ROW = [10, 20, 30, 40]     # shape [1, 4]

# ── helpers ──────────────────────────────────────────────────────────────────
def rect(x, y, w, h, fill, stroke, sw=1.4, rx=6):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def txt(x, y, s, anchor="middle", size=12, fill=INK, family=FONT_SANS, weight="normal"):
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{family}" '
            f'font-size="{size}" font-weight="{weight}" fill="{fill}">{s}</text>')


def cell(cx, cy, w, h, value, fill, stroke, text_color, value_size=18, ghost=False):
    out = [rect(cx, cy, w, h, fill, stroke, sw=1.4, rx=6)]
    if value is not None:
        weight = "normal" if ghost else "bold"
        out.append(txt(cx + w / 2, cy + h / 2 + 6, str(value),
                       size=value_size, fill=text_color, family=FONT_MONO, weight=weight))
    return out


def grid(x0, y0, rows, cols, value_fn, fill_fn, stroke_fn, color_fn, cell_w=46, cell_h=46, gap=4, ghost_fn=None):
    """value_fn(i,j) -> value or None; fill_fn(i,j) -> fill; stroke_fn(i,j) -> stroke; color_fn(i,j) -> text color."""
    out = []
    for i in range(rows):
        for j in range(cols):
            cx = x0 + j * (cell_w + gap)
            cy = y0 + i * (cell_h + gap)
            out.extend(cell(cx, cy, cell_w, cell_h,
                            value_fn(i, j),
                            fill_fn(i, j),
                            stroke_fn(i, j),
                            color_fn(i, j),
                            ghost=(ghost_fn(i, j) if ghost_fn else False)))
    return out


# ── frame builders ───────────────────────────────────────────────────────────
def frame_origin(x0, y0, tile_w, tile_h):
    """Frame 1: just the two source tensors, drawn separately."""
    out = []
    # A on the left (label below the badge row)
    out.append(txt(x0 + 75, y0 + 38, "A · shape [3, 1]",
                   size=11, fill=SLATE, family=FONT_MONO, weight="bold"))
    out.extend(grid(x0 + 50, y0 + 50, 3, 1,
                    value_fn=lambda i, j: A_COL[i],
                    fill_fn=lambda i, j: PURPLE_L,
                    stroke_fn=lambda i, j: PURPLE,
                    color_fn=lambda i, j: PURPLE))
    # plus sign between them
    out.append(txt(x0 + tile_w / 2, y0 + 130, "+", size=28, fill=INK, weight="bold"))
    # B on the right (label above its grid, well clear of the badge)
    out.append(txt(x0 + tile_w - 110, y0 + 38, "B · shape [1, 4]",
                   size=11, fill=SLATE, family=FONT_MONO, weight="bold"))
    out.extend(grid(x0 + tile_w - 220, y0 + 110, 1, 4,
                    value_fn=lambda i, j: B_ROW[j],
                    fill_fn=lambda i, j: TEAL_L,
                    stroke_fn=lambda i, j: TEAL,
                    color_fn=lambda i, j: TEAL))
    return out


def frame_stretch_a(x0, y0, tile_w, tile_h):
    """Frame 2: A reuses its single column across all 4 columns."""
    out = []
    out.append(txt(x0 + tile_w / 2, y0 + 18, "A reuses its column 4 times", size=12, fill=SLATE, family=FONT_MONO, weight="bold"))
    # First column = original (bold), other columns = ghosted copies
    grid_w = 4 * 46 + 3 * 4
    gx = x0 + (tile_w - grid_w) / 2
    out.extend(grid(gx, y0 + 38, 3, 4,
                    value_fn=lambda i, j: A_COL[i],
                    fill_fn=lambda i, j: PURPLE_L if j == 0 else GHOST,
                    stroke_fn=lambda i, j: PURPLE if j == 0 else "#cdc8b8",
                    color_fn=lambda i, j: PURPLE if j == 0 else "#a8a394",
                    ghost_fn=lambda i, j: j != 0))
    # Arrows from first column to the rest (one per row)
    for i in range(3):
        cy = y0 + 38 + i * (46 + 4) + 23
        out.append(f'<path d="M {gx + 46 + 2} {cy} L {gx + 4 * 46 + 3 * 4 - 2} {cy}" '
                   f'stroke="{PURPLE}" stroke-width="1" stroke-dasharray="3 3" fill="none" opacity="0.6"/>')
    return out


def frame_stretch_b(x0, y0, tile_w, tile_h):
    """Frame 3: B reuses its single row across all 3 rows."""
    out = []
    out.append(txt(x0 + tile_w / 2, y0 + 18, "B reuses its row 3 times", size=12, fill=SLATE, family=FONT_MONO, weight="bold"))
    grid_w = 4 * 46 + 3 * 4
    gx = x0 + (tile_w - grid_w) / 2
    out.extend(grid(gx, y0 + 38, 3, 4,
                    value_fn=lambda i, j: B_ROW[j],
                    fill_fn=lambda i, j: TEAL_L if i == 0 else GHOST,
                    stroke_fn=lambda i, j: TEAL if i == 0 else "#cdc8b8",
                    color_fn=lambda i, j: TEAL if i == 0 else "#a8a394",
                    ghost_fn=lambda i, j: i != 0))
    # Arrows from first row down to the rest
    for j in range(4):
        cx = gx + j * (46 + 4) + 23
        out.append(f'<path d="M {cx} {y0 + 38 + 46 + 2} L {cx} {y0 + 38 + 3 * 46 + 2 * 4 - 2}" '
                   f'stroke="{TEAL}" stroke-width="1" stroke-dasharray="3 3" fill="none" opacity="0.6"/>')
    return out


def frame_sum(x0, y0, tile_w, tile_h):
    """Frame 4: the actual A + B sum, both stretched, with answer cells."""
    out = []
    out.append(txt(x0 + tile_w / 2, y0 + 18, "A + B · shape [3, 4]", size=12, fill=AMBER, family=FONT_MONO, weight="bold"))
    grid_w = 4 * 46 + 3 * 4
    gx = x0 + (tile_w - grid_w) / 2
    out.extend(grid(gx, y0 + 38, 3, 4,
                    value_fn=lambda i, j: A_COL[i] + B_ROW[j],
                    fill_fn=lambda i, j: AMBER_L,
                    stroke_fn=lambda i, j: AMBER,
                    color_fn=lambda i, j: AMBER))
    return out


# ── filmstrip canvas ─────────────────────────────────────────────────────────
def build_filmstrip():
    TILE_W = 320
    TILE_H = 220
    GAP_X = 16
    GAP_Y = 26
    COLS = 2
    ROWS = 2

    W = COLS * TILE_W + (COLS + 1) * GAP_X
    H = 70 + ROWS * TILE_H + (ROWS - 1) * GAP_Y + 30

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="{FONT_SANS}">',
           f'<rect width="{W}" height="{H}" fill="{PAPER}"/>']

    # Title
    out.append(txt(W / 2, 32, "Broadcasting [3, 1] + [1, 4] → [3, 4]", size=18, fill=INK, weight="bold"))
    out.append(txt(W / 2, 52, "four frames · same arithmetic, just reuse along the size-1 axis",
                   size=12, fill=SLATE))

    frame_builders = [frame_origin, frame_stretch_a, frame_stretch_b, frame_sum]
    for idx, builder in enumerate(frame_builders):
        col = idx % COLS
        row = idx // COLS
        x0 = GAP_X + col * (TILE_W + GAP_X)
        y0 = 70 + row * (TILE_H + GAP_Y)
        # Tile bg
        out.append(rect(x0, y0, TILE_W, TILE_H, WHITE, "#e6e1d2", sw=1.0, rx=10))
        # Frame label badge
        out.append(rect(x0 + 12, y0 + 8, 64, 18, AMBER_L, AMBER, sw=1.0, rx=4))
        out.append(txt(x0 + 44, y0 + 21, f"frame {idx + 1}", size=10, fill=AMBER, family=FONT_MONO, weight="bold"))
        out.extend(builder(x0, y0 + 8, TILE_W, TILE_H - 8))

    out.append('</svg>')
    return '\n'.join(out)


# ── animated SVG (SMIL) ──────────────────────────────────────────────────────
def build_animation():
    """One canvas, four phases, opacity-driven SMIL animation. Loops forever."""
    W, H = 540, 360
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="{FONT_SANS}">',
           f'<rect width="{W}" height="{H}" fill="{PAPER}"/>']

    out.append(txt(W / 2, 32, "Broadcasting [3, 1] + [1, 4] → [3, 4]", size=16, fill=INK, weight="bold"))
    cap = txt(W / 2, 54, "frame 1 / 4 · just the two source tensors", size=11, fill=SLATE)
    # We'll layer captions and animate visibility per phase.

    DUR = 8  # seconds total
    FRAMES = 4
    PHASE = DUR / FRAMES  # 2s each

    # Captions per phase
    captions = [
        "frame 1 / 4 · just the two source tensors",
        "frame 2 / 4 · A reuses its column 4 times",
        "frame 3 / 4 · B reuses its row 3 times",
        "frame 4 / 4 · A + B · shape [3, 4]",
    ]
    for i, c in enumerate(captions):
        # Each caption is visible during its phase only
        out.append(f'<g opacity="0">'
                   f'{txt(W / 2, 54, c, size=11, fill=SLATE)}'
                   f'<animate attributeName="opacity" '
                   f'values="0;1;1;0" '
                   f'keyTimes="0;{i / FRAMES + 0.001:.4f};{(i + 1) / FRAMES - 0.001:.4f};1" '
                   f'dur="{DUR}s" repeatCount="indefinite"/>'
                   f'</g>')

    # Geometry for the master grid (3 rows × 4 cols)
    cell_w = cell_h = 50
    gap = 5
    grid_w = 4 * cell_w + 3 * gap
    grid_h = 3 * cell_h + 2 * gap
    gx = (W - grid_w) / 2
    gy = 90

    # Helper to write a cell with a phase-keyed opacity animation.
    def animated_cell(cx, cy, value, fill, stroke, color, keyframes, ghost=False):
        """keyframes: list of opacity values (length FRAMES + 1, looped)."""
        weight = "normal" if ghost else "bold"
        opa_values = ";".join(f"{v:.2f}" for v in keyframes)
        key_times = ";".join(f"{i / FRAMES:.4f}" for i in range(FRAMES + 1))
        return (f'<g opacity="{keyframes[0]:.2f}">'
                f'<rect x="{cx}" y="{cy}" width="{cell_w}" height="{cell_h}" rx="6" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>'
                f'<text x="{cx + cell_w / 2}" y="{cy + cell_h / 2 + 6}" text-anchor="middle" '
                f'font-family="{FONT_MONO}" font-size="18" font-weight="{weight}" fill="{color}">'
                f'{value}</text>'
                f'<animate attributeName="opacity" values="{opa_values}" '
                f'keyTimes="{key_times}" dur="{DUR}s" repeatCount="indefinite"/>'
                f'</g>')

    # Phase 1: only first column of A (left-anchored, but we render in master grid coords)
    # Master grid column 0 = A's column. Master grid row 0 = B's row.
    # We will display:
    #   Frame 1: column 0 (purple, real A) + row 0 (teal, real B), with col0/row0 overlap shown teal (top-left cell)
    #   Frame 2: full purple grid (A stretched), teal first row still visible
    #   Frame 3: A grid hidden, teal full grid (B stretched)
    #   Frame 4: amber full grid (sum)
    #
    # Simplest implementation: stack four full grids with phase-keyed opacity.

    def fade_keyframes(visible_phase):
        """Return opacity keyframes [v0,v1,v2,v3,v4] with 1 only during visible_phase."""
        kf = [0.0] * (FRAMES + 1)
        kf[visible_phase] = 1.0
        kf[visible_phase + 1] = 1.0
        return kf

    # ── Frame 1 layer: only A column 0 + B row 0 ──
    for i in range(3):
        cx = gx + 0 * (cell_w + gap)
        cy = gy + i * (cell_h + gap)
        out.append(animated_cell(cx, cy, A_COL[i], PURPLE_L, PURPLE, PURPLE, fade_keyframes(0)))
    for j in range(1, 4):  # skip col 0 to avoid stacking with A
        cx = gx + j * (cell_w + gap)
        cy = gy + 0 * (cell_h + gap)
        out.append(animated_cell(cx, cy, B_ROW[j], TEAL_L, TEAL, TEAL, fade_keyframes(0)))

    # ── Frame 2 layer: full A grid (purple) ──
    for i in range(3):
        for j in range(4):
            cx = gx + j * (cell_w + gap)
            cy = gy + i * (cell_h + gap)
            ghost = j != 0
            fill = PURPLE_L if j == 0 else GHOST
            stroke = PURPLE if j == 0 else "#cdc8b8"
            color = PURPLE if j == 0 else "#a8a394"
            out.append(animated_cell(cx, cy, A_COL[i], fill, stroke, color, fade_keyframes(1), ghost=ghost))

    # ── Frame 3 layer: full B grid (teal) ──
    for i in range(3):
        for j in range(4):
            cx = gx + j * (cell_w + gap)
            cy = gy + i * (cell_h + gap)
            ghost = i != 0
            fill = TEAL_L if i == 0 else GHOST
            stroke = TEAL if i == 0 else "#cdc8b8"
            color = TEAL if i == 0 else "#a8a394"
            out.append(animated_cell(cx, cy, B_ROW[j], fill, stroke, color, fade_keyframes(2), ghost=ghost))

    # ── Frame 4 layer: full sum (amber) ──
    for i in range(3):
        for j in range(4):
            cx = gx + j * (cell_w + gap)
            cy = gy + i * (cell_h + gap)
            value = A_COL[i] + B_ROW[j]
            out.append(animated_cell(cx, cy, value, AMBER_L, AMBER, AMBER, fade_keyframes(3)))

    out.append('</svg>')
    return '\n'.join(out)


def main():
    out_dir = Path(__file__).resolve().parents[2] / "docs/assets/ch-03"
    out_dir.mkdir(parents=True, exist_ok=True)

    film = out_dir / "broadcasting-merge-frames.svg"
    film.write_text(build_filmstrip())
    print(f"wrote {film}")

    anim = out_dir / "broadcasting-merge.svg"
    anim.write_text(build_animation())
    print(f"wrote {anim}")


if __name__ == "__main__":
    main()
