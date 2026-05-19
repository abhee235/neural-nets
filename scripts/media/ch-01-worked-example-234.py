"""Generate ch-01 figure: worked example for shape [2, 3, 4].

Three stacked panels:
  1. Logical 3-D view: two outer slabs (each 3 rows x 4 cols) holding offsets 0..23.
  2. Flat buffer: all 24 cells in a row, with colour-coded chunk markers.
  3. Formula card: offset = 1*12 + 2*4 + 3*1 = 23, with each term colour-matched
     to the axis it came from.

Pure-Python SVG generator. No external dependencies. Writes a single deterministic
file into docs/assets/ch-01/worked-example-234.svg, overwriting on re-run.
"""

from pathlib import Path

# --- Palette (shared with other ch-01 figures) ---
PAPER  = "#faf7f0"
INK    = "#1d1d1b"
SLATE  = "#64748b"
PURPLE = "#7c3aed"   # outer axis (stride 12)
PURPLE_L = "#ede9fe"
TEAL   = "#0f766e"   # middle axis (stride 4)
TEAL_L = "#d8f0eb"
AMBER  = "#b45309"   # inner axis (stride 1) AND highlight colour
AMBER_L = "#fef3c7"

W, H = 1040, 780
OUT = Path(__file__).resolve().parents[2] / "docs/assets/ch-01/worked-example-234.svg"


def rect(x, y, w, h, fill=PAPER, stroke=INK, sw=1.3, rx=4, opacity=1.0, dash=None):
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" '
            f'opacity="{opacity}"{extra}/>')


def txt(x, y, s, anchor="middle", size=13, fill=INK, weight="normal",
        family="ui-sans-serif, -apple-system, Helvetica, Arial", rotate=None):
    r = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
            f'font-family="{family}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}"{r}>{s}</text>')


def line(x1, y1, x2, y2, color=SLATE, sw=1.2, dash=None):
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{color}" stroke-width="{sw}"{extra}/>')


def arrow(x1, y1, x2, y2, color=SLATE, sw=1.4):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{color}" stroke-width="{sw}" marker-end="url(#arr)"/>')


def brace_h(x1, x2, y, color, label, label_above=True, sw=1.6, tick=8):
    """Simple horizontal range marker: |---- label ----| with tick legs."""
    mid = (x1 + x2) / 2
    leg_y = y + tick if label_above else y - tick
    parts = [
        line(x1, y, x2, y, color=color, sw=sw),
        line(x1, y, x1, leg_y, color=color, sw=sw),
        line(x2, y, x2, leg_y, color=color, sw=sw),
    ]
    ly = y - 8 if label_above else y + 18
    parts.append(txt(mid, ly, label, size=12, fill=color, weight="600"))
    return "\n  ".join(parts)


def build():
    out = []
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
               f'width="{W}" height="{H}" font-family="ui-sans-serif, -apple-system, Helvetica, Arial">')
    out.append(f'<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" '
               f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
               f'<path d="M0,0 L10,5 L0,10 z" fill="{SLATE}"/></marker></defs>')
    out.append(rect(0, 0, W, H, fill=PAPER, stroke=PAPER, sw=0))

    # ====== TITLE ======
    out.append(txt(W / 2, 30, "Where does element [1, 2, 3] live in flat memory?",
                   size=18, weight="700"))
    out.append(txt(W / 2, 50,
                   "Shape [2, 3, 4]  \u2014  axis 0 size 2 (purple)  \u00b7  axis 1 size 3 (teal)  \u00b7  axis 2 size 4 (amber)",
                   size=12, fill=SLATE))

    # ====== KEY: what the numbers inside cells mean ======
    key_x = 30
    key_y = 74
    key_w = W - 60
    out.append(rect(key_x, key_y, key_w, 34, fill=PAPER, stroke=SLATE, sw=1.0,
                    rx=6, dash="3 3"))
    out.append(txt(key_x + 14, key_y + 14, "Reading key:",
                   size=11, fill=SLATE, weight="700", anchor="start"))
    out.append(txt(key_x + 14, key_y + 28,
                   "the number printed inside every cell below is that cell\u2019s OWN flat-buffer offset (not a stored value). "
                   "So the cell at position [1, 2, 3] showing \u201823\u2019 means: \u2018flat offset 23\u2019.",
                   size=11, fill=INK, anchor="start"))

    # ====== PANEL 1: LOGICAL 3-D VIEW (two slabs) ======
    panel_top = 124
    cw, ch = 46, 36                       # cell width/height
    slab_w = cw * 4                       # 4 inner cells per row
    slab_h = ch * 3                       # 3 middle rows per slab
    slab_gap = 80
    total_w = slab_w * 2 + slab_gap
    slab0_x = (W - total_w) / 2
    slab1_x = slab0_x + slab_w + slab_gap
    slab_y = panel_top + 50               # leave room for top labels

    # Panel header
    out.append(txt(W / 2, panel_top - 4,
                   "1.  The logical 3-D view  \u2014  shape [2, 3, 4] = two 3\u00d74 slabs",
                   size=13, fill=INK, weight="700"))

    # Inner-axis (axis 2) tick labels above each slab
    for sx in (slab0_x, slab1_x):
        out.append(txt(sx + slab_w / 2, panel_top + 18,
                       "axis 2 (cols)  \u2192",
                       size=11, fill=AMBER, weight="600"))
        for j in range(4):
            out.append(txt(sx + j * cw + cw / 2, panel_top + 36, str(j),
                           size=11, fill=AMBER))

    # Draw the two slabs
    for s_idx, sx in enumerate((slab0_x, slab1_x)):
        # Slab label (axis 0)
        out.append(txt(sx + slab_w / 2, slab_y + slab_h + 28,
                       f"axis 0 = {s_idx}", size=13, fill=PURPLE, weight="700"))
        # Axis 1 labels on left of first slab
        if s_idx == 0:
            out.append(txt(sx - 34, slab_y + slab_h / 2 - 8, "axis 1",
                           size=11, fill=TEAL, weight="600", anchor="middle"))
            out.append(txt(sx - 34, slab_y + slab_h / 2 + 8, "(rows)",
                           size=11, fill=TEAL, weight="600", anchor="middle"))
        for i in range(3):
            out.append(txt(sx - 10, slab_y + i * ch + ch / 2 + 4, str(i),
                           size=11, fill=TEAL, anchor="end"))
            for j in range(4):
                offset = s_idx * 12 + i * 4 + j
                # Highlight target cell [1, 2, 3] -> offset 23
                is_target = (s_idx == 1 and i == 2 and j == 3)
                fill_color = AMBER_L if is_target else PAPER
                stroke_color = AMBER if is_target else INK
                sw = 2.2 if is_target else 1.3
                out.append(rect(sx + j * cw, slab_y + i * ch, cw, ch,
                                fill=fill_color, stroke=stroke_color, sw=sw))
                weight = "700" if is_target else "500"
                color = AMBER if is_target else INK
                out.append(txt(sx + j * cw + cw / 2,
                               slab_y + i * ch + ch / 2 + 4,
                               str(offset), size=12, fill=color, weight=weight))

    # Annotation arrow pointing at the highlighted cell
    tgt_cx = slab1_x + 3 * cw + cw / 2
    tgt_cy = slab_y + 2 * ch + ch / 2
    ann_x = slab1_x + slab_w + 20
    ann_y = slab_y + slab_h + 4
    out.append(arrow(ann_x + 90, ann_y - 4, tgt_cx + 4, tgt_cy + 8,
                     color=AMBER, sw=1.6))
    out.append(txt(ann_x + 95, ann_y + 12,
                   "target: [1, 2, 3]",
                   size=12, fill=AMBER, weight="700", anchor="start"))
    out.append(txt(ann_x + 95, ann_y + 28,
                   "(offset = 23)",
                   size=11, fill=AMBER, anchor="start"))

    # ====== PANEL 2: FLAT BUFFER ======
    buf_top = 370
    buf_cw, buf_ch = 32, 38
    n = 24
    buf_w = buf_cw * n
    buf_x = (W - buf_w) / 2
    buf_y = buf_top + 64                  # extra room so brace labels do not collide with title

    out.append(txt(W / 2, buf_top + 8,
                   "2.  The flat buffer  \u2014  Float64Array(24)  \u2014  same 24 numbers, single row in memory",
                   size=13, fill=INK, weight="700"))

    # Cells
    for k in range(n):
        is_target = (k == 23)
        fill_color = AMBER_L if is_target else PAPER
        stroke_color = AMBER if is_target else INK
        sw = 2.2 if is_target else 1.2
        out.append(rect(buf_x + k * buf_cw, buf_y, buf_cw, buf_ch,
                        fill=fill_color, stroke=stroke_color, sw=sw, rx=3))
        weight = "700" if is_target else "500"
        color = AMBER if is_target else INK
        out.append(txt(buf_x + k * buf_cw + buf_cw / 2, buf_y + buf_ch / 2 + 4,
                       str(k), size=11, fill=color, weight=weight))

    # Purple brackets ABOVE the buffer: the two 12-element axis-0 blocks
    purple_y = buf_y - 14
    out.append(brace_h(buf_x + 0, buf_x + 12 * buf_cw - 2, purple_y,
                       color=PURPLE, label="axis 0 = 0  (one step skips 12 cells = stride[0])",
                       label_above=True))
    out.append(brace_h(buf_x + 12 * buf_cw + 2, buf_x + 24 * buf_cw, purple_y,
                       color=PURPLE, label="axis 0 = 1  (next 12-cell block)",
                       label_above=True))

    # Teal brackets BELOW axis-0 block 1: three 4-element axis-1 chunks
    teal_y = buf_y + buf_ch + 14
    for m in range(3):
        x_start = buf_x + (12 + m * 4) * buf_cw + 2
        x_end   = buf_x + (12 + (m + 1) * 4) * buf_cw - 2
        label = f"axis 1 = {m}  (4 cells = stride[1])" if m == 0 else f"axis 1 = {m}"
        out.append(brace_h(x_start, x_end, teal_y,
                           color=TEAL, label=label, label_above=False))

    # Amber tick under the target cell
    amber_y = buf_y + buf_ch + 56
    out.append(line(buf_x + 23 * buf_cw + 4, amber_y,
                    buf_x + 23 * buf_cw + buf_cw - 4, amber_y,
                    color=AMBER, sw=2.4))
    out.append(txt(buf_x + 23 * buf_cw + buf_cw / 2, amber_y + 16,
                   "axis 2 = 3  (1 cell = stride[2])", size=11, fill=AMBER, weight="600"))

    # ====== PANEL 3: FORMULA CARD ======
    card_top = 570
    card_w = 800
    card_h = 190
    card_x = (W - card_w) / 2
    out.append(rect(card_x, card_top, card_w, card_h,
                    fill=PAPER, stroke=INK, sw=1.5, rx=10, dash="6 4"))
    out.append(txt(card_x + 24, card_top + 26,
                   "3.  The formula  \u2014  add up (index along each axis)  \u00d7  (its stride)",
                   size=14, weight="700", anchor="start"))

    # Color-coded equation rendered as positioned tspans (kept simple as text runs).
    eq_y = card_top + 70
    # We hand-place the pieces so each axis gets its own colour.
    eq_x = card_x + 40
    parts = [
        ("offset  =  ",         INK,    "600"),
        ("1",                   PURPLE, "700"),
        (" \u00d7 ",             INK,    "600"),
        ("12",                  PURPLE, "700"),
        ("    +    ",           INK,    "600"),
        ("2",                   TEAL,   "700"),
        (" \u00d7 ",             INK,    "600"),
        ("4",                   TEAL,   "700"),
        ("    +    ",           INK,    "600"),
        ("3",                   AMBER,  "700"),
        (" \u00d7 ",             INK,    "600"),
        ("1",                   AMBER,  "700"),
        ("    =    ",           INK,    "600"),
        ("23",                  AMBER,  "800"),
    ]
    cursor = eq_x
    for s, color, weight in parts:
        # Approximate per-char width at size 18, monospace-ish.
        out.append(txt(cursor, eq_y, s.replace(" ", "\u00a0"),
                       anchor="start", size=18, fill=color, weight=weight,
                       family="ui-monospace, SFMono-Regular, Menlo, monospace"))
        cursor += len(s) * 10.2

    # Annotation row underneath
    note_y = card_top + 108
    out.append(txt(card_x + 40, note_y,
                   "index along  axis 0  \u00d7  stride[0]   +   "
                   "index along  axis 1  \u00d7  stride[1]   +   "
                   "index along  axis 2  \u00d7  stride[2]",
                   anchor="start", size=12, fill=SLATE))
    out.append(txt(card_x + 40, note_y + 22,
                   "\u201cskip 1 full axis-0 slab (12 cells), then 2 axis-1 rows (8 cells), then 3 axis-2 steps (3 cells).\u201d",
                   anchor="start", size=12, fill=INK, weight="500"))
    out.append(txt(card_x + 40, note_y + 44,
                   "Strides come from the shape:  stride[i] = product of shape[i+1..end].  "
                   "For [2,3,4] that is 12, 4, 1.",
                   anchor="start", size=12, fill=SLATE))

    out.append('</svg>')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    build()
