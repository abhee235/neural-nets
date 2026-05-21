"""Generate ch-01 figure: shape [2, 3, 4] read as two tiny greyscale images.

The figure makes the "tensor numbers ARE the picture" idea concrete:

  Row 1: two image cards side by side. Each card shows a 4x3 grid of NUMBERS
         on the left, an arrow, and the same grid rendered as GREYSCALE PIXELS
         on the right. Cells are tinted from black (0) to white (255).

  Row 2: the same 24 values laid out as a single flat Float64Array, with each
         cell tinted to match the pixel value above it. This is the visual proof
         that the "two images" and the "flat buffer" are the same data.

Pure-Python SVG generator. Writes to docs/assets/ch-01/shape-as-pixels.svg.
"""

from pathlib import Path

# Palette
PAPER  = "#faf7f0"
INK    = "#1d1d1b"
SLATE  = "#64748b"
TEAL   = "#0f766e"
AMBER  = "#b45309"

W, H = 940, 600
OUT = Path(__file__).resolve().parents[2] / "docs/assets/ch-01/shape-as-pixels.svg"

# Sample pixel values for the two 3x4 "images". Chosen to be visually recognisable:
# image 0 is a horizontal-and-vertical gradient, image 1 is a corner-bright wedge.
IMAGE_0 = [
    [ 20,  80, 140, 200],
    [ 40, 100, 160, 220],
    [ 60, 120, 180, 240],
]
IMAGE_1 = [
    [240, 180, 120,  60],
    [200, 140,  80,  40],
    [160, 100,  40,  20],
]


def grey(v: int) -> str:
    """Convert a 0..255 intensity to an RGB hex colour."""
    return f"#{v:02x}{v:02x}{v:02x}"


def text_color_on(v: int) -> str:
    """Pick black or white text so it stays readable on the greyscale background."""
    return "#ffffff" if v < 128 else "#111111"


def rect(x, y, w, h, fill=PAPER, stroke=INK, sw=1.2, rx=3, opacity=1.0, dash=None):
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" '
            f'opacity="{opacity}"{extra}/>')


def txt(x, y, s, anchor="middle", size=13, fill=INK, weight="normal",
        family="ui-sans-serif, -apple-system, Helvetica, Arial"):
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
            f'font-family="{family}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}">{s}</text>')


def arrow(x1, y1, x2, y2, color=SLATE, sw=1.6):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{color}" stroke-width="{sw}" marker-end="url(#arr)"/>')


def line(x1, y1, x2, y2, color=SLATE, sw=1.2):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{color}" stroke-width="{sw}"/>')


def draw_value_grid(parts, x0, y0, values, cw=42, ch=36):
    """Draw a 3x4 grid of numbers (no shading)."""
    for i, row in enumerate(values):
        for j, v in enumerate(row):
            parts.append(rect(x0 + j * cw, y0 + i * ch, cw, ch,
                              fill=PAPER, stroke=INK, sw=1.2))
            parts.append(txt(x0 + j * cw + cw / 2,
                             y0 + i * ch + ch / 2 + 4,
                             str(v), size=12, fill=INK, weight="500",
                             family="ui-monospace, SFMono-Regular, Menlo, monospace"))


def draw_pixel_grid(parts, x0, y0, values, cw=42, ch=36, show_numbers=True):
    """Draw a 3x4 grid where each cell is tinted to its greyscale value."""
    for i, row in enumerate(values):
        for j, v in enumerate(row):
            parts.append(rect(x0 + j * cw, y0 + i * ch, cw, ch,
                              fill=grey(v), stroke="#444", sw=0.8, rx=2))
            if show_numbers:
                parts.append(txt(x0 + j * cw + cw / 2,
                                 y0 + i * ch + ch / 2 + 4,
                                 str(v), size=11, fill=text_color_on(v),
                                 weight="500",
                                 family="ui-monospace, SFMono-Regular, Menlo, monospace"))


def build():
    out = []
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
               f'width="{W}" height="{H}" font-family="ui-sans-serif, -apple-system, Helvetica, Arial">')
    out.append('<defs>'
               f'<marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" '
               f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
               f'<path d="M0,0 L10,5 L0,10 z" fill="{SLATE}"/></marker>'
               # Black -> white linear gradient used in the brightness legend.
               '<linearGradient id="greyramp" x1="0%" y1="0%" x2="100%" y2="0%">'
               '<stop offset="0%" stop-color="#000000"/>'
               '<stop offset="100%" stop-color="#ffffff"/>'
               '</linearGradient>'
               '</defs>')
    out.append(rect(0, 0, W, H, fill=PAPER, stroke=PAPER, sw=0))

    # ====== TITLE ======
    out.append(txt(W / 2, 32, "Shape [2, 3, 4] as 2 tiny greyscale images  (each 3 rows \u00d7 4 cols)",
                   size=17, weight="700"))
    out.append(txt(W / 2, 52,
                   "Each cell of the tensor is one pixel. The number IS the brightness.",
                   size=12, fill=SLATE))

    # ====== BRIGHTNESS LEGEND (the rule: 0 = black, 255 = white, in between = grey) ======
    legend_w = 480
    legend_h = 22
    legend_x = (W - legend_w) / 2
    legend_y = 74
    # Header above the bar
    out.append(txt(W / 2, legend_y - 6,
                   "How a number becomes a shade  (8-bit greyscale)",
                   size=11, fill=INK, weight="600"))
    # The gradient bar itself
    out.append(f'<rect x="{legend_x}" y="{legend_y}" width="{legend_w}" height="{legend_h}" '
               f'fill="url(#greyramp)" stroke="{INK}" stroke-width="1" rx="2"/>')
    # Tick labels below the bar at 0, 64, 128, 192, 255
    ticks = [(0, "0", "black"),
             (64, "64", ""),
             (128, "128", "mid-grey"),
             (192, "192", ""),
             (255, "255", "white")]
    for v, num, name in ticks:
        tx = legend_x + (v / 255) * legend_w
        # Tick mark
        out.append(line(tx, legend_y + legend_h, tx, legend_y + legend_h + 4,
                        color=INK, sw=1.2))
        out.append(txt(tx, legend_y + legend_h + 16, num, size=10, fill=INK,
                       weight="600",
                       family="ui-monospace, SFMono-Regular, Menlo, monospace"))
        if name:
            out.append(txt(tx, legend_y + legend_h + 30, name, size=10, fill=SLATE))
    # Explainer to the side
    out.append(txt(W / 2, legend_y + legend_h + 48,
                   "a value of v paints a cell that is  v/255  of the way from black to white",
                   size=11, fill=INK))

    # ====== ROW 1: TWO IMAGE CARDS ======
    cw, ch = 42, 36
    grid_w, grid_h = cw * 4, ch * 3
    card_w = grid_w + 40 + grid_w + 28      # value grid + arrow gap + pixel grid + padding
    card_h = grid_h + 70
    card_gap = 30
    total_w = card_w * 2 + card_gap
    card0_x = (W - total_w) / 2
    card1_x = card0_x + card_w + card_gap
    card_y = 160

    for idx, (cx, vals) in enumerate([(card0_x, IMAGE_0), (card1_x, IMAGE_1)]):
        # Card background
        out.append(rect(cx, card_y, card_w, card_h,
                        fill=PAPER, stroke=SLATE, sw=1.2, rx=8, dash="4 4"))
        out.append(txt(cx + card_w / 2, card_y + 22,
                       f"image {idx}  \u2014  tensor[{idx}, :, :]",
                       size=13, fill=TEAL, weight="700"))

        # Value grid (left)
        vg_x = cx + 14
        vg_y = card_y + 36
        draw_value_grid(out, vg_x, vg_y, vals)
        out.append(txt(vg_x + grid_w / 2, vg_y + grid_h + 22,
                       "12 numbers", size=11, fill=SLATE, weight="500"))

        # Arrow
        ax1 = vg_x + grid_w + 8
        ax2 = ax1 + 24
        ay  = vg_y + grid_h / 2
        out.append(arrow(ax1, ay, ax2, ay, color=AMBER, sw=2.0))
        out.append(txt((ax1 + ax2) / 2, ay - 8, "render",
                       size=10, fill=AMBER, weight="600"))

        # Pixel grid (right)
        pg_x = ax2 + 8
        pg_y = vg_y
        draw_pixel_grid(out, pg_x, pg_y, vals)
        # Heavy outer frame so the pixel block reads as a single picture
        out.append(rect(pg_x - 1, pg_y - 1, grid_w + 2, grid_h + 2,
                        fill="none", stroke=INK, sw=1.6, rx=2))
        out.append(txt(pg_x + grid_w / 2, pg_y + grid_h + 22,
                       "the actual picture", size=11, fill=INK, weight="600"))

    # ====== ROW 2: FLAT BUFFER WITH MATCHING SHADING ======
    flat_top = card_y + card_h + 40
    flat_cw, flat_ch = 32, 36
    n = 24
    flat_w = flat_cw * n
    flat_x = (W - flat_w) / 2
    flat_y = flat_top + 36

    out.append(txt(W / 2, flat_top + 16,
                   "...and underneath it is one flat Float64Array(24) \u2014 same numbers, same shades",
                   size=13, weight="600"))

    # Flatten the two images in row-major order
    flat_vals = []
    for img in (IMAGE_0, IMAGE_1):
        for row in img:
            flat_vals.extend(row)

    for k, v in enumerate(flat_vals):
        out.append(rect(flat_x + k * flat_cw, flat_y, flat_cw, flat_ch,
                        fill=grey(v), stroke="#444", sw=0.8, rx=2))
        out.append(txt(flat_x + k * flat_cw + flat_cw / 2,
                       flat_y + flat_ch / 2 + 4,
                       str(v), size=10, fill=text_color_on(v),
                       weight="500",
                       family="ui-monospace, SFMono-Regular, Menlo, monospace"))

    # Offset labels under every 4th cell (start of each image-row)
    for k in (0, 4, 8, 12, 16, 20, 23):
        out.append(txt(flat_x + k * flat_cw + flat_cw / 2,
                       flat_y + flat_ch + 14,
                       str(k), size=10, fill=SLATE,
                       family="ui-monospace, SFMono-Regular, Menlo, monospace"))
    out.append(txt(flat_x - 18, flat_y + flat_ch / 2 + 4,
                   "data:", size=11, fill=SLATE, weight="600", anchor="end"))
    out.append(txt(flat_x - 18, flat_y + flat_ch + 14,
                   "index:", size=10, fill=SLATE, weight="600", anchor="end"))

    # Brackets marking image 0 and image 1 chunks
    bracket_y = flat_y + flat_ch + 34
    for img_idx in (0, 1):
        x1 = flat_x + img_idx * 12 * flat_cw + 2
        x2 = flat_x + (img_idx + 1) * 12 * flat_cw - 2
        out.append(f'<line x1="{x1}" y1="{bracket_y}" x2="{x2}" y2="{bracket_y}" '
                   f'stroke="{TEAL}" stroke-width="1.8"/>')
        out.append(f'<line x1="{x1}" y1="{bracket_y}" x2="{x1}" y2="{bracket_y - 6}" '
                   f'stroke="{TEAL}" stroke-width="1.8"/>')
        out.append(f'<line x1="{x2}" y1="{bracket_y}" x2="{x2}" y2="{bracket_y - 6}" '
                   f'stroke="{TEAL}" stroke-width="1.8"/>')
        out.append(txt((x1 + x2) / 2, bracket_y + 16,
                       f"image {img_idx}  (12 numbers)",
                       size=11, fill=TEAL, weight="600"))

    out.append('</svg>')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    build()
