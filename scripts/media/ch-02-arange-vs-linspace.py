"""
Chapter: 02 — Tensor Creation
Figure:  arange-vs-linspace.svg
Concept: Two number lines comparing arange(0, 1, 0.25) vs linspace(0, 1, 5).
Output:  docs/assets/ch-02/arange-vs-linspace.svg

Teaches: arange holds the step fixed and stops BEFORE the endpoint (open dot at 1.0).
         linspace holds the count fixed and LANDS ON the endpoint (closed dot at 1.0).

Run:
    python3 scripts/media/ch-02-arange-vs-linspace.py
"""

import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "assets", "ch-02")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "arange-vs-linspace.svg")

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
FONT    = "ui-monospace, SFMono-Regular, Menlo, monospace"
SERIF   = "Georgia, 'Iowan Old Style', serif"

W, H = 1100, 460
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
       f'font-family="{SERIF}">',
       f'<rect width="{W}" height="{H}" fill="{PAPER}"/>']

svg.append(f'<text x="{W/2}" y="42" text-anchor="middle" font-size="20" '
           f'font-weight="bold" fill="{INK}">arange  vs  linspace</text>')
svg.append(f'<text x="{W/2}" y="62" text-anchor="middle" font-size="12" '
           f'fill="{SLATE}">Same span [0, 1].  Different rule for the right endpoint.</text>')

# ── shared geometry ─────────────────────────────────────────────────────────
LINE_X0 = 110
LINE_X1 = 760
def x_for(v, vmin=0.0, vmax=1.0):
    return LINE_X0 + (v - vmin) / (vmax - vmin) * (LINE_X1 - LINE_X0)

def number_line(y, accent_dark, accent_light, title, signature, values,
                stop_inclusive, n, step):
    out = []
    # title
    out.append(f'<text x="{LINE_X0}" y="{y - 64}" font-size="15" '
               f'font-weight="bold" font-family="{FONT}" fill="{accent_dark}">{title}</text>')
    out.append(f'<text x="{LINE_X0}" y="{y - 46}" font-size="12" '
               f'font-family="{FONT}" fill="{SLATE}">{signature}</text>')

    # background pill of the span
    out.append(f'<rect x="{LINE_X0 - 8}" y="{y - 10}" '
               f'width="{LINE_X1 - LINE_X0 + 16}" height="20" rx="10" '
               f'fill="{accent_light}" stroke="{accent_dark}" '
               f'stroke-width="1" opacity="0.6"/>')
    # the line itself
    out.append(f'<line x1="{LINE_X0 - 14}" y1="{y}" x2="{LINE_X1 + 14}" y2="{y}" '
               f'stroke="{INK}" stroke-width="1.4"/>')

    # endpoint marker at 0 (always closed circle)
    out.append(f'<circle cx="{x_for(0)}" cy="{y}" r="7" fill="{accent_dark}"/>')
    out.append(f'<text x="{x_for(0)}" y="{y + 28}" text-anchor="middle" '
               f'font-size="11" font-family="{FONT}" fill="{INK}">0</text>')

    # endpoint marker at 1 (closed if inclusive, open if exclusive)
    if stop_inclusive:
        out.append(f'<circle cx="{x_for(1)}" cy="{y}" r="7" fill="{accent_dark}"/>')
    else:
        out.append(f'<circle cx="{x_for(1)}" cy="{y}" r="7" fill="{PAPER}" '
                   f'stroke="{accent_dark}" stroke-width="2"/>')
    label_1 = "1.0  ✓ included" if stop_inclusive else "1.0  ✗ stops before"
    out.append(f'<text x="{x_for(1)}" y="{y + 28}" text-anchor="middle" '
               f'font-size="11" font-family="{FONT}" fill="{INK}">{label_1}</text>')

    # internal sample points
    for v in values:
        if v == 0 or v == 1:
            continue
        out.append(f'<circle cx="{x_for(v)}" cy="{y}" r="6" fill="{accent_dark}"/>')
        out.append(f'<text x="{x_for(v)}" y="{y - 16}" text-anchor="middle" '
                   f'font-size="11" font-family="{FONT}" fill="{accent_dark}">{v}</text>')

    # step / gap brace under first interval
    if values and len(values) >= 2:
        a, b = values[0], values[1]
        bx0, bx1 = x_for(a), x_for(b)
        by = y + 50
        out.append(f'<path d="M {bx0} {by - 6} L {bx0} {by} L {bx1} {by} L {bx1} {by - 6}" '
                   f'fill="none" stroke="{AMBER}" stroke-width="1.4"/>')
        if step is not None:
            label = f"step = {step}"
        else:
            label = f"Δ = (b−a)/(n−1)"
        out.append(f'<text x="{(bx0 + bx1)/2}" y="{by + 16}" text-anchor="middle" '
                   f'font-size="11" font-family="{FONT}" fill="{AMBER}">{label}</text>')

    # output array box on right
    box_x = LINE_X1 + 26
    box_y = y - 24
    arr_text = "[" + ", ".join(str(v) for v in values) + "]"
    box_w = max(180, 9.5 * len(arr_text))
    out.append(f'<rect x="{box_x}" y="{box_y}" width="{box_w}" height="46" '
               f'rx="8" fill="{AMBER_L}" stroke="{AMBER}" stroke-width="1.4"/>')
    out.append(f'<text x="{box_x + box_w/2}" y="{box_y + 18}" '
               f'text-anchor="middle" font-size="11" font-family="{FONT}" '
               f'fill="{AMBER}">length = {n}</text>')
    out.append(f'<text x="{box_x + box_w/2}" y="{box_y + 36}" '
               f'text-anchor="middle" font-size="13" font-family="{FONT}" '
               f'fill="{AMBER}">{arr_text}</text>')

    return "\n".join(out)

# Top: arange (exclusive, 4 points)
svg.append(number_line(y=170, accent_dark=PURPLE, accent_light=PURPLE_L,
                       title="arange(0, 1, 0.25)",
                       signature="fixed step  ·  stop is exclusive",
                       values=[0, 0.25, 0.5, 0.75],
                       stop_inclusive=False, n=4, step=0.25))

# Bottom: linspace (inclusive, 5 points)
svg.append(number_line(y=340, accent_dark=TEAL, accent_light=TEAL_L,
                       title="linspace(0, 1, 5)",
                       signature="fixed count  ·  stop is inclusive",
                       values=[0, 0.25, 0.5, 0.75, 1.0],
                       stop_inclusive=True, n=5, step=None))

# Between-line legend
svg.append(f'<text x="{W/2}" y="252" text-anchor="middle" font-size="12" '
           f'font-style="italic" fill="{SLATE}">'
           f'Same internal points 0.25, 0.50, 0.75. The right endpoint is the only difference.</text>')

# footer
svg.append(f'<text x="{W/2}" y="{H - 14}" text-anchor="middle" '
           f'font-size="11" fill="{SLATE}" font-style="italic">'
           f'positional encoding (Ch 19) uses both: arange for indices, linspace for frequencies.</text>')

svg.append('</svg>')
with open(OUT_FILE, "w") as f:
    f.write("\n".join(svg))
print(f"wrote {OUT_FILE}")
