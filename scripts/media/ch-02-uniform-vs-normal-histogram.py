"""
Chapter: 02 — Tensor Creation
Figure:  uniform-vs-normal-histogram.svg
Concept: 10000 samples from rand() vs randn(), shown as side-by-side histograms.
Output:  docs/assets/ch-02/uniform-vs-normal-histogram.svg

Teaches: rand() spreads samples evenly across [0, 1). randn() (Box-Muller) clusters
         them around 0 in a bell curve. Same RNG underneath, different shaping.

Run:
    python3 scripts/media/ch-02-uniform-vs-normal-histogram.py
"""

import math
import os
import random

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "assets", "ch-02")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "uniform-vs-normal-histogram.svg")

# ── palette ─────────────────────────────────────────────────────────────────
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

# ── deterministic sample ────────────────────────────────────────────────────
random.seed(42)
N = 10_000

uniform = [random.random() for _ in range(N)]

# Box-Muller; mirrors what src/tensor/creation.ts will compute
normal = []
i = 0
while len(normal) < N:
    u1 = max(random.random(), 1e-300)
    u2 = random.random()
    r = math.sqrt(-2 * math.log(u1))
    theta = 2 * math.pi * u2
    normal.append(r * math.cos(theta))
    if len(normal) < N:
        normal.append(r * math.sin(theta))

# ── histogram bins ──────────────────────────────────────────────────────────
def histogram(samples, lo, hi, nbins):
    bins = [0] * nbins
    width = (hi - lo) / nbins
    for s in samples:
        if lo <= s < hi:
            idx = int((s - lo) / width)
            if 0 <= idx < nbins:
                bins[idx] += 1
    return bins

UNI_LO, UNI_HI, UNI_BINS = 0.0, 1.0, 25
NRM_LO, NRM_HI, NRM_BINS = -4.0, 4.0, 41

uni_hist = histogram(uniform, UNI_LO, UNI_HI, UNI_BINS)
nrm_hist = histogram(normal,  NRM_LO, NRM_HI, NRM_BINS)

# ── canvas ──────────────────────────────────────────────────────────────────
W, H = 980, 460
PANEL_W = 420
PANEL_H = 320
PAD = 50
GAP = 60
PANEL_Y = 90

L_X = PAD
R_X = PAD + PANEL_W + GAP

# Plot area inside each panel
PLOT_PAD_L = 50
PLOT_PAD_R = 24
PLOT_PAD_T = 36
PLOT_PAD_B = 50
plot_w = PANEL_W - PLOT_PAD_L - PLOT_PAD_R
plot_h = PANEL_H - PLOT_PAD_T - PLOT_PAD_B

svg = []
svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
           f'font-family="{SERIF}">')
svg.append(f'<rect width="{W}" height="{H}" fill="{PAPER}"/>')

# ── title ───────────────────────────────────────────────────────────────────
svg.append(f'<text x="{W/2}" y="44" text-anchor="middle" font-size="20" '
           f'font-weight="bold" fill="{INK}">10 000 samples · two distributions</text>')
svg.append(f'<text x="{W/2}" y="64" text-anchor="middle" font-size="12" '
           f'fill="{SLATE}">Same `Math.random()` underneath. The right panel '
           f'reshapes uniform samples into a bell via Box–Muller.</text>')

def draw_panel(x, y, title, accent_dark, accent_light, hist, lo, hi, xticks,
               overlay_curve=None, mean=None, std=None):
    out = []
    # Panel frame
    out.append(f'<rect x="{x}" y="{y}" width="{PANEL_W}" height="{PANEL_H}" '
               f'rx="14" fill="#ffffff" stroke="{GRID_S}" stroke-width="1.2"/>')

    # Title
    out.append(f'<text x="{x + PANEL_W/2}" y="{y + 24}" text-anchor="middle" '
               f'font-size="14" font-weight="bold" font-family="{FONT}" '
               f'fill="{accent_dark}">{title}</text>')

    # Plot frame
    px = x + PLOT_PAD_L
    py = y + PLOT_PAD_T
    pw = plot_w
    ph = plot_h

    # Y-axis gridlines (5 levels)
    max_h = max(hist)
    # Round y-max up nicely
    def nice_ceil(v):
        for step in [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]:
            if v <= step:
                return step
        return v
    y_max = nice_ceil(max_h)

    for i in range(6):
        gy = py + ph - (i / 5) * ph
        out.append(f'<line x1="{px}" y1="{gy:.1f}" x2="{px+pw}" y2="{gy:.1f}" '
                   f'stroke="{GRID_S}" stroke-width="0.8"/>')
        out.append(f'<text x="{px - 8}" y="{gy + 4:.1f}" text-anchor="end" '
                   f'font-size="10" font-family="{FONT}" fill="{SLATE}">'
                   f'{int(i/5*y_max)}</text>')

    # Bars
    nb = len(hist)
    bw = pw / nb
    for i, h in enumerate(hist):
        bx = px + i * bw
        bh = (h / y_max) * ph if y_max > 0 else 0
        by = py + ph - bh
        out.append(f'<rect x="{bx + 0.6:.2f}" y="{by:.2f}" '
                   f'width="{bw - 1.2:.2f}" height="{bh:.2f}" '
                   f'fill="{accent_light}" stroke="{accent_dark}" '
                   f'stroke-width="0.8"/>')

    # Optional smooth normal-density overlay
    if overlay_curve is not None:
        path_pts = []
        for i in range(120):
            t = i / 119
            x_data = lo + t * (hi - lo)
            y_data = overlay_curve(x_data)
            # Density scaled to bin width and N so the curve sits on top of bars
            bin_width = (hi - lo) / nb
            expected_count = y_data * N * bin_width
            ny = py + ph - (expected_count / y_max) * ph
            nx = px + t * pw
            path_pts.append(f'{nx:.1f},{ny:.1f}')
        out.append(f'<polyline points="{" ".join(path_pts)}" fill="none" '
                   f'stroke="{AMBER}" stroke-width="2"/>')

    # X-axis line and ticks
    out.append(f'<line x1="{px}" y1="{py + ph}" x2="{px + pw}" y2="{py + ph}" '
               f'stroke="{INK}" stroke-width="1"/>')
    for tv in xticks:
        tx = px + (tv - lo) / (hi - lo) * pw
        out.append(f'<line x1="{tx:.1f}" y1="{py + ph}" x2="{tx:.1f}" '
                   f'y2="{py + ph + 5}" stroke="{INK}" stroke-width="1"/>')
        label = f'{tv:g}' if tv != int(tv) else f'{int(tv)}'
        out.append(f'<text x="{tx:.1f}" y="{py + ph + 18}" text-anchor="middle" '
                   f'font-size="10" font-family="{FONT}" fill="{SLATE}">{label}</text>')

    # Stats box
    if mean is not None and std is not None:
        bx0 = x + PANEL_W - 138
        by0 = y + 36
        out.append(f'<rect x="{bx0}" y="{by0}" width="124" height="40" rx="8" '
                   f'fill="{AMBER_L}" stroke="{AMBER}" stroke-width="1.4"/>')
        out.append(f'<text x="{bx0 + 62}" y="{by0 + 17}" text-anchor="middle" '
                   f'font-size="11" font-family="{FONT}" fill="{AMBER}">'
                   f'mean = {mean:+.3f}</text>')
        out.append(f'<text x="{bx0 + 62}" y="{by0 + 32}" text-anchor="middle" '
                   f'font-size="11" font-family="{FONT}" fill="{AMBER}">'
                   f'std  = {std:.3f}</text>')

    # Bottom caption inside panel
    out.append(f'<text x="{x + PANEL_W/2}" y="{y + PANEL_H - 8}" '
               f'text-anchor="middle" font-size="11" fill="{SLATE}">'
               f'value of sample</text>')

    return "\n".join(out)


# Compute empirical stats
def stats(samples):
    n = len(samples)
    m = sum(samples) / n
    v = sum((s - m) ** 2 for s in samples) / n
    return m, math.sqrt(v)

uni_m, uni_s = stats(uniform)
nrm_m, nrm_s = stats(normal)

# Standard normal pdf for overlay
def std_normal(x):
    return math.exp(-x * x / 2) / math.sqrt(2 * math.pi)

svg.append(draw_panel(L_X, PANEL_Y, "rand([10000])  →  Uniform(0, 1)",
                      PURPLE, PURPLE_L, uni_hist, UNI_LO, UNI_HI,
                      [0.0, 0.25, 0.5, 0.75, 1.0],
                      overlay_curve=None, mean=uni_m, std=uni_s))

svg.append(draw_panel(R_X, PANEL_Y, "randn([10000])  →  N(0, 1)",
                      TEAL, TEAL_L, nrm_hist, NRM_LO, NRM_HI,
                      [-4, -2, 0, 2, 4],
                      overlay_curve=std_normal, mean=nrm_m, std=nrm_s))

# Footer line
svg.append(f'<text x="{W/2}" y="{H - 14}" text-anchor="middle" '
           f'font-size="11" fill="{SLATE}" font-style="italic">'
           f'amber curve on the right panel: ideal N(0, 1) density.  '
           f'amber stat box: empirical mean and std of the 10 000 samples.</text>')

svg.append('</svg>')

with open(OUT_FILE, "w") as f:
    f.write("\n".join(svg))

print(f"wrote {OUT_FILE}")
