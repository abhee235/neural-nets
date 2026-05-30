"""
Chapter: 02 — Tensor Creation
Figure:  box-muller-geometry.svg
Concept: Box-Muller as a 3-step geometric pipeline:
         (u1, u2) in unit square  →  (r, θ) on disk  →  (z0, z1) in normal plane.
Output:  docs/assets/ch-02/box-muller-geometry.svg

Teaches: The transform is not magic — it is uniform-square → polar → cartesian,
         where r = sqrt(-2 ln u1) is exactly the radius distribution that produces
         a 2-D rotationally symmetric standard normal.

Run:
    python3 scripts/media/ch-02-box-muller-geometry.py
"""

import math
import os
import random

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "assets", "ch-02")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "box-muller-geometry.svg")

# palette
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

# canvas
W, H = 1080, 540
svg = []
svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
           f'font-family="{SERIF}">')
svg.append(f'<rect width="{W}" height="{H}" fill="{PAPER}"/>')

# arrow marker
svg.append('<defs>')
svg.append(f'<marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" '
           f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
           f'<path d="M0,0 L10,5 L0,10 Z" fill="{SLATE}"/></marker>')
svg.append(f'<marker id="arrAmber" viewBox="0 0 10 10" refX="9" refY="5" '
           f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
           f'<path d="M0,0 L10,5 L0,10 Z" fill="{AMBER}"/></marker>')
svg.append(f'<marker id="arrPurple" viewBox="0 0 10 10" refX="9" refY="5" '
           f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
           f'<path d="M0,0 L10,5 L0,10 Z" fill="{PURPLE}"/></marker>')
svg.append('</defs>')

# title
svg.append(f'<text x="{W/2}" y="40" text-anchor="middle" font-size="20" '
           f'font-weight="bold" fill="{INK}">Box–Muller geometry  ·  uniform → polar → cartesian</text>')
svg.append(f'<text x="{W/2}" y="60" text-anchor="middle" font-size="12" '
           f'fill="{SLATE}">One concrete sample, shown in three coordinate systems.  '
           f'u₁ = 0.500, u₂ = 0.250.</text>')

# concrete numbers we will trace through all three panels
u1, u2 = 0.500, 0.250
r = math.sqrt(-2 * math.log(u1))      # ≈ 1.177
theta = 2 * math.pi * u2               # = π/2
z0 = r * math.cos(theta)               # ≈ 0
z1 = r * math.sin(theta)               # ≈ 1.177

PANEL_W = 320
PANEL_H = 360
PAD = 24
PY = 90
P1_X = PAD
P2_X = PAD + PANEL_W + PAD + 18
P3_X = PAD + 2 * (PANEL_W + PAD) + 36

def panel_frame(x, y, title, subtitle):
    return [
        f'<rect x="{x}" y="{y}" width="{PANEL_W}" height="{PANEL_H}" rx="14" '
        f'fill="#ffffff" stroke="{GRID_S}" stroke-width="1.2"/>',
        f'<text x="{x + PANEL_W/2}" y="{y + 24}" text-anchor="middle" '
        f'font-size="14" font-weight="bold" font-family="{FONT}" fill="{INK}">{title}</text>',
        f'<text x="{x + PANEL_W/2}" y="{y + 42}" text-anchor="middle" '
        f'font-size="11" fill="{SLATE}">{subtitle}</text>',
    ]

# ── panel 1: unit square (u1, u2) ───────────────────────────────────────────
def panel_uniform(x, y):
    out = panel_frame(x, y, "1.  Pick (u₁, u₂)", "uniformly from the unit square")
    # plotting frame
    sw = 220
    sx = x + (PANEL_W - sw) / 2
    sy = y + 70
    out.append(f'<rect x="{sx}" y="{sy}" width="{sw}" height="{sw}" '
               f'fill="{PURPLE_L}" stroke="{PURPLE}" stroke-width="1.4" rx="4"/>')
    # gridlines at 0.25 spacing
    for i in range(1, 4):
        gx = sx + i * sw / 4
        out.append(f'<line x1="{gx}" y1="{sy}" x2="{gx}" y2="{sy + sw}" '
                   f'stroke="{PURPLE}" stroke-width="0.6" opacity="0.35"/>')
        gy = sy + i * sw / 4
        out.append(f'<line x1="{sx}" y1="{gy}" x2="{sx + sw}" y2="{gy}" '
                   f'stroke="{PURPLE}" stroke-width="0.6" opacity="0.35"/>')
    # axes labels
    out.append(f'<text x="{sx - 10}" y="{sy + sw + 4}" text-anchor="end" '
               f'font-size="10" font-family="{FONT}" fill="{SLATE}">0</text>')
    out.append(f'<text x="{sx - 10}" y="{sy + 4}" text-anchor="end" '
               f'font-size="10" font-family="{FONT}" fill="{SLATE}">1</text>')
    out.append(f'<text x="{sx + sw}" y="{sy + sw + 18}" text-anchor="middle" '
               f'font-size="10" font-family="{FONT}" fill="{SLATE}">1</text>')
    out.append(f'<text x="{sx}" y="{sy + sw + 18}" text-anchor="middle" '
               f'font-size="10" font-family="{FONT}" fill="{SLATE}">0</text>')
    # axis names
    out.append(f'<text x="{sx + sw/2}" y="{sy + sw + 32}" text-anchor="middle" '
               f'font-size="11" fill="{PURPLE}" font-family="{FONT}">u₁</text>')
    out.append(f'<text x="{sx - 26}" y="{sy + sw/2}" text-anchor="middle" '
               f'font-size="11" fill="{PURPLE}" font-family="{FONT}" '
               f'transform="rotate(-90 {sx - 26} {sy + sw/2})">u₂</text>')
    # the sampled point
    px = sx + u1 * sw
    py = sy + (1 - u2) * sw
    out.append(f'<line x1="{sx}" y1="{py}" x2="{px}" y2="{py}" '
               f'stroke="{AMBER}" stroke-width="1.2" stroke-dasharray="4 3"/>')
    out.append(f'<line x1="{px}" y1="{sy + sw}" x2="{px}" y2="{py}" '
               f'stroke="{AMBER}" stroke-width="1.2" stroke-dasharray="4 3"/>')
    out.append(f'<circle cx="{px}" cy="{py}" r="6" fill="{AMBER}" '
               f'stroke="#ffffff" stroke-width="1.4"/>')
    out.append(f'<text x="{px + 10}" y="{py - 8}" font-size="11" '
               f'font-family="{FONT}" fill="{AMBER}">(0.50, 0.25)</text>')
    # answer box at bottom
    bx = x + 30
    by = y + PANEL_H - 50
    out.append(f'<rect x="{bx}" y="{by}" width="{PANEL_W - 60}" height="36" '
               f'rx="8" fill="{AMBER_L}" stroke="{AMBER}" stroke-width="1.4"/>')
    out.append(f'<text x="{x + PANEL_W/2}" y="{by + 22}" text-anchor="middle" '
               f'font-size="13" font-family="{FONT}" fill="{AMBER}">'
               f'u₁ = 0.500   ·   u₂ = 0.250</text>')
    return "\n".join(out)

# ── panel 2: polar disk ─────────────────────────────────────────────────────
def panel_polar(x, y):
    out = panel_frame(x, y, "2.  Re-read as polar (r, θ)",
                      "r = √(−2 ln u₁)   ·   θ = 2π u₂")
    cx = x + PANEL_W / 2
    cy = y + 70 + 110     # disk center
    R = 100               # display radius (covers up to ~r=2.5σ)
    # gridlines: rings at 1, 2, 3 sigma equivalents
    R_PER_SIGMA = R / 2.5
    for k, label in [(1, "1σ"), (2, "2σ")]:
        out.append(f'<circle cx="{cx}" cy="{cy}" r="{k*R_PER_SIGMA:.1f}" '
                   f'fill="none" stroke="{TEAL}" stroke-width="0.8" '
                   f'opacity="0.45" stroke-dasharray="3 3"/>')
        out.append(f'<text x="{cx + k*R_PER_SIGMA + 4}" y="{cy - 4}" '
                   f'font-size="9" font-family="{FONT}" fill="{TEAL}">{label}</text>')
    # outer disk frame (lightly)
    out.append(f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="{TEAL_L}" '
               f'stroke="{TEAL}" stroke-width="1.2" opacity="0.55"/>')
    # angle reference: x axis from center
    out.append(f'<line x1="{cx - R}" y1="{cy}" x2="{cx + R}" y2="{cy}" '
               f'stroke="{SLATE}" stroke-width="0.8" stroke-dasharray="2 3"/>')
    out.append(f'<line x1="{cx}" y1="{cy + R}" x2="{cx}" y2="{cy - R}" '
               f'stroke="{SLATE}" stroke-width="0.8" stroke-dasharray="2 3"/>')
    # the sampled radius / angle
    px = cx + r * R_PER_SIGMA * math.cos(theta)
    py = cy - r * R_PER_SIGMA * math.sin(theta)
    # radius arrow
    out.append(f'<line x1="{cx}" y1="{cy}" x2="{px}" y2="{py}" '
               f'stroke="{AMBER}" stroke-width="2.2" marker-end="url(#arrAmber)"/>')
    # angle arc (from +x axis ccw to theta)
    arc_r = 30
    ax2 = cx + arc_r * math.cos(theta)
    ay2 = cy - arc_r * math.sin(theta)
    out.append(f'<path d="M {cx + arc_r} {cy} A {arc_r} {arc_r} 0 0 0 '
               f'{ax2:.1f} {ay2:.1f}" fill="none" stroke="{AMBER}" '
               f'stroke-width="1.4"/>')
    out.append(f'<text x="{cx + arc_r + 4}" y="{cy - 12}" font-size="11" '
               f'font-family="{FONT}" fill="{AMBER}">θ</text>')
    out.append(f'<text x="{(cx + px)/2 - 22}" y="{(cy + py)/2 - 6}" '
               f'font-size="11" font-family="{FONT}" fill="{AMBER}">r</text>')
    # endpoint dot
    out.append(f'<circle cx="{px}" cy="{py}" r="6" fill="{AMBER}" '
               f'stroke="#ffffff" stroke-width="1.4"/>')
    # answer box
    bx = x + 18
    by = y + PANEL_H - 64
    out.append(f'<rect x="{bx}" y="{by}" width="{PANEL_W - 36}" height="50" '
               f'rx="8" fill="{AMBER_L}" stroke="{AMBER}" stroke-width="1.4"/>')
    out.append(f'<text x="{x + PANEL_W/2}" y="{by + 20}" text-anchor="middle" '
               f'font-size="12.5" font-family="{FONT}" fill="{AMBER}">'
               f'r = √(−2 ln 0.5) ≈ {r:.3f}</text>')
    out.append(f'<text x="{x + PANEL_W/2}" y="{by + 38}" text-anchor="middle" '
               f'font-size="12.5" font-family="{FONT}" fill="{AMBER}">'
               f'θ = 2π · 0.25 = π/2</text>')
    return "\n".join(out)

# ── panel 3: cartesian (z0, z1) ─────────────────────────────────────────────
def panel_cartesian(x, y):
    out = panel_frame(x, y, "3.  Project to cartesian",
                      "z₀ = r cos θ   ·   z₁ = r sin θ")
    cx = x + PANEL_W / 2
    cy = y + 70 + 110
    R = 100
    R_PER_SIGMA = R / 2.5
    # 1σ, 2σ density rings
    for k, label in [(1, "1σ"), (2, "2σ")]:
        out.append(f'<circle cx="{cx}" cy="{cy}" r="{k*R_PER_SIGMA:.1f}" '
                   f'fill="none" stroke="{AMBER}" stroke-width="1" '
                   f'opacity="0.45" stroke-dasharray="3 3"/>')
        out.append(f'<text x="{cx + k*R_PER_SIGMA + 4}" y="{cy - 4}" '
                   f'font-size="9" font-family="{FONT}" fill="{AMBER}">{label}</text>')
    # axes
    out.append(f'<line x1="{cx - R - 8}" y1="{cy}" x2="{cx + R + 8}" y2="{cy}" '
               f'stroke="{INK}" stroke-width="1" marker-end="url(#arr)"/>')
    out.append(f'<line x1="{cx}" y1="{cy + R + 8}" x2="{cx}" y2="{cy - R - 8}" '
               f'stroke="{INK}" stroke-width="1" marker-end="url(#arr)"/>')
    out.append(f'<text x="{cx + R + 16}" y="{cy + 4}" font-size="11" '
               f'font-family="{FONT}" fill="{INK}">z₀</text>')
    out.append(f'<text x="{cx - 4}" y="{cy - R - 14}" text-anchor="end" '
               f'font-size="11" font-family="{FONT}" fill="{INK}">z₁</text>')
    # tick marks at -2, -1, 1, 2
    for k in [-2, -1, 1, 2]:
        tx = cx + k * R_PER_SIGMA
        out.append(f'<line x1="{tx}" y1="{cy - 3}" x2="{tx}" y2="{cy + 3}" '
                   f'stroke="{INK}" stroke-width="1"/>')
        out.append(f'<text x="{tx}" y="{cy + 16}" text-anchor="middle" '
                   f'font-size="9" font-family="{FONT}" fill="{SLATE}">{k}</text>')
        ty = cy - k * R_PER_SIGMA
        out.append(f'<line x1="{cx - 3}" y1="{ty}" x2="{cx + 3}" y2="{ty}" '
                   f'stroke="{INK}" stroke-width="1"/>')
        out.append(f'<text x="{cx - 8}" y="{ty + 4}" text-anchor="end" '
                   f'font-size="9" font-family="{FONT}" fill="{SLATE}">{k}</text>')

    # the sampled point
    px = cx + z0 * R_PER_SIGMA
    py = cy - z1 * R_PER_SIGMA
    # dashed projections to axes
    out.append(f'<line x1="{cx}" y1="{py}" x2="{px}" y2="{py}" '
               f'stroke="{AMBER}" stroke-width="1.2" stroke-dasharray="4 3"/>')
    out.append(f'<line x1="{px}" y1="{cy}" x2="{px}" y2="{py}" '
               f'stroke="{AMBER}" stroke-width="1.2" stroke-dasharray="4 3"/>')
    out.append(f'<circle cx="{px}" cy="{py}" r="6" fill="{AMBER}" '
               f'stroke="#ffffff" stroke-width="1.4"/>')
    out.append(f'<text x="{px + 10}" y="{py - 8}" font-size="11" '
               f'font-family="{FONT}" fill="{AMBER}">(z₀, z₁)</text>')
    # answer box
    bx = x + 18
    by = y + PANEL_H - 64
    out.append(f'<rect x="{bx}" y="{by}" width="{PANEL_W - 36}" height="50" '
               f'rx="8" fill="{AMBER_L}" stroke="{AMBER}" stroke-width="1.4"/>')
    out.append(f'<text x="{x + PANEL_W/2}" y="{by + 20}" text-anchor="middle" '
               f'font-size="12.5" font-family="{FONT}" fill="{AMBER}">'
               f'z₀ = {z0:+.3f}   ·   z₁ = {z1:+.3f}</text>')
    out.append(f'<text x="{x + PANEL_W/2}" y="{by + 38}" text-anchor="middle" '
               f'font-size="11" font-family="{FONT}" fill="{AMBER}">'
               f'two independent N(0, 1) samples</text>')
    return "\n".join(out)

svg.append(panel_uniform(P1_X, PY))
svg.append(panel_polar(P2_X, PY))
svg.append(panel_cartesian(P3_X, PY))

# arrows between panels
A_Y = PY + PANEL_H / 2
def between_arrow(x1, x2, label):
    mid = (x1 + x2) / 2
    return (f'<line x1="{x1+4}" y1="{A_Y}" x2="{x2-4}" y2="{A_Y}" '
            f'stroke="{SLATE}" stroke-width="1.6" marker-end="url(#arr)"/>'
            f'<text x="{mid}" y="{A_Y - 8}" text-anchor="middle" '
            f'font-size="10" font-family="{FONT}" fill="{SLATE}">{label}</text>')

svg.append(between_arrow(P1_X + PANEL_W, P2_X, "polar"))
svg.append(between_arrow(P2_X + PANEL_W, P3_X, "cartesian"))

# footer
svg.append(f'<text x="{W/2}" y="{H - 12}" text-anchor="middle" '
           f'font-size="11" fill="{SLATE}" font-style="italic">'
           f'The amber dot is the same sample throughout — only its coordinate frame changes.</text>')

svg.append('</svg>')
with open(OUT_FILE, "w") as f:
    f.write("\n".join(svg))
print(f"wrote {OUT_FILE}")
