---
description: "Use when adding, reviewing, or referencing visuals (diagrams, graphs, GIFs) in chapter docs. Defines folder layout, format priority, generation-script rules, and Markdown reference syntax for the docs media system."
applyTo: "docs/**/*.md"
---

# Chapter Media Standard

The course is plain Markdown plus **static media files**. No MDX, no embedded apps,
no JavaScript-driven charts in `.md`. Chapter docs stay clean; generation logic
lives elsewhere.

This file is the source of truth. When adding or reviewing visuals in any chapter,
follow it exactly.

---

## Where things live

```text
docs/
  part-N-name/
    ch-NN-name.md          ← chapter prose (this file you edit)
  assets/
    ch-NN/                 ← rendered media for that chapter
      *.svg | *.png | *.gif
      media.json           ← optional per-asset metadata

scripts/
  media/
    ch-NN-short-topic.py   ← reproducible generator that writes into docs/assets/ch-NN/
    README.md              ← script conventions
```

- Chapter `.md` references only **finished assets** under `docs/assets/ch-NN/`.
- Generation details (data, code, prompts) live in `scripts/media/` — never in the chapter.
- Per-asset captions or alt text may live in `docs/assets/ch-NN/media.json` if non-trivial.

---

## Format priority

1. **SVG** — first choice. Use for shape diagrams, computation graphs,
   tensor layouts, attention maps, masks, encoder/decoder data flow.
2. **PNG** — only when the figure is genuinely raster (e.g. a rendered numerical plot).
3. **GIF** — only when motion is essential (e.g. stepwise index walk).
   Keep file size small (< ~1 MB), keep frame count low.
4. **KaTeX math** — for formulas, written inline in `.md`. Never render math as an image.

Do **not** use:

- Embedded JavaScript or interactive widgets in `.md`.
- Mermaid as a hard dependency (VS Code preview support varies; treat as optional only).
- Video as a primary teaching mechanism.
- Decorative PNGs that do not teach a concept.

---

## How to reference media in a chapter

Use a Markdown image, path relative to the chapter file:

```markdown
![A 2x3 tensor laid out as a flat buffer](../assets/ch-01/tensor-flat-buffer.svg)

*Figure: the same six numbers, reinterpreted by shape.*
```

Rules:

- Provide a real **alt text** describing what the figure shows.
- Add a *Figure: ...* italic caption below when it helps the reader.
- Do not size with HTML unless layout genuinely requires it; prefer clean SVG dimensions.
- One figure per teaching idea. Avoid figure dumps.

---

## When to add a visual

Add a visual only when **one of these is true**:

- A shape or indexing relationship is hard to picture (Ch 01–06, Ch 18–21).
- A data flow is multi-step (autograd graphs, encoder/decoder, attention).
- A numerical effect needs to be seen (softmax temperature, attention weights, gradient flow).
- A mask pattern is structural (causal mask, padding mask).

Do **not** add visuals to decorate prose. Each figure must teach.

---

## Generation script rules

Every non-trivial figure must have a matching generator in `scripts/media/`:

- Filename: `ch-NN-short-topic.py` (or `.ts`).
- Output path: `docs/assets/ch-NN/<asset>.svg` (or `.png` / `.gif`).
- Deterministic: fixed seeds, no network calls.
- Re-runnable: overwrites output, does not append.

Generation scripts may use any tooling (matplotlib, svgwrite, cairosvg, pillow).
The **zero-dependency constraint applies to `src/`**, not to media tooling.

For details, see `scripts/media/README.md`.

---

## Optional metadata

If a chapter has multiple figures with non-trivial sourcing, add
`docs/assets/ch-NN/media.json`:

```json
{
  "tensor-grid.svg": {
    "alt": "2x3 tensor laid out as a grid with row and column labels",
    "caption": "A tensor as a shape-labeled view of a flat buffer.",
    "source": "scripts/media/ch-01-tensor-grid.py"
  }
}
```

Skip the metadata file for trivial single figures.

---

## Visual conventions (shared palette + answer-box pattern)

All chapter media should share a consistent visual language so figures across
chapters feel like pages from the same book.

**Shared palette** (use these exact hex values):

| Token     | Hex       | Role                                          |
|-----------|-----------|-----------------------------------------------|
| PAPER     | `#faf7f0` | Background fill                               |
| INK       | `#1d1d1b` | Primary text, default strokes                 |
| SLATE     | `#64748b` | Captions, secondary labels                    |
| PURPLE    | `#7c3aed` | Axis 0 / leading dimension                    |
| PURPLE_L  | `#ede9fe` | Axis 0 light fill                             |
| TEAL      | `#0f766e` | Axis 1                                        |
| TEAL_L    | `#d8f0eb` | Axis 1 light fill                             |
| AMBER     | `#b45309` | Axis 2 / **highlight + answer-box border**    |
| AMBER_L   | `#fef3c7` | **Answer-box fill**                           |

**Answer-box pattern** (the SVG counterpart to KaTeX `\boxed{}` in prose):

When a figure presents a "key result" formula or computed answer, wrap it in
a tinted callout box so it reads as a landmark when the reader scrolls:

- Fill: `AMBER_L` (`#fef3c7`)
- Stroke: `AMBER` (`#b45309`), stroke-width 1.4–1.5
- Rounded corners: `rx=8–10`
- Formula font: monospace (`ui-monospace, SFMono-Regular, Menlo, monospace`),
  size 17 pt (filmstrip tiles) to 22 pt (full-canvas animations)
- Center the formula text vertically inside the box

This matches the `\Large \boxed{...}` treatment used for key equations in chapter
prose (see `.github/instructions/chapter-doc.instructions.md` → *Math Formatting Conventions*).

Reserve the answer box for **one or two formulas per figure**. If every label is
boxed, the highlight loses meaning.

---

## Forbidden patterns in `.md`

- `<script>` tags or inline JavaScript.
- MDX-style imports or components.
- External image URLs (always commit assets locally).
- Images without alt text.
- Embedding raw base64 image data inline.
- Putting generation code in the chapter file.

---

## Review checklist

When reviewing a chapter that contains visuals, verify:

1. Every image lives under `docs/assets/ch-NN/` for the right chapter.
2. Every reference uses a relative path (`../assets/ch-NN/...`).
3. Every image has descriptive alt text.
4. SVG is used unless raster or motion is genuinely required.
5. A matching generator exists in `scripts/media/` for non-trivial figures.
6. No interactive embedding, no remote URLs, no inline scripts.
</content>
</invoke>
<invoke name="multi_replace_string_in_file">
<parameter name="explanation">Update copilot-instructions.md to reference the new media instruction file