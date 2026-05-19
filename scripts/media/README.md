# Media Generation Scripts

Reproducible generators for chapter visuals.

## Conventions

- One file per figure or per closely-related figure group:
  `ch-NN-short-topic.py` or `ch-NN-short-topic.ts`.
- Output path always under `docs/assets/ch-NN/`.
- Deterministic: fix random seeds, no network calls.
- Re-runnable: overwrite the output file, do not append.
- Header comment must state:
  - chapter the figure belongs to
  - what concept the figure teaches
  - exact output path

## Tooling

Generators may use any tooling (matplotlib, svgwrite, cairosvg, pillow, plain Python,
or Bun/TypeScript). The **zero-dependency rule applies to `src/`, not to media scripts.**

## Running

```bash
python3 scripts/media/ch-01-tensor-grid.py
# or
bun run scripts/media/ch-01-tensor-grid.ts
```

The script writes directly into `docs/assets/ch-NN/`. Commit both the script and
the rendered asset.

## Full standard

See `.github/instructions/chapter-media.instructions.md`.
