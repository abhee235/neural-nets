# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

A 30-chapter, build-from-scratch course implementing a transformer library in TypeScript using Bun. **No external ML, tensor, math, tokenizer, or plotting libraries.** `bun:test` is the only allowed testing tool. The goal is to understand every layer — from raw tensor arithmetic to GPT-style autoregressive generation.

See `docs/` for chapter content, `.github/instructions/` for code style rules. The tutor role, chapter map, and chapter-generation protocol live in `AGENTS.md` (read it when generating or implementing chapters).

## Slash Commands (type these in Claude Code)

| Command | What it does |
|---|---|
| `/chapter 04 matrix-ops` | Scaffold a new chapter stub (two files: index.ts + index.test.ts) |
| `/implement 04` | Implement a chapter stub from its doc, then run tests |
| `/test 04` | Run chapter tests and give a structured pass/fail report |
| `/explain broadcasting` | Tutor-style explanation: intuition → math → transformer connection → example → pitfalls |
| `/verify 04` | Check verification gate before moving to the next chapter |

## Commands

```bash
bun test                                   # run all tests
bun test src/tensor/linalg                 # run tests for a specific file (prefix match)
bun test --watch                           # watch mode
bun run exercises/ch-04-matrix-ops.ts      # run an exercise file directly
```

## Architecture

The library is built in strict dependency order:

```
src/tensor/     → src/autograd/     → src/nn/
                                      src/optim/
                                      src/tokenizer/
                                      src/utils/
```

**`src/tensor/`** — Foundation (Ch 01–06). `Tensor` is a plain interface: `{ data: Float64Array, shape: number[], ndim, size }` stored row-major. Modules: `types.ts` (core type + `flatIndex`), `creation.ts` (zeros/ones/randn via Box-Muller/linspace), `ops.ts` (elementwise + broadcasting), `math.ts` (exp/log/sqrt/tanh/clip), `linalg.ts` (matMul/matMulBatch/transpose/reshape/concat/stack), `reduce.ts` (sum/mean/max along axes with `keepDims`).

**`src/autograd/`** — Two-level autodiff (Ch 07–10). `value.ts` implements scalar `Value` nodes with a computation graph and `backward()`. `engine.ts` provides `topoSort()`. `grad.ts` lifts this to `TensorValue`: wraps a `Tensor` with `grad: Tensor | null` and the same graph machinery. **`TensorValue` is what all neural net layers operate on.**

**`src/nn/`** — Neural net layers and transformer (Ch 11–30). All layers take `TensorValue` inputs and expose `parameters(): TensorValue[]`. Key files: `linear.ts` (Q/K/V projections + FFN), `activations.ts` (ReLU/GELU/softmax), `losses.ts` (cross-entropy with log-sum-exp trick), `layernorm.ts`, `dropout.ts`, `embedding.ts`, `positional.ts` (sinusoidal PE), `attention.ts` (self/multi-head/cross), `feedforward.ts`, `transformer.ts` (encoder+decoder blocks), `gpt.ts` (decoder-only).

**`src/optim/`** — `sgd.ts` and `adam.ts`. Both accept `parameters: TensorValue[]` and a learning rate.

**`src/tokenizer/`** — `char.ts` (character-level), `bpe.ts` (byte-pair encoding), `masks.ts` (causal mask, padding mask).

**`src/utils/`** — `numerical.ts` (numerical gradient check helpers), `data.ts` (batch/shuffle utilities).

## Code Style (from `.github/instructions/chapter-code.instructions.md`)

**Comments ARE the documentation** — every exported function must have a JSDoc block explaining what it does and where it appears in the transformer. This is the opposite of the default "write no comments" rule. Inline math comments are required for formulas.

**Required file header** for every `index.ts`:
```typescript
/**
 * CHAPTER N: Title
 * ════════════════════════════════════════
 * Part X of 6: Part Name
 *
 * WHAT WE'RE BUILDING:  [what this exports]
 * WHY IT MATTERS:       [which later chapter calls this and why]
 * WHAT THIS UNLOCKS:    → Ch N+1 (Next Chapter Title)
 *
 * REFERENCE: docs/part-X-partname/ch-NN-chaptername.md
 */
```

**Naming conventions:**
- Functions: ML/math concepts — `matMul` not `multiply`, `softmax` not `normalize`
- Variables: full names — `learningRate` not `lr`, `batchSize`, `seqLen`, `dModel`, `numHeads`
- Types and interfaces go before functions in each file

**No shortcuts:**
- Box-Muller for normal distribution (no `Math.random()` scaled)
- No `.reduce()` hiding an algorithm; make the loop explicit
- No `as any` or `as unknown` casts
- All functions throw `new Error("Not implemented — read the chapter doc first")` as stubs

## Test Style (from `.github/instructions/test-style.instructions.md`)

```typescript
// Tests serve verification AND education.
// Name tests after mathematical properties:
it("is associative: (AB)C === A(BC)", () => { ... });         // ✓
it("returns a number", () => { ... });                         // ✗

// One describe per exported function; nested for sub-behaviors.
// Add a comment above each expect explaining the mathematical fact.
// Use it.todo(...) for unimplemented tests.
// Floating-point comparisons use a tolerance helper (EPSILON = 1e-6).
// Ch 07+: every differentiable op needs a numerical gradient check test.
```

## TypeScript Notes

Strict mode with `noUncheckedIndexedAccess` — array accesses return `T | undefined` and must be narrowed (`shape[i]!` or nullish coalescing). Module resolution is `bundler` (Bun-native); imports use explicit `.ts` extensions.

## Branch Model

Each chapter lives on its own branch:
- `starter/ch-NN-*` — stubs + tests (student starting point)
- `solution/ch-NN-*` — reference solution

Current branch: `starter/ch-04-matrix-ops`

### Branch sync protocol — MUST follow to avoid divergence

The course is built strictly cumulatively. Every chapter branch must start from the latest `main`, and every finished chapter must be merged back to `main` before the next chapter branch is created. Skipping this caused a real divergence in Ch 04 (the branch was cut from the original scaffold and missed ch-01/02/03 content that lived only on `main`).

**Starting a new chapter branch (e.g. ch-05):**

```bash
git checkout main && git pull origin main          # 1. main must be up to date FIRST
git checkout -b starter/ch-05-reductions main      # 2. branch FROM main, never from an old commit
```

**Finishing a chapter branch (before starting the next):**

```bash
# on the chapter branch
git push -u origin starter/ch-NN-*                 # 1. push the branch
# open PR, merge to main (squash or merge commit — both fine)
# only AFTER main has the merge:
git checkout main && git pull origin main          # 2. pull the merged state locally
# now main carries this chapter + all prior chapters — the next branch can start
```

**Red flags to check before any chapter work:**
- `git merge-base HEAD main` should equal a recent main commit, not an old scaffold commit
- `git diff main --stat docs/ src/` should show only the in-progress chapter's files. If unrelated chapters differ, this branch is stale and must rebase onto main first.

## Simulations and Visual Assets

The course is a **book** — readers read the `.md` files in VS Code Markdown Preview, they do not run anything. All visuals must render inline in the doc. No terminal simulations. No "run this script to see the output."

**Rule: every animation or diagram lives as an SVG file in `docs/assets/ch-NN/` and is embedded in the chapter `.md` via an `<img>` or inline `<svg>` tag.**

### The only tool: SMIL-animated SVG

VS Code Markdown Preview renders SVGs natively, including SMIL animations (`<animate>`, `<animateTransform>`). No browser, no extension, no npm package needed.

- File location: `docs/assets/ch-NN/description.svg`
- Embed in doc: `<img src="../assets/ch-NN/description.svg" alt="..." />`
- No external libraries, no JS inside the SVG — pure declarative SMIL
- See `docs/assets/ch-04/matmul-animation.svg` as the reference

### 3D in SVG (isometric projection)

SVG is 2D, but isometric projection creates convincing 3D. Project `(x, y, z)` → `(u, v)`:

```
u = (x - z) * cos(30°)  ≈  (x - z) * 0.866
v = (x + z) * sin(30°) - y  ≈  (x + z) * 0.5 - y
```

Draw each face as a `<polygon>`. Animate with `<animate>` on `fill` or `<animateTransform>` for rotation. This is how tensor "cube" visualizations work — a 3D tensor appears as a stack of parallelogram-faced planes.

## Claude Code Config

`.claude/settings.json` — pre-approves `bun test`, `bun run`, `git` read commands, and `ls`/`find` to reduce permission prompts. A PostToolUse hook prints a reminder to run tests after editing a `.ts` source file.

`AGENTS.md` is **not** auto-loaded by Claude Code (Claude reads only `CLAUDE.md`). It is a detailed reference — read it explicitly when you need the full chapter map, tutor role protocol, or chapter generation template.
