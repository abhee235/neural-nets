---
description: "Use when writing, reviewing, or scaffolding TypeScript source files in src/**. Covers chapter header block comment format, function naming conventions, inline math comment standards, and export rules for this learning library."
applyTo: "src/**/*.ts"
---

# Chapter Code Standards

Every `.ts` file in `src/` is **implementation and documentation at the same time**.
A reader should be able to understand the math and the code by reading the file top to bottom,
like a textbook page. There is no separate documentation — the comments ARE the docs.

## File Header (Required)

Every `index.ts` must start with this block comment:

```typescript
/**
 * CHAPTER N: Chapter Title
 * ════════════════════════════════════════
 * Part X of 6: Part Name
 *
 * WHAT WE'RE BUILDING:
 *   One or two sentences. What TypeScript constructs and functions does this file export?
 *
 * WHY IT MATTERS:
 *   Connect this module to the transformer. Which later chapter will call these functions?
 *   Example: "matMul will be called hundreds of times per forward pass in Ch 22–26."
 *
 * WHAT THIS UNLOCKS:
 *   List the next chapter(s) that depend on this one.
 *   Example: "→ Ch 07 (Calculus for ML) uses Tensor as its primary type."
 *
 * REFERENCE: docs/part-X-partname/ch-NN-chaptername.md
 */
```

## Function Comments (Required)

Every exported function must have:

1. A JSDoc `/** */` block explaining WHAT it does and WHERE it's used in ML/transformers.
2. An inline math comment for non-trivial formulas (use ASCII math or LaTeX-style notation).
3. Named intermediate variables — never chain complex expressions into one line.

```typescript
/**
 * Matrix multiplication of two 2D tensors.
 *
 * This is THE core operation of neural networks. Every Linear layer, every
 * attention score computation, every projection is a matMul under the hood.
 *
 * Math: C[i][j] = Σ_k A[i][k] * B[k][j]
 * Shapes: (M, K) × (K, N) → (M, N)
 */
export function matMul(a: Tensor, b: Tensor): Tensor {
  // ... implementation
}
```

## Naming Rules

- Name functions after their **mathematical/ML concept**: `matMul` not `multiply`, `softmax` not `normalize`.
- Use full descriptive names in variables: `learningRate` not `lr`, `gradient` not `grad`.
- Shape dimension variables: `batchSize`, `seqLen`, `dModel`, `numHeads`, `vocabSize`.
- Loop indices: `i`, `j`, `k` are fine for standard matrix indices.

## Export Rules

- Export **every** function, type, and interface. Future chapters import from earlier ones.
- Put types/interfaces first, then creation functions, then computation functions.
- Re-export everything from `index.ts` so imports are always `from "../ch-01-.../index.ts"`.

## No Shortcuts

- Implement Box-Muller for normal distribution — do NOT just use `Math.random()` scaled.
- Do NOT use `.reduce()` to hide a loop whose algorithm should be explicitly visible.
- Do NOT use `as any` or `as unknown` casts to silence type errors — fix the types.
- The learning is in the explicit implementation, not in reaching the answer quickly.

## Print Helpers

Include a `toString(t: Tensor): string` or `print(t: Tensor): void` helper so shapes and
values can be inspected from the command line. This is essential for debugging.
