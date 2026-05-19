/**
 * tensor/reduce.ts
 * ══════════════════════════════════════════════════════════
 * Axis-wise reductions: sum, mean, max, min, argmax, softmax.
 * Equivalent to torch.sum, torch.mean, torch.softmax, torch.argmax.
 *
 * Chapter: 05 — Reductions & Statistical Ops
 * Doc:     docs/part-1-tensor-library/ch-05-reductions.md
 */
import type { Tensor } from "./types.ts";

/**
 * Sum along an axis (or all elements when axis is omitted).
 * keepDims=true inserts a size-1 axis at the reduced position.
 */
export function sum(t: Tensor, axis?: number, keepDims?: boolean): Tensor {
  throw new Error("sum not implemented");
}

/**
 * Mean along an axis. Used in LayerNorm to compute per-token mean.
 * Math: mean(x) = sum(x) / n
 */
export function mean(t: Tensor, axis?: number, keepDims?: boolean): Tensor {
  throw new Error("mean not implemented");
}

/**
 * Maximum value along an axis.
 * Used in numerically-stable softmax (subtract max before exp).
 */
export function max(t: Tensor, axis?: number, keepDims?: boolean): Tensor {
  throw new Error("max not implemented");
}

/** Minimum value along an axis. */
export function min(t: Tensor, axis?: number, keepDims?: boolean): Tensor {
  throw new Error("min not implemented");
}

/**
 * Index of the maximum value along an axis.
 * Used during greedy decoding: argmax(logits) → predicted token ID.
 */
export function argmax(t: Tensor, axis?: number): Tensor {
  throw new Error("argmax not implemented");
}

/**
 * Numerically stable softmax:
 *   softmax(x)_i = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
 *
 * Output is a probability distribution summing to 1.
 * Final step of every attention head.
 */
export function softmax(t: Tensor, axis?: number): Tensor {
  throw new Error("softmax not implemented");
}
