/**
 * tensor/math.ts
 * ══════════════════════════════════════════════════════════
 * Elementwise math functions applied to tensors.
 * Equivalent to torch.exp, torch.log, torch.tanh, etc.
 *
 * Chapter: 06 — Math Primitives
 * Doc:     docs/part-1-tensor-library/ch-06-math-primitives.md
 */
import type { Tensor } from "./types.ts";

/** Elementwise e^x. Used in softmax, cross-entropy, and GELU. */
export function exp(t: Tensor): Tensor {
  throw new Error("exp not implemented");
}

/**
 * Elementwise ln(x). Guard against log(0) with a small epsilon clamp.
 * Used in cross-entropy loss.
 */
export function log(t: Tensor): Tensor {
  throw new Error("log not implemented");
}

/**
 * Elementwise √x.
 * Used in LayerNorm (÷ std) and attention scaling (÷ √dHead).
 */
export function sqrt(t: Tensor): Tensor {
  throw new Error("sqrt not implemented");
}

/** Elementwise x^exponent. */
export function pow(t: Tensor, exponent: number): Tensor {
  throw new Error("pow not implemented");
}

/** Elementwise |x|. */
export function abs(t: Tensor): Tensor {
  throw new Error("abs not implemented");
}

/**
 * Clamp every element to [min, max].
 * Used to prevent log(0) and clip activations during training.
 */
export function clip(t: Tensor, min: number, max: number): Tensor {
  throw new Error("clip not implemented");
}

/**
 * Elementwise tanh(x) = (e^x - e^{-x}) / (e^x + e^{-x}).
 * Output in (-1, 1). Used in the GELU approximation.
 */
export function tanh(t: Tensor): Tensor {
  throw new Error("tanh not implemented");
}

/**
 * Elementwise σ(x) = 1 / (1 + e^{-x}).
 * Output in (0, 1). Gradient: σ(x)(1 − σ(x)).
 */
export function sigmoid(t: Tensor): Tensor {
  throw new Error("sigmoid not implemented");
}
