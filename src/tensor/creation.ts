/**
 * tensor/creation.ts
 * ══════════════════════════════════════════════════════════
 * Factory functions that produce Tensors filled with specific values.
 * Equivalent to numpy.zeros, numpy.ones, numpy.random.randn, numpy.eye.
 *
 * Chapter: 02 — Tensor Creation
 * Doc:     docs/part-1-tensor-library/ch-02-tensor-creation.md
 */
import type { Tensor } from "./types.ts";

/** Tensor filled with 0.0. Used to initialise bias vectors and masks. */
export function zeros(shape: number[]): Tensor {
  throw new Error("zeros not implemented");
}

/** Tensor filled with 1.0. Used for LayerNorm gamma initialisation. */
export function ones(shape: number[]): Tensor {
  throw new Error("ones not implemented");
}

/** Tensor filled with a constant value. */
export function fill(shape: number[], value: number): Tensor {
  throw new Error("fill not implemented");
}

/**
 * Tensor of N(0,1) samples using Box-Muller transform.
 * Do NOT use Math.random() scaled — implement Box-Muller explicitly:
 *   Z = sqrt(-2 ln U1) * cos(2π U2),  where U1, U2 ~ Uniform(0,1)
 *
 * Used to initialise weight matrices in Linear and Embedding layers.
 */
export function randn(shape: number[]): Tensor {
  throw new Error("randn not implemented — use Box-Muller, not Math.random() scaling");
}

/** n×n identity matrix. 1 on diagonal, 0 elsewhere. */
export function eye(n: number): Tensor {
  throw new Error("eye not implemented");
}

/**
 * 1-D tensor [start, start+step, ...] up to (not including) stop.
 * Like numpy.arange.
 */
export function arange(start: number, stop: number, step?: number): Tensor {
  throw new Error("arange not implemented");
}

/**
 * 1-D tensor of n evenly-spaced values from start to stop (inclusive).
 * Like numpy.linspace.
 */
export function linspace(start: number, stop: number, n: number): Tensor {
  throw new Error("linspace not implemented");
}

/** New tensor with the same shape as t, filled with value. */
export function fullLike(t: Tensor, value: number): Tensor {
  throw new Error("fullLike not implemented");
}
