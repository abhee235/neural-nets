/**
 * tensor/types.ts
 * ══════════════════════════════════════════════════════════
 * The core Tensor type: a flat Float64Array + shape descriptor.
 * Equivalent to numpy.ndarray's raw layout.
 *
 * Chapter: 01 — Tensor Type System
 * Doc:     docs/part-1-tensor-library/ch-01-tensor-type-system.md
 */

// ─── Core type ────────────────────────────────────────────────────────────────

/**
 * A multi-dimensional array stored in row-major (C-contiguous) flat layout.
 *
 * Example — 2×3 matrix:
 *   shape = [2, 3]
 *   data  = Float64Array [1,2,3, 4,5,6]
 *   element[i,j] lives at data[i * 3 + j]
 */
export interface Tensor {
  /** Flat storage, row-major order. */
  readonly data: Float64Array;
  /** Size of each dimension, e.g. [batch, seq, dModel]. */
  readonly shape: number[];
  /** Number of dimensions — shape.length. */
  readonly ndim: number;
  /** Total element count — product of all shape values. */
  readonly size: number;
}

// ─── Construction ────────────────────────────────────────────────────────────

/**
 * Create a Tensor from a flat number array and a shape.
 * Validates data.length === product(shape).
 */
export function createTensor(data: number[], shape: number[]): Tensor {
  throw new Error("createTensor not implemented — docs/part-1-tensor-library/ch-01-tensor-type-system.md");
}

/**
 * Create a rank-0 scalar Tensor (shape = [], size = 1).
 */
export function scalar(value: number): Tensor {
  throw new Error("scalar not implemented");
}

// ─── Utilities ────────────────────────────────────────────────────────────────

/**
 * Runtime type guard — narrows unknown to Tensor.
 */
export function isTensor(value: unknown): value is Tensor {
  throw new Error("isTensor not implemented");
}

/**
 * Convert a multi-dimensional index array to the flat offset into data[].
 * Row-major formula:  offset = Σ_i  indices[i] * stride[i]
 */
export function flatIndex(shape: number[], indices: number[]): number {
  throw new Error("flatIndex not implemented");
}

/**
 * Human-readable representation showing shape and sampled values.
 * Use this instead of console.log(tensor) for debugging.
 */
export function toString(t: Tensor): string {
  throw new Error("toString not implemented");
}
