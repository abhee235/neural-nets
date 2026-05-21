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
  if (data.length !== shape.reduce((a, b) => a * b, 1)) {
    throw new Error("Data length does not match shape product");
  }
  return {
    data: new Float64Array(data),
    shape,
    ndim: shape.length,
    size: data.length,
  };
}

/**
 * Create a rank-0 scalar Tensor (shape = [], size = 1).
 */
export function scalar(value: number): Tensor {
  return createTensor([value], []);
}

// ─── Utilities ────────────────────────────────────────────────────────────────

/**
 * Runtime type guard — narrows unknown to Tensor.
 */
export function isTensor(value: unknown): value is Tensor {
  return (
    typeof value === "object" &&
    value !== null &&
    "data" in value &&
    "shape" in value &&
    "ndim" in value &&
    "size" in value &&
    value.data instanceof Float64Array &&
    Array.isArray(value.shape) &&
    typeof value.ndim === "number" &&
    typeof value.size === "number"
  );
}

/**
 * Convert a multi-dimensional index array to the flat offset into data[].
 * Row-major formula:  offset = Σ_i  indices[i] * stride[i]
 */
export function flatIndex(shape: number[], indices: number[]): number {
  if (shape.length !== indices.length) {
    throw new Error("Shape and indices must have the same length");
  }
  let offset = 0;
  let stride = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    // noUncheckedIndexedAccess makes arr[i] return T | undefined even inside a
    // loop whose bounds we control. Read once into typed locals so the rest of
    // the loop body sees plain `number`.
    const dim = shape[i] as number;
    const idx = indices[i] as number;
    if (idx < 0 || idx >= dim) {
      throw new Error(`Index ${idx} out of bounds for dimension ${i} with size ${dim}`);
    }
    offset += idx * stride;
    stride *= dim;
  }
  return offset;
}

/**
 * Human-readable representation showing shape and sampled values.
 * Use this instead of console.log(tensor) for debugging.
 */
export function toString(t: Tensor): string {
  const sampleSize = Math.min(5, t.size);
  const sampleValues = Array.from(t.data.slice(0, sampleSize)).join(", ");
  return `Tensor(shape=[${t.shape.join(", ")}], data=[${sampleValues}${t.size > sampleSize ? ", ..." : ""}])`;
}
