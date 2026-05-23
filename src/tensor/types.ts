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
  // Compute the expected total size: product of all shape dimensions.
  // Empty shape [] means a scalar with size 1.
  const size = shape.length === 0 ? 1 : shape.reduce((acc, d) => acc * d, 1);

  if (data.length !== size) {
    throw new Error(
      `createTensor: data length ${data.length} does not match shape [${shape}] (expected ${size})`,
    );
  }

  return {
    data: new Float64Array(data),  // copy into typed array for numeric precision
    shape: [...shape],             // defensive copy so callers can't mutate
    ndim: shape.length,
    size,
  };
}

/**
 * Create a rank-0 scalar Tensor (shape = [], size = 1).
 */
export function scalar(value: number): Tensor {
  // A rank-0 tensor has no axes; its flat data holds exactly one element.
  return createTensor([value], []);
}

// ─── Utilities ────────────────────────────────────────────────────────────────

/**
 * Runtime type guard — narrows unknown to Tensor.
 */
export function isTensor(value: unknown): value is Tensor {
  if (value === null || typeof value !== "object") return false;
  const t = value as Record<string, unknown>;
  return (
    t["data"] instanceof Float64Array &&
    Array.isArray(t["shape"]) &&
    typeof t["ndim"] === "number" &&
    typeof t["size"] === "number"
  );
}

/**
 * Convert a multi-dimensional index array to the flat offset into data[].
 * Row-major formula:  offset = Σ_i  indices[i] * stride[i]
 */
export function flatIndex(shape: number[], indices: number[]): number {
  // Row-major (C-order) formula:
  //   flat = i_0*s_1*s_2*...*s_n  +  i_1*s_2*...*s_n  +  ...  +  i_n
  // Each axis contributes its index scaled by the product of all trailing dims.
  //
  // Example: shape=[2,3,4], indices=[1,2,3]
  //   stride_0 = 3*4 = 12,  stride_1 = 4,  stride_2 = 1
  //   flat = 1*12 + 2*4 + 3*1 = 23
  let flat = 0;
  let stride = 1;
  // Walk RIGHT to LEFT: accumulate stride as we go.
  for (let i = shape.length - 1; i >= 0; i--) {
    flat += (indices[i] as number) * stride;
    stride *= shape[i] as number;
  }
  return flat;
}

/**
 * Human-readable representation showing shape and sampled values.
 * Use this instead of console.log(tensor) for debugging.
 */
export function toString(t: Tensor): string {
  // Show shape and up to 8 values so the output is readable even for large tensors.
  const preview = Array.from(t.data.slice(0, 8)).map(v => v.toFixed(4));
  const suffix = t.size > 8 ? `, ... (${t.size} total)` : "";
  return `Tensor(shape=[${t.shape}], [${preview.join(", ")}${suffix}])`;
}
