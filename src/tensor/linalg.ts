/**
 * tensor/linalg.ts
 * ══════════════════════════════════════════════════════════
 * Linear algebra: matmul, transpose, reshape, flatten, squeeze/unsqueeze.
 * These are the structural ops used in every attention and linear layer.
 *
 * Chapter: 04 — Matrix Operations
 * Doc:     docs/part-1-tensor-library/ch-04-matrix-ops.md
 */
import type { Tensor } from "./types.ts";

/**
 * Matrix multiply.
 *   2-D:  (M,K) × (K,N) → (M,N)
 *   N-D:  batched matmul on the last two axes
 *
 * Math: C[i,j] = Σ_k A[i,k] * B[k,j]
 *
 * This is the dominant compute operation in transformers —
 * every Q/K/V projection and FFN layer is a matMul.
 */
export function matMul(a: Tensor, b: Tensor): Tensor {
  throw new Error("matMul not implemented");
}

/**
 * Permute tensor axes. Default: reverse all axes.
 * Pass axes to specify a custom permutation.
 *
 * Used to swap head/seq dims in multi-head attention.
 */
export function transpose(t: Tensor, axes?: number[]): Tensor {
  throw new Error("transpose not implemented");
}

/**
 * Return a view with a new shape. Total element count must be unchanged.
 * Pass -1 for at most one dimension to infer its size automatically.
 */
export function reshape(t: Tensor, newShape: number[]): Tensor {
  throw new Error("reshape not implemented");
}

/** Collapse all axes from startAxis onward into a single dimension. */
export function flatten(t: Tensor, startAxis?: number): Tensor {
  throw new Error("flatten not implemented");
}

/** Remove a size-1 dimension at the given axis. */
export function squeeze(t: Tensor, axis: number): Tensor {
  throw new Error("squeeze not implemented");
}

/**
 * Insert a size-1 dimension at the given axis.
 * Used to go [batch, seq, dModel] → [batch, 1, seq, dModel] for masks.
 */
export function unsqueeze(t: Tensor, axis: number): Tensor {
  throw new Error("unsqueeze not implemented");
}
