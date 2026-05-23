/**
 * CHAPTER 04: Matrix Operations
 * ════════════════════════════════════════
 * Part 1 of 6: Tensor Library
 *
 * WHAT WE'RE BUILDING:
 *   matMul (2-D and batched), transpose, reshape, flatten, squeeze, unsqueeze,
 *   concat, and stack — the structural layer that every neural network layer
 *   sits on top of.
 *
 * WHY IT MATTERS:
 *   matMul alone accounts for the majority of compute in a transformer. Every
 *   linear projection, every attention score, every FFN layer is a matMul.
 *   The shape-manipulation helpers (reshape, unsqueeze, concat) are the glue
 *   that connects those matMuls together by keeping axes aligned.
 *
 * WHAT THIS UNLOCKS:
 *   → Ch 13 (Linear layer — forward pass is matMul + bias add)
 *   → Ch 21 (Masks — unsqueeze to create broadcastable mask dims)
 *   → Ch 22 (Self-attention — QK^T is matMulBatch(Q, transpose(K)))
 *   → Ch 23 (Multi-head attention — reshape to split heads, concat to merge)
 *
 * REFERENCE: docs/part-1-tensor-library/ch-04-matrix-ops.md
 */
import { createTensor, flatIndex, type Tensor } from "./types.ts";

// ─── helpers ────────────────────────────────────────────────────────────────

/**
 * Return the product of all elements in an array.
 * Used to compute tensor sizes and stride products.
 */
function product(arr: number[]): number {
  let result = 1;
  for (const n of arr) result *= n;
  return result;
}

/**
 * Convert a flat index back to per-axis indices for `shape` (row-major).
 * Inverse of flatIndex from Ch 01.
 *
 *   unravel(9, [3,4]) → [2, 1]
 *   Walk RIGHT to LEFT: peel off `% dim`, then `÷ dim`.
 */
function unravel(flat: number, shape: number[]): number[] {
  const indices = new Array<number>(shape.length);
  let remainder = flat;
  for (let axis = shape.length - 1; axis >= 0; axis--) {
    const dim = shape[axis] as number;
    indices[axis] = remainder % dim;
    remainder = Math.floor(remainder / dim);
  }
  return indices;
}

// ─── matMul ─────────────────────────────────────────────────────────────────

/**
 * 2-D matrix multiplication: (M×K) × (K×N) → (M×N).
 *
 * This is THE core operation of neural networks. Every Linear layer, every
 * attention score, every projection reduces to a matMul.
 *
 * Math: C[i,j] = Σ_k A[i,k] * B[k,j]
 *
 * The inner dimension K is consumed. Only the outer dimensions M and N
 * survive into the output shape.
 */
export function matMul(a: Tensor, b: Tensor): Tensor {
  if (a.ndim !== 2 || b.ndim !== 2) {
    throw new Error(`matMul requires 2-D tensors; got [${a.shape}] and [${b.shape}]`);
  }

  // Destructure into named dimension variables — shape errors are much
  // easier to read when variables say M/K/N than shape[0]/shape[1].
  const M = a.shape[0] as number;
  const K = a.shape[1] as number;
  const K2 = b.shape[0] as number;
  const N = b.shape[1] as number;

  if (K !== K2) {
    throw new Error(`matMul inner dims mismatch: A has ${K} columns, B has ${K2} rows`);
  }

  const out = new Array<number>(M * N).fill(0);

  for (let i = 0; i < M; i++) {
    for (let j = 0; j < N; j++) {
      // Accumulate the dot product of row i of A with column j of B.
      let sum = 0;
      for (let k = 0; k < K; k++) {
        // Row-major offsets: A[i,k] = data[i*K + k],  B[k,j] = data[k*N + j].
        const aVal = a.data[i * K + k] as number;
        const bVal = b.data[k * N + j] as number;
        sum += aVal * bVal;
      }
      out[i * N + j] = sum;  // C[i,j] = sum
    }
  }

  return createTensor(out, [M, N]);
}

/**
 * Batched matrix multiplication over all leading dimensions.
 * The last two axes of each tensor are treated as the matrix.
 *
 * Shape rule: [..., M, K] × [..., K, N] → [..., M, N]
 * All leading dimensions must match exactly.
 *
 * Used in multi-head attention where a and b are [batch, heads, seq, dHead].
 */
export function matMulBatch(a: Tensor, b: Tensor): Tensor {
  if (a.ndim < 2 || b.ndim < 2 || a.ndim !== b.ndim) {
    throw new Error(
      `matMulBatch requires same-rank tensors with ndim ≥ 2; got [${a.shape}] and [${b.shape}]`,
    );
  }

  // Split shape into leading batch dims + the 2-D matrix dims at the end.
  const batchShape = a.shape.slice(0, -2);
  const M = a.shape[a.ndim - 2] as number;
  const K = a.shape[a.ndim - 1] as number;
  const K2 = b.shape[b.ndim - 2] as number;
  const N = b.shape[b.ndim - 1] as number;

  if (K !== K2) {
    throw new Error(`matMulBatch inner dims mismatch: ${K} vs ${K2}`);
  }

  // Validate that every leading (batch) dimension matches.
  for (let i = 0; i < batchShape.length; i++) {
    if (a.shape[i] !== b.shape[i]) {
      throw new Error(
        `matMulBatch batch dims mismatch at axis ${i}: ${a.shape[i]} vs ${b.shape[i]}`,
      );
    }
  }

  const batchSize = product(batchShape);   // number of 2-D matrix pairs
  const outShape = [...batchShape, M, N];
  const out = new Array<number>(batchSize * M * N).fill(0);

  for (let b_ = 0; b_ < batchSize; b_++) {
    // Offsets into the flat data arrays for the b_-th matrix pair.
    const aBase = b_ * M * K;
    const bBase = b_ * K * N;
    const outBase = b_ * M * N;

    for (let i = 0; i < M; i++) {
      for (let j = 0; j < N; j++) {
        let sum = 0;
        for (let k = 0; k < K; k++) {
          const aVal = a.data[aBase + i * K + k] as number;
          const bVal = b.data[bBase + k * N + j] as number;
          sum += aVal * bVal;
        }
        out[outBase + i * N + j] = sum;
      }
    }
  }

  return createTensor(out, outShape);
}

// ─── transpose ───────────────────────────────────────────────────────────────

/**
 * Permute the axes of a tensor.
 * Default (no `axes`): reverse all axes — for 2-D this is the classic transpose.
 * With `axes`: e.g. [0, 2, 1] swaps the last two axes of a 3-D tensor.
 *
 * Math: out[i_0, i_1, ..., i_n] = t[i_{axes[0]}, i_{axes[1]}, ..., i_{axes[n]}]
 *
 * Used in attention to turn K [batch, seq, dHead] into [batch, dHead, seq]
 * before multiplying with Q.
 */
export function transpose(t: Tensor, axes?: number[]): Tensor {
  // Default permutation: reverse all axes  [0,1,2] → [2,1,0].
  const perm = axes ?? t.shape.map((_, i) => t.ndim - 1 - i);

  if (perm.length !== t.ndim) {
    throw new Error(`transpose: axes length ${perm.length} must match ndim ${t.ndim}`);
  }

  // New shape: pick t's dim sizes in the permuted order.
  const newShape = perm.map(axis => t.shape[axis] as number);
  const out = new Array<number>(t.size);

  for (let flatOut = 0; flatOut < t.size; flatOut++) {
    // Decode output flat index → output indices in the transposed shape.
    const outputIdx = unravel(flatOut, newShape);

    // Map back to input indices: outputIdx[i] lives on axis perm[i] of the input.
    const inputIdx = new Array<number>(t.ndim);
    for (let i = 0; i < t.ndim; i++) {
      inputIdx[perm[i] as number] = outputIdx[i] as number;
    }

    out[flatOut] = t.data[flatIndex(t.shape, inputIdx)] as number;
  }

  return createTensor(out, newShape);
}

// ─── reshape ─────────────────────────────────────────────────────────────────

/**
 * Return a tensor with `newShape` but the same flat data.
 * At most one element of `newShape` may be -1; it is inferred automatically.
 *
 * Total element count must be unchanged: product(oldShape) === product(newShape).
 */
export function reshape(t: Tensor, newShape: number[]): Tensor {
  // Find the -1 slot if present, and count known dimensions.
  const inferAxis = newShape.indexOf(-1);
  let resolvedShape: number[];

  if (inferAxis === -1) {
    // No inference needed — just validate.
    resolvedShape = [...newShape];
  } else {
    // Count how many -1s there are — only one allowed.
    if (newShape.filter(d => d === -1).length > 1) {
      throw new Error("reshape: at most one dimension may be -1");
    }
    const knownProduct = newShape.reduce((acc, d) => (d === -1 ? acc : acc * d), 1);
    if (t.size % knownProduct !== 0) {
      throw new Error(
        `reshape: cannot infer dim for shape [${newShape}] from size ${t.size}`,
      );
    }
    resolvedShape = newShape.map((d, i) =>
      i === inferAxis ? t.size / knownProduct : d,
    );
  }

  if (product(resolvedShape) !== t.size) {
    throw new Error(
      `reshape: new shape [${resolvedShape}] has ${product(resolvedShape)} elements, but tensor has ${t.size}`,
    );
  }

  // Data is identical — copy the array so mutations don't alias.
  return createTensor(Array.from(t.data), resolvedShape);
}

/**
 * Collapse all axes from `startAxis` onward into a single dimension.
 * Default startAxis=0 flattens everything to a 1-D tensor of length `t.size`.
 *
 * flatten(t, 1) on [2, 3, 4] → [2, 12]  (keeps axis 0, flattens axes 1+2)
 */
export function flatten(t: Tensor, startAxis = 0): Tensor {
  const leadingShape = t.shape.slice(0, startAxis);
  const trailingSize = product(t.shape.slice(startAxis));
  return reshape(t, [...leadingShape, trailingSize]);
}

// ─── squeeze / unsqueeze ─────────────────────────────────────────────────────

/**
 * Remove the size-1 dimension at `axis`.
 * Throws if that dimension is not 1 — squeezing a larger axis would change the data layout.
 *
 * Used after reductions that leave a leftover size-1 axis.
 */
export function squeeze(t: Tensor, axis: number): Tensor {
  if (t.shape[axis] !== 1) {
    throw new Error(
      `squeeze: axis ${axis} has size ${t.shape[axis]}, not 1`,
    );
  }
  const newShape = t.shape.filter((_, i) => i !== axis);
  return createTensor(Array.from(t.data), newShape);
}

/**
 * Insert a size-1 dimension at `axis`.
 * Shape [3, 4] with axis=0 → [1, 3, 4].
 * Shape [3, 4] with axis=1 → [3, 1, 4].
 *
 * Used to align a tensor's rank for broadcasting —
 * e.g. turning a mask from [seq] into [1, 1, seq] for attention.
 */
export function unsqueeze(t: Tensor, axis: number): Tensor {
  const newShape = [...t.shape];
  newShape.splice(axis, 0, 1);
  return createTensor(Array.from(t.data), newShape);
}

// ─── concat / stack ──────────────────────────────────────────────────────────

/**
 * Concatenate tensors along an existing axis.
 * All other axes must have the same size.
 *
 * concat([A [3,4], B [2,4]], axis=0) → [5, 4]
 * concat([A [3,4], B [3,2]], axis=1) → [3, 6]
 *
 * Used in multi-head attention to merge per-head outputs along the feature axis.
 */
export function concat(tensors: Tensor[], axis: number): Tensor {
  if (tensors.length === 0) throw new Error("concat: needs at least one tensor");

  const first = tensors[0] as Tensor;

  // Validate that all tensors agree on every axis except the concat axis.
  for (let t = 1; t < tensors.length; t++) {
    const cur = tensors[t] as Tensor;
    if (cur.ndim !== first.ndim) {
      throw new Error(`concat: all tensors must have the same ndim`);
    }
    for (let ax = 0; ax < first.ndim; ax++) {
      if (ax !== axis && cur.shape[ax] !== first.shape[ax]) {
        throw new Error(
          `concat: shape mismatch at axis ${ax}: ${first.shape[ax]} vs ${cur.shape[ax]}`,
        );
      }
    }
  }

  // Build the output shape: same everywhere except the concat axis.
  const outShape = [...first.shape];
  outShape[axis] = tensors.reduce((sum, t) => sum + (t.shape[axis] as number), 0);

  const out = new Array<number>(product(outShape));

  // For each output position, figure out which input tensor and offset it comes from.
  for (let flatOut = 0; flatOut < out.length; flatOut++) {
    const idx = unravel(flatOut, outShape);

    // Walk the tensors in order; find which one owns this concat-axis position.
    let concatPos = idx[axis] as number;
    let sourceTensor: Tensor | undefined;
    let sourceAxis: number | undefined;

    for (const t of tensors) {
      const dimSize = t.shape[axis] as number;
      if (concatPos < dimSize) {
        sourceTensor = t;
        sourceAxis = concatPos;
        break;
      }
      concatPos -= dimSize;
    }

    if (sourceTensor === undefined || sourceAxis === undefined) {
      throw new Error("concat: internal index error");
    }

    // Build the input index — same as output except the concat axis is local.
    const inputIdx = [...idx];
    inputIdx[axis] = sourceAxis;
    out[flatOut] = sourceTensor.data[flatIndex(sourceTensor.shape, inputIdx)] as number;
  }

  return createTensor(out, outShape);
}

/**
 * Stack tensors along a new axis.
 * All tensors must have identical shape.
 *
 * stack([A [3], B [3]], axis=0) → [2, 3]
 * stack([A [3], B [3]], axis=1) → [3, 2]
 *
 * Implemented as: unsqueeze each tensor at `axis`, then concat.
 */
export function stack(tensors: Tensor[], axis: number): Tensor {
  if (tensors.length === 0) throw new Error("stack: needs at least one tensor");

  const first = tensors[0] as Tensor;
  for (const t of tensors) {
    if (t.shape.join(",") !== first.shape.join(",")) {
      throw new Error(`stack: all tensors must have identical shape`);
    }
  }

  // Insert a new size-1 axis at `axis` in each tensor, then concat along it.
  const expanded = tensors.map(t => unsqueeze(t, axis));
  return concat(expanded, axis);
}

