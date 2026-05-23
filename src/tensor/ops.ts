/**
 * CHAPTER 03: Elementwise Ops & Broadcasting
 * ════════════════════════════════════════
 * Part 1 of 6: Tensor Library
 *
 * WHAT WE'RE BUILDING:
 *   The arithmetic layer on top of the Tensor type — elementwise add/sub/mul/div,
 *   NumPy-style broadcasting (broadcastShapes, broadcast), scalar wrappers
 *   (addScalar, mulScalar), and the unary map applyFn.
 *
 * WHY IT MATTERS:
 *   Every later module speaks this language. Residual connections are add(x, block(x)).
 *   Bias adds, normalization stats, attention masks, and activation functions all
 *   reduce to a broadcasted elementwise rule plus, sometimes, one unary scalar function.
 *
 * WHAT THIS UNLOCKS:
 *   → Ch 09 (gradient descent — parameter -= lr * grad)
 *   → Ch 11 (activations built on applyFn)
 *   → Ch 13 (linear layer: matMul + broadcasted bias add)
 *   → Ch 20–22 (normalization, dropout, scaled attention scores)
 *
 * REFERENCE: docs/part-1-tensor-library/ch-03-elementwise-ops-broadcasting.md
 */
import { createTensor, flatIndex, type Tensor } from "./types.ts";

/**
 * Compute the output shape when broadcasting a and b (NumPy rules).
 * Align right; each dim must be equal or one of them is 1.
 *   broadcastShapes([3,1], [1,4]) → [3,4]
 */
export function broadcastShapes(a: number[], b: number[]): number[] {
  const out: number[] = [];
  const maxRank = Math.max(a.length, b.length);
  for (let i = 0; i < maxRank; i++) {
    const dimA = a[a.length - 1 - i] ?? 1;
    const dimB = b[b.length - 1 - i] ?? 1;
    if (dimA !== dimB && dimA !== 1 && dimB !== 1) {
      throw new Error(`Incompatible shapes for broadcasting: ${a} and ${b}`);
    }
    out.unshift(Math.max(dimA, dimB));
  }
  return out;
}

/**
 * Expand t to a larger shape by repeating along size-1 axes.
 *
 * Picture: we have t with shape [3, 1] and we want shape [3, 4].
 *
 *     t (shape [3,1])              broadcast(t, [3,4])
 *     ┌───┐                        ┌──────────────┐
 *     │ 1 │                        │ 1  1  1  1   │   row 0
 *     │ 2 │       ───────▶         │ 2  2  2  2   │   row 1
 *     │ 3 │                        │ 3  3  3  3   │   row 2
 *     └───┘                        └──────────────┘
 *
 *     t.data = [1, 2, 3]           out.data = [1,1,1,1, 2,2,2,2, 3,3,3,3]
 *
 * THE WHOLE ALGORITHM in plain English:
 *   For every position in the output, figure out which position in t feeds it,
 *   then copy that value over. Size-1 axes "stretch" — they always read from
 *   index 0 of t no matter where we are in the output.
 *
 * THE CODE follows that sentence literally:
 *
 *     for flatOut in 0..size-1:
 *         outputIdx = unravelIndex(flatOut, shape)      // where am I in the output?
 *         inputIdx  = stretchToInput(outputIdx, t.shape) // where do I read from in t?
 *         out[flatOut] = t.data[ flatIndex(t.shape, inputIdx) ]
 *
 * Two helpers below do the index bookkeeping so this main loop stays tiny.
 * Use pen and paper to work through a small example and convince yourself you understand the
 * index math before diving into the code.
 */
export function broadcast(t: Tensor, shape: number[]): Tensor {
  validateBroadcastTarget(t.shape, shape);

  const size = shape.reduce((product, dim) => product * dim, 1);
  const out = new Array<number>(size);

  for (let flatOut = 0; flatOut < size; flatOut++) {
    const outputIdx = unravelIndex(flatOut, shape);
    const inputIdx = stretchToInput(outputIdx, t.shape);
    out[flatOut] = t.data[flatIndex(t.shape, inputIdx)] as number;
  }

  return createTensor(out, [...shape]);
}

/**
 * Guard: refuse to broadcast t to `shape` if any axis of t is larger than the
 * matching axis of `shape`, or if t has more axes than `shape`. Broadcasting
 * only ever stretches 1 → N; it never shrinks.
 */
function validateBroadcastTarget(tShape: number[], shape: number[]): void {
  const merged = broadcastShapes(tShape, shape);
  if (merged.length !== shape.length) {
    throw new Error(`Cannot broadcast shape ${tShape} to ${shape}`);
  }
  for (let i = 0; i < shape.length; i++) {
    if (merged[i] !== shape[i]) {
      throw new Error(`Cannot broadcast shape ${tShape} to ${shape}`);
    }
  }
}

/**
 * Turn a flat index back into a list of per-axis indices for a given shape.
 * This is the inverse of row-major flattening (flatIndex from Ch 01).
 *
 *   shape   = [3, 4]
 *   flat=9  → indices [2, 1]   (row 2, column 1)
 *
 * How: in row-major layout the LAST axis varies fastest, so we peel axes off
 * RIGHT to LEFT using `% dim` (the index on this axis) and `÷ dim` (what's
 * left to assign to the more significant axes).
 *
 *   iter 1: axis=1, dim=4 → indices[1] = 9 % 4 = 1; remainder = ⌊9/4⌋ = 2
 *   iter 2: axis=0, dim=3 → indices[0] = 2 % 3 = 2; remainder = ⌊2/3⌋ = 0
 *   → [2, 1]
 */
function unravelIndex(flat: number, shape: number[]): number[] {
  const indices = new Array<number>(shape.length);
  let remainder = flat;
  for (let axis = shape.length - 1; axis >= 0; axis--) {
    const dim = shape[axis] as number;
    indices[axis] = remainder % dim;
    remainder = Math.floor(remainder / dim);
  }
  return indices;
}

/**
 * Translate an output position into the position to read from in t.
 *
 *   Output:  outputIdx = [2, 1]   (3×4 grid, row 2 col 1)
 *   Input t: t.shape   = [3, 1]
 *   Result:  inputIdx  = [2, 0]   (col collapsed to 0 because t's col-axis
 *                                  has size 1 — "the stretch")
 *
 * Two rules (NumPy broadcasting):
 *   (1) Rank alignment. If t has fewer axes than the output, t's axes are
 *       right-aligned. rankOffset = shape.length − t.shape.length. So
 *       t's axis k maps to outputIdx[k + rankOffset]. Output axes to the
 *       LEFT of that just disappear here — they're the implicit leading 1s.
 *
 *   (2) Size-1 stretch. If t's axis has size 1, read from slot 0 regardless
 *       of where we are in the output. Otherwise the axes match in size and
 *       we copy the output index straight across.
 */
function stretchToInput(outputIdx: number[], tShape: number[]): number[] {
  const inputIdx = new Array<number>(tShape.length);
  const rankOffset = outputIdx.length - tShape.length;
  for (let axis = 0; axis < tShape.length; axis++) {
    const dim = tShape[axis] as number;
    const fromOutput = outputIdx[axis + rankOffset] as number;
    inputIdx[axis] = dim === 1 ? 0 : fromOutput;
  }
  return inputIdx;
}

/**
 * Shared engine for binary elementwise ops with broadcasting.
 *
 * The chapter says: once both tensors are broadcast to the common shape, the
 * operation collapses to one explicit loop where position k of the output
 * depends only on a.data[k] and b.data[k]. This helper is that loop.
 *
 * Math: C[i_1..i_n] = op(A[i_1..i_n], B[i_1..i_n]),
 *       after both inputs are broadcast to shape = broadcastShapes(a.shape, b.shape).
 */
function elementwise(
  a: Tensor,
  b: Tensor,
  op: (x: number, y: number) => number,
): Tensor {
  const shape = broadcastShapes(a.shape, b.shape);
  const aB = broadcast(a, shape);
  const bB = broadcast(b, shape);

  const out = new Array<number>(aB.size);
  for (let k = 0; k < aB.size; k++) {
    // Read once into typed locals — strict TS treats indexed access as T | undefined.
    const x = aB.data[k] as number;
    const y = bB.data[k] as number;
    out[k] = op(x, y);
  }
  return createTensor(out, [...shape]);
}

/** Elementwise a + b with broadcasting. Used for biases and residual connections. */
export function add(a: Tensor, b: Tensor): Tensor {
  return elementwise(a, b, (x, y) => x + y);
}

/** Elementwise a - b with broadcasting. Used for parameter updates and loss differences. */
export function sub(a: Tensor, b: Tensor): Tensor {
  return elementwise(a, b, (x, y) => x - y);
}

/** Elementwise a * b with broadcasting. Used for gating, masks, and elementwise scaling. */
export function mul(a: Tensor, b: Tensor): Tensor {
  return elementwise(a, b, (x, y) => x * y);
}

/** Elementwise a / b with broadcasting. Used for normalization (divide by stddev, etc). */
export function div(a: Tensor, b: Tensor): Tensor {
  return elementwise(a, b, (x, y) => x / y);
}

/**
 * Add scalar s to every element of t.
 * Math: out[i] = t.data[i] + s
 */
export function addScalar(t: Tensor, s: number): Tensor {
  const out = new Array<number>(t.size);
  for (let i = 0; i < t.size; i++) {
    const value = t.data[i] as number;
    out[i] = value + s;
  }
  return createTensor(out, [...t.shape]);
}

/**
 * Multiply every element of t by scalar s.
 * Used for learning-rate steps and the 1/sqrt(d_k) factor in scaled attention.
 * Math: out[i] = t.data[i] * s
 */
export function mulScalar(t: Tensor, s: number): Tensor {
  const out = new Array<number>(t.size);
  for (let i = 0; i < t.size; i++) {
    const value = t.data[i] as number;
    out[i] = value * s;
  }
  return createTensor(out, [...t.shape]);
}

/**
 * Apply unary function fn to every element of t, preserving shape.
 *
 * This is the bridge from plain arithmetic to neural-network behaviour: ReLU,
 * tanh, sigmoid, exp for softmax — all of them are applyFn with a different fn.
 *
 * Math: applyFn(T, f)[i_1..i_n] = f(T[i_1..i_n])
 */
export function applyFn(t: Tensor, fn: (x: number) => number): Tensor {
  const out = new Array<number>(t.size);
  for (let i = 0; i < t.size; i++) {
    const value = t.data[i] as number;
    out[i] = fn(value);
  }
  return createTensor(out, [...t.shape]);
}
