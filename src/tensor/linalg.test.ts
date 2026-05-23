/**
 * Tests for tensor/linalg.ts
 * Chapter 04 — Matrix Operations
 *
 * Run: bun test src/tensor/linalg.test.ts
 */
import { describe, it, expect } from "bun:test";
import {
  matMul, matMulBatch,
  transpose,
  reshape, flatten,
  squeeze, unsqueeze,
  concat, stack,
} from "./linalg.ts";
import { createTensor } from "./types.ts";

const data = (t: ReturnType<typeof createTensor>) => Array.from(t.data);

// ─── matMul ──────────────────────────────────────────────────────────────────

describe("matMul", () => {
  it("(M×K) × (K×N) produces shape (M×N): inner dim is consumed", () => {
    const a = createTensor(new Array(6).fill(0), [2, 3]);
    const b = createTensor(new Array(12).fill(0), [3, 4]);
    // 2×3 times 3×4 → 2×4: the shared K=3 disappears
    expect(matMul(a, b).shape).toEqual([2, 4]);
  });

  it("[[1,2],[3,4]] × [[1],[1]] === [[3],[7]]: row dot column gives a scalar", () => {
    const a = createTensor([1, 2, 3, 4], [2, 2]);
    const b = createTensor([1, 1], [2, 1]);
    // Row 0: 1*1 + 2*1 = 3.   Row 1: 3*1 + 4*1 = 7.
    expect(data(matMul(a, b))).toEqual([3, 7]);
  });

  it("multiplying by the identity matrix is a no-op: A × I === A", () => {
    const a = createTensor([1, 2, 3, 4], [2, 2]);
    const identity = createTensor([1, 0, 0, 1], [2, 2]);
    expect(data(matMul(a, identity))).toEqual([1, 2, 3, 4]);
  });

  it("throws when inner dimensions do not match: [2,3] × [2,4] is invalid", () => {
    const a = createTensor(new Array(6).fill(0), [2, 3]);
    const b = createTensor(new Array(8).fill(0), [2, 4]);
    // a has 3 columns, b has 2 rows — incompatible
    expect(() => matMul(a, b)).toThrow();
  });
});

// ─── matMulBatch ──────────────────────────────────────────────────────────────

describe("matMulBatch", () => {
  it("[B,M,K] × [B,K,N] produces [B,M,N]: leading batch dim passes through", () => {
    const a = createTensor(new Array(2 * 3 * 4).fill(1), [2, 3, 4]);
    const b = createTensor(new Array(2 * 4 * 5).fill(1), [2, 4, 5]);
    // 2 batches of (3×4) × (4×5) → 2 batches of (3×5)
    expect(matMulBatch(a, b).shape).toEqual([2, 3, 5]);
  });

  it("each batch slice is multiplied independently: second slice is separate from first", () => {
    // Batch 0: [[1,0],[0,1]] × [[2,0],[0,2]] = [[2,0],[0,2]]
    // Batch 1: [[1,1],[1,1]] × [[1,0],[0,1]] = [[1,1],[1,1]]
    const a = createTensor([1,0,0,1,  1,1,1,1], [2, 2, 2]);
    const b = createTensor([2,0,0,2,  1,0,0,1], [2, 2, 2]);
    expect(data(matMulBatch(a, b))).toEqual([2,0,0,2, 1,1,1,1]);
  });
});

// ─── transpose ───────────────────────────────────────────────────────────────

describe("transpose", () => {
  it("2-D transpose swaps shape: [2,3] becomes [3,2]", () => {
    const a = createTensor([1,2,3,4,5,6], [2, 3]);
    expect(transpose(a).shape).toEqual([3, 2]);
  });

  it("element A[i,j] moves to A^T[j,i]: rows become columns", () => {
    const a = createTensor([1,2,3, 4,5,6], [2, 3]);
    // A = [[1,2,3],[4,5,6]].  A^T = [[1,4],[2,5],[3,6]].
    expect(data(transpose(a))).toEqual([1,4, 2,5, 3,6]);
  });

  it("double transpose is the identity: (A^T)^T === A", () => {
    const a = createTensor([1,2,3,4,5,6], [2, 3]);
    expect(data(transpose(transpose(a)))).toEqual([1,2,3,4,5,6]);
    expect(transpose(transpose(a)).shape).toEqual([2, 3]);
  });

  it("3-D transpose with axes [0,2,1] swaps the last two axes", () => {
    const t = createTensor(new Array(24).fill(0), [2, 3, 4]);
    expect(transpose(t, [0, 2, 1]).shape).toEqual([2, 4, 3]);
  });
});

// ─── reshape ─────────────────────────────────────────────────────────────────

describe("reshape", () => {
  it("preserves total element count: data is unchanged, shape is not", () => {
    const t = createTensor([1,2,3,4,5,6], [2, 3]);
    const r = reshape(t, [3, 2]);
    expect(r.shape).toEqual([3, 2]);
    // Row-major order is preserved — same flat sequence, new shape
    expect(data(r)).toEqual([1,2,3,4,5,6]);
  });

  it("infers -1 dimension: reshape([2,3], [-1,2]) → shape [3,2]", () => {
    const t = createTensor([1,2,3,4,5,6], [2, 3]);
    expect(reshape(t, [-1, 2]).shape).toEqual([3, 2]);
  });

  it("reshape to [t.size] is equivalent to flatten: all elements in one row", () => {
    const t = createTensor([1,2,3,4,5,6], [2, 3]);
    expect(reshape(t, [6]).shape).toEqual([6]);
  });

  it("throws when new shape has incompatible element count", () => {
    const t = createTensor([1,2,3,4,5,6], [2, 3]);
    expect(() => reshape(t, [2, 4])).toThrow();
  });
});

// ─── flatten ─────────────────────────────────────────────────────────────────

describe("flatten", () => {
  it("default (startAxis=0) collapses all dimensions to 1-D", () => {
    const t = createTensor([1,2,3,4,5,6], [2, 3]);
    expect(flatten(t).shape).toEqual([6]);
    expect(data(flatten(t))).toEqual([1,2,3,4,5,6]);
  });

  it("startAxis=1 keeps the first dim and flattens the rest", () => {
    const t = createTensor(new Array(24).fill(1), [2, 3, 4]);
    // Flatten axes 1 and 2: [2, 3, 4] → [2, 12]
    expect(flatten(t, 1).shape).toEqual([2, 12]);
  });
});

// ─── squeeze / unsqueeze ─────────────────────────────────────────────────────

describe("unsqueeze", () => {
  it("inserts a size-1 axis at axis=0: [3,4] → [1,3,4]", () => {
    const t = createTensor(new Array(12).fill(0), [3, 4]);
    expect(unsqueeze(t, 0).shape).toEqual([1, 3, 4]);
  });

  it("inserts a size-1 axis at axis=1: [3,4] → [3,1,4]", () => {
    const t = createTensor(new Array(12).fill(0), [3, 4]);
    expect(unsqueeze(t, 1).shape).toEqual([3, 1, 4]);
  });

  it("data is unchanged: unsqueeze is pure metadata", () => {
    const t = createTensor([1,2,3], [3]);
    expect(data(unsqueeze(t, 0))).toEqual([1,2,3]);
  });
});

describe("squeeze", () => {
  it("removes the size-1 axis: [1,3,4] → [3,4]", () => {
    const t = createTensor(new Array(12).fill(0), [1, 3, 4]);
    expect(squeeze(t, 0).shape).toEqual([3, 4]);
  });

  it("squeeze after unsqueeze is a round-trip: shape is restored", () => {
    const t = createTensor([1,2,3,4], [2, 2]);
    expect(squeeze(unsqueeze(t, 0), 0).shape).toEqual([2, 2]);
  });

  it("throws when axis is not size 1", () => {
    const t = createTensor(new Array(6).fill(0), [2, 3]);
    expect(() => squeeze(t, 0)).toThrow();
  });
});

// ─── concat ──────────────────────────────────────────────────────────────────

describe("concat", () => {
  it("concat along axis=0 stacks rows: [3] + [3] → [6]", () => {
    const a = createTensor([1,2,3], [3]);
    const b = createTensor([4,5,6], [3]);
    expect(concat([a, b], 0).shape).toEqual([6]);
    expect(data(concat([a, b], 0))).toEqual([1,2,3,4,5,6]);
  });

  it("concat along axis=1 stacks columns: [2,2] + [2,3] → [2,5]", () => {
    const a = createTensor([1,2, 3,4], [2, 2]);
    const b = createTensor([5,6,7, 8,9,10], [2, 3]);
    const out = concat([a, b], 1);
    expect(out.shape).toEqual([2, 5]);
    // Row 0: [1,2,5,6,7]   Row 1: [3,4,8,9,10]
    expect(data(out)).toEqual([1,2,5,6,7, 3,4,8,9,10]);
  });

  it("throws when non-concat axes have mismatched sizes", () => {
    const a = createTensor(new Array(6).fill(0), [2, 3]);
    const b = createTensor(new Array(8).fill(0), [4, 2]);
    expect(() => concat([a, b], 0)).toThrow();
  });
});

// ─── stack ───────────────────────────────────────────────────────────────────

describe("stack", () => {
  it("stack along axis=0 creates a new leading dim: two [3]s → [2,3]", () => {
    const a = createTensor([1,2,3], [3]);
    const b = createTensor([4,5,6], [3]);
    const out = stack([a, b], 0);
    expect(out.shape).toEqual([2, 3]);
    expect(data(out)).toEqual([1,2,3, 4,5,6]);
  });

  it("stack along axis=1 interleaves columns: two [3]s → [3,2]", () => {
    const a = createTensor([1,2,3], [3]);
    const b = createTensor([4,5,6], [3]);
    const out = stack([a, b], 1);
    expect(out.shape).toEqual([3, 2]);
    // Each row pairs one element from a with one from b
    expect(data(out)).toEqual([1,4, 2,5, 3,6]);
  });

  it("throws when tensors have different shapes", () => {
    const a = createTensor([1,2,3], [3]);
    const b = createTensor([1,2,3,4], [4]);
    expect(() => stack([a, b], 0)).toThrow();
  });
});

