/**
 * Tests for tensor/ops.ts
 * Chapter 03 — Elementwise Ops & Broadcasting
 *
 * Run: bun test src/tensor/ops.test.ts
 */
import { describe, it, expect } from "bun:test";
import {
  add, sub, mul, div,
  addScalar, mulScalar,
  applyFn,
  broadcast, broadcastShapes,
} from "./ops.ts";
import { createTensor } from "./types.ts";

// ─── helpers ────────────────────────────────────────────────────────────────
// Converts Float64Array to a plain number[] so toEqual comparisons are readable.
const data = (t: ReturnType<typeof createTensor>) => Array.from(t.data);

// ─── broadcastShapes ─────────────────────────────────────────────────────────

describe("broadcastShapes", () => {
  it("[3,1] and [1,4] merge to [3,4]: each axis takes the non-1 size", () => {
    // When one side has 1 on an axis, the other side's size wins.
    expect(broadcastShapes([3, 1], [1, 4])).toEqual([3, 4]);
  });

  it("[5] and [3,5] merge to [3,5]: lower-rank input gets implicit leading 1", () => {
    // [5] is treated as [1,5], so axis 0 → max(1,3)=3, axis 1 → max(5,5)=5.
    expect(broadcastShapes([5], [3, 5])).toEqual([3, 5]);
  });

  it("identical shapes broadcast to themselves: no stretching needed", () => {
    expect(broadcastShapes([2, 3], [2, 3])).toEqual([2, 3]);
  });

  it("throws when both sides are >1 but different: [3] and [4] are incompatible", () => {
    // Neither is 1, so there is no rule to resolve the conflict.
    expect(() => broadcastShapes([3], [4])).toThrow();
  });
});

// ─── broadcast ───────────────────────────────────────────────────────────────

describe("broadcast", () => {
  it("stretches a column-vector [3,1] to a matrix [3,4] by repeating each value", () => {
    // t = [[1],[2],[3]]; after broadcast every row becomes 4 copies of its value.
    const t = createTensor([1, 2, 3], [3, 1]);
    const out = broadcast(t, [3, 4]);
    expect(out.shape).toEqual([3, 4]);
    // Row 0: four 1s, row 1: four 2s, row 2: four 3s.
    expect(data(out)).toEqual([1,1,1,1, 2,2,2,2, 3,3,3,3]);
  });

  it("broadcasting to the same shape is a no-op: every value stays in place", () => {
    const t = createTensor([1, 2, 3, 4], [2, 2]);
    expect(data(broadcast(t, [2, 2]))).toEqual([1, 2, 3, 4]);
  });

  it("throws when the target shape is smaller than t on any axis", () => {
    // [2,3] cannot broadcast to [3] — would have to shrink the first axis.
    const t = createTensor([1,2,3,4,5,6], [2, 3]);
    expect(() => broadcast(t, [3])).toThrow();
  });
});

// ─── add / sub ────────────────────────────────────────────────────────────────

describe("add", () => {
  it("same-shape addition is pointwise: each pair of elements summed once", () => {
    const a = createTensor([1, 2, 3, 4], [2, 2]);
    const b = createTensor([10, 20, 30, 40], [2, 2]);
    // Each position: a[i] + b[i].
    expect(data(add(a, b))).toEqual([11, 22, 33, 44]);
  });

  it("[3,1] + [1,4] broadcasts to [3,4]: outer product of values via add", () => {
    // a = [[1],[2],[3]], b = [[10,20,30,40]]
    // After broadcasting both to [3,4], result[r][c] = a[r][0] + b[0][c].
    const a = createTensor([1, 2, 3], [3, 1]);
    const b = createTensor([10, 20, 30, 40], [1, 4]);
    const out = add(a, b);
    expect(out.shape).toEqual([3, 4]);
    expect(data(out)).toEqual([
      11, 21, 31, 41,  // row 0: 1 + each of [10,20,30,40]
      12, 22, 32, 42,  // row 1: 2 + each of [10,20,30,40]
      13, 23, 33, 43,  // row 2: 3 + each of [10,20,30,40]
    ]);
  });

  it("addition is commutative: a + b === b + a", () => {
    const a = createTensor([1, 2, 3], [3]);
    const b = createTensor([4, 5, 6], [3]);
    expect(data(add(a, b))).toEqual(data(add(b, a)));
  });
});

describe("sub", () => {
  it("sub(a, a) is always zero: a tensor minus itself cancels out", () => {
    const a = createTensor([5, 10, 15], [3]);
    // a - a = 0 for every element.
    expect(data(sub(a, a))).toEqual([0, 0, 0]);
  });
});

// ─── mul / div ────────────────────────────────────────────────────────────────

describe("mul", () => {
  it("elementwise multiplication scales each position independently", () => {
    const a = createTensor([1, 2, 3, 4], [2, 2]);
    const b = createTensor([2, 3, 4, 5], [2, 2]);
    expect(data(mul(a, b))).toEqual([2, 6, 12, 20]);
  });

  it("multiplying by a scalar-shaped tensor [1] broadcasts to all elements", () => {
    const a = createTensor([1, 2, 3], [3]);
    const scale = createTensor([10], [1]);
    expect(data(mul(a, scale))).toEqual([10, 20, 30]);
  });
});

describe("div", () => {
  it("div(a, a) is all-ones for nonzero a: a number divided by itself is 1", () => {
    const a = createTensor([2, 4, 8], [3]);
    // Every element divided by itself.
    expect(data(div(a, a))).toEqual([1, 1, 1]);
  });
});

// ─── addScalar / mulScalar ────────────────────────────────────────────────────

describe("addScalar", () => {
  it("adds s to every element, preserving shape", () => {
    const t = createTensor([1, 2, 3], [3]);
    expect(data(addScalar(t, 10))).toEqual([11, 12, 13]);
    expect(addScalar(t, 10).shape).toEqual([3]);
  });
});

describe("mulScalar", () => {
  it("multiplying by 2 doubles every element", () => {
    const t = createTensor([1, 2, 3, 4], [2, 2]);
    expect(data(mulScalar(t, 2))).toEqual([2, 4, 6, 8]);
  });

  it("multiplying by 0 produces all zeros: the zero element of multiplication", () => {
    const t = createTensor([1, 2, 3], [3]);
    expect(data(mulScalar(t, 0))).toEqual([0, 0, 0]);
  });

  it("output shape equals input shape: scalar multiply does not reshape", () => {
    const t = createTensor([1, 2, 3, 4, 5, 6], [2, 3]);
    expect(mulScalar(t, 5).shape).toEqual([2, 3]);
  });
});

// ─── applyFn ─────────────────────────────────────────────────────────────────

describe("applyFn", () => {
  it("applies the function independently to every element", () => {
    const t = createTensor([1, 4, 9, 16], [4]);
    // sqrt is applied to each value; no values influence each other.
    expect(data(applyFn(t, Math.sqrt))).toEqual([1, 2, 3, 4]);
  });

  it("output shape exactly matches input shape: applyFn never reshapes", () => {
    const t = createTensor([1, 2, 3, 4, 5, 6], [2, 3]);
    expect(applyFn(t, x => x * 2).shape).toEqual([2, 3]);
  });

  it("applying the identity function returns an identical tensor", () => {
    const t = createTensor([7, 8, 9], [3]);
    expect(data(applyFn(t, x => x))).toEqual([7, 8, 9]);
  });
});
