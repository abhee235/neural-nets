/**
 * Tests for tensor/reduce.ts
 * Chapter 05 — Reductions & Statistical Ops
 *
 * Run: bun test src/tensor/reduce.test.ts
 *
 * These tests encode the BEHAVIOUR the chapter specifies — they are the target
 * your implementation must hit. Tests are named after the mathematical property
 * they check, and each expect() has a comment stating the fact being verified.
 */
import { describe, it, expect } from "bun:test";
import { sum, mean, max, min, argmax, argmin, variance, std, softmax } from "./reduce.ts";
import { createTensor } from "./types.ts";

const EPSILON = 1e-6;
const data = (t: { data: Float64Array }) => Array.from(t.data);
/** Element-wise closeness for floating-point arrays (tolerance defaults to EPSILON). */
const closeAll = (a: number[], b: number[], tol = EPSILON) =>
  a.length === b.length && a.every((v, i) => Math.abs(v - (b[i] as number)) < tol);

describe("sum", () => {
  it("sum of all elements equals the scalar total", () => {
    const t = createTensor([1, 2, 3, 4, 5, 6], [2, 3]);
    // 1+2+3+4+5+6 = 21, returned as a rank-0 scalar tensor
    expect(data(sum(t))).toEqual([21]);
    expect(sum(t).shape).toEqual([]);
  });

  it("sum along axis=0 collapses rows: column sums", () => {
    const t = createTensor([1, 2, 3, 4, 5, 6], [2, 3]);
    // [[1,2,3],[4,5,6]] → [1+4, 2+5, 3+6] = [5,7,9], shape [3]
    expect(sum(t, 0).shape).toEqual([3]);
    expect(data(sum(t, 0))).toEqual([5, 7, 9]);
  });

  it("sum along axis=1 collapses columns: row sums", () => {
    const t = createTensor([1, 2, 3, 4, 5, 6], [2, 3]);
    // [[1,2,3],[4,5,6]] → [1+2+3, 4+5+6] = [6,15], shape [2]
    expect(sum(t, 1).shape).toEqual([2]);
    expect(data(sum(t, 1))).toEqual([6, 15]);
  });

  it("keepDims=true preserves the reduced axis as size 1", () => {
    const t = createTensor([1, 2, 3, 4, 5, 6], [2, 3]);
    // axis 1 reduced but kept → shape [2,1] instead of [2]
    expect(sum(t, 1, true).shape).toEqual([2, 1]);
    expect(data(sum(t, 1, true))).toEqual([6, 15]);
  });
});

describe("mean", () => {
  it("mean of [1,2,3,4] === 2.5", () => {
    const t = createTensor([1, 2, 3, 4], [4]);
    // (1+2+3+4)/4 = 2.5
    expect(data(mean(t))).toEqual([2.5]);
  });

  it("mean along axis produces the per-column average", () => {
    const t = createTensor([1, 2, 3, 4, 5, 6], [2, 3]);
    // columns: (1+4)/2, (2+5)/2, (3+6)/2 = [2.5, 3.5, 4.5]
    expect(closeAll(data(mean(t, 0)), [2.5, 3.5, 4.5])).toBe(true);
  });
});

describe("max", () => {
  it("max of all elements equals the largest value", () => {
    const t = createTensor([3, 1, 4, 1, 5, 9, 2, 6], [8]);
    expect(data(max(t))).toEqual([9]);
  });

  it("max along axis=1 picks the largest per row", () => {
    const t = createTensor([1, 5, 3, 4, 2, 6], [2, 3]);
    // rows [1,5,3] and [4,2,6] → [5, 6]
    expect(data(max(t, 1))).toEqual([5, 6]);
  });
});

describe("min", () => {
  it("min of all elements equals the smallest value", () => {
    const t = createTensor([3, 1, 4, 1, 5, 9, 2, 6], [8]);
    expect(data(min(t))).toEqual([1]);
  });

  it("min along axis=1 picks the smallest per row", () => {
    const t = createTensor([1, 5, 3, 4, 2, 6], [2, 3]);
    // rows [1,5,3] and [4,2,6] → [1, 2]
    expect(data(min(t, 1))).toEqual([1, 2]);
  });
});

describe("argmax", () => {
  it("returns the index of the maximum element", () => {
    const t = createTensor([1, 5, 3], [3]);
    // biggest value 5 is at position 1
    expect(data(argmax(t))).toEqual([1]);
  });

  it("argmax along axis=1 returns the winning position per row", () => {
    const t = createTensor([1, 5, 3, 4, 2, 6], [2, 3]);
    // row 0: max at index 1; row 1: max at index 2
    expect(data(argmax(t, 1))).toEqual([1, 2]);
  });
});

describe("argmin", () => {
  it("returns the index of the minimum element", () => {
    const t = createTensor([1, 5, 3], [3]);
    // smallest value 1 is at position 0
    expect(data(argmin(t))).toEqual([0]);
  });

  it("argmin along axis=1 returns the lowest position per row", () => {
    const t = createTensor([1, 5, 3, 4, 2, 6], [2, 3]);
    // row 0: min at index 0; row 1: min at index 1
    expect(data(argmin(t, 1))).toEqual([0, 1]);
  });
});

describe("variance", () => {
  it("variance([2,4,4,4,5,5,7,9]) === 4 (population variance)", () => {
    const t = createTensor([2, 4, 4, 4, 5, 5, 7, 9], [8]);
    // mean = 5; squared deviations sum to 32; 32/8 = 4
    expect(closeAll(data(variance(t)), [4])).toBe(true);
  });

  it("variance is zero for a constant tensor", () => {
    const t = createTensor([7, 7, 7, 7], [4]);
    // no spread → variance 0
    expect(closeAll(data(variance(t)), [0])).toBe(true);
  });
});

describe("std", () => {
  it("std === sqrt(variance + epsilon)", () => {
    const t = createTensor([2, 4, 4, 4, 5, 5, 7, 9], [8]);
    // variance 4 → std ≈ 2
    expect(closeAll(data(std(t)), [2])).toBe(true);
  });

  it("std of a constant tensor is ~0 (epsilon floor)", () => {
    const t = createTensor([7, 7, 7, 7], [4]);
    // variance 0 → std = sqrt(eps) ≈ 0
    expect(data(std(t))[0]!).toBeLessThan(1e-3);
  });
});

describe("softmax", () => {
  it("output sums to 1.0", () => {
    const t = createTensor([2, 1, 3], [3]);
    const total = data(softmax(t)).reduce((a, b) => a + b, 0);
    // a probability distribution must sum to 1
    expect(Math.abs(total - 1)).toBeLessThan(EPSILON);
  });

  it("matches the worked example [2,1,3] → [0.245, 0.090, 0.665]", () => {
    const t = createTensor([2, 1, 3], [3]);
    // exp(-1),exp(-2),exp(0) normalised by their sum (see Figure 3); 2-decimal precision
    expect(closeAll(data(softmax(t)), [0.245, 0.09, 0.665], 1e-3)).toBe(true);
  });

  it("shift invariance: softmax(x) === softmax(x + c)", () => {
    const a = data(softmax(createTensor([1, 2, 3], [3])));
    const b = data(softmax(createTensor([1001, 1002, 1003], [3])));
    // adding a constant to every logit leaves the probabilities unchanged
    expect(closeAll(a, b)).toBe(true);
  });

  it("largest logit gets the highest probability", () => {
    const p = data(softmax(createTensor([2, 1, 3], [3])));
    // index 2 had the largest logit, so it must hold the largest probability
    expect(p[2]! > p[0]! && p[2]! > p[1]!).toBe(true);
  });

  it("identical logits produce a uniform distribution", () => {
    const p = data(softmax(createTensor([5, 5, 5], [3])));
    // equal scores → equal probabilities
    expect(closeAll(p, [1 / 3, 1 / 3, 1 / 3])).toBe(true);
  });

  it("along axis=1, EACH ROW sums to 1 independently", () => {
    const t = createTensor([1, 2, 3, 1, 2, 3], [2, 3]);
    const out = data(softmax(t, 1));
    const row0 = out.slice(0, 3).reduce((a, b) => a + b, 0);
    const row1 = out.slice(3, 6).reduce((a, b) => a + b, 0);
    // a row-wise softmax makes every row a distribution, not the whole tensor
    expect(Math.abs(row0 - 1)).toBeLessThan(EPSILON);
    expect(Math.abs(row1 - 1)).toBeLessThan(EPSILON);
  });
});
