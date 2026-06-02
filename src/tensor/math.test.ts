/**
 * Tests for tensor/math.ts
 * Chapter 06 — Math Primitives
 *
 * Run: bun test src/tensor/math.test.ts
 *
 * Tests are named after the mathematical property they check. Each expect()
 * has a comment stating the fact being verified. Pitfall cases (log(0),
 * sigmoid overflow, shape preservation) are checked explicitly.
 */
import { describe, it, expect } from "bun:test";
import { exp, log, sqrt, pow, abs, clip, tanh, sigmoid } from "./math.ts";
import { createTensor } from "./types.ts";

const EPSILON = 1e-6;
const data = (t: { data: Float64Array }) => Array.from(t.data);
const close = (a: number, b: number) => Math.abs(a - b) < EPSILON;
const closeAll = (a: number[], b: number[], tol = EPSILON) =>
  a.length === b.length && a.every((v, i) => Math.abs(v - (b[i] as number)) < tol);

describe("exp", () => {
  it("exp(0) === 1 for every element", () => {
    const t = createTensor([0, 0, 0], [3]);
    // e⁰ = 1
    expect(data(exp(t))).toEqual([1, 1, 1]);
  });

  it("exp(1) ≈ 2.71828", () => {
    // e¹ = e
    expect(close(data(exp(createTensor([1], [1])))[0]!, Math.E)).toBe(true);
  });

  it("preserves shape (element-wise never reshapes)", () => {
    const t = createTensor([0, 1, 2, 3, 4, 5], [2, 3]);
    expect(exp(t).shape).toEqual([2, 3]);
  });
});

describe("log", () => {
  it("log(1) === 0", () => {
    // ln(1) = 0
    expect(data(log(createTensor([1], [1])))).toEqual([0]);
  });

  it("log(exp(x)) ≈ x — round-trip identity", () => {
    const t = createTensor([-2, 0.5, 3], [3]);
    // log undoes exp
    expect(closeAll(data(log(exp(t))), [-2, 0.5, 3])).toBe(true);
  });

  it("exp(log(x)) ≈ x for positive x", () => {
    const t = createTensor([0.1, 1, 5], [3]);
    // exp undoes log on positive inputs
    expect(closeAll(data(exp(log(t))), [0.1, 1, 5])).toBe(true);
  });

  it("pitfall: log(0) is -Infinity (pure log does NOT clamp)", () => {
    // the primitive is pure; clipping is the caller's responsibility
    expect(data(log(createTensor([0], [1])))[0]!).toBe(-Infinity);
  });
});

describe("sqrt", () => {
  it("sqrt(4) === 2 and sqrt(9) === 3 element-wise", () => {
    expect(data(sqrt(createTensor([4, 9, 16], [3])))).toEqual([2, 3, 4]);
  });

  it("sqrt(0) === 0", () => {
    expect(data(sqrt(createTensor([0], [1])))).toEqual([0]);
  });
});

describe("pow", () => {
  it("pow(x, 2) squares each element", () => {
    // (x)² — the squared deviations used in variance
    expect(data(pow(createTensor([1, 2, 3, -4], [4]), 2))).toEqual([1, 4, 9, 16]);
  });

  it("pow(x, 0.5) equals sqrt", () => {
    expect(closeAll(data(pow(createTensor([4, 9], [2]), 0.5)), [2, 3])).toBe(true);
  });
});

describe("abs", () => {
  it("|x| is non-negative and drops the sign", () => {
    expect(data(abs(createTensor([-3, 0, 2, -0.5], [4])))).toEqual([3, 0, 2, 0.5]);
  });
});

describe("clip", () => {
  it("values below min become min", () => {
    expect(data(clip(createTensor([-5, -1], [2]), 0, 1))).toEqual([0, 0]);
  });

  it("values above max become max", () => {
    expect(data(clip(createTensor([9, 100], [2]), 0, 1))).toEqual([1, 1]);
  });

  it("values within [min, max] are unchanged", () => {
    expect(data(clip(createTensor([0, 0.5, 1], [3]), 0, 1))).toEqual([0, 0.5, 1]);
  });

  it("clip(x, 0, Infinity) is ReLU (keep positives, zero the rest)", () => {
    expect(data(clip(createTensor([-2, -0.1, 0, 3], [4]), 0, Infinity))).toEqual([0, 0, 0, 3]);
  });
});

describe("tanh", () => {
  it("tanh(0) === 0", () => {
    expect(data(tanh(createTensor([0], [1])))).toEqual([0]);
  });

  it("output stays in (-1, 1) for typical inputs", () => {
    // mathematically the range is the OPEN interval (-1, 1); for very large |x|
    // float64 saturates to exactly ±1, so we check moderate inputs here and the
    // saturation separately in the next test.
    const out = data(tanh(createTensor([-3, -1, 0, 1, 3], [5])));
    expect(out.every((v) => v > -1 && v < 1)).toBe(true);
  });

  it("large inputs squash toward ±1", () => {
    const out = data(tanh(createTensor([-10, 10], [2])));
    expect(close(out[0]!, -1) && close(out[1]!, 1)).toBe(true);
  });
});

describe("sigmoid", () => {
  it("sigmoid(0) === 0.5", () => {
    expect(close(data(sigmoid(createTensor([0], [1])))[0]!, 0.5)).toBe(true);
  });

  it("output stays in (0, 1) for typical inputs", () => {
    // open interval (0, 1) mathematically; float64 saturates to exactly 0/1 for
    // extreme |x|, so we check moderate inputs here.
    const out = data(sigmoid(createTensor([-10, -2, 0, 2, 10], [5])));
    expect(out.every((v) => v > 0 && v < 1)).toBe(true);
  });

  it("matches 1/(1+e⁻ˣ): sigmoid(2) ≈ 0.8808", () => {
    expect(close(data(sigmoid(createTensor([2], [1])))[0]!, 0.880797)).toBe(true);
  });

  it("pitfall: very negative input does NOT overflow to NaN", () => {
    // naive 1/(1+exp(-x)) would compute exp(1000)=Infinity; the stable branch
    // keeps this finite and ≈ 0
    const out = data(sigmoid(createTensor([-1000], [1])))[0]!;
    expect(Number.isFinite(out) && close(out, 0)).toBe(true);
  });
});
