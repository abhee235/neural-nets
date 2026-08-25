/**
 * Tests for nn/activations.ts
 * Chapter 11 — Activation Functions
 *
 * Run: bun test src/nn/activations.test.ts
 *
 * The fixtures are the chapter's own running row, so every expected number
 * below is one the doc derives by hand:
 *
 *   x           = [ -2       -1       0       1       2      ]
 *   relu(x)     = [  0.0000   0.0000  0.0000  1.0000  2.0000 ]
 *   gelu(x)     = [ -0.0454  -0.1588  0.0000  0.8412  1.9546 ]
 *   gelu'(x)    = [ -0.0861  -0.0830  0.5000  1.0830  1.0861 ]
 *   sigmoid(x)  = [  0.1192   0.2689  0.5000  0.7311  0.8808 ]
 *   sigmoid'(x) = [  0.1050   0.1966  0.2500  0.1966  0.1050 ]
 *
 *   softmax([1,2,3]) = [ 0.090031  0.244728  0.665241 ]   (sums to 1)
 */
import { describe, it, expect } from "bun:test";
import { relu, gelu, sigmoid, softmax } from "./activations.ts";
import { TensorValue, checkTensorGradient } from "../autograd/grad.ts";
import { createTensor } from "../tensor/types.ts";

const EPSILON = 1e-6;

/** The chapter's running row, as a fresh leaf. A new graph per backward pass. */
const row = () => new TensorValue(createTensor([-2, -1, 0, 1, 2], [5]));
const vals = (t: TensorValue) => Array.from(t.data.data);
const grads = (t: TensorValue) => Array.from(t.grad!.data);

/** Compare two rows elementwise within a tolerance. */
function expectRow(actual: number[], expected: number[], tol = EPSILON): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    expect(Math.abs(actual[i]! - expected[i]!)).toBeLessThan(tol);
  }
}

/**
 * Run a backward pass with an upstream gradient of all ones.
 * Summing makes every output cell's upstream exactly 1, which is the
 * "out.grad = all ones" row the chapter traces by hand.
 */
function backwardOfSum(f: (x: TensorValue) => TensorValue, x: TensorValue): void {
  f(x).sum().backward();
}

describe("relu", () => {
  it("relu(2) === 2 — a positive input passes through untouched", () => {
    const x = new TensorValue(createTensor([2], [1]));
    expect(relu(x).data.data[0]).toBe(2);
  });

  it("relu(-2) === 0 — a negative input is flattened to zero", () => {
    const x = new TensorValue(createTensor([-2], [1]));
    expect(relu(x).data.data[0]).toBe(0);
  });

  it("reproduces the chapter's forward row", () => {
    // max(0, v) cell by cell: the three non-positive entries collapse to 0.
    expectRow(vals(relu(row())), [0, 0, 0, 1, 2]);
  });

  it("is a gate, not a scale: the gradient is 0 or 1, never in between", () => {
    const x = row();
    backwardOfSum(relu, x);
    // relu'(x) ⊙ [1,1,1,1,1]. At x = 0 exactly we chose the 0 convention (Ch 08).
    expectRow(grads(x), [0, 0, 0, 1, 1]);
  });

  it("preserves shape — elementwise ops never change it", () => {
    // Output cell i depends on input cell i alone, so no broadcasting, no reduction.
    const m = new TensorValue(createTensor([-1, 2, -3, 4, 5, -6], [2, 3]));
    const out = relu(m);
    expect(out.data.shape).toEqual([2, 3]);
    out.sum().backward();
    expect(m.grad!.shape).toEqual([2, 3]);
    expectRow(Array.from(out.data.data), [0, 2, 0, 4, 5, 0]);
  });

  it("accumulates when the same node is used twice", () => {
    // x reaches the loss by two routes, so the contributions SUM (Ch 08b's `+=`).
    const y = new TensorValue(createTensor([1, 2], [2]));
    relu(y).add(relu(y)).sum().backward();
    expectRow(grads(y), [2, 2]);
  });

  it("keeps the graph connected — out records its parent", () => {
    // A missing _inputs severs backward silently; Ch 10's matMul had this bug.
    const x = row();
    expect(relu(x)._inputs.length).toBe(1);
    expect(relu(x)._inputs[0]).toBe(x);
  });

  it("numerical gradient check passes (x ≠ 0)", () => {
    // Away from the corner the analytic gradient must match finite differences.
    // At x = 0 a centered difference straddles the kink and averages the two
    // one-sided slopes to 0.5, which disagrees with every convention.
    const x = new TensorValue(createTensor([-2.5, -1, 0.5, 1, 3.7], [5]));
    expect(checkTensorGradient((ins) => relu(ins[0]!), [x])).toBe(true);
  });
});

describe("gelu", () => {
  it("gelu(0) === 0 exactly", () => {
    // gelu(x) = x · fraction(x), and x is 0, so the product is 0 whatever the fraction.
    const x = new TensorValue(createTensor([0], [1]));
    expect(gelu(x).data.data[0]).toBe(0);
  });

  it("reproduces the chapter's forward row", () => {
    // The tanh approximation, accurate to ~1e-4 against exact Φ.
    expectRow(vals(gelu(row())), [-0.0454, -0.1588, 0.0, 0.8412, 1.9546], 1e-4);
  });

  it("reproduces the chapter's derivative row", () => {
    const x = row();
    backwardOfSum(gelu, x);
    expectRow(grads(x), [-0.0861, -0.083, 0.5, 1.083, 1.0861], 1e-4);
  });

  it("gelu'(0) === 0.5 — half open, not 0 or 1", () => {
    // From the product rule: fraction(0) + 0·rate = 0.5 + 0. The second term
    // vanishes because x is zero, leaving the fraction alone.
    const x = new TensorValue(createTensor([0], [1]));
    backwardOfSum(gelu, x);
    expect(Math.abs(x.grad!.data[0]! - 0.5)).toBeLessThan(EPSILON);
  });

  it("has a NEGATIVE derivative at x = -1 — gelu is not monotonic", () => {
    // fraction(-1) + (-1)·rate = 0.1587 - 0.2420 < 0: the second term wins,
    // so the curve dips below zero before climbing.
    const x = new TensorValue(createTensor([-1], [1]));
    backwardOfSum(gelu, x);
    expect(x.grad!.data[0]!).toBeLessThan(0);
  });

  it("dips below zero on the left, bottoming near x = -0.75", () => {
    // The dip is real, not an artifact of the tanh approximation.
    const probe = new TensorValue(createTensor([-0.75], [1]));
    expect(gelu(probe).data.data[0]!).toBeLessThan(-0.16);
    expect(gelu(probe).data.data[0]!).toBeGreaterThan(-0.18);
  });

  it("is smooth at x = 0, unlike relu's corner", () => {
    // Approaching 0 from both sides gives nearly the same slope; relu's
    // one-sided slopes there are 0 and 1 and never agree.
    const h = 1e-4;
    const g = (v: number) => gelu(new TensorValue(createTensor([v], [1]))).data.data[0]!;
    const leftSlope = (g(-h) - g(-2 * h)) / h;
    const rightSlope = (g(2 * h) - g(h)) / h;
    expect(Math.abs(leftSlope - rightSlope)).toBeLessThan(1e-3);
  });

  it("approaches relu for large positive x", () => {
    // At x = 5 the fraction kept is ~1, so gelu returns almost all of x.
    const x = new TensorValue(createTensor([5], [1]));
    expect(Math.abs(gelu(x).data.data[0]! - 5)).toBeLessThan(1e-5);
  });

  it("numerical gradient check passes", () => {
    // The derivative has the most terms in this file — a dropped constant
    // produces plausible-looking numbers that only this test catches.
    const x = new TensorValue(createTensor([-2, -0.5, 0.3, 1, 2.4], [5]));
    expect(checkTensorGradient((ins) => gelu(ins[0]!), [x])).toBe(true);
  });
});

describe("sigmoid", () => {
  it("sigmoid(0) === 0.5", () => {
    const x = new TensorValue(createTensor([0], [1]));
    expect(sigmoid(x).data.data[0]).toBe(0.5);
  });

  it("output is strictly inside (0,1) across the usable range", () => {
    // The bounded range is the whole point — and the cause of saturation.
    // The bound is tested on [-30, 30] rather than "any finite input" because
    // the strict version is not reachable in float64: 1 + e^-40 rounds to 1.0
    // (machine epsilon is 2.2e-16), so sigmoid(40) is exactly 1 however it is
    // computed. Mathematically strict, numerically not. See the saturation
    // test below for what happens past the edge.
    const x = new TensorValue(createTensor([-30, -10, -1, 0, 1, 10, 30], [7]));
    for (const v of vals(sigmoid(x))) {
      expect(v).toBeGreaterThan(0);
      expect(v).toBeLessThan(1);
    }
  });

  it("saturates to exactly 0 and 1 outside that range — and never NaNs", () => {
    // Documents the real numerical limit rather than pretending it away.
    // 1/(1+e^-x) with x = -1000 computes 1/(1+Infinity) = 0: no NaN, but the
    // output has hit the floor and its gradient is gone.
    const x = new TensorValue(createTensor([-1000, 1000], [2]));
    const out = vals(sigmoid(x));
    expect(out[0]).toBe(0);
    expect(out[1]).toBe(1);
    for (const v of out) expect(Number.isNaN(v)).toBe(false);
  });

  it("reproduces the chapter's forward row", () => {
    expectRow(vals(sigmoid(row())), [0.1192, 0.2689, 0.5, 0.7311, 0.8808], 1e-4);
  });

  it("reproduces the chapter's derivative row", () => {
    const x = row();
    backwardOfSum(sigmoid, x);
    expectRow(grads(x), [0.105, 0.1966, 0.25, 0.1966, 0.105], 1e-4);
  });

  it("its derivative never exceeds 0.25, and peaks exactly at x = 0", () => {
    // The ceiling that starves deep networks: 0.25^10 ≈ 9.5e-7 (section 10).
    const x = new TensorValue(createTensor([-6, -3, -1, 0, 1, 3, 6], [7]));
    backwardOfSum(sigmoid, x);
    const g = grads(x);
    for (const v of g) expect(v).toBeLessThanOrEqual(0.25 + EPSILON);
    // index 3 is x = 0
    expect(Math.abs(g[3]! - 0.25)).toBeLessThan(EPSILON);
  });

  it("saturates: the gradient at x = 10 is nearly zero", () => {
    // The curve has nowhere left to rise, so nudging the input barely moves it.
    const x = new TensorValue(createTensor([10], [1]));
    backwardOfSum(sigmoid, x);
    expect(x.grad!.data[0]!).toBeLessThan(1e-4);
  });

  it("backward reads out.data, not x.data — checked away from the origin", () => {
    // σ'(1) = σ(1)(1-σ(1)) = 0.19661. The popular bug x·(1-x) gives 1·(1-1) = 0
    // here, and happens to be right only at x = 0 — which is why this is at x = 1.
    const x = new TensorValue(createTensor([1], [1]));
    backwardOfSum(sigmoid, x);
    expect(Math.abs(x.grad!.data[0]! - 0.1966119)).toBeLessThan(1e-6);
  });

  it("numerical gradient check passes", () => {
    const x = new TensorValue(createTensor([-2, -0.5, 0, 1, 2.5], [5]));
    expect(checkTensorGradient((ins) => sigmoid(ins[0]!), [x])).toBe(true);
  });
});

describe("softmax", () => {
  it("output sums to 1.0 along the target axis", () => {
    // That is what makes it a probability distribution.
    const x = new TensorValue(createTensor([1, 2, 3], [3]));
    const total = vals(softmax(x)).reduce((a, b) => a + b, 0);
    expect(Math.abs(total - 1)).toBeLessThan(EPSILON);
  });

  it("reproduces the chapter's row", () => {
    const x = new TensorValue(createTensor([1, 2, 3], [3]));
    expectRow(vals(softmax(x)), [0.090031, 0.244728, 0.665241], 1e-6);
  });

  it("shift invariance: softmax(x) === softmax(x + c)", () => {
    // Subtracting the max is therefore safe, and is what keeps exp(1000) from
    // ever appearing. Proof: docs/deep-dives/ch-05-why-subtract-the-max.md
    const a = vals(softmax(new TensorValue(createTensor([1, 2, 3], [3]))));
    const b = vals(softmax(new TensorValue(createTensor([1000, 1001, 1002], [3]))));
    expectRow(a, b);
  });

  it("is not elementwise: changing one input moves every output", () => {
    // The denominator sums over the whole axis, so the outputs are coupled.
    const a = vals(softmax(new TensorValue(createTensor([1, 2, 3], [3]))));
    const b = vals(softmax(new TensorValue(createTensor([1, 2, 5], [3]))));
    expect(Math.abs(a[0]! - b[0]!)).toBeGreaterThan(1e-3);
  });

  it("preserves shape", () => {
    const x = new TensorValue(createTensor([1, 2, 3, 4, 5, 6], [2, 3]));
    expect(softmax(x).data.shape).toEqual([2, 3]);
  });

  it("normalises each row independently on a 2-D tensor", () => {
    // Default axis is the last one, so each row sums to 1 on its own.
    const x = new TensorValue(createTensor([1, 2, 3, 10, 20, 30], [2, 3]));
    const out = vals(softmax(x));
    expect(Math.abs(out[0]! + out[1]! + out[2]! - 1)).toBeLessThan(EPSILON);
    expect(Math.abs(out[3]! + out[4]! + out[5]! - 1)).toBeLessThan(EPSILON);
  });

  it("numerical gradient check passes", () => {
    // The Jacobian collapses to s ⊙ (out.grad − Σ(out.grad ⊙ s)). Getting the
    // weighted sum wrong still produces a plausible-looking tensor, so this is
    // the test that actually pins the formula down.
    const x = new TensorValue(createTensor([0.5, -1.2, 2, 0.3], [4]));
    expect(checkTensorGradient((ins) => softmax(ins[0]!), [x])).toBe(true);
  });
});
