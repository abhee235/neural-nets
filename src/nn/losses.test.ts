/**
 * Tests for nn/losses.ts
 * Chapter 12 — Loss Functions
 *
 * Run: bun test src/nn/losses.test.ts
 *
 * The fixtures are the chapter's own examples, so every expected number is
 * one the doc derives by hand:
 *
 *   Regression (doc section 1) — the temperatures:
 *     predicted [32, 28, 31]  vs  actual [35, 28, 30]
 *     MSE = 3.333333    gradient = [-2, 0, 0.666667]
 *
 *   Classification (doc sections 5–15) — "the cat ___", logits [1, 2, 3]:
 *     softmax [0.090031, 0.244728, 0.665241]
 *     loss (truth sat/ran/flew) = 2.407606 / 1.407606 / 0.407606
 *     gradient (truth sat)      = [-0.909969, 0.244728, 0.665241]
 *
 *   logSumExp: [1,2,3] → 3.407606,  [1000,1001,1002] → 1002.407606
 */
import { describe, it, expect } from "bun:test";
import { mseLoss, logSumExp, crossEntropyFromLogits } from "./losses.ts";
import { TensorValue } from "../autograd/grad.ts";
import { createTensor } from "../tensor/types.ts";

const EPSILON = 1e-6;

/** Compare two rows elementwise within a tolerance. */
function expectRow(actual: number[], expected: number[], tol = EPSILON): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    expect(Math.abs(actual[i]! - expected[i]!)).toBeLessThan(tol);
  }
}

/** Centered-difference gradient check for a scalar-valued loss of one input. */
function numericalGradientMatches(
  lossOf: (t: TensorValue) => TensorValue,
  values: number[],
  shape: number[],
  tol = 1e-5,
): boolean {
  const h = 1e-6;
  const analytic = new TensorValue(createTensor(values, shape));
  lossOf(analytic).backward();
  for (let i = 0; i < values.length; i++) {
    const up = [...values], down = [...values];
    up[i]! += h;
    down[i]! -= h;
    const numeric =
      (lossOf(new TensorValue(createTensor(up, shape))).data.data[0]! -
        lossOf(new TensorValue(createTensor(down, shape))).data.data[0]!) /
      (2 * h);
    if (Math.abs(numeric - analytic.grad!.data[i]!) > tol) return false;
  }
  return true;
}

describe("mseLoss", () => {
  it("is zero exactly when predictions equal targets", () => {
    // A perfect model pays nothing — requirement 2 of a loss.
    const predictions = new TensorValue(createTensor([35, 28, 30], [3]));
    const loss = mseLoss(predictions, createTensor([35, 28, 30], [3]));
    expect(loss.data.data[0]).toBe(0);
    // And has nothing to teach: the gradient at the minimum is zero.
    loss.backward();
    expectRow(Array.from(predictions.grad!.data), [0, 0, 0]);
  });

  it("reproduces the chapter's temperatures: MSE = 3.333333", () => {
    // differences [-3, 0, 1] → squared [9, 0, 1] → mean 10/3.
    const predictions = new TensorValue(createTensor([32, 28, 31], [3]));
    const loss = mseLoss(predictions, createTensor([35, 28, 30], [3]));
    expect(Math.abs(loss.data.data[0]! - 10 / 3)).toBeLessThan(EPSILON);
  });

  it("gradient is (2/n)(p − y): points every prediction at its target", () => {
    const predictions = new TensorValue(createTensor([32, 28, 31], [3]));
    mseLoss(predictions, createTensor([35, 28, 30], [3])).backward();
    // Monday 3° too low → negative → pushed up. Wednesday too high → down.
    // [-6, 0, 2] here would mean the mean was lost (sum-not-mean bug).
    expectRow(Array.from(predictions.grad!.data), [-2, 0, 2 / 3]);
  });

  it("averages over n — the loss does not grow with more predictions", () => {
    // Same per-element error, twice the elements, same loss. This is what
    // keeps the learning rate independent of batch size.
    const three = mseLoss(new TensorValue(createTensor([1, 1, 1], [3])), createTensor([0, 0, 0], [3]));
    const six = mseLoss(new TensorValue(createTensor([1, 1, 1, 1, 1, 1], [6])), createTensor([0, 0, 0, 0, 0, 0], [6]));
    expect(Math.abs(three.data.data[0]! - six.data.data[0]!)).toBeLessThan(EPSILON);
  });

  it("is symmetric in the error: +e and −e cost the same", () => {
    // Squaring removes direction — being too high and too low both count.
    const over = mseLoss(new TensorValue(createTensor([12], [1])), createTensor([10], [1]));
    const under = mseLoss(new TensorValue(createTensor([8], [1])), createTensor([10], [1]));
    expect(over.data.data[0]).toBe(under.data.data[0]);
  });

  it("keeps the graph connected: gradient reaches a weight BEHIND the predictions", () => {
    // The doc's one-weight model: p = x·w with x=2, w=3, target 10.
    // dL/dw = 2(p−y)·x = 2(6−10)·2 = -16 — the section 2 walkthrough.
    const w = new TensorValue(createTensor([3], [1]));
    const x = new TensorValue(createTensor([2], [1]));
    mseLoss(x.mul(w), createTensor([10], [1])).backward();
    expect(Math.abs(w.grad!.data[0]! - -16)).toBeLessThan(EPSILON);
  });

  it("numerical gradient check passes", () => {
    expect(
      numericalGradientMatches(
        (t) => mseLoss(t, createTensor([35, 28, 30], [3])),
        [32, 28, 31],
        [3],
      ),
    ).toBe(true);
  });
});

describe("logSumExp", () => {
  it("logSumExp([1,2,3]) = 3.407606 — matches log(e¹+e²+e³) done directly", () => {
    const out = logSumExp(new TensorValue(createTensor([1, 2, 3], [3])));
    const direct = Math.log(Math.E ** 1 + Math.E ** 2 + Math.E ** 3);
    expect(Math.abs(out.data.data[0]! - direct)).toBeLessThan(EPSILON);
  });

  it("survives logits the naive route cannot: [1000,1001,1002] → 1002.407606, finite", () => {
    // exp(1000) alone is Infinity; the max subtraction keeps every exponent ≤ 0.
    const out = logSumExp(new TensorValue(createTensor([1000, 1001, 1002], [3])));
    expect(Number.isFinite(out.data.data[0]!)).toBe(true);
    expect(Math.abs(out.data.data[0]! - 1002.407606)).toBeLessThan(1e-5);
  });

  it("shifts exactly with its input: logSumExp(z + c) = logSumExp(z) + c", () => {
    // The fractional parts of 3.407606 and 1002.407606 match — this is why.
    const small = logSumExp(new TensorValue(createTensor([1, 2, 3], [3])));
    const big = logSumExp(new TensorValue(createTensor([1000, 1001, 1002], [3])));
    expect(Math.abs(big.data.data[0]! - small.data.data[0]! - 999)).toBeLessThan(1e-5);
  });

  it("is a smooth max: ≥ max(z), and → max(z) as the gap widens", () => {
    // With one dominant logit the sum is ~all that term: LSE([0,10]) ≈ 10.
    const out = logSumExp(new TensorValue(createTensor([0, 10], [2])));
    expect(out.data.data[0]!).toBeGreaterThan(10);
    expect(out.data.data[0]! - 10).toBeLessThan(1e-4);
  });

  it("its gradient IS softmax — the doc's section 15 fact", () => {
    const x = new TensorValue(createTensor([1, 2, 3], [3]));
    logSumExp(x).backward();
    expectRow(Array.from(x.grad!.data), [0.090031, 0.244728, 0.665241], 1e-6);
  });

  it("reduces per row on a batch, keepDims, and gradients stay per-row", () => {
    const m = new TensorValue(createTensor([1, 2, 3, 1000, 1001, 1002], [2, 3]));
    const out = logSumExp(m);
    // keepDims: [2, 1], so it broadcasts against full rows downstream.
    expect(out.data.shape).toEqual([2, 1]);
    out.sum().backward();
    // Each row's gradient is that row's softmax — the huge row saturates
    // to the same softmax as the small one (shift invariance again).
    expectRow(Array.from(m.grad!.data), [0.090031, 0.244728, 0.665241, 0.090031, 0.244728, 0.665241], 1e-6);
  });

  it("numerical gradient check passes", () => {
    expect(
      numericalGradientMatches((t) => logSumExp(t).sum(), [0.5, -1.2, 2, 0.3], [4]),
    ).toBe(true);
  });
});

describe("crossEntropyFromLogits", () => {
  const logits123 = () => new TensorValue(createTensor([1, 2, 3], [1, 3]));

  it("loss = −log(p of the true class): truth sat → 2.407606", () => {
    // The stable form logSumExp(z) − z_y must equal the long way exactly.
    const loss = crossEntropyFromLogits(logits123(), createTensor([1, 0, 0], [1, 3]));
    expect(Math.abs(loss.data.data[0]! - -Math.log(0.090031))).toBeLessThan(1e-5);
  });

  it("losses for the three truths differ by exactly the logit gaps", () => {
    // logSumExp(z) is the same whichever class is true; only z_y changes.
    const sat = crossEntropyFromLogits(logits123(), createTensor([1, 0, 0], [1, 3])).data.data[0]!;
    const ran = crossEntropyFromLogits(logits123(), createTensor([0, 1, 0], [1, 3])).data.data[0]!;
    const flew = crossEntropyFromLogits(logits123(), createTensor([0, 0, 1], [1, 3])).data.data[0]!;
    expect(Math.abs(sat - ran - 1)).toBeLessThan(EPSILON);
    expect(Math.abs(ran - flew - 1)).toBeLessThan(EPSILON);
  });

  it("shift invariance: logits [1000,1001,1002] give the identical loss", () => {
    // The reason the naive log(softmax) breaks and this form does not.
    const small = crossEntropyFromLogits(logits123(), createTensor([0, 0, 1], [1, 3])).data.data[0]!;
    const big = crossEntropyFromLogits(
      new TensorValue(createTensor([1000, 1001, 1002], [1, 3])),
      createTensor([0, 0, 1], [1, 3]),
    ).data.data[0]!;
    expect(Number.isFinite(big)).toBe(true);
    expect(Math.abs(big - small)).toBeLessThan(1e-5);
  });

  it("a confidently wrong model gets an unbounded loss, not a capped one", () => {
    // Truth first, all confidence on the last class: loss ≈ the logit gap.
    // (MSE would cap at 0.667 here — the doc's section 10 measurement.)
    const loss = crossEntropyFromLogits(
      new TensorValue(createTensor([0, 0, 20], [1, 3])),
      createTensor([1, 0, 0], [1, 3]),
    );
    expect(loss.data.data[0]!).toBeGreaterThan(19);
  });

  it("gradient is p − y: the truth is pulled up, the rest pushed down", () => {
    const logits = logits123();
    crossEntropyFromLogits(logits, createTensor([1, 0, 0], [1, 3])).backward();
    // softmax − one-hot, from the chapter's row. flew took the most
    // probability it should not have, so it is pushed down hardest.
    expectRow(Array.from(logits.grad!.data), [-0.909969, 0.244728, 0.665241], 1e-6);
  });

  it("gradient sums to zero: the loss only redistributes confidence", () => {
    // p sums to 1 and y sums to 1, so p − y sums to 0 — cross-entropy can
    // never push every logit the same way, matching shift invariance.
    const logits = logits123();
    crossEntropyFromLogits(logits, createTensor([0, 1, 0], [1, 3])).backward();
    const total = Array.from(logits.grad!.data).reduce((a, b) => a + b, 0);
    expect(Math.abs(total)).toBeLessThan(1e-12);
  });

  it("averages over the batch, and each row's gradient still sums to ~0", () => {
    // Two rows, different truths: the loss is the mean of the row losses.
    const logits = new TensorValue(createTensor([1, 2, 3, 3, 1, 0.5], [2, 3]));
    const loss = crossEntropyFromLogits(logits, createTensor([1, 0, 0, 1, 0, 0], [2, 3]));
    const rowSat = -Math.log(0.090031);
    const lse2 = Math.log(Math.exp(0) + Math.exp(-2) + Math.exp(-2.5)) + 3; // logSumExp([3,1,0.5])
    expect(Math.abs(loss.data.data[0]! - (rowSat + (lse2 - 3)) / 2)).toBeLessThan(1e-5);
    loss.backward();
    const g = Array.from(logits.grad!.data);
    expect(Math.abs(g[0]! + g[1]! + g[2]!)).toBeLessThan(1e-12);
    expect(Math.abs(g[3]! + g[4]! + g[5]!)).toBeLessThan(1e-12);
  });

  it("numerical gradient check passes", () => {
    expect(
      numericalGradientMatches(
        (t) => crossEntropyFromLogits(t, createTensor([0, 1, 0], [1, 3])),
        [0.5, -1.2, 2],
        [1, 3],
      ),
    ).toBe(true);
  });
});
