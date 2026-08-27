/**
 * Tests for optim/sgd.ts — the TensorValue rebuild
 * Chapter 14 — Optimizers
 *
 * Run: bun test src/optim/sgd.test.ts
 *
 * Fixtures are the chapter's own numbers:
 *
 *   the bowl      L = (θ − 5)², θ₀ = 0, lr = 0.1
 *                 → 1, 1.8, 2.44, 2.952, each step 0.8× the last
 *
 *   momentum      β = 0.9, v₀ = 0, gradient +1 every step
 *                 → 1, 1.9, 2.71, 3.439 … 5.695 after 8
 *                 gradient alternating ±1 → 1, −0.1, 0.91, −0.181 … −0.300
 *
 * The scalar originals live in sgd-scalar.ts and have their own tests; these
 * check the same rules on tensors, plus the things only tensors can get
 * wrong — shape, graph leakage, and per-parameter state.
 */
import { describe, it, expect } from "bun:test";
import { SGD, SGDMomentum } from "./sgd.ts";
import { TensorValue } from "../autograd/grad.ts";
import { createTensor } from "../tensor/types.ts";
import { Linear } from "../nn/linear.ts";
import { mseLoss } from "../nn/losses.ts";

const EPSILON = 1e-9;

function expectRow(actual: number[], expected: number[], tol = EPSILON): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    expect(Math.abs(actual[i]! - expected[i]!)).toBeLessThan(tol);
  }
}

/** A leaf holding one number, so hand-computed traces are readable. */
const scalarParam = (v: number) => new TensorValue(createTensor([v], [1]));

/** Put a gradient on a parameter directly, standing in for a backward pass. */
function setGrad(p: TensorValue, ...values: number[]): void {
  p.grad = createTensor(values, p.data.shape);
}

describe("SGD", () => {
  it("takes the chapter's bowl trace: 1, 1.8, 2.44, 2.952", () => {
    // L = (θ−5)², so dL/dθ = 2(θ−5). mseLoss on one element is exactly that.
    const theta = scalarParam(0);
    const opt = new SGD([theta], 0.1);
    const target = createTensor([5], [1]);
    const seen: number[] = [];
    for (let i = 0; i < 4; i++) {
      opt.zeroGrad();
      mseLoss(theta, target).backward();
      opt.step();
      seen.push(theta.data.data[0]!);
    }
    expectRow(seen, [1, 1.8, 2.44, 2.952], 1e-9);
  });

  it("each step is 0.8× the previous one, because the slope flattens", () => {
    const theta = scalarParam(0);
    const opt = new SGD([theta], 0.1);
    const target = createTensor([5], [1]);
    const steps: number[] = [];
    let previous = 0;
    for (let i = 0; i < 4; i++) {
      opt.zeroGrad();
      mseLoss(theta, target).backward();
      opt.step();
      steps.push(theta.data.data[0]! - previous);
      previous = theta.data.data[0]!;
    }
    for (let i = 1; i < steps.length; i++) {
      expect(Math.abs(steps[i]! / steps[i - 1]! - 0.8)).toBeLessThan(1e-9);
    }
  });

  it("moves a parameter DOWN when its gradient is positive", () => {
    // The sign is the whole rule: θ ← θ − lr·g.
    const p = scalarParam(3);
    setGrad(p, 2);
    new SGD([p], 0.1).step();
    expect(Math.abs(p.data.data[0]! - 2.8)).toBeLessThan(EPSILON);
  });

  it("updates every element of a multi-element parameter independently", () => {
    const p = new TensorValue(createTensor([1, 2, 3], [3]));
    setGrad(p, 10, 0, -10);
    new SGD([p], 0.1).step();
    // element 1 has zero gradient and must not move at all
    expectRow(Array.from(p.data.data), [0, 2, 4]);
  });

  it("preserves parameter shape", () => {
    const p = new TensorValue(createTensor([1, 2, 3, 4, 5, 6], [3, 2]));
    setGrad(p, 1, 1, 1, 1, 1, 1);
    new SGD([p], 0.1).step();
    expect(p.data.shape).toEqual([3, 2]);
  });

  it("never builds graph nodes — the parameter stays a leaf", () => {
    // If step() used TensorValue methods instead of tensor functions, the
    // graph would grow by a node per parameter per iteration, forever.
    const theta = scalarParam(0);
    const opt = new SGD([theta], 0.1);
    const target = createTensor([5], [1]);
    for (let i = 0; i < 5; i++) {
      opt.zeroGrad();
      mseLoss(theta, target).backward();
      opt.step();
    }
    expect(theta._inputs.length).toBe(0);
  });

  it("skips a parameter whose gradient is still null", () => {
    // A parameter that took no part in the forward pass must not crash.
    const used = scalarParam(1);
    const unused = scalarParam(7);
    setGrad(used, 1);
    new SGD([used, unused], 0.1).step();
    expect(unused.data.data[0]).toBe(7);
  });

  it("zeroGrad resets to null, not to a zeros tensor", () => {
    // Ch 10's accumulate() treats null as "first contribution" and assigns.
    // A zeros tensor would make it take the add-branch instead.
    const p = scalarParam(0);
    setGrad(p, -10);
    const opt = new SGD([p], 0.1);
    expect(p.grad).not.toBe(null);
    opt.zeroGrad();
    expect(p.grad).toBe(null);
  });

  it("holds the caller's own objects, not copies", () => {
    // An optimizer given copies would update tensors nothing reads.
    const p = scalarParam(1);
    expect(new SGD([p], 0.1).params[0]).toBe(p);
  });

  it("trains a real Ch 13 Linear layer through parameters()", () => {
    // The contract closing: the optimizer walks a flat list and never knows
    // a layer exists.
    const layer = new Linear(2, 3);
    const opt = new SGD(layer.parameters(), 0.05);
    const x = new TensorValue(createTensor([1, 2], [1, 2]));
    const want = createTensor([1, 2, 3], [1, 3]);
    for (let i = 0; i < 60; i++) {
      opt.zeroGrad();
      mseLoss(layer.forward(x), want).backward();
      opt.step();
    }
    expectRow(Array.from(layer.forward(x).data.data), [1, 2, 3], 1e-3);
  });
});

describe("SGDMomentum", () => {
  it("builds velocity on agreement: 1, 1.9, 2.71, 3.439", () => {
    // v ← 0.9v + g with g = +1 every step. Read the velocity back out of the
    // parameter: with lr = 1, each step moves θ by exactly −v.
    const p = scalarParam(0);
    const opt = new SGDMomentum([p], 1, 0.9);
    const seen: number[] = [];
    let previous = 0;
    for (let i = 0; i < 4; i++) {
      setGrad(p, 1);
      opt.step();
      seen.push(previous - p.data.data[0]!);
      previous = p.data.data[0]!;
    }
    expectRow(seen, [1, 1.9, 2.71, 3.439], 1e-9);
  });

  it("reaches 5.695 after eight agreeing steps", () => {
    const p = scalarParam(0);
    const opt = new SGDMomentum([p], 1, 0.9);
    let previous = 0, last = 0;
    for (let i = 0; i < 8; i++) {
      setGrad(p, 1);
      opt.step();
      last = previous - p.data.data[0]!;
      previous = p.data.data[0]!;
    }
    expect(Math.abs(last - 5.695)).toBeLessThan(1e-3);
  });

  it("cancels on disagreement: alternating gradients stay near zero", () => {
    // Same rule, same |g|, only the signs differ — and after eight steps the
    // velocity is −0.300 instead of 5.695.
    const p = scalarParam(0);
    const opt = new SGDMomentum([p], 1, 0.9);
    let previous = 0, last = 0;
    for (let i = 0; i < 8; i++) {
      setGrad(p, i % 2 === 0 ? 1 : -1);
      opt.step();
      last = previous - p.data.data[0]!;
      previous = p.data.data[0]!;
    }
    expect(Math.abs(last + 0.300)).toBeLessThan(1e-3);
  });

  it("with β = 0 it is exactly plain SGD", () => {
    // v ← 0·v + g = g, so the update collapses to θ − lr·g.
    const a = scalarParam(3);
    const b = scalarParam(3);
    setGrad(a, 2);
    setGrad(b, 2);
    new SGDMomentum([a], 0.1, 0).step();
    new SGD([b], 0.1).step();
    expect(a.data.data[0]).toBe(b.data.data[0]!);
  });

  it("defaults momentum to 0.9", () => {
    expect(new SGDMomentum([scalarParam(0)], 0.1).momentum).toBe(0.9);
  });

  it("keeps one velocity per parameter, never shared", () => {
    // Different shapes and different histories: a shared velocity would
    // either crash on shape or contaminate one parameter with the other's
    // gradient.
    const w = new TensorValue(createTensor([0, 0, 0, 0, 0, 0], [3, 2]));
    const b = new TensorValue(createTensor([0, 0, 0], [3]));
    const opt = new SGDMomentum([w, b], 1, 0.9);
    setGrad(w, 1, 1, 1, 1, 1, 1);
    setGrad(b, 2, 2, 2);
    opt.step();
    opt.step();
    // w's velocity: 1 then 1.9  → moved 2.9 total.  b's: 2 then 3.8 → 5.8.
    expectRow(Array.from(w.data.data), [-2.9, -2.9, -2.9, -2.9, -2.9, -2.9], 1e-9);
    expectRow(Array.from(b.data.data), [-5.8, -5.8, -5.8], 1e-9);
  });

  it("velocity persists across steps — it is not rebuilt each time", () => {
    // The silent bug: allocating velocities inside step() gives 1, 1, 1, 1
    // instead of 1, 1.9, 2.71 — plain SGD wearing a momentum costume.
    const p = scalarParam(0);
    const opt = new SGDMomentum([p], 1, 0.9);
    setGrad(p, 1);
    opt.step();
    const first = -p.data.data[0]!;
    const before = p.data.data[0]!;
    setGrad(p, 1);
    opt.step();
    const second = before - p.data.data[0]!;
    expect(second).toBeGreaterThan(first);
    expect(Math.abs(second - 1.9)).toBeLessThan(1e-9);
  });

  it("zeroGrad clears gradients but NOT the velocity", () => {
    // Gradients are per-iteration scratch; velocity is the memory.
    const p = scalarParam(0);
    const opt = new SGDMomentum([p], 1, 0.9);
    setGrad(p, 1);
    opt.step();
    const before = p.data.data[0]!;
    opt.zeroGrad();
    expect(p.grad).toBe(null);
    setGrad(p, 1);
    opt.step();
    // velocity survived, so this step is 1.9, not another 1.0
    expect(Math.abs(before - p.data.data[0]! - 1.9)).toBeLessThan(1e-9);
  });

  it("never builds graph nodes", () => {
    const theta = scalarParam(0);
    const opt = new SGDMomentum([theta], 0.01, 0.9);
    const target = createTensor([5], [1]);
    for (let i = 0; i < 5; i++) {
      opt.zeroGrad();
      mseLoss(theta, target).backward();
      opt.step();
    }
    expect(theta._inputs.length).toBe(0);
  });

  it("skips a parameter whose gradient is still null", () => {
    const unused = scalarParam(7);
    new SGDMomentum([unused], 0.1, 0.9).step();
    expect(unused.data.data[0]).toBe(7);
  });

  it("trains a real Linear layer", () => {
    const layer = new Linear(2, 3);
    const opt = new SGDMomentum(layer.parameters(), 0.01, 0.9);
    const x = new TensorValue(createTensor([1, 2], [1, 2]));
    const want = createTensor([1, 2, 3], [1, 3]);
    for (let i = 0; i < 200; i++) {
      opt.zeroGrad();
      mseLoss(layer.forward(x), want).backward();
      opt.step();
    }
    expectRow(Array.from(layer.forward(x).data.data), [1, 2, 3], 1e-2);
  });
});
