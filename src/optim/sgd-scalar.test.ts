/**
 * Tests for optim/sgd.ts
 * Chapters 09 & 14 — Gradient Descent / Optimizers
 *
 * Run: bun test src/optim/sgd.test.ts
 *
 * The loss throughout is the chapter's bowl, L(w) = (w − c)², whose gradient
 * 2(w − c) can be checked by hand at every step. Where a test asserts an exact
 * trajectory, those numbers are the ones worked out in the chapter doc.
 */
import { describe, it, expect } from "bun:test";
import { SGDScalar as SGD, SGDMomentumScalar as SGDMomentum } from "./sgd-scalar.ts";
import { Value } from "../autograd/value.ts";

const EPSILON = 1e-6;
const close = (a: number, b: number) => Math.abs(a - b) < EPSILON;

/** A fresh graph for L = (w − target)². Rebuilt every step, as the loop requires. */
function bowlLoss(w: Value, target: number): Value {
  return w.add(new Value(-target)).pow(2);
}

/** Run the five-stage loop and return |w − target| at the end. */
function descend(
  makeOpt: (p: Value[]) => { step(): void; zeroGrad(): void },
  target: number,
  start: number,
  steps: number,
): number {
  const w = new Value(start);
  const opt = makeOpt([w]);
  for (let i = 0; i < steps; i++) {
    bowlLoss(w, target).backward();
    opt.step();
    opt.zeroGrad();
  }
  return Math.abs(w.data - target);
}

describe("SGD", () => {
  it("step moves a parameter in the direction that reduces loss", () => {
    const w = new Value(0);
    const opt = new SGD([w], 0.1);
    const before = bowlLoss(w, 5);
    before.backward();
    // ∂/∂w (w−5)² = 2(w−5) = −10 at w = 0. Negative, so the loss falls as w rises.
    expect(close(w.grad, -10)).toBe(true);
    opt.step();
    // θ ← 0 − 0.1×(−10) = 1.0. Subtracting a negative moves w UP, toward the minimum.
    expect(close(w.data, 1.0)).toBe(true);
    // …and the loss really did decrease: 25 → 16.
    expect(before.data).toBe(25);
    expect(close(bowlLoss(w, 5).data, 16)).toBe(true);
  });

  it("reproduces the chapter's hand-computed trajectory", () => {
    const w = new Value(0);
    const opt = new SGD([w], 0.1);
    const trace: number[] = [];
    for (let i = 0; i < 3; i++) {
      bowlLoss(w, 5).backward();
      opt.step();
      opt.zeroGrad();
      trace.push(w.data);
    }
    // The doc's §3 trace: 1.0, 1.8, 2.44 — each step 0.8× the last, because the
    // distance to the minimum shrinks by exactly (1 − 2η) = 0.8 per step.
    expect(close(trace[0]!, 1.0)).toBe(true);
    expect(close(trace[1]!, 1.8)).toBe(true);
    expect(close(trace[2]!, 2.44)).toBe(true);
  });

  it("on L=(w−3)², SGD converges w toward 3", () => {
    // The error obeys eₙ = e₀(1 − 2η)ⁿ, so at η = 0.1 it decays by 0.8 per step
    // and after 100 steps is ~1e-9 — indistinguishable from arrival.
    expect(descend((p) => new SGD(p, 0.1), 3, 0, 100)).toBeLessThan(1e-6);
  });

  it("converges from either side of the minimum", () => {
    // The sign of the gradient chooses the direction; no per-parameter logic
    // is involved, so starting above the minimum must work identically.
    expect(descend((p) => new SGD(p, 0.1), 3, 10, 100)).toBeLessThan(1e-6);
  });

  it("zeroGrad resets all parameter gradients to 0", () => {
    const a = new Value(1);
    const b = new Value(2);
    const opt = new SGD([a, b], 0.1);
    a.mul(b).backward();
    // ∂(ab)/∂a = b = 2 and ∂(ab)/∂b = a = 1 — both non-zero before clearing.
    expect(a.grad).toBe(2);
    expect(b.grad).toBe(1);
    opt.zeroGrad();
    // Every owned parameter is cleared, not just the first.
    expect(a.grad).toBe(0);
    expect(b.grad).toBe(0);
  });

  it("step does not clear gradients — that is zeroGrad's job", () => {
    const w = new Value(0);
    const opt = new SGD([w], 0.1);
    bowlLoss(w, 5).backward();
    opt.step();
    // The two responsibilities stay separate, so gradients can still be
    // inspected after a step. Fusing them would make debugging impossible.
    expect(close(w.grad, -10)).toBe(true);
  });

  it("larger learning rate produces a larger parameter change", () => {
    // Same starting point, same gradient, two learning rates.
    const wSmall = new Value(0);
    const wBig = new Value(0);
    bowlLoss(wSmall, 5).backward();
    bowlLoss(wBig, 5).backward();
    new SGD([wSmall], 0.01).step();
    new SGD([wBig], 0.1).step();
    // The gradient is −10 for both, so the moves are 0.1 and 1.0. The step is
    // exactly proportional to η — a 10× learning rate gives a 10× step.
    expect(close(wSmall.data, 0.1)).toBe(true);
    expect(close(wBig.data, 1.0)).toBe(true);
    expect(Math.abs(wBig.data)).toBeGreaterThan(Math.abs(wSmall.data));
  });

  it("updates parameters in place, without extending the graph", () => {
    const w = new Value(0);
    const opt = new SGD([w], 0.1);
    bowlLoss(w, 5).backward();
    opt.step();
    // The update must write to .data directly. If it went through Value ops it
    // would return a NEW node, leaving the model pointing at the old one — and
    // the parameter would still look like a leaf here while never training.
    expect(w._inputs).toHaveLength(0);
    expect(w._op).toBe("");
    // The optimizer holds the same object it was given, not a copy of its value.
    expect(opt.params[0]).toBe(w);
  });

  it("updates every parameter it owns, from one backward pass", () => {
    // L = (a−3)² + (b+1)², the chapter's two-parameter checkpoint.
    const a = new Value(0);
    const b = new Value(0);
    const opt = new SGD([a, b], 0.1);
    for (let i = 0; i < 200; i++) {
      bowlLoss(a, 3).add(bowlLoss(b, -1)).backward();
      opt.step();
      opt.zeroGrad();
    }
    // Both descend together, from a single backward() call, with nothing in
    // step() aware that there is more than one parameter.
    expect(close(a.data, 3)).toBe(true);
    expect(close(b.data, -1)).toBe(true);
  });
});

describe("SGDMomentum", () => {
  it("velocity accumulates across steps", () => {
    const w = new Value(0);
    const opt = new SGDMomentum([w], 0.1, 0.9);
    const vs: number[] = [];
    for (let i = 0; i < 3; i++) {
      bowlLoss(w, 5).backward();
      opt.step();
      opt.zeroGrad();
      vs.push(opt.velocities[0]!);
    }
    // v ← βv − η·grad, so with grads −10, −8, −4.6 the velocity grows
    // 1.0 → 1.7 → 1.99 even as the gradient shrinks. That growth is the whole
    // point: it only happens because the previous velocity survived the call.
    expect(close(vs[0]!, 1.0)).toBe(true);
    expect(close(vs[1]!, 1.7)).toBe(true);
    expect(close(vs[2]!, 1.99)).toBe(true);
    // The parameter moves BY the velocity, so w tracks the running sum.
    expect(close(w.data, 4.69)).toBe(true);
  });

  it("allocates velocity once, so it survives between steps", () => {
    const a = new Value(0);
    const b = new Value(0);
    const opt = new SGDMomentum([a, b], 0.1, 0.9);
    // One entry per parameter, all starting at rest.
    expect(opt.velocities).toHaveLength(2);
    expect(opt.velocities[0]).toBe(0);
    const first = opt.velocities;
    bowlLoss(a, 5).add(bowlLoss(b, 5)).backward();
    opt.step();
    // Re-allocating inside step() would reset the velocity to 0 every call and
    // silently degrade momentum into vanilla SGD — which still converges, so
    // only a test that inspects the array itself catches it.
    expect(opt.velocities).toBe(first);
    expect(opt.velocities[0]).not.toBe(0);
  });

  it("momentum=0 is equivalent to vanilla SGD", () => {
    // With β = 0 the previous velocity is discarded entirely and the rule
    // collapses to θ ← θ − η·grad. The two classes must then agree exactly,
    // not merely closely — this is an algebraic identity, not an approximation.
    const withMomentum = descend((p) => new SGDMomentum(p, 0.1, 0), 5, 0, 20);
    const vanilla = descend((p) => new SGD(p, 0.1), 5, 0, 20);
    expect(withMomentum).toBe(vanilla);
  });

  it("defaults to momentum 0.9 when omitted", () => {
    // The default is what every worked example in the chapter assumes.
    expect(new SGDMomentum([new Value(0)], 0.1).momentum).toBe(0.9);
    // An explicit 0 must survive as 0, not be replaced by the default — the
    // reason the constructor tests for undefined rather than falsiness.
    expect(new SGDMomentum([new Value(0)], 0.1, 0).momentum).toBe(0);
  });

  it("converges faster than vanilla SGD when the learning rate is conservative", () => {
    // Momentum's steady-state step is η/(1−β) — ten times vanilla's at β = 0.9.
    // That is a win only when η is SMALL. At η = 0.01 over 100 steps vanilla is
    // still 0.66 away while momentum is 0.021 away, a 30× improvement.
    const vanilla = descend((p) => new SGD(p, 0.01), 5, 0, 100);
    const momentum = descend((p) => new SGDMomentum(p, 0.01, 0.9), 5, 0, 100);
    expect(momentum).toBeLessThan(vanilla);
    expect(momentum).toBeLessThan(vanilla / 10);
  });

  it("overshoots the minimum when the learning rate is already well tuned", () => {
    // The honest other half, and the reason Ch 14 spends effort on tuning β.
    // At η = 0.1 the effective rate becomes ~1.0, past this bowl's stability
    // threshold for plain descent, so momentum oscillates while vanilla — whose
    // error decays by a clean 0.8 per step — has essentially arrived.
    const vanilla = descend((p) => new SGD(p, 0.1), 5, 0, 100);
    const momentum = descend((p) => new SGDMomentum(p, 0.1, 0.9), 5, 0, 100);
    expect(vanilla).toBeLessThan(momentum);
    // It is still converging, just slowly and from alternating sides.
    expect(momentum).toBeLessThan(0.1);
  });

  it("zeroGrad resets all parameter gradients to 0", () => {
    const a = new Value(1);
    const b = new Value(2);
    const opt = new SGDMomentum([a, b], 0.1, 0.9);
    a.mul(b).backward();
    opt.zeroGrad();
    // Clearing gradients must not disturb the velocity — they are separate state.
    expect(a.grad).toBe(0);
    expect(b.grad).toBe(0);
    expect(opt.velocities[0]).toBe(0);
  });
});
