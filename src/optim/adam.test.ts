/**
 * Tests for optim/adam.ts
 * Chapter 14 — Optimizers
 *
 * Run: bun test src/optim/adam.test.ts
 *
 * Fixtures are the chapter's own numbers:
 *
 *   first step, g = 1, defaults
 *     m₁ = 0.9·0   + 0.1·1   = 0.1        m̂₁ = 0.1   / (1−0.9)   = 1
 *     v₁ = 0.999·0 + 0.001·1 = 0.001      v̂₁ = 0.001 / (1−0.999) = 1
 *     update = 1 / (√1 + ε) ≈ 1, so the parameter moves by ≈ lr
 *
 *   the m̂/√v̂ ratio, from the doc's table
 *     all +1     → 1.000        all +1000 → 1.000     (scale cancels)
 *     ±3 mixed   → ≈ ±0.05      (small step: the gradients disagree)
 *
 *   AdamW, θ = 10, lr = 0.1, λ = 0.01, g = 0  →  10 − 0.1·0.01·10 = 9.99
 */
import { describe, it, expect } from "bun:test";
import { Adam, AdamW } from "./adam.ts";
import { TensorValue } from "../autograd/grad.ts";
import { createTensor } from "../tensor/types.ts";
import { Linear } from "../nn/linear.ts";
import { mseLoss } from "../nn/losses.ts";

/** A leaf holding one number, so hand-computed traces stay readable. */
const param = (v: number) => new TensorValue(createTensor([v], [1]));

/** Put a gradient on a parameter directly, standing in for a backward pass. */
function setGrad(p: TensorValue, ...values: number[]): void {
  p.grad = createTensor(values, p.data.shape);
}

/**
 * Run `grads` through an optimizer with lr = 1 and report how far the
 * parameter moved on the last step. With lr = 1 that distance IS the m̂/√v̂
 * ratio, which is the quantity the chapter reasons about.
 */
function lastRatio(grads: number[]): number {
  const p = param(0);
  const opt = new Adam([p], 1);
  let previous = 0, moved = 0;
  for (const g of grads) {
    setGrad(p, g);
    opt.step();
    moved = previous - p.data.data[0]!;
    previous = p.data.data[0]!;
  }
  return moved;
}

describe("Adam — construction", () => {
  it("uses the standard defaults", () => {
    const opt = new Adam([param(0)]);
    expect(opt.learningRate).toBe(1e-3);
    expect(opt.beta1).toBe(0.9);
    expect(opt.beta2).toBe(0.999);
    expect(opt.epsilon).toBe(1e-8);
  });

  it("accepts overrides", () => {
    const opt = new Adam([param(0)], 0.05, 0.5, 0.9, 1e-6);
    expect(opt.learningRate).toBe(0.05);
    expect(opt.beta1).toBe(0.5);
    expect(opt.beta2).toBe(0.9);
    expect(opt.epsilon).toBe(1e-6);
  });

  it("holds the caller's own objects, not copies", () => {
    const p = param(1);
    expect(new Adam([p], 0.1).params[0]).toBe(p);
  });
});

describe("Adam — the first step", () => {
  it("moves by ≈ lr, which is what bias correction is for", () => {
    // Both hats come out to exactly 1, so the update is 1 · lr.
    // WITHOUT correction it would be 0.1/√0.001 · lr = 3.162e-3 — three times
    // too large — so this number proves the correction is applied.
    const p = param(0);
    setGrad(p, 1);
    new Adam([p], 1e-3).step();
    expect(Math.abs(-p.data.data[0]! - 1e-3)).toBeLessThan(1e-9);
  });

  it("is nowhere near the uncorrected value", () => {
    const p = param(0);
    setGrad(p, 1);
    new Adam([p], 1e-3).step();
    const uncorrected = (0.1 / Math.sqrt(0.001)) * 1e-3;
    expect(Math.abs(-p.data.data[0]! - uncorrected)).toBeGreaterThan(1e-3);
  });

  it("steps against the gradient, not with it", () => {
    // A positive gradient must move the parameter down.
    const p = param(0);
    setGrad(p, 1);
    new Adam([p], 1e-3).step();
    expect(p.data.data[0]!).toBeLessThan(0);
  });
});

describe("Adam — the m̂/√v̂ ratio", () => {
  it("is ≈ 1 for a steady gradient", () => {
    expect(Math.abs(lastRatio(Array(10).fill(1)) - 1)).toBeLessThan(1e-6);
  });

  it("ignores the SIZE of the gradients entirely", () => {
    // The chapter's central claim: ×1 and ×1000 give the same answer, because
    // m̂ scales by 1000, √v̂ scales by 1000, and the ratio cancels.
    const small = lastRatio(Array(10).fill(1));
    const large = lastRatio(Array(10).fill(1000));
    expect(Math.abs(small - large)).toBeLessThan(1e-6);
  });

  it("takes a SMALL step when gradients disagree", () => {
    // Alternating ±3: v notices they are large, m notices they cancel.
    // The doc's table puts this near ±0.05.
    const alternating = lastRatio(Array.from({ length: 20 }, (_, i) => (i % 2 ? -3 : 3)));
    expect(Math.abs(alternating)).toBeLessThan(0.15);
  });

  it("takes a FULL step when they agree, on the same gradient magnitude", () => {
    // Same |g| = 3 as above, all one sign — the contrast is agreement alone.
    const agreeing = lastRatio(Array(20).fill(3));
    expect(Math.abs(agreeing - 1)).toBeLessThan(1e-6);
  });
});

describe("Adam — state", () => {
  it("keeps one moment pair per parameter, never shared", () => {
    // Different shapes and different gradients: a shared pair would either
    // crash on shape or contaminate one parameter with the other's history.
    const w = new TensorValue(createTensor([0, 0, 0, 0, 0, 0], [3, 2]));
    const b = new TensorValue(createTensor([0, 0, 0], [3]));
    const opt = new Adam([w, b], 1);
    setGrad(w, 1, 1, 1, 1, 1, 1);
    setGrad(b, 100, 100, 100);
    opt.step();
    // Both ratios are ≈1 despite gradients differing 100-fold.
    for (const v of w.data.data) expect(Math.abs(-v - 1)).toBeLessThan(1e-6);
    for (const v of b.data.data) expect(Math.abs(-v - 1)).toBeLessThan(1e-6);
  });

  it("advances t once per step, not once per parameter", () => {
    // If t incremented per parameter, the second one would get a different
    // bias correction than the first and they would move differently.
    const a = param(0), b = param(0);
    const opt = new Adam([a, b], 1e-3);
    setGrad(a, 1);
    setGrad(b, 1);
    opt.step();
    expect(a.data.data[0]).toBe(b.data.data[0]!);
  });

  it("persists the moments across steps", () => {
    // Rebuilding m and v inside step() would make every step look like a
    // first step. Here a decaying gradient must produce a decaying ratio.
    const p = param(0);
    const opt = new Adam([p], 1);
    const ratios: number[] = [];
    let previous = 0;
    for (const g of [1, 0, 0, 0]) {
      setGrad(p, g);
      opt.step();
      ratios.push(previous - p.data.data[0]!);
      previous = p.data.data[0]!;
    }
    // With no fresh gradient the remembered direction fades rather than
    // resetting — each ratio strictly smaller than the last.
    for (let i = 1; i < ratios.length; i++) {
      expect(ratios[i]!).toBeLessThan(ratios[i - 1]!);
    }
  });

  it("preserves parameter shape", () => {
    const p = new TensorValue(createTensor([1, 2, 3, 4, 5, 6], [3, 2]));
    setGrad(p, 1, 1, 1, 1, 1, 1);
    new Adam([p], 1e-3).step();
    expect(p.data.shape).toEqual([3, 2]);
  });
});

describe("Adam — integration", () => {
  it("skips a parameter whose gradient is still null", () => {
    const unused = param(7);
    new Adam([unused], 1e-3).step();
    expect(unused.data.data[0]).toBe(7);
  });

  it("zeroGrad resets to null", () => {
    const p = param(0);
    setGrad(p, 1);
    const opt = new Adam([p], 1e-3);
    opt.zeroGrad();
    expect(p.grad).toBe(null);
  });

  it("never builds graph nodes", () => {
    const theta = param(0);
    const opt = new Adam([theta], 0.1);
    const target = createTensor([5], [1]);
    for (let i = 0; i < 5; i++) {
      opt.zeroGrad();
      mseLoss(theta, target).backward();
      opt.step();
    }
    expect(theta._inputs.length).toBe(0);
  });

  it("trains a real Ch 13 Linear layer through parameters()", () => {
    const layer = new Linear(2, 3);
    const opt = new Adam(layer.parameters(), 0.05);
    const x = new TensorValue(createTensor([1, 2], [1, 2]));
    const want = createTensor([1, 2, 3], [1, 3]);
    for (let i = 0; i < 400; i++) {
      opt.zeroGrad();
      mseLoss(layer.forward(x), want).backward();
      opt.step();
    }
    const out = Array.from(layer.forward(x).data.data);
    for (let i = 0; i < 3; i++) expect(Math.abs(out[i]! - (i + 1))).toBeLessThan(1e-2);
  });
});

describe("AdamW", () => {
  it("with λ = 0 it is Adam exactly", () => {
    // The identity that makes the subclass honest.
    const a = param(5), b = param(5);
    const optA = new Adam([a], 0.1);
    const optW = new AdamW([b], 0.1, 0);
    for (let i = 0; i < 10; i++) {
      setGrad(a, 1);
      setGrad(b, 1);
      optA.step();
      optW.step();
    }
    expect(b.data.data[0]).toBe(a.data.data[0]!);
  });

  it("decays a parameter that has no gradient at all", () => {
    // θ = 10, lr = 0.1, λ = 0.01, g = 0 → Adam's part is 0, so only the
    // decay moves it: 10 − 0.1·0.01·10 = 9.99.
    const p = param(10);
    setGrad(p, 0);
    new AdamW([p], 0.1, 0.01).step();
    expect(Math.abs(p.data.data[0]! - 9.99)).toBeLessThan(1e-9);
  });

  it("the decay is DECOUPLED — it never enters m or v", () => {
    // Adam's share of the step must be identical for θ = 1 and θ = 100.
    // Had the decay been folded into the gradient (PyTorch's Adam
    // weight_decay), the two would differ a hundredfold.
    const adamShare = (theta: number) => {
      const p = param(theta);
      const opt = new AdamW([p], 0.1, 0.01);
      setGrad(p, 1);
      opt.step();
      const moved = theta - p.data.data[0]!;
      return moved - 0.1 * 0.01 * theta; // remove the decay, leave Adam's part
    };
    expect(Math.abs(adamShare(1) - adamShare(100))).toBeLessThan(1e-9);
  });

  it("pulls a large weight toward zero over many steps", () => {
    const p = param(50);
    const opt = new AdamW([p], 0.1, 0.05);
    for (let i = 0; i < 40; i++) {
      setGrad(p, 0);
      opt.step();
    }
    expect(p.data.data[0]!).toBeLessThan(50);
    expect(p.data.data[0]!).toBeGreaterThan(0);
  });

  it("defaults λ to 0.01 and inherits Adam's other defaults", () => {
    const opt = new AdamW([param(0)]);
    expect(opt.weightDecay).toBe(0.01);
    expect(opt.learningRate).toBe(1e-3);
    expect(opt.beta1).toBe(0.9);
    expect(opt.beta2).toBe(0.999);
  });

  it("is an Adam, so anything taking an optimizer accepts it", () => {
    expect(new AdamW([param(0)])).toBeInstanceOf(Adam);
  });
});
