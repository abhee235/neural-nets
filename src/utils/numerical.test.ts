/**
 * Tests for utils/numerical.ts
 * Chapter 07 — Calculus for ML
 *
 * Run: bun test src/utils/numerical.test.ts
 *
 * Tests are named after the property they verify; each expect() has a comment
 * stating the mathematical fact. Pitfall cases (full-tensor coverage, no input
 * mutation) are checked explicitly.
 */
import { describe, it, expect } from "bun:test";
import { numericalGradient, numericalGradientTensor, checkGradient } from "./numerical.ts";
import { createTensor } from "../tensor/types";

const EPSILON = 1e-4; // finite-difference estimates are approximate
const close = (a: number, b: number, tol = EPSILON) => Math.abs(a - b) < tol;
const data = (t: { data: Float64Array }) => Array.from(t.data);

describe("numericalGradient", () => {
  it("derivative of x² at x=3 ≈ 6", () => {
    // d/dx x² = 2x → 6 at x=3
    expect(close(numericalGradient((x) => x * x, 3), 6)).toBe(true);
  });

  it("derivative of sin(x) at x=0 ≈ 1", () => {
    // d/dx sin(x) = cos(x) → 1 at x=0
    expect(close(numericalGradient(Math.sin, 0), 1)).toBe(true);
  });

  it("derivative of exp(x) at x=0 ≈ 1", () => {
    // d/dx eˣ = eˣ → 1 at x=0
    expect(close(numericalGradient(Math.exp, 0), 1)).toBe(true);
  });

  it("matches the sigmoid's analytical derivative at x=0.5", () => {
    const sig = (x: number) => 1 / (1 + Math.exp(-x));
    const s = sig(0.5);
    // σ'(x) = σ(x)(1 − σ(x))
    expect(close(numericalGradient(sig, 0.5), s * (1 - s))).toBe(true);
  });

  it("respects a custom h", () => {
    // cube: d/dx x³ = 3x² → 12 at x=2
    expect(close(numericalGradient((x) => x ** 3, 2, 1e-4), 12)).toBe(true);
  });
});

describe("numericalGradientTensor", () => {
  it("gradient of sum(x²) is 2x, element-wise, for EVERY element", () => {
    const t = createTensor([1, 2, 3, 4, 5, 6], [2, 3]);
    // f(x) = Σ xᵢ²  ⇒  ∂f/∂xᵢ = 2xᵢ
    const f = (x: { data: Float64Array }) =>
      Array.from(x.data).reduce((acc, v) => acc + v * v, 0);
    const g = numericalGradientTensor(f, t);
    // every element must be filled (the shape[0] bug would leave most at 0)
    expect(g.shape).toEqual([2, 3]);
    expect(close(data(g)[0]!, 2) && close(data(g)[5]!, 12)).toBe(true);
    expect(data(g).every((v, i) => close(v, 2 * (i + 1)))).toBe(true);
  });

  it("gradient of a weighted sum equals the weights", () => {
    const t = createTensor([0, 0, 0], [3]);
    const w = [2, -3, 5];
    // f(x) = Σ wᵢ·xᵢ  ⇒  ∂f/∂xᵢ = wᵢ
    const f = (x: { data: Float64Array }) =>
      Array.from(x.data).reduce((acc, v, i) => acc + w[i]! * v, 0);
    expect(data(numericalGradientTensor(f, t)).every((v, i) => close(v, w[i]!))).toBe(true);
  });

  it("does NOT mutate the input tensor (perturbs copies)", () => {
    const t = createTensor([1, 2, 3], [3]);
    const before = data(t);
    numericalGradientTensor((x) => Array.from(x.data).reduce((a, v) => a + v * v, 0), t);
    // the caller's data must be byte-for-byte unchanged afterward
    expect(data(t)).toEqual(before);
  });
});

describe("checkGradient", () => {
  it("returns true when analytical and numerical agree within tolerance", () => {
    expect(checkGradient(6.0, 6.0000001)).toBe(true);
  });

  it("returns false when gradients differ beyond tolerance", () => {
    // off by 0.1, well past the 1e-5 default
    expect(checkGradient(6.0, 6.1)).toBe(false);
  });

  it("respects a custom tolerance", () => {
    expect(checkGradient(6.0, 6.05, 0.1)).toBe(true);
  });
});
