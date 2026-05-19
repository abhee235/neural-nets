/**
 * Tests for utils/numerical.ts
 * Chapter 07 — Calculus for ML
 *
 * Run: bun test src/utils/numerical.test.ts
 */
import { describe, it, expect } from "bun:test";
// import { numericalGradient, checkGradient } from "./numerical.ts";

describe("numericalGradient", () => {
  it.todo("derivative of x² at x=3 ≈ 6");
  it.todo("derivative of sin(x) at x=0 ≈ 1");
  it.todo("derivative of exp(x) at x=0 ≈ 1");
  it.todo("smaller h gives a more accurate estimate for smooth functions");
});

describe("checkGradient", () => {
  it.todo("returns true when analytical and numerical agree within tolerance");
  it.todo("returns false when gradients differ beyond tolerance");
});
