/**
 * Tests for tensor/math.ts
 * Chapter 06 — Math Primitives
 *
 * Run: bun test src/tensor/math.test.ts
 */
import { describe, it, expect } from "bun:test";
// import { exp, log, sqrt, pow, abs, clip, tanh, sigmoid } from "./math.ts";
// import { scalar } from "./types.ts";

const EPSILON = 1e-6;
const close = (a: number, b: number) => Math.abs(a - b) < EPSILON;

describe("exp", () => {
  it.todo("exp(0) === 1 for every element");
  it.todo("exp(1) ≈ 2.71828");
});

describe("log", () => {
  it.todo("log(exp(x)) ≈ x — round-trip identity");
  it.todo("exp(log(x)) ≈ x for positive x");
});

describe("sqrt", () => {
  it.todo("sqrt(4) === 2 elementwise");
  it.todo("sqrt(0) === 0");
});

describe("clip", () => {
  it.todo("values below min become min");
  it.todo("values above max become max");
  it.todo("values within [min, max] are unchanged");
});

describe("tanh", () => {
  it.todo("tanh(0) === 0");
  it.todo("output is in (-1, 1) for any finite input");
});

describe("sigmoid", () => {
  it.todo("sigmoid(0) === 0.5");
  it.todo("output is in (0, 1) for any finite input");
});
