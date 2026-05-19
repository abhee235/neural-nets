/**
 * Tests for tensor/creation.ts
 * Chapter 02 — Tensor Creation
 *
 * Run: bun test src/tensor/creation.test.ts
 */
import { describe, it, expect } from "bun:test";
// import { zeros, ones, fill, randn, eye, arange, linspace } from "./creation.ts";

const EPSILON = 1e-6;
const close = (a: number, b: number) => Math.abs(a - b) < EPSILON;

describe("zeros", () => {
  it.todo("every element is 0.0");
  it.todo("shape matches the argument");
});

describe("ones", () => {
  it.todo("every element is 1.0");
});

describe("randn", () => {
  it.todo("output shape matches the requested shape");
  it.todo("large sample mean ≈ 0 (Box-Muller property)");
  it.todo("large sample standard deviation ≈ 1 (Box-Muller property)");
  it.todo("values are not all identical — genuine randomness");
});

describe("eye", () => {
  it.todo("diagonal elements are 1.0");
  it.todo("off-diagonal elements are 0.0");
  it.todo("shape is [n, n]");
});

describe("arange", () => {
  it.todo("arange(0, 5, 1) produces [0, 1, 2, 3, 4]");
  it.todo("step defaults to 1");
});

describe("linspace", () => {
  it.todo("linspace(0, 1, 5) produces [0, 0.25, 0.5, 0.75, 1.0]");
  it.todo("first element === start, last element === stop");
});
