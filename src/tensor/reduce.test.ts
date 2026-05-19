/**
 * Tests for tensor/reduce.ts
 * Chapter 05 — Reductions & Statistical Ops
 *
 * Run: bun test src/tensor/reduce.test.ts
 */
import { describe, it, expect } from "bun:test";
// import { sum, mean, max, argmax, softmax } from "./reduce.ts";
// import { createTensor } from "./types.ts";

const EPSILON = 1e-6;
const close = (a: number, b: number) => Math.abs(a - b) < EPSILON;

describe("sum", () => {
  it.todo("sum of all elements equals the scalar total");
  it.todo("sum along axis=0 of a (3,4) tensor produces shape (4,)");
  it.todo("keepDims=true preserves the reduced axis as size 1");
});

describe("mean", () => {
  it.todo("mean of [1,2,3,4] === 2.5");
  it.todo("mean along axis produces the correct per-row average");
});

describe("softmax", () => {
  it.todo("output sums to 1.0 along the target axis");
  it.todo("shift invariance: softmax(x) === softmax(x + c) for any constant c");
  it.todo("largest logit gets the highest probability");
  it.todo("identical logits produce a uniform distribution");
});

describe("argmax", () => {
  it.todo("returns the index of the maximum element");
  it.todo("argmax along axis=1 produces shape (batch,)");
});
