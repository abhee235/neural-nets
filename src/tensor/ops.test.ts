/**
 * Tests for tensor/ops.ts
 * Chapter 03 — Elementwise Ops & Broadcasting
 *
 * Run: bun test src/tensor/ops.test.ts
 */
import { describe, it, expect } from "bun:test";
// import { add, sub, mul, div, mulScalar, broadcastShapes, broadcast } from "./ops.ts";
// import { createTensor } from "./types.ts";

describe("broadcastShapes", () => {
  it.todo("[3,1] and [1,4] broadcast to [3,4]");
  it.todo("[5] and [3,5] broadcast to [3,5]");
  it.todo("equal shapes broadcast to themselves");
  it.todo("throws for incompatible shapes, e.g. [3] and [4]");
});

describe("add", () => {
  it.todo("add([1,2],[3,4]) produces [4,6]");
  it.todo("add broadcasts [3,1] + [1,4] to shape [3,4]");
});

describe("mul / div", () => {
  it.todo("elementwise multiplication is correct");
  it.todo("div(a, a) produces all-ones for nonzero a");
});

describe("mulScalar", () => {
  it.todo("doubles every element when s=2");
  it.todo("produces all zeros when s=0");
});

describe("applyFn", () => {
  it.todo("applies the function independently to every element");
  it.todo("output shape matches input shape");
});
