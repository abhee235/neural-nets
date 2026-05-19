/**
 * Tests for autograd/value.ts
 * Chapters 08a & 08b — Scalar Autograd
 *
 * Run: bun test src/autograd/value.test.ts
 */
import { describe, it, expect } from "bun:test";
// import { Value } from "./value.ts";

const EPSILON = 1e-5;
const close = (a: number, b: number) => Math.abs(a - b) < EPSILON;

describe("Value — forward pass", () => {
  it.todo("add produces the correct scalar sum");
  it.todo("mul produces the correct scalar product");
  it.todo("_inputs records both operands of a binary op");
  it.todo("chained ops build a graph of depth > 1");
  it.todo("exp(0).data === 1");
  it.todo("tanh(0).data === 0");
});

describe("Value — backward pass", () => {
  it.todo("z = x*y: backward gives dz/dx = y and dz/dy = x");
  it.todo("z = x^2: backward gives dz/dx = 2x");
  it.todo("z = exp(x): backward gives dz/dx = exp(x)");
  it.todo("gradients accumulate correctly through a chain of ops");
  it.todo("zeroGrad resets grad to 0");
});

describe("numerical gradient checks", () => {
  it.todo("add gradient matches finite differences");
  it.todo("mul gradient matches finite differences");
  it.todo("pow gradient matches finite differences");
  it.todo("exp gradient matches finite differences");
  it.todo("tanh gradient matches finite differences");
  it.todo("relu gradient matches finite differences (x ≠ 0)");
});
