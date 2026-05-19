/**
 * Tests for optim/adam.ts
 * Chapter 14 — Optimizers
 *
 * Run: bun test src/optim/adam.test.ts
 */
import { describe, it, expect } from "bun:test";
// import { Adam } from "./adam.ts";
// import { Value } from "../autograd/value.ts";

describe("Adam", () => {
  it.todo("step reduces loss on a simple quadratic");
  it.todo("bias correction dominates at step t=1");
  it.todo("zeroGrad resets all gradients to 0");
  it.todo("converges faster than plain SGD on L=(w−3)²");
  it.todo("moment estimates are initialised to 0");
});
