/**
 * Tests for optim/sgd.ts
 * Chapters 09 & 14 — Gradient Descent / Optimizers
 *
 * Run: bun test src/optim/sgd.test.ts
 */
import { describe, it, expect } from "bun:test";
// import { SGD, SGDMomentum } from "./sgd.ts";
// import { Value } from "../autograd/value.ts";

describe("SGD", () => {
  it.todo("step moves a parameter in the direction that reduces loss");
  it.todo("on L=(w−3)², SGD converges w toward 3");
  it.todo("zeroGrad resets all parameter gradients to 0");
  it.todo("larger learning rate produces a larger parameter change");
});

describe("SGDMomentum", () => {
  it.todo("velocity accumulates across steps");
  it.todo("momentum=0 is equivalent to vanilla SGD");
  it.todo("converges faster than vanilla SGD on a quadratic");
});
