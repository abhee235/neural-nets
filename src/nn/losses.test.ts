/**
 * Tests for nn/losses.ts
 * Chapter 12 — Loss Functions
 *
 * Run: bun test src/nn/losses.test.ts
 */
import { describe, it, expect } from "bun:test";
// import { mseLoss, logSumExp, crossEntropyFromLogits } from "./losses.ts";

describe("mseLoss", () => {
  it.todo("loss is 0 when predictions exactly equal targets");
  it.todo("gradient points toward the targets");
  it.todo("numerical gradient check passes");
});

describe("logSumExp", () => {
  it.todo("shift invariance: logSumExp(x) === logSumExp(x+c) + c");
  it.todo("numerical gradient check passes");
});

describe("crossEntropyFromLogits", () => {
  it.todo("lower loss when the correct class has a higher logit");
  it.todo("gradient is softmax(logits) − one_hot(targets)");
  it.todo("numerical gradient check passes");
});
