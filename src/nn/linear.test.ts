/**
 * Tests for nn/linear.ts
 * Chapter 13 — Linear Layer
 *
 * Run: bun test src/nn/linear.test.ts
 */
import { describe, it, expect } from "bun:test";
// import { Linear } from "./linear.ts";

describe("Linear", () => {
  it.todo("forward output shape is [*, outputDim]");
  it.todo("forward with bias=true adds a learnable offset");
  it.todo("forward with bias=false has no bias parameter");
  it.todo("parameters() returns weight and bias");
  it.todo("He init: weight variance ≈ 2/inputDim for large layers");
  it.todo("numerical gradient check passes w.r.t. weight and bias");
});
