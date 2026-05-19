/**
 * Tests for nn/activations.ts
 * Chapter 11 — Activation Functions
 *
 * Run: bun test src/nn/activations.test.ts
 */
import { describe, it, expect } from "bun:test";
// import { relu, gelu, sigmoid, softmax } from "./activations.ts";

describe("relu", () => {
  it.todo("relu(2) === 2 (positive input passes through)");
  it.todo("relu(−2) === 0 (negative input is zeroed)");
  it.todo("numerical gradient check passes (x ≠ 0)");
});

describe("gelu", () => {
  it.todo("gelu(0) ≈ 0");
  it.todo("gelu is smooth near x=0 (no kink unlike ReLU)");
  it.todo("numerical gradient check passes");
});

describe("sigmoid", () => {
  it.todo("sigmoid(0) === 0.5");
  it.todo("output is strictly in (0,1) for any finite input");
  it.todo("numerical gradient check passes");
});

describe("softmax", () => {
  it.todo("output sums to 1.0 along the target axis");
  it.todo("shift invariance: softmax(x) === softmax(x + c)");
  it.todo("numerical gradient check passes");
});
