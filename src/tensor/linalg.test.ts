/**
 * Tests for tensor/linalg.ts
 * Chapter 04 — Matrix Operations
 *
 * Run: bun test src/tensor/linalg.test.ts
 */
import { describe, it, expect } from "bun:test";
// import { matMul, transpose, reshape, flatten } from "./linalg.ts";
// import { createTensor } from "./types.ts";

describe("matMul", () => {
  it.todo("(M×K) × (K×N) produces shape (M×N)");
  it.todo("[[1,2],[3,4]] × [[1],[1]] === [[3],[7]]");
  it.todo("batched matMul (B,M,K) × (B,K,N) produces (B,M,N)");
  it.todo("throws when inner dimensions do not match");
});

describe("transpose", () => {
  it.todo("transpose of (3,4) produces shape (4,3)");
  it.todo("element at [i,j] moves to [j,i]");
  it.todo("double transpose is the identity");
});

describe("reshape", () => {
  it.todo("preserves total element count");
  it.todo("reshape to [-1] flattens to 1-D");
  it.todo("throws when shape has incompatible element count");
});

describe("flatten", () => {
  it.todo("collapses all dimensions into one 1-D tensor");
  it.todo("startAxis=1 keeps the first dim, flattens the rest");
});
