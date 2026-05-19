/**
 * Tests for tensor/types.ts
 * Chapter 01 — Tensor Type System
 *
 * Run: bun test src/tensor/types.test.ts
 */
import { describe, it, expect } from "bun:test";
// import { createTensor, scalar, isTensor, flatIndex } from "./types.ts";

describe("createTensor", () => {
  it.todo("stores data flat in the same order as the input array");
  it.todo("records shape exactly as given");
  it.todo("size equals the product of all shape dimensions");
  it.todo("ndim equals shape.length");
  it.todo("throws when data.length does not equal the shape product");
});

describe("scalar", () => {
  it.todo("creates a tensor with shape [] (rank 0)");
  it.todo("stores exactly one element");
});

describe("flatIndex", () => {
  it.todo("[0,0] maps to offset 0 for any shape");
  it.todo("row-major: [i,j] maps to i*cols + j for a 2-D tensor");
  it.todo("throws when an index is out of bounds");
});

describe("isTensor", () => {
  it.todo("returns true for a valid Tensor object");
  it.todo("returns false for a plain number");
  it.todo("returns false for null and undefined");
});
