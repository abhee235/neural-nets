/**
 * Tests for tensor/types.ts
 * Chapter 01 — Tensor Type System
 *
 * Run: bun test src/tensor/types.test.ts
 */
import { describe, it, expect } from "bun:test";
import { createTensor, scalar, isTensor, flatIndex } from "./types";

describe("createTensor", () => {
  it("stores data flat in the same order as the input array", () => {
    const t = createTensor([10, 20, 30, 40, 50, 60], [2, 3]);
    // Row-major: first row [10,20,30] then second row [40,50,60]
    expect(Array.from(t.data)).toEqual([10, 20, 30, 40, 50, 60]);
  });

  it("records shape exactly as given", () => {
    const t = createTensor([1, 2, 3, 4, 5, 6], [2, 3]);
    // shape must reflect the logical dimensionality
    expect(t.shape).toEqual([2, 3]);
  });

  it("size equals the product of all shape dimensions", () => {
    const t = createTensor(new Array(24).fill(0), [2, 3, 4]);
    // 2 * 3 * 4 = 24 elements total
    expect(t.size).toBe(24);
  });

  it("ndim equals shape.length", () => {
    const t = createTensor(new Array(24).fill(0), [2, 3, 4]);
    // A rank-3 tensor has 3 axes
    expect(t.ndim).toBe(3);
  });

  it("throws when data.length does not equal the shape product", () => {
    // 5 elements cannot represent a [2,3] tensor that needs 6
    expect(() => createTensor([1, 2, 3, 4, 5], [2, 3])).toThrow();
  });
});

describe("scalar", () => {
  it("creates a tensor with shape [] (rank 0)", () => {
    const s = scalar(3.14);
    // A scalar has no axes — shape is the empty array
    expect(s.shape).toEqual([]);
    expect(s.ndim).toBe(0);
  });

  it("stores exactly one element", () => {
    const s = scalar(42);
    // A scalar is a single number wrapped as a Tensor
    expect(s.size).toBe(1);
    expect(s.data[0]).toBe(42);
  });
});

describe("flatIndex", () => {
  it("[0,0] maps to offset 0 for any shape", () => {
    // The origin of any tensor is always at flat offset 0
    expect(flatIndex([3, 4], [0, 0])).toBe(0);
    expect(flatIndex([2, 3, 4], [0, 0, 0])).toBe(0);
  });

  it("row-major: [i,j] maps to i*cols + j for a 2-D tensor", () => {
    // For shape [3,4]: offset = row * 4 + col
    expect(flatIndex([3, 4], [1, 2])).toBe(1 * 4 + 2); // 6
    expect(flatIndex([3, 4], [2, 3])).toBe(2 * 4 + 3); // 11
  });

  it("3-D: [d,r,c] maps to d*(R*C) + r*C + c", () => {
    // For shape [2,3,4]: stride[0]=12, stride[1]=4, stride[2]=1
    expect(flatIndex([2, 3, 4], [1, 2, 3])).toBe(1 * 12 + 2 * 4 + 3 * 1); // 23
  });

  it("throws when an index is out of bounds", () => {
    // row index 3 is out of bounds for a shape with only 3 rows (0,1,2)
    expect(() => flatIndex([3, 4], [3, 0])).toThrow();
    // negative indices are never valid
    expect(() => flatIndex([3, 4], [-1, 0])).toThrow();
  });
});

describe("isTensor", () => {
  it("returns true for a valid Tensor object", () => {
    const t = createTensor([1, 2, 3], [3]);
    // A properly constructed Tensor must pass the type guard
    expect(isTensor(t)).toBe(true);
  });

  it("returns false for a plain number", () => {
    // A bare number has none of the required Tensor fields
    expect(isTensor(42)).toBe(false);
  });

  it("returns false for null and undefined", () => {
    expect(isTensor(null)).toBe(false);
    expect(isTensor(undefined)).toBe(false);
  });
});
