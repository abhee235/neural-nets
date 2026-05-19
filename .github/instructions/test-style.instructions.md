---
description: "Use when writing test files (*.test.ts). Covers bun:test import style, naming tests after mathematical properties, and educational comment standards for this learning library."
applyTo: "src/**/*.test.ts"
---

# Test File Standards

Tests in this project serve two purposes: **verification** and **education**.
A test should explain WHAT mathematical property it's checking, not just assert a value.
Someone reading the test file should learn about the concept being tested.

## Imports

Always import from `bun:test` — never from any external test library.

```typescript
import { describe, it, expect, beforeEach } from "bun:test";
```

## Structure

One `describe` block per exported function. Nested `describe` for distinct sub-behaviors.

```typescript
describe("matMul", () => {
  describe("shape rules", () => {
    it("(M×K) × (K×N) produces shape (M×N)", () => { ... });
  });
  describe("mathematical properties", () => {
    it("is associative: (AB)C === A(BC)", () => { ... });
  });
});
```

## Test Names

Name tests after the **mathematical property** being verified, not the implementation detail.

```typescript
// ✓ Good — the test name teaches the reader about math
it("is commutative: dot(a, b) === dot(b, a)", () => { ... });
it("applying softmax to identical values produces a uniform distribution", () => { ... });
it("the gradient of MSE with respect to prediction is 2(pred - target)/n", () => { ... });

// ✗ Bad — says nothing about math
it("returns a number", () => { ... });
it("works correctly", () => { ... });
```

## Assertion Comments

Add a short comment above each `expect` explaining what mathematical fact the assertion proves.

```typescript
it("(M×K) × (K×N) produces shape (M×N)", () => {
  const a = zeros([3, 4]);
  const b = zeros([4, 5]);
  const c = matMul(a, b);
  // Matrix multiplication contracts the inner dimension K and keeps the outer M×N
  expect(c.shape).toEqual([3, 5]);
});
```

## Numerical Tolerance

Floating-point arithmetic is never exact. Use a tolerance helper for float comparisons.

```typescript
// Define once at the top of the test file
const EPSILON = 1e-6;
const close = (a: number, b: number) => Math.abs(a - b) < EPSILON;

it("exp(0) === 1", () => {
  // Euler's identity: e^0 = 1
  expect(close(exp(scalar(0)).data[0], 1.0)).toBe(true);
});
```

## Gradient Checks (Ch 07+)

For every differentiable operation, include at least one numerical gradient check test.
This verifies that the analytical backward pass matches the finite-difference approximation.

```typescript
it("numerical gradient check passes for [functionName]", () => {
  // f'(x) ≈ (f(x+h) - f(x-h)) / 2h
  const h = 1e-5;
  // ... compute numerical and analytical gradients and assert they're close
});
```

## What to Test

- Correct output shape for typical inputs
- Correct output values for small, hand-calculable examples
- Edge cases: 1D tensors, batch dimensions, zero values, large values
- Mathematical properties where they apply: commutativity, associativity, distributivity
- Gradient correctness (numerical gradient check) for all differentiable ops
