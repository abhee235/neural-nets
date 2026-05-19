/**
 * tensor/ops.ts
 * ══════════════════════════════════════════════════════════
 * Elementwise arithmetic and NumPy-style broadcasting.
 * Equivalent to torch's +, -, *, / tensor operators.
 *
 * Chapter: 03 — Elementwise Ops & Broadcasting
 * Doc:     docs/part-1-tensor-library/ch-03-elementwise-ops-broadcasting.md
 */
import type { Tensor } from "./types.ts";

/**
 * Compute the output shape when broadcasting a and b (NumPy rules).
 * Align right; each dim must be equal or one of them is 1.
 *   broadcastShapes([3,1], [1,4]) → [3,4]
 */
export function broadcastShapes(a: number[], b: number[]): number[] {
  throw new Error("broadcastShapes not implemented");
}

/**
 * Expand t to the target shape by repeating along size-1 dimensions.
 */
export function broadcast(t: Tensor, shape: number[]): Tensor {
  throw new Error("broadcast not implemented");
}

/** Elementwise a + b with broadcasting. */
export function add(a: Tensor, b: Tensor): Tensor {
  throw new Error("add not implemented");
}

/** Elementwise a - b with broadcasting. */
export function sub(a: Tensor, b: Tensor): Tensor {
  throw new Error("sub not implemented");
}

/** Elementwise a * b with broadcasting. */
export function mul(a: Tensor, b: Tensor): Tensor {
  throw new Error("mul not implemented");
}

/** Elementwise a / b with broadcasting. */
export function div(a: Tensor, b: Tensor): Tensor {
  throw new Error("div not implemented");
}

/** Add scalar s to every element. */
export function addScalar(t: Tensor, s: number): Tensor {
  throw new Error("addScalar not implemented");
}

/** Multiply every element by scalar s. */
export function mulScalar(t: Tensor, s: number): Tensor {
  throw new Error("mulScalar not implemented");
}

/**
 * Apply function fn to every element.
 * The building block for all activation functions and math primitives.
 */
export function applyFn(t: Tensor, fn: (x: number) => number): Tensor {
  throw new Error("applyFn not implemented");
}
