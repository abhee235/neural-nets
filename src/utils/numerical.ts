/**
 * utils/numerical.ts
 * ══════════════════════════════════════════════════════════
 * Numerical gradient checking via finite differences.
 * Use this to verify every backward pass you implement.
 *
 * Chapter: 07 — Calculus for ML
 * Doc:     docs/part-2-autodiff/ch-07-calculus-for-ml.md
 */
import type { Tensor } from "../tensor/index.ts";

/**
 * Symmetric finite-difference approximation of the scalar derivative:
 *   f′(x) ≈ ( f(x+h) − f(x−h) ) / 2h
 *
 * More accurate than the one-sided formula for smooth functions.
 */
export function numericalGradient(
  fn: (x: number) => number,
  x: number,
  h?: number
): number {
  throw new Error("numericalGradient not implemented");
}

/**
 * Compute the numerical gradient of a scalar-valued function w.r.t.
 * every element of a Tensor, perturbing one element at a time.
 */
export function numericalGradientTensor(
  fn: (t: Tensor) => number,
  t: Tensor,
  h?: number
): Tensor {
  throw new Error("numericalGradientTensor not implemented");
}

/**
 * Return true when |analytical − numerical| < tolerance.
 * Default tolerance: 1e-5.
 */
export function checkGradient(
  analytical: number,
  numerical: number,
  tolerance?: number
): boolean {
  throw new Error("checkGradient not implemented");
}
