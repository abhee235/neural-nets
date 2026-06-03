/**
 * utils/numerical.ts
 * ══════════════════════════════════════════════════════════
 * Numerical gradient checking via finite differences.
 * Use this to verify every backward pass you implement.
 *
 * Chapter: 07 — Calculus for ML
 * Doc:     docs/part-2-autodiff/ch-07-calculus-for-ml.md
 */
import { createTensor, type Tensor } from "../tensor/types";

/**
 * Symmetric (centered) finite-difference approximation of the scalar derivative:
 *   f′(x) ≈ ( f(x+h) − f(x−h) ) / 2h
 *
 * Centered (both-sided) is O(h²) accurate — far better than the one-sided O(h)
 * form for the same h. Default h = 1e-5 is the round-off/truncation sweet spot.
 */
export function numericalGradient(
  fn: (x: number) => number,
  x: number,
  h?: number,
): number {
  const delta = h ?? 1e-5;
  return (fn(x + delta) - fn(x - delta)) / (2 * delta);
}

/**
 * Numerical gradient of a scalar-valued function w.r.t. every element of a
 * Tensor, perturbing one element at a time with the centered difference.
 *
 * Pitfall handled: each perturbation is applied to an isolated COPY of the data,
 * so the caller's tensor is never mutated (its `data` is conceptually readonly,
 * and an exception inside `fn` must not leave the input corrupted).
 */
export function numericalGradientTensor(
  fn: (t: Tensor) => number,
  t: Tensor,
  h?: number,
): Tensor {
  const delta = h ?? 1e-5;
  const grad = new Array<number>(t.size);

  // ── SIMPLE VERSION (clearest — what to write first) ──────────────────────
  // A gradient is one partial derivative per input: for each element i, nudge
  // ONLY element i (up, then down) while every other element stays at its
  // original value, and take the centered difference. The obvious way is to
  // make two fresh copies of the data each iteration and bump element i:
  //
  //   for (let i = 0; i < t.size; i++) {
  //     const plus  = Array.from(t.data);   // a clean copy, every iteration
  //     const minus = Array.from(t.data);
  //     plus[i]  = plus[i]!  + delta;        // only element i differs
  //     minus[i] = minus[i]! - delta;
  //     const fPlus  = fn(createTensor(plus,  t.shape));
  //     const fMinus = fn(createTensor(minus, t.shape));
  //     grad[i] = (fPlus - fMinus) / (2 * delta);
  //   }
  //
  // Correct and easy to read, but it allocates 2·N arrays for an N-element
  // tensor — wasteful when all we ever change is one element at a time.

  // ── OPTIMIZED VERSION (one working copy, restored each step) ─────────────
  // Keep a single private copy of the data. Perturb element i, evaluate, then
  // restore it before moving on — so the other elements are always original.
  // We mutate our OWN copy, never the caller's tensor (its data stays intact).
  const work = Array.from(t.data); // one allocation for the whole loop
  for (let i = 0; i < t.size; i++) {
    const original = work[i]!;

    work[i] = original + delta;
    const fPlus = fn(createTensor(work, t.shape)); // createTensor copies internally

    work[i] = original - delta;
    const fMinus = fn(createTensor(work, t.shape));

    work[i] = original; // restore before the next element
    grad[i] = (fPlus - fMinus) / (2 * delta);
  }

  return createTensor(grad, t.shape);
}

/**
 * Return true when |analytical − numerical| < tolerance.
 * Default tolerance: 1e-5.
 */
export function checkGradient(
  analytical: number,
  numerical: number,
  tolerance?: number,
): boolean {
  const tol = tolerance ?? 1e-5;
  return Math.abs(analytical - numerical) < tol;
}
