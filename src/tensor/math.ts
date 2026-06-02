/**
 * tensor/math.ts
 * ══════════════════════════════════════════════════════════
 * Element-wise math functions applied to tensors.
 * Equivalent to torch.exp, torch.log, torch.sqrt, torch.tanh, torch.sigmoid, …
 *
 * Chapter: 06 — Math Primitives
 * Doc:     docs/part-1-tensor-library/ch-06-math-primitives.md
 *
 * Every function here is the SAME pattern: stamp a one-number function `f`
 * onto every element, leaving the shape unchanged. The only real thought is
 * numerical hygiene (log(0), exp overflow, sigmoid overflow) — handled below.
 */
import { createTensor, type Tensor } from "./types";

// ─── helper ───────────────────────────────────────────────────────────────

/**
 * Lift a scalar function to a whole tensor: apply `fn` to every element.
 * The output has the SAME shape as the input — element-wise never reshapes.
 */
function applyElementwise(t: Tensor, fn: (x: number) => number): Tensor {
  const out = new Array<number>(t.size);
  for (let i = 0; i < t.size; i++) out[i] = fn(t.data[i]!);
  return createTensor(out, t.shape);
}

// ─── exponential / logarithm ────────────────────────────────────────────────

/**
 * Element-wise eˣ. Always positive; grows fast. Used in softmax and sigmoid.
 *
 * Pitfall: exp overflows to Infinity past ~709. We do NOT clamp here — the
 * caller controls stability by subtracting the max first (softmax does this).
 * Keeping exp pure means log(exp(x)) === x exactly where it should.
 */
export function exp(t: Tensor): Tensor {
  return applyElementwise(t, Math.exp);
}

/**
 * Element-wise natural log ln(x). The inverse of exp. Used in cross-entropy.
 *
 * Pitfall: this is the PURE log — log(0) = -Infinity, log(negative) = NaN.
 * Guarding is the caller's job: clip the input first, e.g. log(clip(p, 1e-7, 1)).
 * Keeping log pure preserves the identities log(exp(x)) ≈ x and exp(log(x)) ≈ x.
 */
export function log(t: Tensor): Tensor {
  return applyElementwise(t, Math.log);
}

// ─── powers / roots / magnitude ──────────────────────────────────────────────

/**
 * Element-wise √x. Used in LayerNorm (÷ std) and attention scaling (÷ √dHead).
 *
 * Pitfall: sqrt of a negative is NaN, and the NEXT step usually divides by the
 * result — so callers add epsilon under the root: sqrt(variance + ε).
 */
export function sqrt(t: Tensor): Tensor {
  return applyElementwise(t, Math.sqrt);
}

/** Element-wise xⁿ. Most common case: pow(t, 2) to square (e.g. variance). */
export function pow(t: Tensor, exponent: number): Tensor {
  return applyElementwise(t, (x) => x ** exponent);
}

/** Element-wise |x| — distance from zero, always non-negative. */
export function abs(t: Tensor): Tensor {
  return applyElementwise(t, Math.abs);
}

// ─── clamping ────────────────────────────────────────────────────────────────

/**
 * Clamp every element into [min, max]: below min → min, above max → max.
 * Used as a safety rail — e.g. clip(p, 1e-7, 1) before log to avoid log(0),
 * or clip(grad, -1, 1) to tame exploding gradients. clip(x, 0, ∞) is ReLU.
 */
export function clip(t: Tensor, min: number, max: number): Tensor {
  return applyElementwise(t, (x) => Math.min(Math.max(x, min), max));
}

// ─── activation curves ───────────────────────────────────────────────────────

/**
 * Element-wise tanh(x), output in (-1, 1). A non-linear "bend" between layers.
 * Appears inside the GELU approximation the transformer FFN uses (Ch 25).
 * Math.tanh is already stable, so we use it directly.
 */
export function tanh(t: Tensor): Tensor {
  return applyElementwise(t, Math.tanh);
}

/**
 * Element-wise sigmoid σ(x) = 1 / (1 + e⁻ˣ), output in (0, 1).
 *
 * Pitfall: the naive form computes e⁻ˣ, which OVERFLOWS for very negative x.
 * The branch below keeps the exponent's argument ≤ 0 in both cases, so it
 * never overflows (and never produces NaN):
 *   x ≥ 0:  1 / (1 + e⁻ˣ)
 *   x < 0:  eˣ / (1 + eˣ)      (mathematically identical, but eˣ is small here)
 */
export function sigmoid(t: Tensor): Tensor {
  return applyElementwise(t, (x) => {
    if (x >= 0) {
      return 1 / (1 + Math.exp(-x));
    }
    const ex = Math.exp(x);
    return ex / (1 + ex);
  });
}
