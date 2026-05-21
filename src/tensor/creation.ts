/**
 * tensor/creation.ts
 * ══════════════════════════════════════════════════════════
 * Factory functions that produce Tensors filled with specific values.
 * Equivalent to numpy.zeros, numpy.ones, numpy.random.randn, numpy.eye,
 * numpy.arange, numpy.linspace.
 *
 * Chapter: 02 — Tensor Creation
 * Doc:     docs/part-1-tensor-library/ch-02-tensor-creation.md
 *
 * Design rule: every factory below boils down to "compute size, allocate buffer,
 * fill it, hand to createTensor". The only thing that varies is the fill step.
 */
import type { Tensor } from "./types.ts";
import { createTensor } from "./types";

// ─── Internal helpers ────────────────────────────────────────────────────────

/**
 * Total element count for a shape. Empty shape `[]` represents a scalar with
 * size 1. Negative or non-integer dims are rejected up-front so allocation
 * cannot silently produce a zero-element tensor when the caller meant
 * something else.
 */
function sizeOf(shape: number[]): number {
  let size = 1;
  for (let i = 0; i < shape.length; i++) {
    const dim = shape[i] as number;
    if (!Number.isInteger(dim) || dim < 0) {
      throw new Error(`shape contains an invalid dimension at index ${i}: ${dim}`);
    }
    size *= dim;
  }
  return size;
}

/**
 * Build a Tensor whose buffer is produced by `gen(i)` for each flat index i.
 * Used by the random factories below; the constant factories take a faster
 * path that calls Float64Array.fill directly.
 */
function build(shape: number[], gen: (i: number) => number): Tensor {
  const size = sizeOf(shape);
  const buf = new Float64Array(size);
  for (let i = 0; i < size; i++) buf[i] = gen(i);
  // createTensor takes number[], not Float64Array — Array.from bridges them.
  // The conversion cost is one-time and keeps Ch 01 invariants in one place.
  return createTensor(Array.from(buf), [...shape]);
}

// ─── Constant factories ──────────────────────────────────────────────────────

/**
 * Tensor of given shape with every element equal to `value`.
 *
 * `zeros`, `ones`, and `fullLike` are all thin wrappers over this — keeps the
 * fill loop in exactly one place.
 */
export function fill(shape: number[], value: number): Tensor {
  const size = sizeOf(shape);
  // Float64Array.fill is the fastest way to memset a typed array in JS engines.
  const buf = new Float64Array(size);
  buf.fill(value);
  return createTensor(Array.from(buf), [...shape]);
}

/** Tensor filled with 0.0. Used for bias vectors and padding masks. */
export function zeros(shape: number[]): Tensor {
  return fill(shape, 0);
}

/** Tensor filled with 1.0. Used for LayerNorm gamma initialisation (Ch 26). */
export function ones(shape: number[]): Tensor {
  return fill(shape, 1);
}

/**
 * New tensor with the same shape as `t`, filled with `value`.
 * Used in autograd to allocate a gradient tensor matching a parameter's shape.
 */
export function fullLike(t: Tensor, value: number): Tensor {
  // Spread to give createTensor a fresh shape array — t.shape may be aliased.
  return fill([...t.shape], value);
}

// ─── Identity matrix ─────────────────────────────────────────────────────────

/**
 * The n×n identity matrix Iₙ.   I[i, j] = 1 if i == j else 0.
 *
 * Allocate a zeroed buffer and write a single 1 per row at offset i*n + i.
 * One write per diagonal cell — cheaper than the naïve double loop that
 * visits every cell. (See Self-Check Q6 in the chapter doc.)
 */
export function eye(n: number): Tensor {
  if (!Number.isInteger(n) || n < 0) {
    throw new Error(`eye(n): n must be a non-negative integer, got ${n}`);
  }
  const buf = new Float64Array(n * n); // already zero-filled
  for (let i = 0; i < n; i++) {
    buf[i * n + i] = 1;
  }
  return createTensor(Array.from(buf), [n, n]);
}

// ─── Sequence factories ──────────────────────────────────────────────────────

/**
 * 1-D tensor [start, start+step, start+2*step, …] up to (but not including)
 * stop. Like numpy.arange. `step` defaults to 1.
 *
 * Edge cases:
 *  - step === 0          → error (would loop forever).
 *  - sign mismatch       → empty tensor (matches NumPy).
 *  - start === stop      → empty tensor.
 *
 * Compute each value as `start + i * step` rather than accumulating; that
 * keeps floating-point error at O(eps) instead of O(n · eps).
 */
export function arange(start: number, stop: number, step?: number): Tensor {
  const s = step ?? 1;
  if (s === 0) throw new Error("arange: step must not be zero");
  // If (stop - start) and step disagree in sign, the result is empty —
  // matches numpy.arange.
  const span = stop - start;
  if (span === 0 || Math.sign(span) !== Math.sign(s)) {
    return createTensor([], [0]);
  }
  const n = Math.max(0, Math.ceil(span / s));
  const buf = new Float64Array(n);
  for (let i = 0; i < n; i++) buf[i] = start + i * s;
  return createTensor(Array.from(buf), [n]);
}

/**
 * 1-D tensor of `n` evenly spaced values from `start` to `stop` inclusive.
 * Like numpy.linspace.
 *
 * The divisor is `n - 1`, not `n`, because n points create n-1 gaps. Special
 * cases:
 *  - n === 0  → empty tensor.
 *  - n === 1  → single-element [start] (the "lonely point" case).
 *  - start === stop, n > 1 → repeated value.
 *
 * The final element is set to exactly `stop` to defeat float drift —
 * `start + (n-1) · delta` may not equal `stop` due to rounding.
 */
export function linspace(start: number, stop: number, n: number): Tensor {
  if (!Number.isInteger(n) || n < 0) {
    throw new Error(`linspace: n must be a non-negative integer, got ${n}`);
  }
  if (n === 0) return createTensor([], [0]);
  if (n === 1) return createTensor([start], [1]);
  const buf = new Float64Array(n);
  const delta = (stop - start) / (n - 1);
  for (let i = 0; i < n; i++) buf[i] = start + i * delta;
  // Pin the right endpoint exactly to absorb FP rounding.
  buf[n - 1] = stop;
  return createTensor(Array.from(buf), [n]);
}

// ─── Random factories ────────────────────────────────────────────────────────

/**
 * Uniform random tensor — each element is an independent sample from
 * Uniform(0, 1) using JavaScript's built-in Math.random().
 *
 * Used rarely in ML directly; provided as a building block and for tests.
 */
export function rand(shape: number[]): Tensor {
  return build(shape, () => Math.random());
}

// ─── Box-Muller buffer (module-level) ────────────────────────────────────────
// Box-Muller produces two independent N(0,1) samples per (u1, u2) pair. We
// return the first and stash the second here for the next call. This halves
// the calls to Math.random / Math.log / Math.cos / Math.sin.
//
// State machine (see docs/assets/ch-02/box-muller-buffering.svg):
//   STATE A (buffer === null) → compute pair, return z₀, set buffer = z₁.
//   STATE B (buffer !== null) → return buffer, set buffer = null.
let boxMullerBuffer: number | null = null;

/**
 * Produce one N(0, 1) sample. Encapsulates the buffering state machine so
 * callers cannot accidentally desynchronise the cache.
 */
function randnScalar(): number {
  if (boxMullerBuffer !== null) {
    const cached = boxMullerBuffer;
    boxMullerBuffer = null;
    return cached;
  }
  // Math.random() may return 0; guard u1 away from 0 so Math.log is finite.
  // EPSILON gives a sample beyond ~38σ in the worst case — far outside any
  // value a neural net would care about.
  const u1 = Math.max(Math.random(), Number.EPSILON);
  const u2 = Math.random();
  const r = Math.sqrt(-2 * Math.log(u1));
  const theta = 2 * Math.PI * u2;
  const z0 = r * Math.cos(theta);
  const z1 = r * Math.sin(theta);
  boxMullerBuffer = z1;
  return z0;
}

/**
 * Tensor of N(0, 1) samples — the workhorse of weight initialisation.
 *
 * Each element comes from one call to `randnScalar`, so the buffering state
 * machine is consulted exactly `size` times per call. If `size` is odd, the
 * buffer ends up holding one unused sample that the *next* `randn` call will
 * consume — no waste, no correctness issue.
 *
 * Used to initialise weight matrices in Linear (Ch 13) and Embedding (Ch 18).
 */
export function randn(shape: number[]): Tensor {
  return build(shape, () => randnScalar());
}

// ─── Test-only helper ────────────────────────────────────────────────────────

/**
 * Reset the Box-Muller buffer. Used by tests so randn behaviour is independent
 * of test ordering. NOT intended for production use.
 */
export function _resetRandnBuffer(): void {
  boxMullerBuffer = null;
}
