/**
 * tensor/reduce.ts
 * ══════════════════════════════════════════════════════════
 * Axis-wise reductions: sum, mean, max, min, argmax, argmin, variance, std, softmax.
 * Equivalent to torch.sum, torch.mean, torch.max, torch.softmax, torch.argmax.
 *
 * Chapter: 05 — Reductions & Statistical Ops
 * Doc:     docs/part-1-tensor-library/ch-05-reductions.md
 *
 * Every reduction folds ONE axis away and keeps the rest. The axis-walking
 * machinery is identical for all of them — only the per-fold rule differs
 * (running total, running max, running best-index, …). That shared machinery
 * lives in the helpers below; each exported function just supplies its rule.
 */
import { createTensor, flatIndex, type Tensor } from "./types";

// ─── helpers ──────────────────────────────────────────────────────────────

/**
 * Decode a flat (row-major) index back into per-axis indices for `shape`.
 * Inverse of flatIndex. Example: unravel(9, [3,4]) → [2, 1].
 */
function unravel(flat: number, shape: number[]): number[] {
  const indices = new Array<number>(shape.length);
  let remainder = flat;
  for (let axis = shape.length - 1; axis >= 0; axis--) {
    const dim = shape[axis]!;
    indices[axis] = remainder % dim;
    remainder = Math.floor(remainder / dim);
  }
  return indices;
}

/** Product of an array (used for output sizes). Empty array → 1 (a scalar). */
function product(arr: number[]): number {
  let result = 1;
  for (const n of arr) result *= n;
  return result;
}

/** The shape with `axis` removed, e.g. dropAxis([2,3,4], 1) → [2,4]. */
function dropAxis(shape: number[], axis: number): number[] {
  return shape.filter((_, i) => i !== axis);
}

/**
 * Final output shape after reducing `axis`:
 *   keepDims=false → axis removed       ([2,3] reduce axis 1 → [2])
 *   keepDims=true  → axis kept as size 1 ([2,3] reduce axis 1 → [2,1])
 */
function reducedShape(shape: number[], axis: number, keepDims: boolean): number[] {
  if (keepDims) return shape.map((d, i) => (i === axis ? 1 : d));
  return dropAxis(shape, axis);
}

/**
 * Core value-reduction along an axis. For every output position it folds the
 * `axisSize` values lying along the reduced axis using `combine`, starting from
 * `init`. Returns the flat result array indexed by the axis-removed shape.
 */
function foldAxis(
  t: Tensor,
  axis: number,
  init: number,
  combine: (acc: number, value: number) => number,
): number[] {
  const dropped = dropAxis(t.shape, axis);
  const out = new Array<number>(product(dropped)).fill(init);

  // Walk every input element; route it to its output slot (the multi-index with
  // the reduced axis removed) and fold it in.
  for (let f = 0; f < t.size; f++) {
    const idx = unravel(f, t.shape);
    const outFlat = flatIndex(dropped, dropAxis(idx, axis));
    out[outFlat] = combine(out[outFlat]!, t.data[f]!);
  }
  return out;
}

// ─── sum ────────────────────────────────────────────────────────────────────

/**
 * Sum along an axis (or all elements when axis is omitted).
 * keepDims=true keeps the reduced axis as a size-1 dimension.
 */
export function sum(t: Tensor, axis?: number, keepDims = false): Tensor {
  if (axis === undefined) {
    // Reduce everything to one scalar. .reduce() here is a simple running sum.
    const total = t.data.reduce((acc, value) => acc + value, 0);
    return createTensor([total], []);
  }
  const summed = foldAxis(t, axis, 0, (acc, value) => acc + value);
  return createTensor(summed, reducedShape(t.shape, axis, keepDims));
}

// ─── mean ─────────────────────────────────────────────────────────────────

/**
 * Mean along an axis. Mean is just sum divided by the count along that axis.
 * Used in LayerNorm to compute the per-token mean (Ch 20).
 */
export function mean(t: Tensor, axis?: number, keepDims = false): Tensor {
  if (axis === undefined) {
    const total = t.data.reduce((acc, value) => acc + value, 0);
    return createTensor([total / t.size], []);
  }
  const count = t.shape[axis]!;
  const summed = foldAxis(t, axis, 0, (acc, value) => acc + value);
  const averaged = summed.map((s) => s / count);
  return createTensor(averaged, reducedShape(t.shape, axis, keepDims));
}

// ─── max / min ───────────────────────────────────────────────────────────────

/**
 * Maximum value along an axis. Used to stabilise softmax (subtract max before exp).
 */
export function max(t: Tensor, axis?: number, keepDims = false): Tensor {
  if (axis === undefined) {
    // Explicit loop (NOT Math.max(...data) — spreading a big array overflows the stack).
    let m = -Infinity;
    for (let i = 0; i < t.size; i++) if (t.data[i]! > m) m = t.data[i]!;
    return createTensor([m], []);
  }
  const maxed = foldAxis(t, axis, -Infinity, (acc, value) => (value > acc ? value : acc));
  return createTensor(maxed, reducedShape(t.shape, axis, keepDims));
}

/** Minimum value along an axis. The mirror of max. */
export function min(t: Tensor, axis?: number, keepDims = false): Tensor {
  if (axis === undefined) {
    let m = Infinity;
    for (let i = 0; i < t.size; i++) if (t.data[i]! < m) m = t.data[i]!;
    return createTensor([m], []);
  }
  const minned = foldAxis(t, axis, Infinity, (acc, value) => (value < acc ? value : acc));
  return createTensor(minned, reducedShape(t.shape, axis, keepDims));
}

// ─── argmax / argmin ─────────────────────────────────────────────────────────

/**
 * Index (position) of the extreme value along an axis. Shared by argmax/argmin;
 * `isBetter(candidate, current)` decides "bigger" (argmax) or "smaller" (argmin).
 *
 * Ties: because the test is strict, a later equal value never displaces an
 * earlier one — so the FIRST occurrence wins, matching NumPy.
 */
function argExtreme(
  t: Tensor,
  axis: number | undefined,
  isBetter: (candidate: number, current: number) => boolean,
): Tensor {
  if (axis === undefined) {
    let bestIdx = 0;
    for (let i = 1; i < t.size; i++) if (isBetter(t.data[i]!, t.data[bestIdx]!)) bestIdx = i;
    return createTensor([bestIdx], []);
  }

  const dropped = dropAxis(t.shape, axis);
  const positions = new Array<number>(product(dropped)).fill(0);
  const bestVals = new Array<number | undefined>(product(dropped)).fill(undefined);

  for (let f = 0; f < t.size; f++) {
    const idx = unravel(f, t.shape);
    const outFlat = flatIndex(dropped, dropAxis(idx, axis));
    const value = t.data[f]!;
    const current = bestVals[outFlat];
    if (current === undefined || isBetter(value, current)) {
      bestVals[outFlat] = value;
      positions[outFlat] = idx[axis]!; // the index ALONG the reduced axis
    }
  }
  // argmax/argmin never keep dims (they return one index per output position).
  return createTensor(positions, dropped);
}

/**
 * Index of the maximum value along an axis.
 * Used during greedy decoding: argmax(logits) → predicted token ID (Ch 29).
 */
export function argmax(t: Tensor, axis?: number): Tensor {
  return argExtreme(t, axis, (candidate, current) => candidate > current);
}

/**
 * Index of the minimum value along an axis. The mirror of argmax — e.g. nearest
 * neighbour is argmin over distances.
 */
export function argmin(t: Tensor, axis?: number): Tensor {
  return argExtreme(t, axis, (candidate, current) => candidate < current);
}

// ─── variance / std ──────────────────────────────────────────────────────────

/**
 * Population variance along an axis.
 * Math: σ² = (1/N) Σ (x_i − mean)²   (population, N — not sample, N−1)
 *
 * Two-pass and stable: compute the mean first, then average the squared
 * deviations. (The "shortcut" E[x²] − E[x]² loses precision on large numbers.)
 * This is the variance LayerNorm uses to normalise each token (Ch 20).
 */
export function variance(t: Tensor, axis?: number, keepDims = false): Tensor {
  if (axis === undefined) {
    const mu = t.data.reduce((acc, value) => acc + value, 0) / t.size;
    let sq = 0;
    for (let i = 0; i < t.size; i++) sq += (t.data[i]! - mu) ** 2;
    return createTensor([sq / t.size], []);
  }

  const count = t.shape[axis]!;
  const dropped = dropAxis(t.shape, axis);

  // Pass 1: mean of each group (one value per output position).
  const means = foldAxis(t, axis, 0, (acc, value) => acc + value).map((s) => s / count);

  // Pass 2: average the squared deviations from each group's mean.
  const sqDev = new Array<number>(product(dropped)).fill(0);
  for (let f = 0; f < t.size; f++) {
    const idx = unravel(f, t.shape);
    const outFlat = flatIndex(dropped, dropAxis(idx, axis));
    sqDev[outFlat] = sqDev[outFlat]! + (t.data[f]! - means[outFlat]!) ** 2;
  }
  const result = sqDev.map((s) => s / count);
  return createTensor(result, reducedShape(t.shape, axis, keepDims));
}

/**
 * Standard deviation along an axis (population, with epsilon for stability).
 * Math: σ = sqrt(variance + ε)   where ε = 1e-8 prevents sqrt/divide of zero
 * when the spread is zero. Same ε that appears in the LayerNorm denominator.
 */
export function std(t: Tensor, axis?: number, keepDims = false): Tensor {
  const EPSILON = 1e-8;
  const v = variance(t, axis, keepDims);
  // std is just the element-wise sqrt of (variance + ε).
  const rooted = Array.from(v.data, (value) => Math.sqrt(value + EPSILON));
  return createTensor(rooted, v.shape);
}

// ─── softmax ─────────────────────────────────────────────────────────────────

/**
 * Numerically stable softmax along an axis (or over all elements if axis omitted):
 *   softmax(x)_i = exp(x_i - max) / Σ_j exp(x_j - max)
 *
 * Subtracting the max keeps exp from overflowing without changing the result
 * (see docs/deep-dives/ch-05-why-subtract-the-max.md). Output is a probability
 * distribution per slice along `axis`. Final step of every attention head (Ch 22).
 */
export function softmax(t: Tensor, axis?: number): Tensor {
  const out = new Float64Array(t.size);

  if (axis === undefined) {
    // Global softmax over every element (the whole tensor sums to 1).
    let m = -Infinity;
    for (let i = 0; i < t.size; i++) if (t.data[i]! > m) m = t.data[i]!;
    let sumExp = 0;
    for (let i = 0; i < t.size; i++) {
      out[i] = Math.exp(t.data[i]! - m);
      sumExp += out[i]!;
    }
    for (let i = 0; i < t.size; i++) out[i] = out[i]! / sumExp;
    return createTensor(Array.from(out), t.shape);
  }

  // Per-axis softmax: each slice along `axis` becomes its own distribution.
  // Group elements by their output position (multi-index with the axis removed).
  const dropped = dropAxis(t.shape, axis);
  const groups = product(dropped);
  const groupMax = new Array<number>(groups).fill(-Infinity);
  const groupSum = new Array<number>(groups).fill(0);

  const groupOf = (f: number) => flatIndex(dropped, dropAxis(unravel(f, t.shape), axis));

  // Pass 1: max per group (for stability).
  for (let f = 0; f < t.size; f++) {
    const g = groupOf(f);
    if (t.data[f]! > groupMax[g]!) groupMax[g] = t.data[f]!;
  }
  // Pass 2: exponentiate (shifted) and accumulate each group's sum.
  for (let f = 0; f < t.size; f++) {
    const g = groupOf(f);
    out[f] = Math.exp(t.data[f]! - groupMax[g]!);
    groupSum[g] = groupSum[g]! + out[f]!;
  }
  // Pass 3: normalise each element by its group's sum.
  for (let f = 0; f < t.size; f++) out[f] = out[f]! / groupSum[groupOf(f)]!;

  return createTensor(Array.from(out), t.shape);
}
