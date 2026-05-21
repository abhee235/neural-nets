/**
 * Tests for tensor/creation.ts
 * Chapter 02 — Tensor Creation
 *
 * Run: bun test src/tensor/creation.test.ts
 */
import { describe, it, expect } from "bun:test";
import {
  zeros,
  ones,
  fill,
  fullLike,
  eye,
  arange,
  linspace,
  rand,
  randn,
  _resetRandnBuffer,
} from "./creation";

// ── Constant factories ─────────────────────────────────────────────────────

describe("zeros", () => {
  it("produces a tensor whose every element is 0", () => {
    // The defining property of `zeros`: ∀ i, data[i] = 0.
    const t = zeros([2, 3]);
    for (let i = 0; i < t.size; i++) expect(t.data[i]).toBe(0);
  });

  it("matches the requested shape exactly", () => {
    // Shape must be carried through verbatim — no transposes, no padding.
    const t = zeros([4, 5, 6]);
    expect(t.shape).toEqual([4, 5, 6]);
    expect(t.size).toBe(120);
  });

  it("supports a scalar (rank-0) tensor with shape []", () => {
    // Empty shape encodes a scalar; size is the empty product (= 1).
    const t = zeros([]);
    expect(t.size).toBe(1);
    expect(t.data[0]).toBe(0);
  });
});

describe("ones", () => {
  it("produces a tensor whose every element is 1", () => {
    // Mirror of zeros — defining property is ∀ i, data[i] = 1.
    const t = ones([3, 3]);
    for (let i = 0; i < t.size; i++) expect(t.data[i]).toBe(1);
  });
});

describe("fill", () => {
  it("sets every element to the given constant", () => {
    // Both zeros and ones are special cases of fill — verifying the general
    // property here covers them too.
    const t = fill([4], 7.5);
    for (let i = 0; i < t.size; i++) expect(t.data[i]).toBe(7.5);
  });

  it("returns a fresh buffer (no aliasing across calls)", () => {
    // Mutating one tensor must not leak into another — defensive copy check.
    const a = fill([3], 1);
    const b = fill([3], 1);
    a.data[0] = 99;
    expect(b.data[0]).toBe(1);
  });
});

describe("fullLike", () => {
  it("matches the source tensor's shape, not its values", () => {
    // The whole point of *Like is "shape only" — values come from `value`.
    const src = ones([2, 3]);
    const out = fullLike(src, -4);
    expect(out.shape).toEqual([2, 3]);
    for (let i = 0; i < out.size; i++) expect(out.data[i]).toBe(-4);
  });

  it("does not alias the source tensor's shape array", () => {
    // Mutating out.shape must not affect src.shape — the spread inside
    // fullLike is what guarantees this.
    const src = ones([2, 3]);
    const out = fullLike(src, 0);
    out.shape[0] = 99;
    expect(src.shape).toEqual([2, 3]);
  });
});

// ── Identity matrix ────────────────────────────────────────────────────────

describe("eye", () => {
  it("places 1 on the main diagonal and 0 elsewhere", () => {
    // I[i,j] = 1 iff i == j. This is the defining property.
    const I = eye(4);
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        const expected = i === j ? 1 : 0;
        expect(I.data[i * 4 + j]).toBe(expected);
      }
    }
  });

  it("returns shape [n, n]", () => {
    expect(eye(5).shape).toEqual([5, 5]);
  });

  it("eye(0) is a 0×0 tensor with no elements", () => {
    // Edge case — must not crash, must produce an empty tensor.
    const I = eye(0);
    expect(I.size).toBe(0);
    expect(I.shape).toEqual([0, 0]);
  });
});

// ── Sequence factories ─────────────────────────────────────────────────────

describe("arange", () => {
  it("produces [start, start+step, …] strictly less than stop", () => {
    // Defining property: arange is half-open at the right.
    const a = arange(0, 1, 0.25);
    expect(Array.from(a.data)).toEqual([0, 0.25, 0.5, 0.75]);
    expect(a.shape).toEqual([4]);
  });

  it("default step is 1", () => {
    // Convenience overload — arange(0, 5) === arange(0, 5, 1).
    const a = arange(0, 5);
    expect(Array.from(a.data)).toEqual([0, 1, 2, 3, 4]);
  });

  it("supports negative step (descending sequence)", () => {
    // Sign of step controls direction; stop is still exclusive.
    const a = arange(5, 0, -1);
    expect(Array.from(a.data)).toEqual([5, 4, 3, 2, 1]);
  });

  it("returns an empty tensor when step's sign disagrees with span", () => {
    // arange(0, 5, -1) cannot reach 5 going down → empty (matches NumPy).
    const a = arange(0, 5, -1);
    expect(a.size).toBe(0);
  });

  it("returns an empty tensor when start === stop", () => {
    // No room between start and stop → no elements.
    const a = arange(3, 3);
    expect(a.size).toBe(0);
  });

  it("throws when step is zero", () => {
    // Otherwise the loop would never terminate.
    expect(() => arange(0, 5, 0)).toThrow();
  });
});

describe("linspace", () => {
  it("includes both endpoints when n > 1", () => {
    // Defining property: linspace is closed at both ends, unlike arange.
    const a = linspace(0, 1, 5);
    expect(Array.from(a.data)).toEqual([0, 0.25, 0.5, 0.75, 1]);
  });

  it("uses divisor (n-1) so the spacing is (stop-start)/(n-1)", () => {
    // 11 points across [0,10] → spacing of 1.0, last value exactly 10.
    const a = linspace(0, 10, 11);
    expect(a.size).toBe(11);
    expect(a.data[a.size - 1]).toBe(10);
    expect(a.data[1] as number - (a.data[0] as number)).toBeCloseTo(1, 12);
  });

  it("n = 1 returns the single point [start]", () => {
    // Edge case: one point, no gaps. Return [start] (NumPy convention).
    const a = linspace(2, 7, 1);
    expect(Array.from(a.data)).toEqual([2]);
  });

  it("n = 0 returns an empty tensor", () => {
    // Edge case: zero points → empty. Must not divide by zero.
    const a = linspace(0, 1, 0);
    expect(a.size).toBe(0);
  });

  it("pins the right endpoint exactly to defeat float drift", () => {
    // start + (n-1) * delta may not equal stop due to FP rounding.
    // We assert exact equality (not toBeCloseTo) to verify the pin works.
    const a = linspace(0, 1, 7);
    expect(a.data[6]).toBe(1);
  });
});

// ── Random factories ───────────────────────────────────────────────────────

describe("rand", () => {
  it("produces values in [0, 1)", () => {
    // Math.random's range is the contract we propagate.
    const a = rand([1000]);
    for (let i = 0; i < a.size; i++) {
      const v = a.data[i] as number;
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });
});

describe("randn", () => {
  it("has mean ≈ 0 and stddev ≈ 1 over many samples", () => {
    // The whole point of Box-Muller: the output is N(0, 1). We verify the
    // first two moments empirically. Sample size 10 000 + tolerance 0.05
    // gives a reliably-passing test even though Math.random is not seeded.
    _resetRandnBuffer();
    const N = 10000;
    const a = randn([N]);
    let sum = 0;
    for (let i = 0; i < N; i++) sum += a.data[i] as number;
    const mean = sum / N;
    let sumSq = 0;
    for (let i = 0; i < N; i++) {
      const d = (a.data[i] as number) - mean;
      sumSq += d * d;
    }
    const std = Math.sqrt(sumSq / N);
    expect(Math.abs(mean)).toBeLessThan(0.05);
    expect(Math.abs(std - 1)).toBeLessThan(0.05);
  });

  it("buffers the second sample so two scalars share one (u1, u2) draw", () => {
    // After randn([1]) the buffer holds z₁; the next randn([1]) must consume
    // it without calling Math.random again. We assert this indirectly: with
    // a stubbed Math.random, two calls to randn([1]) should consume exactly
    // 2 base samples (u1, u2), not 4.
    _resetRandnBuffer();
    const queue = [0.7, 0.3, 0.9, 0.1]; // more than enough to expose extra draws
    const original = Math.random;
    let consumed = 0;
    (Math as { random: () => number }).random = () => {
      consumed++;
      const v = queue.shift();
      if (v === undefined) throw new Error("randn consumed more samples than expected");
      return v;
    };
    try {
      randn([1]); // STATE A → consumes u1, u2 (2 draws) and stashes z₁
      randn([1]); // STATE B → returns the cached z₁ (0 draws)
      expect(consumed).toBe(2);
    } finally {
      (Math as { random: () => number }).random = original;
      _resetRandnBuffer();
    }
  });
});
