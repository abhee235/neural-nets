# Chapter 06: Math Primitives

> **Part 1 of 6 — Tensor Library**
> Source: [`src/tensor/math.ts`](../../src/tensor/math.ts)
> Tests: [`src/tensor/math.test.ts`](../../src/tensor/math.test.ts)
> Exercise: [`exercises/ch-06-math-primitives.ts`](../../exercises/ch-06-math-primitives.ts)

---

## Learning Goals

By the end of this chapter you can:

- Implement elementwise `exp`, `log`, `sqrt`, `tanh`, `pow`, and `abs` on tensors.
- Identify the three classic numerical-stability traps: `log(0)`, `exp(big)`, and `sqrt(x + ε)`.
- Write a reusable `applyElementwise(fn)` helper so each primitive is one line.
- Verify each primitive against finite differences before depending on it.
- Know which math primitive shows up where in the transformer (softmax, LayerNorm, GELU).

---

## Intuition First

Math primitives are tiny scalar functions stamped onto every cell of a tensor. The interesting work is not the math (JavaScript already has `Math.exp`); the interesting work is **numerical hygiene**. A naive softmax with a single large logit can produce `Infinity / Infinity = NaN` and silently poison your entire training run. We learn the stable forms once and re-use them everywhere.

---

## Mental Model

```text
  applyElementwise(fn):
      for i in 0 .. data.length-1:
          out[i] = fn(data[i])

  guard rails to remember:
      log(x)        →  log(max(x, ε))
      exp(x)        →  exp(x - max(x))   (in softmax)
      sqrt(x)       →  sqrt(x + ε)        (in LayerNorm)
```

---

## Concepts

### The Pattern: Lifting Scalar Functions

A "math primitive" for tensors is just: apply the scalar function to every element.

$$f(T)[i_0, i_1, \ldots] = f(T[i_0, i_1, \ldots])$$

This means all these functions have the same implementation skeleton:

```typescript
function applyElementwise(t: Tensor, fn: (x: number) => number): Tensor {
  return createTensor(t.data.map(fn), t.shape);
}

const sqrt = (t: Tensor) => applyElementwise(t, Math.sqrt);
const exp  = (t: Tensor) => applyElementwise(t, Math.exp);
const log  = (t: Tensor) => applyElementwise(t, Math.log);
const abs  = (t: Tensor) => applyElementwise(t, Math.abs);
const sign = (t: Tensor) => applyElementwise(t, Math.sign);
```

### The Functions and Their Uses

| Function | Formula | Used in |
|----------|---------|---------|
| `sqrt(t)` | $\sqrt{x}$ | LayerNorm denominator, attention scaling ($1/\sqrt{d_k}$) |
| `exp(t)` | $e^x$ | Softmax, Sigmoid numerator |
| `log(t)` | $\ln(x)$ | Cross-entropy loss: $-\log(p)$ |
| `pow(t, n)` | $x^n$ | Variance: $(x - \mu)^2$; gradient norms |
| `abs(t)` | $|x|$ | Gradient clipping, L1 regularization |
| `clip(t, min, max)` | $\min(\max(x, \text{lo}), \text{hi})$ | ReLU clamp, numerical safety |
| `sign(t)` | $-1, 0, +1$ | Subgradient of abs |
| `tanh(t)` | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | GELU approximation, LSTM gates |

### Numerical Stability

Some primitives are numerically unstable in naive form. You must guard against:

**`log(0)` = -Infinity:**
Cross-entropy computes `log(prediction)`. If a prediction is 0 (a probability), this explodes.
Always clip predictions: `log(clip(pred, 1e-7, 1.0))`.

**`exp(x)` overflow for large x:**
For large $x$ (e.g., $x = 1000$), `Math.exp(1000)` = `Infinity`.
Softmax protects against this by subtracting the max before exponentiating:

$$\text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$$

This is mathematically identical (the max cancels in numerator and denominator) but numerically safe.

**`sqrt(0)` = 0 (fine), but `1/sqrt(0)` = Infinity:**
In LayerNorm and attention scaling, you divide by `sqrt(variance + ε)`. The `ε` (epsilon)
prevents division by zero when variance is 0.

### Why `tanh` Is Explicit Here

`tanh` could be written as `div(sub(exp(x), exp(neg(x))), add(exp(x), exp(neg(x))))` using
earlier ops, but this would be numerically poor and slow. Use `Math.tanh` directly.
The pattern continues: when JavaScript already provides a stable, optimized implementation,
use it — but understand what it computes.

---

## What to Implement

| Function | Signature | Notes |
|----------|-----------|-------|
| `sqrt` | `(t: Tensor) => Tensor` | Element-wise square root |
| `exp` | `(t: Tensor) => Tensor` | Element-wise $e^x$ |
| `log` | `(t: Tensor) => Tensor` | Element-wise natural log; `log(0)` = `-Infinity` |
| `log2` | `(t: Tensor) => Tensor` | Log base 2 (occasionally useful) |
| `pow` | `(t: Tensor, n: number) => Tensor` | Element-wise $x^n$ |
| `abs` | `(t: Tensor) => Tensor` | Element-wise absolute value |
| `clip` | `(t, lo, hi) => Tensor` | Clamp each element to `[lo, hi]` |
| `sign` | `(t: Tensor) => Tensor` | $-1$, $0$, or $+1$ |
| `tanh` | `(t: Tensor) => Tensor` | Element-wise hyperbolic tangent |
| `maximum` | `(a, b) => Tensor` | Element-wise max of two tensors (for ReLU: `maximum(t, zeros)`) |
| `minimum` | `(a, b) => Tensor` | Element-wise min of two tensors |

---

## TypeScript Hints

```typescript
// The implementation is a single helper + thin wrappers:
function applyFn(t: Tensor, fn: (x: number) => number): Tensor {
  return createTensor(t.data.map(fn), [...t.shape]);
}

export const sqrt = (t: Tensor): Tensor => applyFn(t, Math.sqrt);
export const exp  = (t: Tensor): Tensor => applyFn(t, Math.exp);
export const log  = (t: Tensor): Tensor => applyFn(t, Math.log);
export const tanh = (t: Tensor): Tensor => applyFn(t, Math.tanh);
export const abs  = (t: Tensor): Tensor => applyFn(t, Math.abs);
export const sign = (t: Tensor): Tensor => applyFn(t, Math.sign);

export const clip = (t: Tensor, lo: number, hi: number): Tensor =>
  applyFn(t, (x) => Math.min(Math.max(x, lo), hi));

export const pow = (t: Tensor, n: number): Tensor =>
  applyFn(t, (x) => Math.pow(x, n));

// maximum and minimum are element-wise between two tensors:
export const maximum = (a: Tensor, b: Tensor): Tensor =>
  elementwise(a, b, Math.max);  // elementwise from Ch 03
```

---

## Common Pitfalls

- Calling `Math.log` on a probability that has rounded to 0; clip with a tiny `eps`.
- Computing `exp(x)` for `x` of size 1000+; subtract the max first whenever you can.
- Using `**` for integer powers but forgetting it works for floats too (it is `Math.pow`).
- Allocating a new output tensor inside a hot loop; pre-allocate and write in place.
- Trusting your op without finite-difference checks — they catch sign bugs early.

---

## How to Verify

Run the tests and the exercise. Both should pass cleanly with no warnings:

```bash
bun test src/tensor/math.test.ts
```
```bash
bun run exercises/ch-06-math-primitives.ts
```

---

## Self-Check Questions

1. What is `exp(log(x))` for any `x > 0`? Why? (This is an identity you'll use for softmax.)
2. Compute `softmax([1, 2, 3])` by hand using `exp` and `sum`. Does it sum to 1?
3. Why does subtracting the max before softmax not change the result? Show algebraically.
4. What does `clip(t, 0, Infinity)` do? Which activation function is this?
5. Write `sigmoid(x)` using only `exp`, `add`, and `div` from Chapters 03 and 06.

---

## End of Part 1

Congratulations — you now have a complete NumPy-like tensor library in TypeScript.

Chapters 01–06 together give you:
- A `Tensor` type with shape, data, and strides *(Ch 01)*
- Factory functions to create any standard tensor *(Ch 02)*
- Element-wise arithmetic with broadcasting *(Ch 03)*
- Matrix multiplication, reshape, concat *(Ch 04)*
- Reductions along axes *(Ch 05)*
- Scalar math functions lifted to tensors *(Ch 06)*

This is everything PyTorch's tensor module provides — and you built it from scratch.

---

## Further Reading

- [Goldberg — What Every Computer Scientist Should Know About Floating-Point](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) — the classic; explains why `(a + b) + c ≠ a + (b + c)` for floats.
- [Nicholas Higham — Accuracy and Stability of Numerical Algorithms](https://epubs.siam.org/doi/book/10.1137/1.9780898718027) — deeper treatment if you want to design stable kernels.
- [Wikipedia — LogSumExp](https://en.wikipedia.org/wiki/LogSumExp) — the trick behind stable softmax and stable cross-entropy.
- [Goodfellow, Bengio, Courville — Deep Learning](https://www.deeplearningbook.org/) — the standard graduate textbook; chapters map cleanly to this course.

---

## Next Chapter

**[Calculus for ML](../part-2-autodiff/ch-07-calculus-for-ml.md)** — equipped with `exp`, `log`, and `sqrt`, we can move on to derivatives and the autograd engine.
