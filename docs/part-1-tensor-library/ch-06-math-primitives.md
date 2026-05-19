# Chapter 06: Math Primitives

> **Part 1 of 6 — Tensor Library (NumPy-like Foundation)**
> `src/ch-06-math-primitives/`

---

## What You're Building

Element-wise scalar math functions lifted to work on tensors: `sqrt`, `exp`, `log`, `pow`,
`abs`, `clip`, `sign`, and `tanh`. These are the primitives that activation functions (Ch 11)
and loss functions (Ch 12) are built from.

---

## Why This Matters

Every activation function is a composition of these primitives:
- ReLU: `max(0, x)` — uses `clip`
- Sigmoid: `1 / (1 + exp(-x))` — uses `exp`
- Softmax: `exp(x) / sum(exp(x))` — uses `exp` and `sum` (Ch 05)
- Cross-entropy loss: `-sum(target * log(prediction))` — uses `log`
- GELU (used in transformers): involves `tanh`

Once these primitives exist and are correct, activation functions write themselves in one line.

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

## → Next Part

**Part 2 — Autodiff Engine (Ch 07–10):** build the automatic differentiation engine that
computes gradients automatically. This turns your tensor library into a learning system.
