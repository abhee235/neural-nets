# Chapter 02: Tensor Creation

> **Part 1 of 6 — Tensor Library (NumPy-like Foundation)**
> `src/ch-02-tensor-creation/`

---

## What You're Building

Standard factory functions for creating tensors: constant-fill, identity, ranges, and random
distributions. These are the `np.zeros`, `np.ones`, `np.eye`, `np.random.randn` equivalents
that will be used everywhere from weight initialization (Ch 13) to positional encoding (Ch 19).

---

## Why This Matters

Neural network weights must start as random values before training (Ch 13). Positional
encodings need a `linspace` of positions. Attention masks are filled with zeros and
negative infinity. Every chapter from here to Ch 30 calls functions you build here.

---

## Concepts

### Constant Tensors

These create tensors where every element is the same value.

- `zeros(shape)` — all 0.0 (used for: bias initialization, padding masks, accumulated gradients)
- `ones(shape)` — all 1.0 (used for: LayerNorm scale parameter γ, multiplicative identity)
- `fill(shape, value)` — arbitrary constant (used for: attention mask with `-Infinity`)

### Eye / Identity Matrix

`eye(n)` produces an $n \times n$ matrix where $I_{ij} = 1$ if $i = j$, else 0:

$$I = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}$$

Multiplying any matrix $A$ by $I$ gives $A$ unchanged: $AI = IA = A$.
Used in: residual connections conceptually; testing that operations compose correctly.

### Range Tensors

`arange(start, stop, step)` — evenly spaced values in `[start, stop)`:

$$[0, 1, 2, \ldots, n-1]$$

`linspace(start, stop, n)` — exactly `n` values evenly spaced between `start` and `stop` (inclusive):

$$[start,\ start + \frac{stop-start}{n-1},\ \ldots,\ stop]$$

Both used in positional encoding (Ch 19): we need a sequence of positions `[0, 1, 2, ..., seqLen-1]`
and a sequence of dimension indices.

### Random Tensors

**Uniform random — `rand(shape)`:**

Each element is sampled independently from $\text{Uniform}(0, 1)$ using JavaScript's built-in
`Math.random()`. Used rarely in ML (weight init usually uses normal distribution).

**Normal random — `randn(shape)`:**

Each element is sampled independently from the standard normal distribution $\mathcal{N}(0, 1)$.
This is the workhorse of weight initialization.

**The Problem:** JavaScript's `Math.random()` gives uniform samples. We need normal samples.

**The Solution: Box-Muller Transform**

Given two independent uniform random variables $u_1, u_2 \sim \text{Uniform}(0, 1)$, generate
two independent standard normal samples $z_0, z_1$:

$$z_0 = \sqrt{-2 \ln u_1} \cdot \cos(2\pi u_2)$$
$$z_1 = \sqrt{-2 \ln u_1} \cdot \sin(2\pi u_2)$$

Why does this work? It's a change of variables from polar to Cartesian coordinates in 2D normal
space. The proof uses the fact that $-2\ln u_1$ follows a $\chi^2(2)$ distribution when $u_1 \sim U(0,1)$.

Implementation note: generate pairs, use $z_0$ and buffer $z_1$ for the next call. This is exactly
what NumPy and most ML libraries do internally.

```
u1 = Math.random()   // must be > 0 to avoid ln(0) = -Infinity
u2 = Math.random()
magnitude = sqrt(-2 * ln(u1))
z0 = magnitude * cos(2π * u2)   // return this
z1 = magnitude * sin(2π * u2)   // buffer this for the next call
```

---

## What to Implement

| Function | Signature | Description |
|----------|-----------|-------------|
| `zeros` | `(shape: number[]) => Tensor` | All zeros |
| `ones` | `(shape: number[]) => Tensor` | All ones |
| `fill` | `(shape: number[], value: number) => Tensor` | Constant fill |
| `eye` | `(n: number) => Tensor` | n×n identity matrix |
| `arange` | `(start: number, stop: number, step?: number) => Tensor` | Range [start, stop) |
| `linspace` | `(start: number, stop: number, n: number) => Tensor` | n evenly spaced values |
| `rand` | `(shape: number[]) => Tensor` | Uniform random [0, 1) |
| `randn` | `(shape: number[]) => Tensor` | Standard normal N(0, 1) — use Box-Muller |

---

## TypeScript Hints

```typescript
// zeros and ones are just special cases of fill:
function zeros(shape: number[]): Tensor {
  return fill(shape, 0);
}

// For eye, you need to place 1s at positions where row === col.
// Use flatIndex from Ch 01 or build the flat array with a nested loop.
function eye(n: number): Tensor {
  const data = new Array<number>(n * n).fill(0);
  for (let i = 0; i < n; i++) {
    data[i * n + i] = 1;  // diagonal: flat index of [i, i] in shape [n, n]
  }
  return createTensor(data, [n, n]);
}

// For randn: buffer the second Box-Muller value to avoid waste.
// Use a module-level variable to hold the buffered value.
let _boxMullerBuffer: number | null = null;

function randnScalar(): number {
  if (_boxMullerBuffer !== null) {
    const val = _boxMullerBuffer;
    _boxMullerBuffer = null;
    return val;
  }
  // Ensure u1 > 0 to avoid ln(0)
  const u1 = Math.max(Math.random(), Number.EPSILON);
  const u2 = Math.random();
  const magnitude = Math.sqrt(-2 * Math.log(u1));
  _boxMullerBuffer = magnitude * Math.sin(2 * Math.PI * u2);
  return magnitude * Math.cos(2 * Math.PI * u2);
}
```

---

## Self-Check Questions

1. `arange(0, 10, 2)` should produce `[0, 2, 4, 6, 8]`. How many elements is that?
2. `linspace(0, 1, 5)` should produce `[0, 0.25, 0.5, 0.75, 1.0]`. What's the formula?
3. Why must `u1 > 0` in Box-Muller? What would `ln(0)` evaluate to?
4. Prove to yourself that `randn` actually produces values near 0 with std ≈ 1:
   generate 1000 samples and compute the mean and standard deviation. Is it close?
5. In Ch 19, positional encoding uses `arange(0, seqLen)` and `arange(0, dModel, 2)`.
   What shapes would those produce for `seqLen=10, dModel=8`?

---

## → Next Chapter

**Ch 03: Element-wise Ops & Broadcasting** — apply arithmetic operations (+, -, *, /)
across tensors of compatible shapes using NumPy-style broadcasting rules.
