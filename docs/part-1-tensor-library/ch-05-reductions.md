# Chapter 05: Reductions

> **Part 1 of 6 — Tensor Library**
> Source: [`src/tensor/reduce.ts`](../../src/tensor/reduce.ts)
> Tests: [`src/tensor/reduce.test.ts`](../../src/tensor/reduce.test.ts)
> Exercise: [`exercises/ch-05-reductions.ts`](../../exercises/ch-05-reductions.ts)

---

## Learning Goals

By the end of this chapter you can:

- Implement `sum`, `mean`, `max`, `min`, `argmax`, and `var` with an `axis` and `keepDims` option.
- Choose between collapsing an axis (`keepDims=false`) and keeping it as size 1.
- Compute mean/variance in a single pass that is numerically stable.
- Use reductions to compute losses (mean of per-sample errors) and normalisation statistics.
- Predict which reductions appear inside softmax, LayerNorm, and cross-entropy.

---

## Intuition First

A reduction collapses one or more axes into a single value per leftover position. Mean of a `[batch, features]` tensor along axis `1` gives `[batch]`, the average feature value per sample. Reductions are how we get from millions of activations down to a single scalar loss that gradient descent can minimise.

---

## Mental Model

```text
  x.shape: [2, 3]            sum(axis=1)
  [[1, 2, 3],                ──────────►   [6, 15]    keepDims=false → [2]
   [4, 5, 6]]                              [[6], [15]] keepDims=true  → [2, 1]
```

---

## Concepts

### What is a Reduction?

A reduction takes a tensor and collapses one or more of its axes, producing a smaller tensor.

Example: summing a `[3, 4]` matrix along `axis=0` (collapse rows) gives a `[4]` vector.
Summing along `axis=1` (collapse columns) gives a `[3]` vector.

$$\text{sum along axis 0}: \quad out[j] = \sum_{i=0}^{M-1} A[i, j]$$

$$\text{sum along axis 1}: \quad out[i] = \sum_{j=0}^{N-1} A[i, j]$$

### The `keepDims` Parameter

After reduction, the collapsed axis normally disappears.
`keepDims=true` keeps it as a size-1 dimension instead:

```
A: shape [3, 4]
sum(A, axis=1)              → shape [3]      (axis 1 removed)
sum(A, axis=1, keepDims=true) → shape [3, 1]   (axis 1 kept as size 1)
```

`keepDims=true` is essential for broadcasting. In LayerNorm, you subtract the mean from
each feature. The mean has shape `[batch, seq, 1]` and the input is `[batch, seq, dModel]`.
Broadcasting makes this work — but only if the mean has the right number of axes.

### Statistical Reductions

**Mean:**
$$\bar{x} = \frac{1}{N} \sum_{i=0}^{N-1} x_i$$

**Variance** (population variance, not sample variance — ML consistently uses population):
$$\sigma^2 = \frac{1}{N} \sum_{i=0}^{N-1} (x_i - \bar{x})^2$$

**Standard deviation:**
$$\sigma = \sqrt{\sigma^2 + \varepsilon}$$

The $\varepsilon$ (epsilon, typically `1e-8`) prevents division by zero when $\sigma^2 = 0$.
This is the same $\varepsilon$ in LayerNorm.

**Numerically stable variance:** Compute `mean` first, then subtract, then square.
Don't use the "computational formula" $E[x^2] - E[x]^2$ — it's numerically unstable for
large values (catastrophic cancellation).

### `argmax` — Index of the Maximum

`argmax(t, axis)` returns the **indices** of the maximum value along an axis.

In language model inference (Ch 29): the model outputs logits of shape `[vocab_size]`.
`argmax` gives the index of the predicted token. That index is the model's prediction.

### Implementing Reductions Along an Axis

The key insight: reduction along an axis $a$ of a tensor with shape $[d_0, d_1, \ldots, d_{n-1}]$
produces output shape $[d_0, \ldots, d_{a-1}, d_{a+1}, \ldots, d_{n-1}]$ (axis $a$ removed).

Strategy: iterate over all output positions, and for each one, iterate over the range of the
reduced axis, accumulating the result.

```typescript
// Pseudocode for sum along axis
for each output index (i, j, ...) in output shape:
    out[i, j, ...] = 0
    for k in range(shape[axis]):
        out[i, j, ...] += input[..., k, ...]   // k inserted at the axis position
```

One clean way to implement this: flatten the tensor's axes around the reduction axis.
For axis $a$ of shape $[d_0, \ldots, d_{n-1}]$:
1. Transpose so axis $a$ is last → shape $[d_0, \ldots, d_{a-1}, d_{a+1}, \ldots, d_{n-1}, d_a]$
2. Flatten all but last → shape $[M, d_a]$ where $M = \text{size} / d_a$
3. Reduce each row of $M$ elements → shape $[M]$
4. Reshape back and transpose

---

## What to Implement

| Function | Signature | Description |
|----------|-----------|-------------|
| `sum` | `(t, axis?, keepDims?) => Tensor` | Sum along axis (or all axes if omitted) |
| `mean` | `(t, axis?, keepDims?) => Tensor` | Mean along axis |
| `max` | `(t, axis?, keepDims?) => Tensor` | Maximum value along axis |
| `min` | `(t, axis?, keepDims?) => Tensor` | Minimum value along axis |
| `argmax` | `(t, axis) => Tensor` | Index of max along axis |
| `argmin` | `(t, axis) => Tensor` | Index of min along axis |
| `variance` | `(t, axis?, keepDims?) => Tensor` | Population variance along axis |
| `std` | `(t, axis?, keepDims?) => Tensor` | Standard deviation (population, with epsilon) |

---

## TypeScript Hints

```typescript
// When axis is omitted (undefined), reduce over ALL elements to a scalar.
function sum(t: Tensor, axis?: number, keepDims = false): Tensor {
  if (axis === undefined) {
    // Reduce everything to a single scalar
    const total = t.data.reduce((acc, v) => acc + v, 0);
    return createTensor([total], []);  // rank-0 scalar tensor
  }
  // ... axis-specific reduction
}

// mean is just sum divided by the count along that axis:
function mean(t: Tensor, axis?: number, keepDims = false): Tensor {
  const s = sum(t, axis, keepDims);
  const count = axis === undefined ? t.size : t.shape[axis]!;
  return mulScalar(s, 1 / count);   // mulScalar from Ch 03
}

// variance: mean of squared deviations
function variance(t: Tensor, axis?: number, keepDims = false): Tensor {
  const mu = mean(t, axis, true);         // keepDims=true for broadcasting
  const diff = sub(t, mu);               // broadcast: (x - mean)
  const squared = mul(diff, diff);       // element-wise square
  return mean(squared, axis, keepDims);  // average the squared differences
}
```

---

## Common Pitfalls

- Computing variance as `mean(x²) - mean(x)²` instead of `mean((x - mean(x))²)`; the first is faster but loses precision.
- Reducing over the wrong axis because you forgot what shape the input has — print shapes.
- Returning a 0-d tensor when the caller wanted a JS `number`; pick one convention and stick to it.
- Forgetting `keepDims=true` and breaking a broadcast that comes later (LayerNorm and softmax need it).
- Iterating in C-order when the reduced axis is not the inner axis; you'll thrash the cache.

---

## How to Verify

Run the tests and the exercise. Both should pass cleanly with no warnings:

```bash
bun test src/tensor/reduce.test.ts
```
```bash
bun run exercises/ch-05-reductions.ts
```

---

## Self-Check Questions

1. `sum([[1,2],[3,4]], axis=0)` — what is the result and shape?
2. `sum([[1,2],[3,4]], axis=1, keepDims=true)` — what shape is the output?
3. Why is `keepDims=true` critical when computing LayerNorm? Describe the broadcast.
4. Compute `variance([2, 4, 4, 4, 5, 5, 7, 9])` by hand. Does your implementation match?
5. What does `argmax([[1,5,3],[4,2,6]], axis=1)` return?

---

## Further Reading

- [NumPy — Reduction operations](https://numpy.org/doc/stable/reference/routines.math.html) — list of standard reductions and their `axis`/`keepdims` semantics.
- [Welford's algorithm for variance](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm) — a streaming, numerically-stable variance; useful for very large batches.
- [PyTorch — `torch.Tensor.sum`](https://pytorch.org/docs/stable/generated/torch.Tensor.sum.html) — exact semantics for `dim` and `keepdim` we will roughly mirror.
- [Goodfellow, Bengio, Courville — Deep Learning](https://www.deeplearningbook.org/) — the standard graduate textbook; chapters map cleanly to this course.

---

## Next Chapter

**[Math Primitives](ch-06-math-primitives.md)** — combine reductions with elementwise math to get softmax, LayerNorm, and friends.
