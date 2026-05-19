# Chapter 04: Matrix Operations

> **Part 1 of 6 — Tensor Library (NumPy-like Foundation)**
> `src/ch-04-matrix-ops/`

---

## What You're Building

The higher-level tensor operations: `matMul`, `transpose`, `reshape`, `flatten`, `concat`,
`stack`, `squeeze`, and `unsqueeze`. Matrix multiplication alone is the most important
single operation in all of deep learning — every linear layer and every attention computation
reduces to a `matMul`.

---

## Why This Matters

Look at the transformer's attention formula:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

That contains **three** `matMul` calls: $QK^T$, then the softmax output times $V$.
Each call on a batch of 8 sequences, 12 heads, and 64-token sequences involves matrices
in the tens of thousands of elements. Getting `matMul` correct — and understanding its
shape contract — is the single most important step in this entire course.

---

## Concepts

### Matrix Multiplication (`matMul`)

For 2D tensors (matrices), matrix multiplication contracts the inner dimension:

$$C = A \cdot B \quad \text{where } A \in \mathbb{R}^{M \times K},\ B \in \mathbb{R}^{K \times N},\ C \in \mathbb{R}^{M \times N}$$

$$C[i, j] = \sum_{k=0}^{K-1} A[i, k] \cdot B[k, j]$$

The shape rule: the **inner** dimensions must match ($K$), the **outer** dimensions ($M, N$)
become the output shape. A common mistake is multiplying `[3, 4]` with `[3, 4]` — this fails
because the inner dims are 4 and 3 (not equal).

**Batched matMul:** In the transformer, $Q$, $K$, $V$ are 3D tensors `[batch, seq, d_k]`.
You need to matrix-multiply the last two dimensions across all batch items.
Shape rule: `[B, M, K] × [B, K, N] → [B, M, N]`. The batch dimension is broadcast.

### Transpose

For a 2D matrix, transpose swaps rows and columns:

$$A^T[i, j] = A[j, i]$$

For an ND tensor, `transpose(t, axes)` permutes the axes. For example,
`transpose(t, [0, 2, 1])` for a `[B, M, N]` tensor produces `[B, N, M]`.

In attention: $QK^T$ means transposing $K$ from `[batch, seq, d_k]` to `[batch, d_k, seq]`
before multiplying with $Q$ `[batch, seq, d_k]`.

**Memory note:** Transposing without strides means creating a new flat array with elements
in a different order. This is a full data copy — O(n). That's acceptable for a learning library.

### Reshape

Reshape changes the shape metadata without changing the data:

```
[1, 2, 3, 4, 5, 6]  →  shape [2, 3]  or  shape [3, 2]  or  shape [6]
```

The only constraint: the total number of elements must stay the same.

$$\prod \text{old shape} = \prod \text{new shape}$$

Reshape is O(1) if you reuse the data array (just copy the reference) or O(n) if you copy.
For a learning library, either is fine. Prefer copying for safety.

### Flatten and Squeeze / Unsqueeze

- `flatten(t)`: reshape to `[size]` — a 1D array of all elements.
- `squeeze(t, axis?)`: remove size-1 dimensions. Shape `[1, 3, 1, 4]` → `[3, 4]`.
- `unsqueeze(t, axis)`: insert a size-1 dimension at `axis`. Shape `[3, 4]` → `[1, 3, 4]` at axis 0.

`unsqueeze` is used constantly in attention to align batch and head dimensions for broadcasting.

### Concatenation and Stacking

`concat(tensors, axis)`: join tensors along an existing axis.
```
concat([[1,2], [3,4]], axis=0)  →  [1,2,3,4]   (shapes [2]+[2]=[4])
concat([[1,2,3], [4,5,6]], axis=0) given each is [1,3] → [2, 3]
```
All tensors must have the same shape in every axis except the concat axis.

`stack(tensors, axis)`: create a **new** axis and stack tensors along it.
```
stack([[1,2,3], [4,5,6]], axis=0)  →  shape [2, 3]   (2 tensors of shape [3])
stack([[1,2,3], [4,5,6]], axis=1)  →  shape [3, 2]   (interleaved)
```

In multi-head attention (Ch 23): each head produces output of shape `[batch, seq, d_k]`.
You `concat` all heads along `axis=-1` (the last axis) to get `[batch, seq, d_model]`.

---

## What to Implement

| Function | Signature | Description |
|----------|-----------|-------------|
| `matMul` | `(a, b) => Tensor` | 2D matrix multiply. Shape: (M,K)×(K,N)→(M,N) |
| `matMulBatch` | `(a, b) => Tensor` | Batched matmul over first N-2 dims |
| `transpose` | `(t, axes?) => Tensor` | Permute axes. Default: reverse all axes |
| `reshape` | `(t, shape) => Tensor` | Change shape, same data. Validates size |
| `flatten` | `(t) => Tensor` | reshape to [t.size] |
| `squeeze` | `(t, axis?) => Tensor` | Remove size-1 axes |
| `unsqueeze` | `(t, axis) => Tensor` | Insert size-1 axis at position |
| `concat` | `(tensors, axis) => Tensor` | Join along existing axis |
| `stack` | `(tensors, axis) => Tensor` | Stack along new axis |

---

## TypeScript Hints

```typescript
// matMul: validate that the inner dims match, then triple-loop
function matMul(a: Tensor, b: Tensor): Tensor {
  // a: [M, K], b: [K, N]
  if (a.ndim !== 2 || b.ndim !== 2) throw new Error("matMul requires 2D tensors");
  const [M, K] = a.shape as [number, number];
  const [K2, N] = b.shape as [number, number];
  if (K !== K2) throw new Error(`Inner dims mismatch: ${K} vs ${K2}`);

  const data = new Array<number>(M * N).fill(0);
  for (let i = 0; i < M; i++) {
    for (let j = 0; j < N; j++) {
      let sum = 0;
      for (let k = 0; k < K; k++) {
        // a[i][k] = a.data[i * K + k], b[k][j] = b.data[k * N + j]
        sum += a.data[i * K + k]! * b.data[k * N + j]!;
      }
      data[i * N + j] = sum;
    }
  }
  return createTensor(data, [M, N]);
}
```

---

## Self-Check Questions

1. What shape does `matMul(zeros([3, 7]), zeros([7, 5]))` produce?
2. Why does `matMul(zeros([3, 4]), zeros([3, 4]))` throw an error?
3. In the attention formula, $K$ has shape `[batch, seq, d_k]`. What must its shape be
   after transpose before you can matMul it with $Q$ of shape `[batch, seq, d_k]`?
4. `unsqueeze(t, 0)` on a `[3, 4]` tensor gives `[1, 3, 4]`. What does `unsqueeze(t, 1)` give?
5. You have 12 attention heads, each producing `[batch, seq, 64]`. You want to
   concat them into `[batch, seq, 768]`. Which axis do you concat along?

---

## → Next Chapter

**Ch 05: Reduction & Stat Ops** — `sum`, `mean`, `max`, `min`, `argmax`, `variance`, `std`
along arbitrary axes. These power softmax, LayerNorm, and loss computation.
