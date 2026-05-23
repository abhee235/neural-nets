# Chapter 04: Matrix Operations

> **Part 1 of 6 — Tensor Library (NumPy-like Foundation)**
> Code: [`src/tensor/linalg.ts`](../../src/tensor/linalg.ts)
> Tests: [`src/tensor/linalg.test.ts`](../../src/tensor/linalg.test.ts)

---

## Learning Goals

By the end of this chapter you will be able to:

- **Explain** why matrix multiplication contracts the inner dimension and what that means for shapes.
- **Implement** `matMul` for 2-D matrices using a triple loop, reading the result of each `C[i,j]` as a dot product.
- **Implement** batched `matMul` that applies 2-D matrix multiply across a leading batch dimension.
- **Implement** `transpose`, `reshape`, `flatten`, `squeeze`, and `unsqueeze` as shape-manipulation helpers.
- **Implement** `concat` and `stack` to join tensors along existing or new axes.
- **Describe** where each of these operations appears in the attention formula and the linear layer.

---

## Intuition First — Why does shape matter so much?

Imagine you have a spreadsheet with 3 rows and 4 columns. You want to combine it with another spreadsheet using matrix multiplication. The rule is simple: **the number of columns in the first sheet must equal the number of rows in the second sheet.** That shared dimension gets "consumed" — it disappears from the output. What you get back is a new spreadsheet whose size is rows-of-first × columns-of-second.

```
Sheet A: 3 rows × 4 columns    (shape [3, 4])
Sheet B: 4 rows × 5 columns    (shape [4, 5])
                  ↑ ↑
           these must match

Result:   3 rows × 5 columns   (shape [3, 5])
```

Every number in the result sheet is a single dot product: you multiply an entire row from A against an entire column from B, and sum all the products. That is the whole algorithm.

The other operations in this chapter — `transpose`, `reshape`, `squeeze`, `concat` — are pure **bookkeeping**. They rearrange or relabel the data without doing any arithmetic. They exist because the shapes coming out of one layer rarely align perfectly with the shapes the next layer expects.

---

## The Mental Model

### matMul: row × column → one number

```
A [3 × 4]          B [4 × 5]              C [3 × 5]
┌─────────────┐    ┌──────────────────┐   ┌──────────────────┐
│ → row i     │    │ ↓  ↓  ↓  ↓  ↓   │   │                  │
│ [a b c d]   │  × │ col              │ = │  C[i,j]          │
│             │    │  j               │   │  = a*B[0,j]      │
│             │    │                  │   │  + b*B[1,j]      │
└─────────────┘    └──────────────────┘   │  + c*B[2,j]      │
                                          │  + d*B[3,j]      │
                                          └──────────────────┘
```

Each output cell `C[i,j]` is the **dot product** of row `i` of A and column `j` of B.

### reshape / transpose: same data, new label

```
data = [1, 2, 3, 4, 5, 6]        (6 numbers, never move)

reshape to [2, 3]:    reshape to [3, 2]:    transpose [2,3] → [3,2]:
┌─────────┐           ┌───────┐             ┌───────┐
│ 1  2  3 │           │ 1  2  │             │ 1  4  │
│ 4  5  6 │           │ 3  4  │             │ 2  5  │
└─────────┘           │ 5  6  │             │ 3  6  │
                      └───────┘             └───────┘
```

`reshape` keeps rows in reading order. `transpose` swaps which axis is "rows" and which is "columns" — the data is copied into a new order.

---

## Concepts

### 1. Matrix Multiplication — `matMul`

`matMul` is the core operation of every neural network layer. It is where numbers actually mix together — in `transpose` and `reshape` they only move around.

> **Use it when** you need to linearly combine every input feature with every output feature.
> **Picture this** You have 3 questions (rows of A) and you want to score them against 5 answer keys (columns of B). Each score is a dot product. The result is a 3×5 score table.

$$\Large \boxed{C[i,j] = \sum_{k=0}^{K-1} A[i,k] \cdot B[k,j]}$$

In plain English: each cell of the output is the sum of products of one row of A with one column of B. $M$ is the number of rows in A, $K$ is the shared inner dimension, $N$ is the number of columns in B.

**Shape rule:**

$$A \in \mathbb{R}^{M \times K},\quad B \in \mathbb{R}^{K \times N} \implies C \in \mathbb{R}^{M \times N}$$

The $K$ dimension is consumed. $M$ and $N$ survive into the output.

**Example:**

```typescript
const A = createTensor([1, 2, 3, 4], [2, 2]); // [[1,2],[3,4]]
const B = createTensor([1, 0, 0, 1], [2, 2]); // identity matrix
matMul(A, B);
// → [[1,2],[3,4]]  (multiplying by identity leaves A unchanged)

const C = createTensor([1, 2, 3, 4, 5, 6], [2, 3]); // [2,3]
const D = createTensor([1, 2, 1, 2, 1, 2], [3, 2]); // [3,2]
matMul(C, D);
// → [2,2] result: C[0,0] = 1*1+2*1+3*1 = 6, C[0,1] = 1*2+2*2+3*2 = 12
```

**Implementation note:** three nested loops — `i` over rows of A, `j` over columns of B, `k` over the shared inner dim. Each inner iteration does one multiply-accumulate into `out[i * N + j]`. The row-major flat offset for A is `i * K + k`; for B it is `k * N + j`.

> **Why this matters for transformers:** In Ch 22 (self-attention), the query–key score matrix is `matMul(Q, transposeK)`. In Ch 13 (linear layer), the forward pass is `matMul(input, weights)`. This single function is called hundreds of times per forward pass.

### 2. Batched Matrix Multiplication — `matMulBatch`

In the transformer, Q, K, V are not 2-D matrices — they are 3-D tensors with a batch dimension: `[batch, seq, dHead]`. You need to apply `matMul` to every "slice" along the batch axis independently.

> **Use it when** you have a stack of matrices (batch of sentences, or multiple attention heads) and need to multiply each matrix pair independently.
> **Picture this** You have 8 spreadsheet pairs stacked on top of each other. `matMulBatch` multiplies each pair and gives you 8 result sheets stacked together.

$$\Large \boxed{\text{matMulBatch}(A, B)[b] = \text{matMul}(A[b],\; B[b])}$$

**Shape rule:** `[B, M, K] × [B, K, N] → [B, M, N]`. The leading batch axes must match; the 2-D multiply happens on the last two axes.

**Implementation note:** slice out each `b`-th 2-D matrix from A and B using `b * M * K` and `b * K * N` as base offsets into the flat data. Call the inner 2-D multiply logic for each slice, then concatenate results.

> **Why this matters for transformers:** Multi-head attention (Ch 23) processes all heads simultaneously as a batched matmul. Shape `[batch, heads, seq, dHead]` — the batch *and* heads dimensions are both leading batch dims.

### 3. Transpose — `transpose`

Transpose swaps the axes of a tensor. For a 2-D matrix, it turns rows into columns and vice versa.

> **Use it when** you need to flip which axis is "rows" and which is "columns" before a matrix multiply.
> **Picture this** A spreadsheet with 4 rows and 3 columns becomes a spreadsheet with 3 rows and 4 columns. The data is the same; the reading direction changes.

$$\Large \boxed{A^T[i,j] = A[j,i]}$$

For N-D tensors, an optional `axes` argument specifies a permutation. `transpose(t, [0, 2, 1])` on a `[B, M, N]` tensor produces `[B, N, M]` — it keeps axis 0 (batch) fixed and swaps axes 1 and 2.

**Example:**

```typescript
const A = createTensor([1, 2, 3, 4, 5, 6], [2, 3]);
// A = [[1,2,3],[4,5,6]]
transpose(A);
// → shape [3,2]: [[1,4],[2,5],[3,6]]
```

**Implementation note:** the new shape is the old shape with axes reordered per `axes` (default: reversed). For each output position, compute its multi-axis index in the new shape, permute those indices back to the original axis order, then read from the input using `flatIndex`.

> **Why this matters for transformers:** `QK^T` in attention is `matMul(Q, transpose(K))`. K has shape `[batch, seq, dHead]`; after `transpose(K, [0, 2, 1])` it becomes `[batch, dHead, seq]`, which is exactly what `matMulBatch` needs.

### 4. Reshape and Flatten

Reshape changes the shape metadata without changing which number is which. The only constraint: total element count must stay equal.

> **Use it when** a layer produces a flat output but the next layer expects a grid, or vice versa.
> **Picture this** You have 12 tiles arranged in a 2×6 grid. Reshape lets you re-frame them as 3×4 or 4×3 or just a row of 12 — the tiles themselves never move.

$$\Large \boxed{\prod \text{oldShape} = \prod \text{newShape}}$$

A `-1` in `newShape` means "infer this dimension from the others."

**Example:**

```typescript
const t = createTensor([1,2,3,4,5,6], [2,3]);
reshape(t, [3,2]);   // → [[1,2],[3,4],[5,6]]
reshape(t, [6]);     // → [1,2,3,4,5,6]  (flatten)
reshape(t, [-1,2]);  // → [3,2]  (-1 inferred as 6÷2=3)
flatten(t);          // → shape [6]  (shorthand for reshape to 1-D)
```

**Implementation note:** validate `product(newShape_without_-1) * (-1 slot)` equals `t.size`. Replace `-1` with the inferred value. Copy `t.data` into a new tensor with the new shape (shallow clone of the data array is fine).

### 5. Squeeze and Unsqueeze

These two helpers add or remove size-1 dimensions. They exist to satisfy the shape contracts between layers without changing any values.

> **Use it when** you need to insert a broadcast-friendly dimension, or remove a leftover size-1 axis after a reduction.
> **Picture this** `unsqueeze` is like putting a tray into a box — the tray doesn't change, but now it has a "slot" label. `squeeze` takes it back out.

$$\Large \boxed{\text{unsqueeze}(t, \text{axis}): [\ldots] \to [\ldots, 1, \ldots]}$$

```typescript
const t = createTensor([1,2,3,4], [2,2]);  // shape [2,2]
unsqueeze(t, 0);  // → shape [1,2,2]
unsqueeze(t, 1);  // → shape [2,1,2]
squeeze(unsqueeze(t, 0), 0);  // → shape [2,2]  (round-trip)
```

**Implementation note:** `unsqueeze` inserts a `1` at position `axis` in the shape array. `squeeze` removes the `1` at position `axis`. Both functions copy `t.data` unchanged — only the shape metadata changes.

> **Why this matters for transformers:** Attention masks have shape `[batch, 1, 1, seq]`. The two size-1 dimensions are there to broadcast across heads and query positions. They are inserted with `unsqueeze`.

### 6. Concat and Stack

`concat` joins tensors along an **existing** axis. `stack` creates a **new** axis and stacks tensors along it.

> **Use it when** you have multiple tensors that belong together — `concat` if they share an axis, `stack` if you want to treat them as a new collection.
> **Picture this** `concat` is stapling pages together lengthwise (pages stay as pages). `stack` is binding separate notebooks into a shelf — each book becomes one slot in a new dimension.

**Concat:**

$$\Large \boxed{\text{concat}([T_1, T_2, \ldots], \text{axis}): \text{size along axis} = \sum_i T_i.\text{shape[axis]}}$$

All other axes must match exactly.

**Stack:**

$$\Large \boxed{\text{stack}([T_1, T_2, \ldots], \text{axis}): \text{new axis of size} = \text{number of tensors}}$$

All tensors must have identical shape.

```typescript
const a = createTensor([1,2,3], [3]);
const b = createTensor([4,5,6], [3]);
concat([a, b], 0);          // → shape [6]: [1,2,3,4,5,6]
stack([a, b], 0);           // → shape [2,3]: [[1,2,3],[4,5,6]]
stack([a, b], 1);           // → shape [3,2]: [[1,4],[2,5],[3,6]]
```

**Implementation note for concat:** compute the output shape (sum of sizes along the concat axis), allocate the output buffer, then copy each input tensor's data into the right slice. The tricky part is computing the correct flat offset for each input — treat all axes except the concat axis as stride multipliers.

**Implementation note for stack:** `unsqueeze` each tensor at `axis` to insert the new dimension, then `concat` them. This reduces `stack` to one `unsqueeze` call per tensor plus one `concat` call.

> **Why this matters for transformers:** Multi-head attention (Ch 23) splits Q/K/V into heads with `reshape`, processes each head, then `concat`s the head outputs along the last axis. `stack` is used when building batches from individual sequences.

### Where you will see these operations again

| Operation | Where it shows up next | Why that chapter needs it |
|---|---|---|
| `matMul` | **Ch 13** (Linear layer) | forward pass: `output = matMul(input, weights)` |
| `matMulBatch` | **Ch 22** (Self-attention) | `QK^T` score matrix across a batch |
| `transpose` | **Ch 22** (Self-attention) | flip K from `[B, seq, d]` to `[B, d, seq]` |
| `reshape` | **Ch 23** (Multi-head attention) | split/merge head dimension |
| `unsqueeze` | **Ch 21** (Masks) | add broadcast dims to mask tensor |
| `concat` | **Ch 23** (Multi-head attention) | merge all head outputs |
| `stack` | **Ch 28** (Sequence data) | build a batch from a list of sequences |

These operations are the plumbing between every layer. You will use them so often they will feel automatic.

---

## What to Implement

| Function | Signature | Description |
|---|---|---|
| `matMul` | `(a, b) => Tensor` | 2-D matrix multiply. Shape: (M,K)×(K,N)→(M,N) |
| `matMulBatch` | `(a, b) => Tensor` | Batched matmul over all leading dims |
| `transpose` | `(t, axes?) => Tensor` | Permute axes. Default: reverse all |
| `reshape` | `(t, shape) => Tensor` | Change shape; validates total size; supports `-1` |
| `flatten` | `(t, startAxis?) => Tensor` | Collapse axes from `startAxis` onward to one dim |
| `squeeze` | `(t, axis) => Tensor` | Remove the size-1 dim at `axis` |
| `unsqueeze` | `(t, axis) => Tensor` | Insert a size-1 dim at `axis` |
| `concat` | `(tensors, axis) => Tensor` | Join along an existing axis |
| `stack` | `(tensors, axis) => Tensor` | Stack along a new axis |

---

## Common Pitfalls

**1. Inner dimension mismatch in matMul**

`matMul(A, B)` requires `A.shape[1] === B.shape[0]`. The most common mistake: multiplying `[3, 4]` by `[3, 4]` because both tensors are "the same shape". They are not — the inner dims are 4 and 3.

```
A [3, 4] × B [3, 4]   ← ERROR: inner dims are 4 ≠ 3
A [3, 4] × B [4, 5]   ← OK: inner dim 4 = 4
```

**2. Forgetting to transpose K before matMul**

In attention, you want $Q \cdot K^T$. K has shape `[batch, seq, dHead]`. If you pass K directly to `matMulBatch(Q, K)` you will get an inner-dim mismatch because K's last axis is `dHead` (same as Q's last axis). You must first `transpose(K, [0, 2, 1])` to get `[batch, dHead, seq]`.

**3. reshape vs transpose confusion**

`reshape([2,3] → [3,2])` re-reads elements in row-major order — the data order changes meaning. `transpose([2,3] → [3,2])` swaps axes so `A^T[i,j] = A[j,i]`. These produce different numbers.

```typescript
const A = createTensor([1,2,3,4,5,6], [2,3]);
reshape(A, [3,2]);   // [[1,2],[3,4],[5,6]] — just re-reads sequentially
transpose(A);        // [[1,4],[2,5],[3,6]] — swaps row/col
```

**4. squeeze on the wrong axis**

`squeeze(t, axis)` throws (or silently does nothing) if `t.shape[axis] !== 1`. Always check the actual shape before squeezing.

**5. Using -1 in reshape more than once**

Only one axis can be `-1` (inferred). Two `-1`s is an error because the system cannot determine both unknowns from a single product equation.

---

## How to Verify

Run the test suite:

```bash
bun test src/tensor/linalg.test.ts
```

Spot-check `matMul` manually:

```bash
bun -e '
import { createTensor } from "./src/tensor/types.ts";
import { matMul } from "./src/tensor/linalg.ts";
const A = createTensor([1,2,3,4,5,6], [2,3]);
const B = createTensor([7,8,9,10,11,12], [3,2]);
console.log(matMul(A, B).shape);  // [2, 2]
console.log(Array.from(matMul(A, B).data));  // [58, 64, 139, 154]
'
```

Check the attention-style pipeline:

```bash
bun -e '
import { createTensor } from "./src/tensor/types.ts";
import { matMul, transpose } from "./src/tensor/linalg.ts";
// Simulate Q [2,3] @ K^T [3,2] → scores [2,2]
const Q = createTensor([1,0,1, 0,1,0], [2,3]);
const K = createTensor([1,0,1, 0,1,0], [2,3]);
const scores = matMul(Q, transpose(K));
console.log(scores.shape);   // [2,2]
console.log(Array.from(scores.data));  // [2,0,0,1]
'
```

Three-step sanity check for all other ops:
1. `reshape(t, [-1])` should produce a flat tensor of the same size.
2. `concat([t, t], 0)` should double the first dimension.
3. `unsqueeze` then `squeeze` at the same axis should round-trip to the original shape.

---

## Self-Check Questions

1. What shape does `matMul(zeros([3, 7]), zeros([7, 5]))` produce? Why?
2. Why does `matMul(zeros([3, 4]), zeros([3, 4]))` throw an error?
3. K has shape `[batch, seq, dHead]`. What must its shape be after `transpose` so that `matMulBatch(Q, K_T)` with Q `[batch, seq, dHead]` produces `[batch, seq, seq]`?
4. `unsqueeze(t, 0)` on a `[3, 4]` tensor gives `[1, 3, 4]`. What does `unsqueeze(t, 1)` give?
5. You have 12 attention heads, each producing `[batch, seq, 64]`. You want to `concat` them into `[batch, seq, 768]`. Which axis do you concat along?
6. `reshape(t, [2, -1])` on a `[3, 4]` tensor — what is the inferred dimension?

---

## Coding Exercises

1. **Verify the identity property:** create a 3×3 identity matrix and confirm that `matMul(A, identity)` equals A for any 3×3 matrix A.
2. **Reconstruct transpose from reshape:** Can you get from `[[1,2,3],[4,5,6]]` (shape `[2,3]`) to `[[1,4],[2,5],[3,6]]` (shape `[3,2]`) using only `reshape`? Try it — you should find you cannot.
3. **Build `matMulBatch` from `matMul`:** slice the batch dimension manually in a loop, call `matMul` on each pair, then `stack` the results. Verify it matches the built-in.
4. **Attention-score sanity test:** with Q = K = ones([2, 4]), compute `matMul(Q, transpose(K))`. What is every element of the result, and why?

---

## Further Reading

- [Deep Dive — Why matMul is O(n³) and why GPUs matter](../deep-dives/ch-04-matmul-complexity.md) *(optional)*
- NumPy broadcasting + matmul docs — the canonical reference for axis semantics.
- [Ch 22 — Self-attention](../part-5-attention/ch-22-self-attention.md) is where all of these ops work together for the first time.

---

## Next Chapter

**Ch 05: Reduction & Stat Ops** — `sum`, `mean`, `max`, `min`, `argmax`, `variance`, `std` along arbitrary axes. These power softmax, LayerNorm, and loss computation.

