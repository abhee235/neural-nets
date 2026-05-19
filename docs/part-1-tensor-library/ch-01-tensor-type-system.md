# Chapter 01: Tensor Type System

> **Part 1 of 6 — Tensor Library (NumPy-like Foundation)**
> `src/ch-01-tensor-type-system/`

---

## What You're Building

A TypeScript type system for tensors — the fundamental data structure of every neural network.
You will define the `Tensor` interface, shape utilities, and index arithmetic that every
subsequent chapter will depend on.

---

## Why This Matters

Every number in a neural network — weights, activations, gradients, attention scores — lives
inside a tensor. Before you can add two matrices or multiply weights with inputs, you need a
precise, consistent way to represent N-dimensional arrays in TypeScript.

This is the foundation. If the shape system is wrong here, bugs will be silent and baffling
in Chapter 20 when your attention scores have shape `[8, 64, 64]` instead of `[batch, heads, seq]`.

---

## Concepts

### What is a Tensor?

A tensor is a generalization of scalars, vectors, and matrices to any number of dimensions.

| Name   | Rank | Shape example | Appears in transformer as |
|--------|------|---------------|--------------------------|
| Scalar | 0    | `[]`          | a single loss value      |
| Vector | 1    | `[512]`       | one token's embedding    |
| Matrix | 2    | `[64, 512]`   | a batch of embeddings    |
| 3-D    | 3    | `[8, 64, 512]`| batch × seq × d_model   |
| 4-D    | 4    | `[2, 8, 64, 64]` | layer × heads × seq × seq (attention maps) |

Rank is the number of dimensions. Shape is the size along each dimension.

### Why a Flat Array Internally?

Conceptually, a 2D matrix of shape `[3, 4]` is a 2D grid. But computers store memory
linearly. Both NumPy and PyTorch store tensor data as a **flat 1D array** with a separate
`shape` metadata array. Shape `[3, 4]` means: interpret the flat array in row-major order,
3 rows, 4 columns.

This design gives:
- Cache-friendly memory access (sequential reads)
- Cheap reshape and transpose (just change metadata, not data)
- A single code path for all operations regardless of rank

### The Tensor Interface

```typescript
interface Tensor {
  data: number[];      // flat row-major storage
  shape: number[];     // e.g. [3, 4] for a 3×4 matrix
  ndim: number;        // shape.length — number of axes
  size: number;        // total elements = product of shape
}
```

### Shape Arithmetic

**Size** = product of all dimensions:

$$\text{size} = \prod_{i=0}^{\text{ndim}-1} \text{shape}[i]$$

A tensor of shape `[2, 3, 4]` has $2 \times 3 \times 4 = 24$ elements.

**Flat index from N-D index (row-major / C-order):**

For a tensor of shape `[d0, d1, d2]`, the element at index `[i, j, k]` lives at:

$$\text{flat index} = i \cdot (d_1 \cdot d_2) + j \cdot d_2 + k$$

The multipliers `(d1 * d2)`, `(d2)`, `(1)` are called **strides**. Strides tell you
how many elements to skip in the flat array when you move one step along each axis.

### Strides

$$\text{stride}[i] = \prod_{j=i+1}^{\text{ndim}-1} \text{shape}[j]$$

For shape `[2, 3, 4]`:
- `stride[0]` = 3 × 4 = 12  (moving along axis 0 skips 12 elements)
- `stride[1]` = 4            (moving along axis 1 skips 4 elements)
- `stride[2]` = 1            (innermost axis, no skip)

You do NOT need to store strides explicitly in your Tensor type — you can compute them
from shape on demand. But understanding them explains how transpose and reshape work cheaply.

---

## What to Implement

| Function / Type | Description |
|-----------------|-------------|
| `interface Tensor` | The core type: `data`, `shape`, `ndim`, `size` |
| `createTensor(data, shape)` | Validate and wrap a flat array into a Tensor |
| `strides(shape)` | Compute stride array from shape |
| `flatIndex(indices, shape)` | Convert N-D index to flat index |
| `nDIndex(flatIdx, shape)` | Convert flat index back to N-D index |
| `shapeSize(shape)` | Compute total element count from shape |
| `assertShape(t, expected)` | Throw a clear error if shape doesn't match |
| `scalar(n)` | Convenience: wrap a single number as a rank-0 tensor |
| `print(t)` | Pretty-print tensor shape and values for debugging |

---

## TypeScript Hints

```typescript
// The Tensor type should be a plain object (not a class) for simplicity.
// A class adds nice method syntax but makes JSON serialization harder.
// We'll keep it as an interface + standalone functions, like functional NumPy.

interface Tensor {
  data: number[];
  shape: number[];
  ndim: number;  // always === shape.length
  size: number;  // always === shapeSize(shape)
}

// Validate that data.length matches the product of shape during creation.
// A mismatch here causes confusing silent bugs downstream.
function createTensor(data: number[], shape: number[]): Tensor {
  const size = shapeSize(shape);
  if (data.length !== size) {
    throw new Error(`data length ${data.length} doesn't match shape ${shape} (size ${size})`);
  }
  return { data: [...data], shape: [...shape], ndim: shape.length, size };
}
```

---

## Self-Check Questions

1. What is the flat index for element `[1, 2]` in a tensor of shape `[3, 4]`?
2. What are the strides of a tensor with shape `[5, 6, 7]`?
3. Why is `ndim` always `shape.length`? Why store it separately?
4. What happens to memory when you reshape a `[4, 3]` tensor to `[12]`? What changes?
5. A transformer attention tensor has shape `[batchSize, numHeads, seqLen, seqLen]`.
   What is the stride of the `numHeads` axis?

---

## → Next Chapter

**Ch 02: Tensor Creation** — now that the type is defined, build the standard factory
functions: `zeros`, `ones`, `eye`, `rand`, `randn`, `arange`, `linspace`.
