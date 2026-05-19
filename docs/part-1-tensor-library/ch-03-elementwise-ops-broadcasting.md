# Chapter 03: Element-wise Ops & Broadcasting

> **Part 1 of 6 — Tensor Library (NumPy-like Foundation)**
> `src/ch-03-elementwise-ops/`

---

## What You're Building

Element-wise arithmetic (`add`, `sub`, `mul`, `div`) between tensors, plus NumPy-style
**broadcasting** — the rules that let you add a vector to every row of a matrix without
manually looping. These operations underlie residual connections, attention masking, and
normalization throughout the transformer.

---

## Why This Matters

In the transformer's encoder block (Ch 26), after the attention output you add it back to
the input: `output = attention_out + residual`. Both are `[batch, seq, d_model]` — same shape,
simple add. But in LayerNorm (Ch 20), you subtract a mean of shape `[batch, seq, 1]` from
a tensor of shape `[batch, seq, d_model]` — that's broadcasting. Without it, you'd need to
manually repeat the mean across 512 columns every single time.

---

## Concepts

### Element-wise Operations (Same Shape)

When two tensors have the **same shape**, element-wise ops pair each element by position:

$$C[i, j] = A[i, j] \odot B[i, j]$$

where $\odot$ is any scalar operation (+, -, ×, ÷).

This is trivially a flat array iteration:
```
for i in range(size):
    C.data[i] = op(A.data[i], B.data[i])
```

### Broadcasting

Broadcasting handles tensors of **different but compatible shapes**. It follows NumPy's rules
exactly. The same rules are used by PyTorch, TensorFlow, and JAX.

#### The Rules (applied left to right)

**Step 1:** If tensors have different `ndim`, prepend `1`s to the smaller shape until both have
the same `ndim`.

```
A shape: [3, 4]    →    [1, 3, 4]
B shape:    [4]    →    [1, 1, 4]
```

**Step 2:** Along each axis, sizes must be either equal, or one of them must be `1`.

```
A: [1, 3, 4]
B: [1, 1, 4]
   ─────────
   ✓  ✓  ✓   all compatible
```

**Step 3:** The output shape is the **maximum** along each axis.

```
output: [max(1,1), max(3,1), max(4,4)] = [1, 3, 4]
```

A size-1 dimension is "stretched" (virtually repeated) to match the other tensor's size.

#### Examples

| A shape | B shape | Output shape | Why |
|---------|---------|--------------|-----|
| `[3, 4]` | `[3, 4]` | `[3, 4]` | Same shape, trivial |
| `[3, 4]` | `[4]` | `[3, 4]` | B is broadcast across rows |
| `[3, 1]` | `[1, 4]` | `[3, 4]` | Both broadcast on different axes |
| `[8, 1, 64, 64]` | `[1, 12, 64, 64]` | `[8, 12, 64, 64]` | Attention mask in transformer |
| `[3, 4]` | `[3]` | ✗ Error | Last dims 4 ≠ 3 and neither is 1 |

#### Implementation Strategy

The cleanest approach: convert both tensors to the output shape by **materializing** the
broadcast — i.e., actually expand size-1 dimensions into full copies.

This is not the most memory-efficient approach (NumPy uses strides to avoid copying), but
it's the simplest to implement correctly and good enough for a learning library.

```
function broadcastTo(t: Tensor, targetShape: number[]): Tensor
```

With this helper, every element-wise op becomes:
```
const [a_, b_] = broadcastShapes(a, b);
const result = a_.data.map((v, i) => op(v, b_.data[i]));
```

### Operator Unification

Rather than writing `add`, `sub`, `mul`, `div` as four separate functions, write one:

```typescript
function elementwise(a: Tensor, b: Tensor, op: (x: number, y: number) => number): Tensor
```

Then define:
```typescript
const add = (a, b) => elementwise(a, b, (x, y) => x + y);
const sub = (a, b) => elementwise(a, b, (x, y) => x - y);
const mul = (a, b) => elementwise(a, b, (x, y) => x * y);
const div = (a, b) => elementwise(a, b, (x, y) => x / y);
```

Also add scalar variants for convenience:
```typescript
const addScalar = (a: Tensor, s: number) => elementwise(a, fill(a.shape, s), (x, y) => x + y);
```

---

## What to Implement

| Function | Description |
|----------|-------------|
| `broadcastShapes(a, b)` | Compute the output shape for a broadcast between shapes `a` and `b` |
| `broadcastTo(t, shape)` | Expand tensor `t` to `shape` by repeating along size-1 axes |
| `elementwise(a, b, op)` | Apply a scalar binary op to two broadcast-compatible tensors |
| `add(a, b)` | Element-wise addition |
| `sub(a, b)` | Element-wise subtraction |
| `mul(a, b)` | Element-wise multiplication (Hadamard product) |
| `div(a, b)` | Element-wise division |
| `addScalar(a, s)` | Add scalar `s` to every element |
| `mulScalar(a, s)` | Multiply every element by scalar `s` |
| `neg(a)` | Negate every element (`mulScalar(a, -1)`) |

---

## TypeScript Hints

```typescript
// broadcastShapes: align from the right, check compatibility, return output shape
function broadcastShapes(shapeA: number[], shapeB: number[]): number[] {
  const ndim = Math.max(shapeA.length, shapeB.length);
  // Pad shorter shape with leading 1s
  const a = [...Array(ndim - shapeA.length).fill(1), ...shapeA];
  const b = [...Array(ndim - shapeB.length).fill(1), ...shapeB];

  return a.map((da, i) => {
    const db = b[i]!;
    if (da === db) return da;
    if (da === 1) return db;
    if (db === 1) return da;
    throw new Error(`Incompatible shapes at axis ${i}: ${da} vs ${db}`);
  });
}
```

---

## Self-Check Questions

1. Can you broadcast `[5, 3]` with `[3]`? What is the output shape?
2. Can you broadcast `[2, 5, 3]` with `[3, 1]`? What is the output shape?
3. In the transformer, after computing `QK^T` of shape `[batch, heads, seq, seq]`,
   you add an attention mask of shape `[1, 1, seq, seq]`. What broadcasting rule applies?
4. What would happen if you tried to broadcast `[3, 4]` with `[2, 4]`? Write the error message.
5. Implement `broadcastTo([1, 3], [4, 3])` by hand for a `[1, 3]` tensor `[[a, b, c]]`.
   What does the output look like?

---

## → Next Chapter

**Ch 04: Matrix Operations** — `matMul`, `transpose`, `reshape`, `flatten`, `concat`, `stack`.
These are the higher-level operations built on top of element-wise ops.
