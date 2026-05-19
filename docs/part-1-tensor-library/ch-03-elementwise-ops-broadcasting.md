# Chapter 03: Elementwise Ops & Broadcasting

> **Part 1 of 6 — Tensor Library**
> Source: [`src/tensor/ops.ts`](../../src/tensor/ops.ts)
> Tests: [`src/tensor/ops.test.ts`](../../src/tensor/ops.test.ts)
> Exercise: [`exercises/ch-03-elementwise-ops.ts`](../../exercises/ch-03-elementwise-ops.ts)

---

## Learning Goals

By the end of this chapter you can:

- State and apply the two broadcasting rules from memory.
- Compute the broadcast result shape of any two shapes (or detect that they are incompatible).
- Implement `add`, `sub`, `mul`, `div`, `neg` for any two broadcast-compatible tensors.
- Recognise where broadcasting will appear in a transformer (bias, mask, scaling).
- Avoid silent shape errors by validating shapes at every entry point.

---

## Intuition First

Broadcasting is a way of pretending a small tensor is a big one without copying any memory. If you add a `[d]` bias to a `[batch, seq, d]` activation, broadcasting stretches the bias along the first two axes — conceptually — so the shapes match.

The two rules:

1. Line up the shapes from the **right**. Missing axes on the left are treated as size `1`.
2. At each axis, the two sizes must be **equal** or **one of them must be `1`**. The `1` stretches to match the other size.

---

## Mental Model

```text
  A.shape:        [8, 1, 64]
  B.shape:           [12, 64]
  ----------------------------- align right, pad with 1s
  A.shape:        [8,  1, 64]
  B.shape:        [1, 12, 64]
  result shape:   [8, 12, 64]   ← each axis = max
```

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

## Common Pitfalls

- Aligning shapes from the *left* — broadcasting always aligns from the **right**.
- Allowing two axes of size 2 and 3 to broadcast — they must be equal or one must be 1.
- Forgetting that broadcasting may produce a result *larger than either input*.
- Materialising the broadcast result for a tiny op when an in-place loop would do.
- Letting NaN/Infinity slip through `div` when the denominator broadcasts to zero.

---

## How to Verify

Run the tests and the exercise. Both should pass cleanly with no warnings:

```bash
bun test src/tensor/ops.test.ts
```
```bash
bun run exercises/ch-03-elementwise-ops.ts
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

## Further Reading

- [NumPy broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) — the canonical rules our tensor ops follow.
- [PyTorch — Broadcasting semantics](https://pytorch.org/docs/stable/notes/broadcasting.html) — same rules, slightly different wording.
- [Jake VanderPlas — Python Data Science Handbook (broadcasting chapter)](https://jakevdp.github.io/PythonDataScienceHandbook/02.05-computation-on-arrays-broadcasting.html) — the clearest visual treatment of the rules.
- [3Blue1Brown — Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) — geometric intuition for vectors, matrices, and linear maps.

---

## Next Chapter

**[Matrix Ops](ch-04-matrix-ops.md)** — once add/mul broadcast correctly, we need matmul and transpose for linear layers.
