# Chapter 23: Multi-Head Attention

> **Part 5 of 6 — Attention Mechanism**
> `src/ch-23-multi-head-attention/`

---

## What You're Building

Multi-head attention: run $H$ attention heads in parallel, each attending to different
aspects of the input, then concatenate and project their outputs. This is the component
at the heart of every transformer encoder and decoder block.

---

## Why This Matters

Single-head attention (Ch 22) learns one type of relationship: "which tokens are relevant
to me?" But language has many simultaneous relationships — syntactic dependencies, semantic
similarity, coreference, topic coherence, etc. A single head can only capture one at a time.

Multi-head attention lets each head specialize. In trained models, you can observe:
- Head 1 attends to recent tokens (local syntax)
- Head 2 attends to matching brackets / clauses
- Head 3 tracks subject-verb agreement
- Head 4 attends to semantically similar content
...and so on. The model discovers these specializations through gradient descent.

---

## Concepts

### The Multi-Head Idea

Instead of one $d_{\text{model}}$-dimensional attention, split into $H$ heads of $d_k = d_{\text{model}} / H$:

Each head $h \in \{1, \ldots, H\}$ has its own projection matrices:
$$Q_h = X W_Q^{(h)}, \quad K_h = X W_K^{(h)}, \quad V_h = X W_V^{(h)}$$

Where $W_Q^{(h)}, W_K^{(h)}, W_V^{(h)} \in \mathbb{R}^{d_{\text{model}} \times d_k}$.

Each head computes attention independently:
$$\text{head}_h = \text{Attention}(Q_h, K_h, V_h)$$

Then concatenate and project:
$$\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) W_O$$

Where $W_O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$ is the output projection.

### Efficient Implementation: Batch All Heads Together

Naively, you'd compute H separate attention calls in a loop. But there's a faster way:
reshape the input so all heads are processed simultaneously as a batch.

Given $X$: `[batch, seq, dModel]`:

1. **Project** to $Q, K, V$: `[batch, seq, dModel]` → `[batch, seq, dModel]`
   (using combined `W_Q` of shape `[dModel, dModel]` — all heads at once)

2. **Reshape** to separate heads: `[batch, seq, dModel]` → `[batch, seq, H, dHead]`

3. **Transpose** for matmul: `[batch, seq, H, dHead]` → `[batch, H, seq, dHead]`
   (so batch and heads are the leading dims for batched matmul)

4. **Apply attention** to all heads: `[batch, H, seq, dHead]` × `[batch, H, dHead, seq]`
   → `[batch, H, seq, dHead]`

5. **Transpose + reshape** back: `[batch, H, seq, dHead]` → `[batch, seq, dModel]`

6. **Output projection**: `[batch, seq, dModel]` × `[dModel, dModel]` → `[batch, seq, dModel]`

Steps 2–5 are the "split → compute → merge" pattern. The transpose and reshape are O(n)
memory operations. The attention computation is the expensive part.

### Dimensions

For a typical transformer with `dModel=512, H=8`:
- `dHead = dModel / H = 64`
- `W_Q, W_K, W_V`: each `[512, 512]` (all 8 heads packed together)
- OR `W_Q, W_K, W_V`: each `[512, 64]` per head (H separate matrices)

Both are equivalent. The packed form (`[512, 512]`) is more efficient — one large matmul
instead of 8 small ones.

### Why H × dHead = dModel?

Total "capacity" is preserved: with 8 heads of 64 dimensions each, you have 8 × 64 = 512
dimensions total — the same as single-head attention with 512 dimensions. The benefit isn't
extra capacity; it's **diverse attention patterns** from the H independent subspaces.

---

## What to Implement

| Symbol | Description |
|--------|-------------|
| `class MultiHeadAttention` | MHA module: `(dModel, numHeads)`. Contains 4 linear layers: `W_Q, W_K, W_V, W_O`. |
| `MultiHeadAttention.forward(x, mask?)` | Full forward pass: project → split → attend → merge → project. |
| `MultiHeadAttention.parameters()` | All 4 weight matrices. |
| `splitHeads(x, numHeads)` | Reshape `[batch, seq, dModel]` → `[batch, numHeads, seq, dHead]` |
| `mergeHeads(x)` | Reverse of `splitHeads`: `[batch, numHeads, seq, dHead]` → `[batch, seq, dModel]` |

---

## TypeScript Hints

```typescript
export class MultiHeadAttention {
  dModel: number;
  numHeads: number;
  dHead: number;

  WQ: Linear;   // [dModel, dModel] — projects to all Q heads at once
  WK: Linear;   // [dModel, dModel] — projects to all K heads at once
  WV: Linear;   // [dModel, dModel] — projects to all V heads at once
  WO: Linear;   // [dModel, dModel] — output projection

  constructor(dModel: number, numHeads: number) {
    if (dModel % numHeads !== 0) {
      throw new Error(`dModel (${dModel}) must be divisible by numHeads (${numHeads})`);
    }
    this.dModel = dModel;
    this.numHeads = numHeads;
    this.dHead = dModel / numHeads;

    this.WQ = new Linear(dModel, dModel);
    this.WK = new Linear(dModel, dModel);
    this.WV = new Linear(dModel, dModel);
    this.WO = new Linear(dModel, dModel);
  }

  forward(x: Value, mask?: Tensor): Value {
    const [batchSize, seqLen] = x.data.shape as [number, number, number];

    // 1. Project to Q, K, V — shape: [batch, seq, dModel]
    const Q = this.WQ.forward(x);
    const K = this.WK.forward(x);
    const V = this.WV.forward(x);

    // 2. Split into heads — shape: [batch, numHeads, seq, dHead]
    const QH = splitHeads(Q, this.numHeads);
    const KH = splitHeads(K, this.numHeads);
    const VH = splitHeads(V, this.numHeads);

    // 3. Scaled dot-product attention (batched over both batch and head dims)
    const attnOut = scaledDotProductAttention(QH, KH, VH, mask);

    // 4. Merge heads back — shape: [batch, seq, dModel]
    const merged = mergeHeads(attnOut);

    // 5. Output projection
    return this.WO.forward(merged);
  }

  parameters(): Value[] {
    return [
      ...this.WQ.parameters(),
      ...this.WK.parameters(),
      ...this.WV.parameters(),
      ...this.WO.parameters(),
    ];
  }
}
```

---

## Self-Check Questions

1. `MultiHeadAttention(dModel=512, numHeads=8)`: what is `dHead`? How many total parameters
   (count W_Q, W_K, W_V, W_O including biases)?
2. After `splitHeads`, the shape is `[batch, H, seq, dHead]`. Why must the head dimension
   come before the sequence dimension for the batched matmul to work correctly?
3. If one head learns to attend only to punctuation and another only to nouns,
   describe what their W_Q and W_K matrices might look like conceptually.
4. The output projection W_O takes `[batch, seq, dModel]` and outputs `[batch, seq, dModel]`.
   Why is this projection useful rather than just returning the concatenated heads directly?
5. What is the total number of floating point operations (multiply-adds) in one
   forward pass of multi-head attention for `batch=1, seq=512, dModel=512, H=8`?

---

## → Next Chapter

**Ch 24: Cross-Attention** — the decoder variant where Query comes from the decoder
and Key/Value come from the encoder output.
