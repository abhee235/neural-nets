# Chapter 23: Multi-Head Attention

> **Part 5 of 6 — Attention Mechanism**
> Source: [`src/nn/attention.ts`](../../src/nn/attention.ts)
> Tests: [`src/nn/attention.test.ts`](../../src/nn/attention.test.ts)
> Exercise: [`exercises/ch-23-multi-head-attention.ts`](../../exercises/ch-23-multi-head-attention.ts)

---

## Learning Goals

By the end of this chapter you can:

- Split `d_model` into `H` heads of size `d_head = d_model / H`.
- Implement projection → reshape → permute to get `[batch, H, seq, d_head]`.
- Run scaled dot-product attention per head, in parallel.
- Concatenate heads back to `[batch, seq, d_model]` and project with `W_O`.
- Predict the parameter count: `4 · d_model²` (for `W_Q`, `W_K`, `W_V`, `W_O`).

---

## Intuition First

One attention head captures one kind of relationship — say, "which token is the subject?". Multi-head attention runs several smaller heads in parallel so the model can attend to different relationships at the same time (syntax, coreference, semantic similarity). After each head has done its work, we concatenate their outputs and let a final projection mix the heads together.

---

## Mental Model

```text
  x : [batch, seq, d_model]
       │
       │  packed projections
       ├── W_Q ──► Q : [batch, seq, d_model]  ──► reshape ──► [batch, H, seq, d_head]
       ├── W_K ──► K : [batch, seq, d_model]  ──► reshape ──► [batch, H, seq, d_head]
       └── W_V ──► V : [batch, seq, d_model]  ──► reshape ──► [batch, H, seq, d_head]

  per-head:    scaled-dot-product-attention(Q_h, K_h, V_h)
  concat heads back ──► [batch, seq, d_model]
  W_O projection    ──► [batch, seq, d_model]
```

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

## Common Pitfalls

- Wrong reshape order — must split *then* permute; permute-then-split breaks per-head identity.
- Forgetting the final `W_O` projection; without it heads stay disjoint.
- Letting `d_model` not divide by `H`; assert this on construction.
- Computing scaling with `√d_model` instead of `√d_head`.
- Materialising 4 separate parameters when one packed `[d_model, 3 · d_model]` works fine.

---

## How to Verify

Run the tests and the exercise. Both should pass cleanly with no warnings:

```bash
bun test src/nn/attention.test.ts
```
```bash
bun run exercises/ch-23-multi-head-attention.ts
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

## Further Reading

- [Vaswani et al. — Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) — the original transformer paper; every formula in Parts 5–6 comes from it.
- [Voita et al. — Analyzing Multi-Head Self-Attention](https://arxiv.org/abs/1905.09418) — what individual heads learn; pruning experiments.
- [Jay Alammar — The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — the multi-head section is the clearest visual we know.
- [Lilian Weng — Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) — history of attention from seq2seq to multi-head.

---

## Next Chapter

**[Cross-Attention](ch-24-cross-attention.md)** — let the decoder query the encoder, the glue between source and target.
