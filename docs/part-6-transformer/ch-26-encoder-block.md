# Chapter 26: Encoder Block

> **Part 6 of 6 — Transformer**
> Source: [`src/nn/transformer.ts`](../../src/nn/transformer.ts)
> Tests: [`src/nn/transformer.test.ts`](../../src/nn/transformer.test.ts)
> Exercise: [`exercises/ch-26-encoder-block.ts`](../../exercises/ch-26-encoder-block.ts)

---

## Learning Goals

By the end of this chapter you can:

- Combine multi-head self-attention, LayerNorm, residual connection, FFN, LayerNorm, residual into one block.
- Place LayerNorm in **pre-norm** position: `x + Sublayer(LN(x))`.
- Stack `N` blocks to form the full encoder.
- Confirm output shape equals input shape: `[batch, seq, d_model]`.
- Explain why residual connections make deep stacks trainable.

---

## Intuition First

An encoder block is two passes through the standard "normalise → transform → add" ritual. The residual connection lets the gradient flow backwards through the identity shortcut even when the inner transform is hard to learn — this is what makes 12-, 24-, even 96-layer transformers actually train.

Pre-norm (normalise *before* the sublayer) is now standard; it is more stable than the original post-norm formulation.

---

## Mental Model

```text
  x ─► LN ─► MHA ─►(+)─► LN ─► FFN ─►(+)─► out
  │              ↑    │             ↑
  └──────────────┘    └─────────────┘
       residual 1          residual 2
```

---

## Concepts

### Structure of the Encoder Block

Using **pre-norm** convention (GPT-2 / modern standard):

```
x ──→ LayerNorm ──→ MultiHeadSelfAttention ──→ (+) ──→ LayerNorm ──→ FFN ──→ (+) ──→ output
│                                               ↑                              ↑
└───────────────────────────────────────────────┘              ────────────────┘
           (residual skip connection 1)                  (residual skip connection 2)
```

In mathematical notation:
$$x' = x + \text{MHA}(\text{LayerNorm}_1(x))$$
$$\text{output} = x' + \text{FFN}(\text{LayerNorm}_2(x'))$$

Note: LayerNorm is applied *before* the sublayer, and the original (unmodified) $x$ is added
back after. This is pre-norm (also called "pre-LN" or "pre-layer normalization").

### Residual Connections

The `+ x` terms are **residual (skip) connections** from ResNets. They have three critical benefits:

1. **Gradient highway:** gradients flow directly back through the `+` to earlier layers without
   being transformed. This prevents vanishing gradients in deep networks (24+ layers).
2. **Identity initialization:** at random initialization, the sublayer outputs are near zero,
   so the block acts like the identity function. Training starts from a stable point.
3. **Ensemble effect:** you can think of each block as learning a correction to the identity.
   The full network is like an ensemble of shallow paths.

### Pre-Norm vs Post-Norm

**Post-Norm** (original transformer, 2017): `LayerNorm(x + sublayer(x))`
- Works well with learning rate warm-up
- Slightly better final performance in some experiments

**Pre-Norm** (GPT-2, LLaMA, most modern): `x + sublayer(LayerNorm(x))`
- More stable at initialization, no warm-up needed
- Slightly worse final perplexity in some settings
- Almost universally preferred in practice

We use pre-norm.

### The Encoder Stack

A full encoder is simply $N$ blocks stacked sequentially. The original transformer uses $N=6$.
BERT-base uses $N=12$, BERT-large uses $N=24$.

---

## What to Implement

| Symbol | Description |
|--------|-------------|
| `class EncoderBlock` | Constructor: `(dModel, numHeads, dFf, dropout?)` |
| `EncoderBlock.forward(x, mask?)` | Full pre-norm encoder block forward pass |
| `EncoderBlock.parameters()` | All sublayer parameters |
| `class Encoder` | Stack of $N$ `EncoderBlock`s |
| `Encoder.forward(x, mask?)` | Pass input through all blocks sequentially |
| `Encoder.parameters()` | All parameters from all blocks |

---

## TypeScript Hints

```typescript
export class EncoderBlock {
  attn:    MultiHeadAttention;
  ffn:     FFN;
  norm1:   LayerNorm;
  norm2:   LayerNorm;
  dropout: Dropout;

  constructor(dModel: number, numHeads: number, dFf: number, dropoutRate = 0.1) {
    this.attn    = new MultiHeadAttention(dModel, numHeads);
    this.ffn     = new FFN(dModel, dFf);
    this.norm1   = new LayerNorm(dModel);
    this.norm2   = new LayerNorm(dModel);
    this.dropout = new Dropout(dropoutRate);
  }

  forward(x: Value, mask?: Tensor): Value {
    // Sub-layer 1: pre-norm self-attention + residual
    const attnOut = this.attn.forward(this.norm1.forward(x), mask);
    x = x.add(this.dropout.forward(attnOut));

    // Sub-layer 2: pre-norm FFN + residual
    const ffnOut = this.ffn.forward(this.norm2.forward(x));
    x = x.add(this.dropout.forward(ffnOut));

    return x;
  }

  parameters(): Value[] {
    return [
      ...this.attn.parameters(),
      ...this.ffn.parameters(),
      ...this.norm1.parameters(),
      ...this.norm2.parameters(),
    ];
  }
}

export class Encoder {
  blocks: EncoderBlock[];

  constructor(numLayers: number, dModel: number, numHeads: number, dFf: number, dropoutRate = 0.1) {
    this.blocks = Array.from({ length: numLayers }, () =>
      new EncoderBlock(dModel, numHeads, dFf, dropoutRate)
    );
  }

  forward(x: Value, mask?: Tensor): Value {
    // Pass through each encoder block sequentially
    for (const block of this.blocks) {
      x = block.forward(x, mask);
    }
    return x;
  }

  parameters(): Value[] {
    return this.blocks.flatMap(b => b.parameters());
  }
}
```

---

## Common Pitfalls

- Using post-norm by accident (`LN(x + Sublayer(x))`); pre-norm is `x + Sublayer(LN(x))`.
- Forgetting the residual connection; deep stacks become untrainable.
- Sharing LayerNorm parameters across blocks; each block has its own `γ`, `β`.
- Letting dropout fire outside training mode.
- Mis-counting parameters; the FFN dominates (about `8 · d_model²` per block).

---

## How to Verify

Run the tests and the exercise. Both should pass cleanly with no warnings:

```bash
bun test src/nn/transformer.test.ts
```
```bash
bun run exercises/ch-26-encoder-block.ts
```

---

## Self-Check Questions

1. Trace the shapes: input `x` is `[2, 10, 512]`. What is the shape after LayerNorm?
   After attention? After the first residual add? After FFN? After the second residual add?
2. The residual `x + sublayer(x)` requires that `x` and `sublayer(x)` have the same shape.
   What constraint does this place on the attention and FFN sublayers?
3. Initialize a random `EncoderBlock`. Before any training, what do you expect the
   norms of `attnOut` and `ffnOut` to be relative to `x`? Why?
4. If you remove both LayerNorms from an encoder block, the model often fails to train.
   Why? What happens to the activations?
5. BERT-large has 24 encoder blocks. How many total parameters are in the encoder?
   Use: dModel=1024, numHeads=16, dFf=4096. (Approximate — ignore biases.)

---

## Further Reading

- [Vaswani et al. — Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) — the original transformer paper; every formula in Parts 5–6 comes from it.
- [Xiong et al. — On Layer Normalization in the Transformer](https://arxiv.org/abs/2002.04745) — pre-norm vs. post-norm analysis.
- [He et al. — Deep Residual Learning (2015)](https://arxiv.org/abs/1512.03385) — the residual-connection paper that made deep nets possible.
- [Jay Alammar — The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — block-level diagrams.

---

## Next Chapter

**[Decoder Block](ch-27-decoder-block.md)** — add masked self-attention and cross-attention to make a generating layer.
