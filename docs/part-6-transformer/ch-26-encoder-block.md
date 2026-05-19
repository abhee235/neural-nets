# Chapter 26: Encoder Block

> **Part 6 of 6 — Transformer**
> `src/ch-26-encoder-block/`

---

## What You're Building

A complete transformer encoder block: LayerNorm → Multi-Head Self-Attention → residual add
→ LayerNorm → FFN → residual add. Stack $N$ of these to form the full encoder.

---

## Why This Matters

This is the first time you're combining all the primitives into a recognizable piece of a
real transformer. The encoder block is the same as in BERT, the original Transformer, T5's
encoder, and every other encoder-only or encoder-decoder model. After this chapter you'll
have a working encoder — it can turn a sequence of tokens into a sequence of rich, contextual
representations.

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

## → Next Chapter

**Ch 27: Decoder Block** — add masked self-attention and cross-attention to build the
full transformer decoder block.
