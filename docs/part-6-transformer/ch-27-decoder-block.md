# Chapter 27: Decoder Block

> **Part 6 of 6 — Transformer**
> Source: [`src/nn/transformer.ts`](../../src/nn/transformer.ts)
> Tests: [`src/nn/transformer.test.ts`](../../src/nn/transformer.test.ts)
> Exercise: [`exercises/ch-27-decoder-block.ts`](../../exercises/ch-27-decoder-block.ts)

---

## Learning Goals

By the end of this chapter you can:

- Add a masked self-attention sublayer with a causal mask.
- Add a cross-attention sublayer with K, V from the encoder output.
- Keep the FFN sublayer; total of three pre-norm + residual passes.
- Stack `N` decoder blocks to form the full decoder.
- Trace the teacher-forcing training loop: shifted inputs, full-sequence parallel processing.

---

## Intuition First

The decoder is a writer with one eye on what it has written so far (masked self-attention) and another on the source it is translating from (cross-attention). During training we cheat with **teacher forcing** — we feed the true shifted target as input and rely on the causal mask to keep each position from peeking at its answer.

---

## Mental Model

```text
  x ─► LN ─► Masked MHA ─►(+)─► LN ─► Cross-Attn ─►(+)─► LN ─► FFN ─►(+)─► out
                    ↑                  ↑                          ↑
                    │                  │                          │
                    │       encoder_out                          │
  └─ residual 1 ────┘       └─ residual 2 ────┘     └─ residual 3 ┘
```

---

## Concepts

### Structure of the Decoder Block

Three sub-layers instead of two:

```
x ──→ LayerNorm ──→ Masked Self-Attention ──→ (+) ──→ LayerNorm ──→ Cross-Attention ──→ (+) ──→ LayerNorm ──→ FFN ──→ (+) ──→ output
│                                              ↑    │                                    ↑    │                          ↑
└──────────────────────────────────────────────┘    └────────────────────────────────────┘    └──────────────────────────┘
       residual 1                                          residual 2                                  residual 3
```

In notation:
$$x' = x + \text{MaskedSelfAttn}(\text{LayerNorm}_1(x))$$
$$x'' = x' + \text{CrossAttn}(\text{LayerNorm}_2(x'), \text{encoderOutput})$$
$$\text{output} = x'' + \text{FFN}(\text{LayerNorm}_3(x''))$$

### Why Masking in Self-Attention

The decoder generates tokens **left-to-right** (autoregressive). When generating token at
position $t$, it must only see positions $0, 1, \ldots, t-1$ (what has been generated so far)
and position $t$ itself. It must not see future positions $t+1, t+2, \ldots$

This is enforced by the causal mask (Ch 22): add $-\infty$ to future positions in the
self-attention score matrix, so after softmax they receive 0 attention weight.

The key insight: **during training** we feed the entire target sequence at once (for efficiency).
The causal mask prevents information leakage, making each position only "see" its past.
This is called **teacher forcing** — we feed the ground-truth tokens as input instead of
the model's own predictions.

### Cross-Attention in the Decoder

After masked self-attention, the decoder state $x''$ has rich contextual representations
of the partially-generated target. Now cross-attention lets each decoder position query
the encoder's understanding of the full source sequence:

- Q from decoder: "I've generated some output, what from the source should I focus on next?"
- K, V from encoder: "Here's the full source, structured for retrieval"

This is the key mechanism that ties source and target together. Without cross-attention,
the decoder can't use the source at all.

### Training: Teacher Forcing

During training:
1. Encoder encodes the full source
2. Decoder is given: `[BOS, token_1, token_2, ..., token_{n-1}]` as input (shifted right)
3. Target to predict: `[token_1, token_2, ..., token_n, EOS]`
4. Loss: cross-entropy between predicted logits and target tokens

All target tokens are processed in parallel (the causal mask handles masking). This is much
faster than running the decoder one token at a time during training.

### Inference: Autoregressive Decoding

During inference (generation):
1. Encoder encodes the source
2. Start with `[BOS]`
3. Run decoder → get logit → sample/argmax → append token
4. Repeat with the grown sequence until `[EOS]` is generated

This is the slow part of inference — $n$ sequential forward passes for $n$ output tokens.
(KV caching makes this faster by avoiding recomputation.)

---

## What to Implement

| Symbol | Description |
|--------|-------------|
| `class DecoderBlock` | Constructor: `(dModel, numHeads, dFf, dropout?)` |
| `DecoderBlock.forward(x, encoderOut, srcMask?, tgtMask?)` | Full decoder block: masked self-attn + cross-attn + FFN |
| `DecoderBlock.parameters()` | All sublayer parameters |
| `class Decoder` | Stack of $N$ `DecoderBlock`s |
| `Decoder.forward(x, encoderOut, srcMask?, tgtMask?)` | Pass through all decoder blocks |
| `Decoder.parameters()` | All parameters from all blocks |

---

## TypeScript Hints

```typescript
export class DecoderBlock {
  selfAttn:   MultiHeadAttention;   // Masked self-attention
  crossAttn:  CrossAttention;       // Cross-attention (Q from decoder, K/V from encoder)
  ffn:        FFN;
  norm1:      LayerNorm;
  norm2:      LayerNorm;
  norm3:      LayerNorm;
  dropout:    Dropout;

  constructor(dModel: number, numHeads: number, dFf: number, dropoutRate = 0.1) {
    this.selfAttn  = new MultiHeadAttention(dModel, numHeads);
    this.crossAttn = new CrossAttention(dModel, numHeads);
    this.ffn       = new FFN(dModel, dFf);
    this.norm1     = new LayerNorm(dModel);
    this.norm2     = new LayerNorm(dModel);
    this.norm3     = new LayerNorm(dModel);
    this.dropout   = new Dropout(dropoutRate);
  }

  forward(x: Value, encoderOut: Value, srcMask?: Tensor, tgtMask?: Tensor): Value {
    // Sub-layer 1: masked self-attention
    // tgtMask is the causal mask — prevents attending to future target tokens
    const selfAttnOut = this.selfAttn.forward(this.norm1.forward(x), tgtMask);
    x = x.add(this.dropout.forward(selfAttnOut));

    // Sub-layer 2: cross-attention
    // Query from decoder, Key/Value from encoder
    // srcMask is the padding mask for the source (if source has padding tokens)
    const crossAttnOut = this.crossAttn.forward(this.norm2.forward(x), encoderOut, srcMask);
    x = x.add(this.dropout.forward(crossAttnOut));

    // Sub-layer 3: FFN
    const ffnOut = this.ffn.forward(this.norm3.forward(x));
    x = x.add(this.dropout.forward(ffnOut));

    return x;
  }

  parameters(): Value[] {
    return [
      ...this.selfAttn.parameters(),
      ...this.crossAttn.parameters(),
      ...this.ffn.parameters(),
      ...this.norm1.parameters(),
      ...this.norm2.parameters(),
      ...this.norm3.parameters(),
    ];
  }
}

export class Decoder {
  blocks: DecoderBlock[];

  constructor(numLayers: number, dModel: number, numHeads: number, dFf: number, dropoutRate = 0.1) {
    this.blocks = Array.from({ length: numLayers }, () =>
      new DecoderBlock(dModel, numHeads, dFf, dropoutRate)
    );
  }

  forward(x: Value, encoderOut: Value, srcMask?: Tensor, tgtMask?: Tensor): Value {
    for (const block of this.blocks) {
      x = block.forward(x, encoderOut, srcMask, tgtMask);
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

- Forgetting the causal mask in the *masked* self-attention; the model trivially copies its target.
- Passing the encoder output of the wrong layer to cross-attention — use the encoder's final output.
- Adding a causal mask to cross-attention; only self-attention needs it.
- Mismatched padding masks for source vs. target; track both separately.
- Sharing weights between encoder and decoder; almost never what you want.

---

## How to Verify

Run the tests and the exercise. Both should pass cleanly with no warnings:

```bash
bun test src/nn/transformer.test.ts
```
```bash
bun run exercises/ch-27-decoder-block.ts
```

---

## Self-Check Questions

1. The decoder has 3 sub-layers; the encoder has 2. Which sub-layer is the extra one,
   and why does only the decoder have it?
2. During training with teacher forcing, all target tokens are processed simultaneously.
   For a target sequence of length 5, draw the 5×5 causal mask that the decoder self-attention uses.
3. For cross-attention in layer $k$ of the decoder: is `encoderOut` the output of
   encoder layer $k$, or the output of the *final* encoder layer? Why?
4. If you remove the causal mask from decoder self-attention during training,
   the model achieves near-zero training loss immediately. Why? Does this help at inference?
5. A decoder stack has 6 blocks, `dModel=512, numHeads=8, dFf=2048`. Count total parameters:
   how many in all self-attention modules? All cross-attention modules? All FFNs? All LayerNorms?

---

## Further Reading

- [Vaswani et al. — Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) — the original transformer paper; every formula in Parts 5–6 comes from it.
- [Harvard NLP — The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/) — the original Transformer re-implemented with line-by-line commentary.
- [Jay Alammar — The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — decoder-block diagrams.
- [Karpathy — Let's build GPT (video)](https://www.youtube.com/watch?v=kCc8FmEb1nY) — decoder-only but illustrates the masking logic exactly.

---

## Next Chapter

**[Sequence Data & Objectives](ch-28-sequence-data-objectives.md)** — build the data pipeline that feeds the full Transformer.
