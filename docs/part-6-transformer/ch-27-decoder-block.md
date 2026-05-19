# Chapter 27: Decoder Block

> **Part 6 of 6 — Transformer**
> `src/ch-27-decoder-block/`

---

## What You're Building

A complete transformer decoder block: masked self-attention (causal), then cross-attention
over the encoder output, then FFN — each wrapped with pre-norm and residual connections.
Stack $N$ of these to form the full decoder.

---

## Why This Matters

The decoder block is the generating half of the transformer. It's what produces translations,
summaries, and answers. It sees the partial output generated so far (via self-attention)
*and* the full encoder representation of the input (via cross-attention) — and uses both
to decide the next token.

Understanding the decoder block makes it easy to also understand GPT-style models,
which are decoder-only (self-attention + FFN, no cross-attention), which we can discuss
after completing the full transformer.

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

## → Next Chapter

**Ch 28: Sequence Data & Training Objectives** — build the data pipeline, shifted targets,
loss masks, validation split, and one-batch overfit test needed before training the full model.
