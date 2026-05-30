# Chapter 29: Full Transformer

> **Part 6 of 6 — Transformer**
> Source: [`src/nn/transformer.ts`](../../src/nn/transformer.ts)
> Tests: [`src/nn/transformer.test.ts`](../../src/nn/transformer.test.ts)
> Exercise: [`exercises/ch-29-full-transformer.ts`](../../exercises/ch-29-full-transformer.ts)

---

## Learning Goals

By the end of this chapter you can:

- Assemble embeddings, positional encoding, encoder stack, decoder stack, final norm, and output projection.
- Run a full forward pass: `forward(src, tgt) → logits [batch, tgt_len, vocab_size]`.
- Scale embeddings by `√d_model` before adding positional encoding.
- Optionally tie embedding weights to the output projection.
- Train on the string-reversal task and achieve loss < 0.1 within ~100 epochs.

---

## Intuition First

Every previous chapter has been one ingredient. This chapter is the **recipe**. The encoder reads the source into a context tensor; the decoder, guided by cross-attention on that tensor, writes the target token by token. Once you can train this on a toy task, you can train it on translation — only the dataset changes.

---

## Mental Model

```text
  source ids   ──► Embed + √d_model ──► +PE ──► Dropout ──► Encoder × N ──► enc_out
                                                                            │
  target ids   ──► Embed + √d_model ──► +PE ──► Dropout ──► Decoder × N ◄───┘
                                                            │
                                                            ▼
                                                       Final LN
                                                            │
                                                            ▼
                                                  Linear(d_model → vocab)
                                                            │
                                                            ▼
                                                          logits
```

---

## Concepts

### Full Architecture

```
Source Tokens (integers)              Target Tokens (integers, shifted right)
        ↓                                          ↓
  [Embedding]                               [Embedding]
        ↓                                          ↓
  [+ Positional Encoding]              [+ Positional Encoding]
        ↓                                          ↓
  [Dropout]                                  [Dropout]
        ↓                                          ↓
  [EncoderBlock × N]                     [DecoderBlock × N] ←── encoder output
        ↓                                          ↓
  Encoder Output                           [LayerNorm]
  (saved for cross-attn)                         ↓
                                      [Linear: dModel → vocabSize]
                                               ↓
                                           Logits
                                               ↓
                                    Cross-Entropy Loss
```

### Input Pipeline

Source and target sequences are integer tensors of token IDs. The pipeline is:

1. **Embedding lookup** (Ch 18): `[batch, seq]` → `[batch, seq, dModel]`
2. **Positional encoding** (Ch 19): add sinusoidal PE — `[batch, seq, dModel]`
3. **Dropout**: `[batch, seq, dModel]` → `[batch, seq, dModel]`
4. **Scale embedding** by $\sqrt{d_{\text{model}}}$ (from the original paper): this re-scales
   the embeddings to match the magnitude of the positional encodings.

### Output Head

After the decoder stack:

5. **Final LayerNorm**: `[batch, seq, dModel]` → `[batch, seq, dModel]`
6. **Output projection**: Linear `dModel → vocabSize`, giving logits `[batch, seq, vocabSize]`
7. **Cross-entropy loss**: compare predicted logits vs target token IDs

### Weight Tying

A common optimization: share the embedding weights between the source/target embedding
lookup and the output projection matrix. The output projection and the embedding table
are both `[vocabSize, dModel]` — you can set them to be the same parameter.

Benefits: fewer parameters, and an interesting property: words that are close in the
embedding space also have similar output projection behavior.

### Toy Task: Sequence Reversal

A minimal but meaningful test task for the full transformer:
- Source: a random sequence of characters, e.g. `"abcde"`
- Target: the reversed sequence, e.g. `"edcba"`

This requires the model to:
1. Encode the source (understanding which tokens are which)
2. Attend to the source from the decoder (cross-attention)
3. Generate tokens in reverse order

A working transformer will achieve near-zero loss on this task. If it doesn't, you have a bug.

### Training Details

```
dModel     = 64      (small for fast training)
numHeads   = 4       (dHead = 16)
dFf        = 256     (4 × dModel)
numLayers  = 2       (both encoder and decoder)
dropout    = 0.1
maxSeqLen  = 32
vocabSize  = 30      (e.g., a–z + special tokens)
optimizer  = Adam (lr=1e-3)
batchSize  = 32
epochs     = 200
```

With these settings, a working implementation should get training loss < 0.1 in ~100 epochs.

### Autoregressive Inference

After training, test with greedy decoding:

```
function generate(model, srcTokens, maxLen):
  encOut = model.encode(srcTokens)
  tgtTokens = [BOS]
  for step in 0..maxLen:
    logits = model.decode(tgtTokens, encOut)
    nextToken = argmax(logits[last_position])
    tgtTokens.append(nextToken)
    if nextToken == EOS: break
  return tgtTokens[1:]  // strip BOS
```

---

## What to Implement

| Symbol | Description |
|--------|-------------|
| `class Transformer` | Constructor: `(config: TransformerConfig)` |
| `Transformer.encode(src)` | Run embedding + PE + dropout + encoder stack |
| `Transformer.decode(tgt, encoderOut, srcMask?, tgtMask?)` | Run embedding + PE + dropout + decoder stack + final norm |
| `Transformer.forward(src, tgt)` | Full forward pass: encode + decode + project → logits |
| `Transformer.parameters()` | All parameters from all sub-modules |
| `TransformerConfig` | Interface: `{ vocabSize, dModel, numHeads, dFf, numEncoderLayers, numDecoderLayers, maxSeqLen, dropout }` |
| `generate(model, srcTokens, maxLen)` | Greedy autoregressive decoding |

---

## TypeScript Hints

```typescript
export interface TransformerConfig {
  vocabSize:         number;
  dModel:            number;
  numHeads:          number;
  dFf:               number;
  numEncoderLayers:  number;
  numDecoderLayers:  number;
  maxSeqLen:         number;
  dropoutRate:       number;
}

export class Transformer {
  cfg:         TransformerConfig;
  srcEmbed:    Embedding;
  tgtEmbed:    Embedding;   // or share weights with srcEmbed if vocabs are the same
  pe:          PositionalEncoding;
  encoder:     Encoder;
  decoder:     Decoder;
  finalNorm:   LayerNorm;
  outputProj:  Linear;      // dModel → vocabSize
  dropout:     Dropout;

  constructor(cfg: TransformerConfig) {
    this.cfg = cfg;
    this.srcEmbed   = new Embedding(cfg.vocabSize, cfg.dModel);
    this.tgtEmbed   = new Embedding(cfg.vocabSize, cfg.dModel);
    this.pe         = new PositionalEncoding(cfg.maxSeqLen, cfg.dModel);
    this.encoder    = new Encoder(cfg.numEncoderLayers, cfg.dModel, cfg.numHeads, cfg.dFf, cfg.dropoutRate);
    this.decoder    = new Decoder(cfg.numDecoderLayers, cfg.dModel, cfg.numHeads, cfg.dFf, cfg.dropoutRate);
    this.finalNorm  = new LayerNorm(cfg.dModel);
    this.outputProj = new Linear(cfg.dModel, cfg.vocabSize, /* bias= */ false);
    this.dropout    = new Dropout(cfg.dropoutRate);
  }

  encode(src: Tensor, srcMask?: Tensor): Value {
    // Embed + scale + positional encode + dropout + encoder stack
    const emb = this.srcEmbed.forward(src);        // [batch, src_len, dModel]
    const scaled = emb.mulScalar(Math.sqrt(this.cfg.dModel));  // scale embeddings
    const withPE = this.pe.forward(new Value(scaled.data));    // add PE
    const dropped = this.dropout.forward(withPE);
    return this.encoder.forward(dropped, srcMask);
  }

  decode(tgt: Tensor, encoderOut: Value, srcMask?: Tensor, tgtMask?: Tensor): Value {
    const emb = this.tgtEmbed.forward(tgt);
    const scaled = emb.mulScalar(Math.sqrt(this.cfg.dModel));
    const withPE = this.pe.forward(new Value(scaled.data));
    const dropped = this.dropout.forward(withPE);
    const decOut = this.decoder.forward(dropped, encoderOut, srcMask, tgtMask);
    return this.finalNorm.forward(decOut);
  }

  forward(src: Tensor, tgt: Tensor, srcMask?: Tensor): Value {
    const seqLen = tgt.shape[1]!;
    const tgtMask = causalMask(seqLen);   // Ch 22
    const encoderOut = this.encode(src, srcMask);
    const decOut = this.decode(tgt, encoderOut, srcMask, tgtMask);
    return this.outputProj.forward(decOut);  // logits: [batch, tgt_len, vocabSize]
  }

  parameters(): Value[] {
    return [
      ...this.srcEmbed.parameters(),
      ...this.tgtEmbed.parameters(),
      ...this.encoder.parameters(),
      ...this.decoder.parameters(),
      ...this.finalNorm.parameters(),
      ...this.outputProj.parameters(),
    ];
  }
}
```

---

## Common Pitfalls

- Forgetting to scale embeddings by `√d_model`; the magnitudes don't match the positional encoding.
- Building one `nn.LayerNorm` and using it everywhere — each location needs its own.
- Feeding raw target tokens (not shifted) into the decoder; trivial copy task results.
- Letting the decoder maxSeqLen exceed the trained PE table; either crop or extend.
- Declaring victory at low train loss without checking generation on unseen examples.

---

## How to Verify

Run the tests and the exercise. Both should pass cleanly with no warnings:

```bash
bun test src/nn/transformer.test.ts
```
```bash
bun run exercises/ch-29-full-transformer.ts
```

---

## Self-Check Questions

1. Trace the full forward pass for one batch of source length 5, target length 5,
   `dModel=64`. Write out the shape at each step from raw token IDs to logits.
2. The output projection is `Linear(dModel, vocabSize)` without bias. Why no bias?
   (Hint: the bias would add a constant to every logit, equally for all positions.)
3. During training, the decoder input is `[BOS, t1, t2, t3, t4]` and the target
   is `[t1, t2, t3, t4, EOS]`. Why the shift? What would go wrong without it?
4. After 200 epochs on the reversal task, your model achieves near-zero training loss.
   Try generating for a source it has never seen. Does it generalize? Why or why not?
5. You've built every piece from scratch. Sketch the full parameter count for the
   toy model (`dModel=64, numHeads=4, dFf=256, numLayers=2, vocabSize=30`).

---

## Further Reading

- [Vaswani et al. — Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) — the original transformer paper; every formula in Parts 5–6 comes from it.
- [Harvard NLP — The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/) — the closest published cousin of what we are building.
- [Sasha Rush — The Annotated Encoder-Decoder](https://bastings.github.io/annotated_encoder_decoder/) — more notes on training dynamics.
- [Jay Alammar — The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — still the best single visual reference for the full model.

---

## Next Chapter

**[Decoder-Only GPT](ch-30-decoder-only-gpt.md)** — strip the encoder away and train next-token prediction directly.
