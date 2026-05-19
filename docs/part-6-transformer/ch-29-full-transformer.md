# Chapter 29: Full Transformer

> **Part 6 of 6 — Transformer**
> `src/ch-29-full-transformer/`

---

## What You're Building

The complete transformer: encoder stack + decoder stack + embedding layers + output
projection, assembled into a single trainable model. Train it on a toy sequence-to-sequence
task (e.g., character-level reversal: "hello" → "olleh") to see all the pieces working together.

---

## Why This Matters

This is the culmination of the encoder-decoder path. Every single piece you have built — tensor arithmetic,
broadcasting, autograd, activations, losses, optimizers, tokenizer, embeddings, positional
encoding, LayerNorm, dropout, attention, multi-head attention, cross-attention, FFN, encoder
blocks, decoder blocks — comes together here.

After this chapter you will have trained a working transformer from scratch, in TypeScript,
with zero external libraries.

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

## Core Transformer Complete

You have built a full transformer from scratch:
- Part 1: Tensor library (Ch 01–06)
- Part 2: Autograd engine (Ch 07–10)
- Part 3: Neural net primitives (Ch 11–15)
- Part 4: Language model inputs (Ch 16–21)
- Part 5: Attention mechanism (Ch 22–24)
- Part 6: Full encoder-decoder transformer (Ch 25–29)

---

## → Next Chapter

**Ch 30: Decoder-Only GPT** — remove the encoder and cross-attention, then train a
GPT-style model with next-token prediction.

---

## Later Extensions

After the decoder-only capstone, natural extensions to explore:

**KV Cache**
- Cache K and V tensors at each decoder step during inference
- Avoid recomputing the full context at every autoregressive step
- Critical for fast inference in production

**Mixture of Experts (MoE)**
- Replace FFN with multiple expert FFNs
- A router network selects which experts to activate per token
- Used in Mixtral, GPT-4 (rumored), and other large models

**LoRA (Low-Rank Adaptation)**
- Fine-tune only small rank-decomposed matrices, not the full weight matrices
- $W' = W + \frac{\alpha}{r} A B$ where $A \in \mathbb{R}^{d \times r}, B \in \mathbb{R}^{r \times d}$
- Massively reduces fine-tuning memory and compute

**Quantization**
- Store weights in 4-bit or 8-bit integers instead of 32-bit floats
- Reduces memory 4–8×, with minimal quality loss
