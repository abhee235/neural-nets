# Chapter 30: Decoder-Only GPT

> **Part 6 of 6 — Transformer**
> Source: [`src/nn/gpt.ts`](../../src/nn/gpt.ts)
> Tests: [`src/nn/gpt.test.ts`](../../src/nn/gpt.test.ts)
> Exercise: [`exercises/ch-30-decoder-only-gpt.ts`](../../exercises/ch-30-decoder-only-gpt.ts)

---

## Learning Goals

By the end of this chapter you can:

- Strip the encoder and cross-attention; keep only masked self-attention + FFN blocks.
- Train on next-token prediction with a sliding-window language-modelling batch.
- Generate text autoregressively with greedy decoding, then with temperature sampling.
- Recognise the architecture family behind GPT-2, GPT-3, LLaMA, and Mistral.
- Identify natural extensions: KV cache, MoE, LoRA, quantization.

---

## Intuition First

If you can already train an encoder-decoder transformer on string reversal, GPT is *simpler*. There is only one sequence, only one stack of blocks (decoder-only), and only one objective: predict the next token. The same self-attention you wrote in Ch 22, the same FFN from Ch 25, the same LayerNorm from Ch 20 — composed into a language model.

---

## Mental Model

```text
  token ids ──► Embed + PE ──► Dropout ──► DecoderOnlyBlock × N ──► LN ──► Linear(d→V)
                                                                                │
                                                                                ▼
                                                                       next-token logits

  DecoderOnlyBlock:  x = x + MaskedMHA(LN(x))
                     x = x + FFN(LN(x))
```

---

## Concepts

### Architecture

```text
Token IDs
   ↓
Embedding + Positional Encoding
   ↓
Dropout
   ↓
[DecoderOnlyBlock × N]
   ↓
Final LayerNorm
   ↓
Linear(dModel → vocabSize)
   ↓
Next-token logits
```

Each decoder-only block has:

```text
x = x + MaskedSelfAttention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

There is no cross-attention because there is no separate source sequence.

### Next-Token Prediction

Given a token sequence:

```text
[t0, t1, t2, t3, t4]
```

The model input is:

```text
[t0, t1, t2, t3]
```

The target is:

```text
[t1, t2, t3, t4]
```

At position 0, predict token 1. At position 1, predict token 2. And so on.

The causal mask prevents the model from seeing future tokens while training all positions in parallel.

### Language Modeling Dataset

For a tiny character-level dataset, concatenate text and cut it into fixed-length blocks:

```text
text: "to be or not to be"
blockSize = 8
input:  "to be or"
target: "o be or "
```

This is much simpler than seq2seq batching: one stream of tokens becomes input/target windows.

### Generation

Generation is autoregressive:

1. Start with a prompt.
2. Run the model and take logits at the final position.
3. Pick the next token using greedy argmax or sampling.
4. Append that token to the context.
5. Repeat.

For learning, start with greedy decoding. Then add temperature sampling.

### Temperature Sampling

Temperature changes how sharp the probability distribution is:

$$p_i = \text{softmax}\left(\frac{z_i}{T}\right)$$

Where:

- $T < 1$ makes output more deterministic.
- $T = 1$ leaves logits unchanged.
- $T > 1$ makes output more random.

---

## What to Implement

| Symbol | Description |
|--------|-------------|
| `class DecoderOnlyBlock` | Pre-norm masked self-attention + FFN. No cross-attention. |
| `class GPT` | Embedding, positional encoding, N decoder-only blocks, final norm, output projection. |
| `GPT.forward(inputIds)` | Return logits `[batch, seq, vocabSize]`. |
| `makeLanguageModelingBatch(tokenIds, blockSize, batchSize)` | Sample input/target windows. |
| `generateText(model, tokenizer, prompt, maxNewTokens)` | Greedy or sampled autoregressive generation. |
| `sampleFromLogits(logits, temperature)` | Convert logits to probabilities and sample a token. |

---

## TypeScript Hints

```typescript
export class DecoderOnlyBlock {
  selfAttn: MultiHeadAttention;
  ffn: FFN;
  norm1: LayerNorm;
  norm2: LayerNorm;
  dropout: Dropout;

  forward(x: Value, causalMask: Tensor): Value {
    const attnOut = this.selfAttn.forward(this.norm1.forward(x), causalMask);
    x = x.add(this.dropout.forward(attnOut));

    const ffnOut = this.ffn.forward(this.norm2.forward(x));
    x = x.add(this.dropout.forward(ffnOut));

    return x;
  }
}
```

---

## Common Pitfalls

- Accidentally importing the full decoder block (with cross-attention) instead of the decoder-only block.
- Forgetting the causal mask during training — the model trivially copies inputs.
- Generating from all logit positions instead of only the final one at inference.
- Letting the context length exceed `maxSeqLen` without cropping.
- Expecting meaningful language from a tiny dataset before it has memorised patterns.

---

## How to Verify

Run the tests and the exercise. Both should pass cleanly with no warnings:

```bash
bun test src/nn/gpt.test.ts
```
```bash
bun run exercises/ch-30-decoder-only-gpt.ts
```

---

## Self-Check Questions

1. Why does a decoder-only model not need an encoder?
2. What is the training target for input tokens `[5, 8, 2, 9]`?
3. Why is causal masking required even though the target is shifted?
4. What happens when temperature approaches 0?
5. How is this chapter connected to KV cache, LoRA, quantization, and Mixture of Experts?

---

## Further Reading

- [Radford et al. — Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — the GPT-2 paper; this is the architecture you just built.
- [Karpathy — nanoGPT](https://github.com/karpathy/nanoGPT) — minimal reference implementation in PyTorch.
- [Karpathy — Let's build GPT (video)](https://www.youtube.com/watch?v=kCc8FmEb1nY) — the live build you can compare your code against.
- [Anthropic — Transformer Circuits](https://transformer-circuits.pub/) — for the next steps in interpretability.

---

## Next Chapter

You have reached the end of the core course. Natural follow-ons: **KV cache** for fast inference, **Mixture of Experts** for sparse FFNs, **LoRA** for parameter-efficient fine-tuning, and **quantization** for low-precision inference.
