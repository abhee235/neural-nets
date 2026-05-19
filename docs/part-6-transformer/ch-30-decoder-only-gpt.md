# Chapter 30: Decoder-Only GPT

> **Post-Core Capstone — Modern Language Modeling**
> `src/ch-30-decoder-only-gpt/`

---

## What You're Building

A GPT-style decoder-only transformer trained with next-token prediction. This model removes the encoder and cross-attention from the full Transformer and keeps only token embeddings, positional encoding, masked self-attention, FFN blocks, LayerNorm, and an output projection.

This is the architecture family behind modern autoregressive language models.

---

## Why This Matters

The original Transformer is encoder-decoder and was designed for sequence transduction such as translation. Modern AI engineering often starts from decoder-only language models.

The conceptual shift is simple but important:

- Encoder-decoder model: source sequence maps to target sequence.
- Decoder-only model: one sequence predicts its own next token.

This chapter turns the course from “I can implement the original Transformer” into “I understand the core architecture behind GPT-style models.”

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

## Required Tests

- Decoder-only block preserves shape `[batch, seq, dModel]`.
- Causal mask prevents position 0 from depending on position 1.
- Language-modeling batch shifts targets by exactly one token.
- Training can overfit a tiny repeated string.
- Greedy generation produces deterministic output with the same prompt and model.

---

## Common Pitfalls

- Accidentally using cross-attention from the full decoder block.
- Forgetting the causal mask during training.
- Generating from all logits instead of only the final position.
- Letting context length exceed `maxSeqLen` without cropping.
- Expecting meaningful language from a tiny dataset before it has memorized patterns.

---

## Self-Check Questions

1. Why does a decoder-only model not need an encoder?
2. What is the training target for input tokens `[5, 8, 2, 9]`?
3. Why is causal masking required even though the target is shifted?
4. What happens when temperature approaches 0?
5. How is this chapter connected to KV cache, LoRA, quantization, and Mixture of Experts?

---

## Course Milestone

After this chapter, you have two transformer families implemented from scratch:

- Encoder-decoder Transformer for sequence-to-sequence tasks.
- Decoder-only GPT-style Transformer for next-token prediction.

That is the minimum architecture foundation expected of a serious AI engineering course.
