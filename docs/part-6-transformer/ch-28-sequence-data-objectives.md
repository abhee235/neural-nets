# Chapter 28: Sequence Data & Training Objectives

> **Part 6 of 6 — Transformer**
> `src/ch-28-sequence-data-objectives/`

---

## What You're Building

The data pipeline for training sequence models: examples, padding, shifted decoder inputs, token-level targets, loss masking, train/validation split, and perplexity.

This chapter turns a transformer architecture into a trainable experiment.

---

## Why This Matters

Most transformer bugs are not in the formula for attention. They are in the data plumbing:

- The target is not shifted correctly.
- Padding tokens contribute to the loss.
- The causal mask is missing.
- BOS/EOS are inconsistent between training and generation.
- Validation data accidentally leaks into training.

If this chapter is clear, the full transformer capstone becomes a controlled engineering task instead of guesswork.

---

## Concepts

### Sequence-to-Sequence Example Format

For encoder-decoder training, each sample has:

```typescript
type Seq2SeqExample = {
  source: number[];
  target: number[];
};
```

For a reversal task:

```text
source text: "hello"
target text: "olleh"
```

With special tokens:

```text
source ids: [BOS, h, e, l, l, o, EOS]
target ids: [BOS, o, l, l, e, h, EOS]
```

### Shifted Decoder Inputs

The decoder input is the target sequence shifted right:

```text
decoder input: [BOS, o, l, l, e, h]
target labels: [  o, l, l, e, h, EOS]
```

Plain English: at each position, the decoder sees previous correct tokens and predicts the next token.

This is teacher forcing.

### Loss Masking

The model should not be penalized for predictions at padding positions.

If target labels are:

```text
[o, l, l, e, h, EOS, PAD, PAD]
```

The loss mask is:

```text
[1, 1, 1, 1, 1,   1,   0,   0]
```

The token-level cross-entropy should be averaged only over positions where the loss mask is 1.

### Perplexity

Perplexity is the exponentiated average cross-entropy:

$$\text{perplexity} = e^{\text{loss}}$$

Plain English: perplexity is roughly the effective number of choices the model is confused between at each token.

Lower is better. A perfectly confident correct model has loss near 0 and perplexity near 1.

### Overfit-One-Batch Test

Before training on a dataset, train on one tiny batch until the model nearly memorizes it.

If the model cannot overfit one batch, there is a bug in:

- Autograd.
- Loss masking.
- Attention masks.
- Optimizer updates.
- Shifted target construction.

This is one of the most important AI engineering habits in the course.

---

## What to Implement

| Symbol | Description |
|--------|-------------|
| `makeReversalDataset(count, minLen, maxLen, alphabet)` | Generate toy source/target string pairs. |
| `splitDataset(examples, validationFraction, seed)` | Deterministic train/validation split. |
| `makeSeq2SeqBatch(examples, tokenizer, maxLen)` | Produce `srcIds`, `decoderInputIds`, `targetIds`, and masks. |
| `shiftRight(targetIds, bosId)` | Build decoder inputs from target labels. |
| `maskedCrossEntropy(logits, targetIds, lossMask)` | Average CE only over non-pad tokens. |
| `perplexity(loss)` | Return `Math.exp(loss)`. |
| `overfitOneBatch(model, batch, opts)` | Debug routine that must drive loss close to zero. |

---

## TypeScript Hints

```typescript
export function shiftRight(targetIds: Tensor, bosId: number): Tensor {
  // targetIds: [batch, seq]
  // decoderInput[:, 0] = BOS
  // decoderInput[:, t] = targetIds[:, t - 1]
  const [batchSize, seqLen] = targetIds.shape as [number, number];
  const data: number[] = [];

  for (let b = 0; b < batchSize; b++) {
    for (let t = 0; t < seqLen; t++) {
      if (t === 0) {
        data.push(bosId);
      } else {
        data.push(targetIds.data[b * seqLen + (t - 1)]!);
      }
    }
  }

  return createTensor(data, [batchSize, seqLen]);
}
```

---

## Required Tests

- `shiftRight([a, b, EOS], BOS)` returns `[BOS, a, b]`.
- Padding positions have loss mask 0.
- `maskedCrossEntropy` ignores pad positions completely.
- Dataset split is deterministic with the same seed.
- One-batch overfit reduces loss sharply on a tiny dataset.
- Validation examples never appear in the training set.

---

## Common Pitfalls

- Training the decoder to copy the token it already sees instead of predicting the next token.
- Including `<pad>` tokens in the loss average.
- Forgetting EOS, so generation has no natural stopping condition.
- Testing only training loss and never validation loss.
- Moving to the full dataset before passing the one-batch overfit test.

---

## Self-Check Questions

1. Why is the decoder input shifted right relative to the target labels?
2. What does teacher forcing give us during training that we do not have during inference?
3. Why should padding tokens be ignored in the loss?
4. If loss decreases on training but validation loss increases, what is happening?
5. What does it mean if the model cannot overfit one batch?

---

## → Next Chapter

**Ch 29: Full Transformer** — now the final architecture has a verified sequence training pipeline.
