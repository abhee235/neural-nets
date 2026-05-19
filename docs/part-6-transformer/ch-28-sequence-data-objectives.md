# Chapter 28: Sequence Data & Training Objectives

> **Part 6 of 6 — Transformer**
> Source: [`src/utils/data.ts`](../../src/utils/data.ts)
> Tests: [`src/utils/data.test.ts`](../../src/utils/data.test.ts)
> Exercise: [`exercises/ch-28-sequence-data.ts`](../../exercises/ch-28-sequence-data.ts)

---

## Learning Goals

By the end of this chapter you can:

- Define a `Seq2SeqExample` type and a tiny toy dataset (e.g. string reversal).
- Build `shiftRight(target, BOS)` so decoder input = `[BOS, t1, …, t_{n-1}]`.
- Build a masked cross-entropy loss that ignores `<pad>` positions.
- Report perplexity = `exp(loss)` alongside cross-entropy.
- Use the one-batch overfit test as a debugging tool *before* training on the full set.

---

## Intuition First

Most transformer bugs live in the data pipeline, not in the model. Mis-shifted targets, padding tokens in the loss, missing causal masks — each of these silently kills training. This chapter is the data-engineering chapter that makes the next one (the full transformer) a controlled experiment instead of a guessing game.

The **overfit-one-batch** test is the single most valuable habit in this part: if the model cannot drive loss near zero on one tiny batch, something is broken upstream.

---

## Mental Model

```text
  raw pair:        source="hello"           target="olleh"
  tokenize:        src_ids =[BOS, h, e, l, l, o, EOS]
                   tgt_ids =[BOS, o, l, l, e, h, EOS]
  shiftRight:      dec_in  =[BOS, o, l, l, e, h]
                   labels  =[o,   l, l, e, h, EOS]
  loss mask:       [1, 1, 1, 1, 1,   1]    ← pad positions get 0
  loss:            masked cross-entropy → scalar
  metric:          perplexity = exp(loss)
```

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

## Common Pitfalls

- Training the decoder to copy the token it already sees instead of predicting the next token.
- Including `<pad>` tokens in the loss average; perplexity becomes nonsense.
- Forgetting EOS, so generation has no natural stopping condition.
- Testing only training loss; without validation you cannot detect overfitting.
- Moving to the full dataset before passing the one-batch overfit test.

---

## How to Verify

Run the tests and the exercise. Both should pass cleanly with no warnings:

```bash
bun test src/utils/data.test.ts
```
```bash
bun run exercises/ch-28-sequence-data.ts
```

---

## Self-Check Questions

1. Why is the decoder input shifted right relative to the target labels?
2. What does teacher forcing give us during training that we do not have during inference?
3. Why should padding tokens be ignored in the loss?
4. If loss decreases on training but validation loss increases, what is happening?
5. What does it mean if the model cannot overfit one batch?

---

## Further Reading

- [Harvard NLP — The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/) — shows the same data pipeline end-to-end.
- [Karpathy — A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) — the overfit-one-batch habit comes from here.
- [Wikipedia — Perplexity](https://en.wikipedia.org/wiki/Perplexity) — the metric and its information-theoretic meaning.
- [Stanford CS224n — Lecture notes](https://web.stanford.edu/class/cs224n/) — standard reference for sequence-modelling data pipelines.

---

## Next Chapter

**[Full Transformer](ch-29-full-transformer.md)** — train the complete encoder-decoder model on a real toy task.
