# Chapter 21: Mask Cookbook

> **Part 4 of 6 — Tokenizer & Inputs**
> Source: [`src/tokenizer/masks.ts`](../../src/tokenizer/masks.ts)
> Tests: [`src/tokenizer/masks.test.ts`](../../src/tokenizer/masks.test.ts)
> Exercise: [`exercises/ch-21-masks.ts`](../../exercises/ch-21-masks.ts)

---

## Learning Goals

By the end of this chapter you can:

- Distinguish binary masks (`1` allowed, `0` blocked) from additive masks (`0` allowed, `-∞` blocked).
- Convert a binary mask to an additive mask with `(1 − mask) * −∞`.
- Build a causal (lower-triangular) mask of shape `[seq, seq]`.
- Expand a padding mask of shape `[batch, seq]` to `[batch, 1, 1, seq]` for multi-head attention.
- Combine padding mask + causal mask with element-wise min (or sum, in additive form).

---

## Intuition First

Attention computes a softmax over scores. To **forbid** a position, set its score to `-∞` before the softmax — `exp(-∞) = 0`. Masks are the language we use to express "please don't look here". Two reasons to mask:

- **Padding**: positions filled with `<pad>` carry no real information.
- **Causality**: when generating, position `t` must not see positions `> t`.

---

## Mental Model

```text
  binary mask                       additive mask
  ─────────────                     ─────────────
  1 = allowed                       0    = allowed
  0 = blocked                       −∞   = blocked

  scores = QKᵀ / √d_k
  scores = scores + additive_mask
  weights = softmax(scores)      ← blocked positions contribute 0
```

---

## Concepts

### Binary Mask

The tokenizer produces a binary mask:

```text
1 = real token
0 = padding token
```

Example:

```text
ids:  [BOS, h, i, EOS, PAD, PAD]
mask: [  1, 1, 1,   1,   0,   0]
```

Binary masks are easy for humans to read, but they are not what softmax consumes directly.

### Additive Mask

Attention scores are real numbers before softmax. To block a position, add a huge negative number:

```text
0         = allowed
-Infinity = blocked
```

Why this works:

$$\exp(-\infty) = 0$$

So after softmax, blocked positions receive probability 0.

### Padding Mask Shape

For self-attention, scores usually have shape:

```text
[batch, heads, queryLen, keyLen]
```

A tokenizer mask usually has shape:

```text
[batch, keyLen]
```

To apply it to attention scores, reshape/broadcast it to:

```text
[batch, 1, 1, keyLen]
```

This means: for every head and every query position, block padded key positions.

### Causal Mask Shape

The causal mask has shape:

```text
[queryLen, keyLen]
```

For decoder self-attention, `queryLen = keyLen = targetLen`, and the mask blocks `j > i`:

```text
[[0, -∞, -∞],
 [0,  0, -∞],
 [0,  0,  0]]
```

Broadcast it to:

```text
[1, 1, targetLen, targetLen]
```

### Combining Masks

Additive masks combine by addition:

```text
combined = paddingMask + causalMask
```

Any position blocked by either mask becomes `-Infinity`.

For practical code, avoid `-Infinity + -Infinity` if it creates `NaN` in your helper operations. You can use a large negative number like `-1e9` instead. The softmax result is effectively the same for learning-scale logits.

---

## What to Implement

| Symbol | Description |
|--------|-------------|
| `binaryMaskFromIds(ids, padId)` | Return `1` for non-pad IDs and `0` for pad IDs. |
| `toAdditiveMask(binaryMask)` | Convert `1/0` mask to `0/-Infinity`. |
| `expandPaddingMask(mask)` | Convert `[batch, keyLen]` to `[batch, 1, 1, keyLen]`. |
| `causalMask(queryLen, keyLen?)` | Return additive causal mask `[queryLen, keyLen]`. |
| `expandCausalMask(mask)` | Convert `[queryLen, keyLen]` to `[1, 1, queryLen, keyLen]`. |
| `combineMasks(...masks)` | Add broadcast-compatible additive masks. |
| `makeDecoderSelfAttentionMask(targetIds, padId)` | Combine target padding mask and causal mask. |
| `makeEncoderSelfAttentionMask(sourceIds, padId)` | Source padding mask for encoder self-attention. |
| `makeCrossAttentionMask(sourceIds, padId)` | Source padding mask for decoder cross-attention. |

---

## TypeScript Hints

```typescript
const MASK_VALUE = -1e9;

export function toAdditiveMask(binaryMask: Tensor): Tensor {
  return applyFn(binaryMask, value => value === 1 ? 0 : MASK_VALUE);
}

export function expandPaddingMask(mask: Tensor): Tensor {
  // [batch, keyLen] -> [batch, 1, 1, keyLen]
  const [batchSize, keyLen] = mask.shape as [number, number];
  return reshape(mask, [batchSize, 1, 1, keyLen]);
}

export function causalMask(queryLen: number, keyLen = queryLen): Tensor {
  const data: number[] = [];
  for (let i = 0; i < queryLen; i++) {
    for (let j = 0; j < keyLen; j++) {
      data.push(j > i ? MASK_VALUE : 0);
    }
  }
  return createTensor(data, [queryLen, keyLen]);
}
```

---

## Common Pitfalls

- Multiplying scores by a binary mask — the softmax then renormalises and the blocked positions still get weight.
- Using `−∞` literally; use a large negative finite number (`-1e9`) to avoid `NaN` from `0 * ∞`.
- Forgetting to expand the padding mask to `[batch, 1, 1, seq]` so it broadcasts over heads and query positions.
- Building the causal mask once per step inside the loop; build it once per max sequence length.
- Combining binary and additive masks without converting one to the other first.

---

## How to Verify

Run the tests and the exercise. Both should pass cleanly with no warnings:

```bash
bun test src/tokenizer/masks.test.ts
```
```bash
bun run exercises/ch-21-masks.ts
```

---

## Self-Check Questions

1. Why does a padding mask usually block key positions rather than query positions?
2. What shape should a source padding mask have before it is added to encoder attention scores?
3. Why can additive masks be combined by addition?
4. What happens if a decoder self-attention mask includes padding but no causal mask?
5. Why should cross-attention not use a causal mask?

---

## Further Reading

- [Vaswani et al. — Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) — the original transformer paper; every formula in Parts 5–6 comes from it.
- [Jay Alammar — The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — diagrams of where masks plug into attention.
- [Andrej Karpathy — Let's build GPT (video)](https://www.youtube.com/watch?v=kCc8FmEb1nY) — live-codes the causal mask and explains the additive trick.
- [PyTorch — `nn.Transformer` source](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html) — reference for how the same masks are wired in production.

---

## Next Chapter

**[Self-Attention](../part-5-attention/ch-22-self-attention.md)** — combine Q, K, V, scaling, and masks into the heart of the transformer.
