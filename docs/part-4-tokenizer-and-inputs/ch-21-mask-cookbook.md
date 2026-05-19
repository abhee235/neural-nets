# Chapter 21: Attention Mask Cookbook

> **Part 4 of 6 — Language Model Inputs**
> `src/ch-21-mask-cookbook/`

---

## What You're Building

A small mask utility layer that converts tokenizer padding masks and causal decoder masks into the additive masks consumed by attention.

This chapter exists because masking bugs are silent. The model may run, the shapes may look correct, and the loss may still refuse to improve because the model is attending to padding or future tokens.

---

## Why This Matters

Transformers use masks in three different places:

- Source padding mask: encoder and cross-attention should ignore `<pad>` tokens in the source.
- Target padding mask: decoder should ignore `<pad>` tokens in the target.
- Causal mask: decoder self-attention must not look at future target tokens.

These masks start as simple binary arrays from the tokenizer, but attention wants additive masks: `0` for allowed positions and `-Infinity` for blocked positions.

The conversion must be explicit and tested.

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

## Required Tests

- Padding IDs become `0` in the binary mask.
- Binary mask `[1, 0, 1]` becomes additive mask `[0, MASK_VALUE, 0]`.
- Padding mask expansion produces `[batch, 1, 1, keyLen]`.
- Causal mask for length 4 blocks only future positions.
- Combined decoder mask blocks both future tokens and pad tokens.
- After softmax, masked attention probabilities are exactly 0 or numerically near 0.

---

## Common Pitfalls

- Treating binary `0/1` masks as if they can be added directly to attention scores.
- Masking query positions when you meant to mask key positions.
- Forgetting the head dimension in `[batch, heads, queryLen, keyLen]`.
- Creating a target causal mask but forgetting target padding.
- Applying a causal mask to cross-attention, which should see the whole source.

---

## Self-Check Questions

1. Why does a padding mask usually block key positions rather than query positions?
2. What shape should a source padding mask have before it is added to encoder attention scores?
3. Why can additive masks be combined by addition?
4. What happens if a decoder self-attention mask includes padding but no causal mask?
5. Why should cross-attention not use a causal mask?

---

## → Next Chapter

**Ch 22: Self-Attention** — now the attention layer can consume masks with clear, tested semantics.
