# Chapter 19: Positional Encoding

> **Part 4 of 6 — Tokenizer & Inputs**
> Source: [`src/nn/positional.ts`](../../src/nn/positional.ts)
> Tests: [`src/nn/transformer.test.ts`](../../src/nn/transformer.test.ts)
> Exercise: [`exercises/ch-19-positional-encoding.ts`](../../exercises/ch-19-positional-encoding.ts)

---

## Learning Goals

By the end of this chapter you can:

- Implement sinusoidal positional encoding with the original Transformer formula.
- Add positional encodings to embeddings so token + position information mix.
- Plot a few PE rows and recognise the wavelength pattern.
- Compare sinusoidal PE to learned PE and to RoPE / ALiBi at a high level.
- Predict PE output shape for any `[batch, seq, dModel]` input.

---

## Intuition First

Self-attention is permutation-invariant: it treats a sequence like a *set*. Without positional information, "the cat sat" and "sat the cat" produce identical representations. Positional encoding injects a deterministic, position-dependent vector that the network can latch onto. Sinusoids give the model a smooth, extrapolatable signal across many wavelengths.

---

## Mental Model

```text
  PE(pos, 2i)   = sin(pos / 10000^(2i / dModel))
  PE(pos, 2i+1) = cos(pos / 10000^(2i / dModel))

      pos = 0  → [sin(0),    cos(0),    sin(0),    cos(0), …]
      pos = 1  → [sin(1/1),  cos(1/1),  sin(1/10000^(2/d)), …]
      pos = 2  → [sin(2/1),  cos(2/1),  …]
```

---

## Concepts

### Why Attention is Order-Blind

Recall the attention formula: $\text{Attention}(Q, K, V) = \text{softmax}(QK^T / \sqrt{d_k}) V$.

Both $Q$, $K$, $V$ are computed from the same input matrix $X$ (just with different weight matrices).
If you permute the rows of $X$ (i.e., shuffle the token order), the output rows are permuted
the same way — the model just processes each position independently. It has no way to know
"this token is at position 3, that one is at position 7."

Positional encoding breaks this symmetry: after adding PE, each row of $X$ is unique to its position.

### Sinusoidal Positional Encoding

The formula from "Attention Is All You Need" (Vaswani et al., 2017):

$$PE(\text{pos}, 2i) = \sin\!\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)$$

$$PE(\text{pos}, 2i+1) = \cos\!\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)$$

Where:
- $\text{pos} \in \{0, 1, 2, \ldots, \text{seqLen}-1\}$ — the position in the sequence
- $i \in \{0, 1, 2, \ldots, d_{\text{model}}/2 - 1\}$ — the dimension index (pair index)
- Even dimensions use $\sin$, odd dimensions use $\cos$

Each position gets a vector in $\mathbb{R}^{d_{\text{model}}}$. Even indices use sine waves,
odd indices use cosine waves. Different dimensions oscillate at different frequencies.

### Why This Works: Frequencies and Relative Positions

The key insight: **relative positions can be expressed as linear transformations of the PE vectors**.

The denominator $10000^{2i/d_{\text{model}}}$ creates wavelengths that scale geometrically:
- Dimension 0: wavelength = $2\pi$ (oscillates every ~6 positions)
- Dimension $d/4$: wavelength = $2\pi \cdot 100$ (~628 positions)
- Dimension $d/2$: wavelength = $2\pi \cdot 10000$ (~62,832 positions)

Lower dimensions capture short-range position information; higher dimensions capture
long-range information. Together, any position up to $10000$+ can be uniquely represented.

The sinusoidal choice specifically allows the model to attend by relative position:
$PE(\text{pos} + k)$ can be expressed as a linear function of $PE(\text{pos})$, using
the sine/cosine angle addition formulas. This means "the token 3 positions ahead of me"
has a predictable PE relationship to "me."

### Visualization

Plot `PE` as a heatmap (seqLen × dModel). You'll see:
- Lower dimensions: high-frequency stripes (alternating quickly)
- Higher dimensions: low-frequency gradients (changing slowly)
- The combination makes each row unique

This is a beautiful example of frequency-domain position encoding — essentially Fourier features.

### Learned vs Sinusoidal

The original paper tested both:
- **Sinusoidal (fixed):** no parameters, generalizes to sequence lengths beyond training
- **Learned:** $\text{seqLen} \times d_{\text{model}}$ learnable parameters, potentially
  more expressive but doesn't generalize to longer sequences

Both produce similar results. Modern models (GPT-2) use learned PE; the original transformer
used sinusoidal. We implement sinusoidal because it requires no training and is more instructive.

### RoPE and ALiBi (Modern Variants)

For context, modern LLMs use more sophisticated positional schemes:
- **RoPE** (Rotary Position Embedding, LLaMA/GPT-NeoX): applies rotation to Q/K inside attention
- **ALiBi** (Attention with Linear Biases, MPT): adds a linear position bias to attention scores

These extend well to longer contexts than the original sinusoidal PE. We implement sinusoidal
for clarity, and you can return to RoPE after completing the full transformer.

---

## What to Implement

| Symbol | Description |
|--------|-------------|
| `sinusoidalPE(maxSeqLen, dModel)` | Compute the full PE matrix `[maxSeqLen, dModel]` using the formula |
| `class PositionalEncoding` | Wraps the PE matrix; `forward(x)` adds PE to token embeddings `x` |
| `PositionalEncoding.forward(x)` | Input: `[batch, seq, dModel]`. Adds PE for positions `[0..seq-1]`. |

---

## TypeScript Hints

```typescript
/**
 * Compute the sinusoidal positional encoding matrix.
 * Shape: [maxSeqLen, dModel]
 * PE[pos, 2i]   = sin(pos / 10000^(2i/dModel))
 * PE[pos, 2i+1] = cos(pos / 10000^(2i/dModel))
 */
export function sinusoidalPE(maxSeqLen: number, dModel: number): Tensor {
  const data: number[] = [];

  for (let pos = 0; pos < maxSeqLen; pos++) {
    for (let dim = 0; dim < dModel; dim++) {
      // Which dimension pair are we in? (i = floor(dim / 2))
      const i = Math.floor(dim / 2);
      // The frequency denominator: 10000^(2i / dModel)
      const denominator = Math.pow(10000, (2 * i) / dModel);

      if (dim % 2 === 0) {
        // Even dimension: use sine
        data.push(Math.sin(pos / denominator));
      } else {
        // Odd dimension: use cosine
        data.push(Math.cos(pos / denominator));
      }
    }
  }

  return createTensor(data, [maxSeqLen, dModel]);
}

export class PositionalEncoding {
  pe: Tensor;   // [maxSeqLen, dModel] — fixed, not a parameter
  dModel: number;

  constructor(maxSeqLen: number, dModel: number) {
    this.dModel = dModel;
    this.pe = sinusoidalPE(maxSeqLen, dModel);
  }

  // x: Value wrapping [batch, seqLen, dModel]
  // Returns x + pe[0..seqLen] broadcast across batch
  forward(x: Value): Value {
    const seqLen = x.data.shape[1]!;
    // Slice PE to seqLen, then broadcast add across batch dimension
    const peSlice = /* first seqLen rows of this.pe */ ...;
    return x.add(new Value(peSlice));  // PE is not a learnable parameter
  }

  // PE is fixed, not learnable — no parameters to return
  parameters(): Value[] {
    return [];
  }
}
```

---

## Common Pitfalls

- Forgetting that PE is added to embeddings, not concatenated — shapes must match.
- Using base 10000 with `i` in the wrong units; the exponent is `2i / dModel`.
- Computing PE inside the training loop every step; cache it once.
- Allowing `pos` to exceed `maxSeqLen`; crop or extend the table explicitly.
- Scaling embeddings by `√dModel` but forgetting to leave PE unscaled (or vice versa).

---

## How to Verify

Run the tests and the exercise. Both should pass cleanly with no warnings:

```bash
bun test src/nn/transformer.test.ts
```
```bash
bun run exercises/ch-19-positional-encoding.ts
```

---

## Self-Check Questions

1. Compute `PE(0, 0)` and `PE(1, 0)` by hand. (Position 0 and 1, dimension 0.)
2. What is `PE(pos, 0)` for pos = 0, 1, 2, 3? What function does this trace?
3. Why does the denominator `10000^(2i/dModel)` grow so large for high i?
   What does this mean for the periodicity of high-dimensional PE components?
4. If `dModel = 8` and `maxSeqLen = 10`, what is the shape of the PE matrix?
   Print it and verify it matches the formula.
5. After adding PE to embeddings, can two different positions ever have the same combined
   vector? Why or why not? (Assume the embedding table has random initial values.)

---

## Further Reading

- [Vaswani et al. — Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) — the original transformer paper; every formula in Parts 5–6 comes from it.
- [Su et al. — RoPE: Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — the modern alternative used by LLaMA and others.
- [Press et al. — ALiBi: Train Short, Test Long](https://arxiv.org/abs/2108.12409) — another modern alternative; bias-based, no learned PE.
- [Jay Alammar — The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — visual treatment that includes positional encodings.

---

## Next Chapter

**[LayerNorm & Dropout](ch-20-layernorm-dropout.md)** — make activations stable and add the only regulariser inside transformer blocks.
