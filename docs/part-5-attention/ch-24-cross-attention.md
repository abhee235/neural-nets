# Chapter 24: Cross-Attention

> **Part 5 of 6 — Attention Mechanism**
> `src/ch-24-cross-attention/`

---

## What You're Building

Cross-attention: a variant of multi-head attention where the Queries come from one sequence
(the decoder) and the Keys/Values come from a different sequence (the encoder output).
This is how the decoder "reads" the encoder's understanding of the input when generating output.

---

## Why This Matters

In a sequence-to-sequence (seq2seq) model — translation, summarization, question answering
— you have two sequences: source (e.g., French sentence) and target (e.g., English translation).

The encoder processes the full source. The decoder generates the target one token at a time.
At each decoder step, the decoder needs to ask: "what part of the source is most relevant
to what I'm generating right now?" That's exactly what cross-attention does.

Without cross-attention, the decoder has no way to look at the source — it can only look
at what it has already generated (via self-attention). Cross-attention is the bridge between
encoder and decoder.

---

## Concepts

### The Difference from Self-Attention

In **self-attention**, Q, K, V all come from the same sequence $X$:
$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

In **cross-attention**, Q comes from the **decoder** state $X_{\text{dec}}$,
while K and V come from the **encoder** output $X_{\text{enc}}$:
$$Q = X_{\text{dec}} W_Q, \quad K = X_{\text{enc}} W_K, \quad V = X_{\text{enc}} W_V$$

The formula is otherwise identical:
$$\text{CrossAttention}(X_{\text{dec}}, X_{\text{enc}}) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

### Shape Analysis

Let `src_len` = encoder sequence length, `tgt_len` = decoder sequence length:
- $Q$: `[batch, tgt_len, dHead]` (from decoder)
- $K$: `[batch, src_len, dHead]` (from encoder)
- $V$: `[batch, src_len, dHead]` (from encoder)
- Attention scores: `[batch, tgt_len, src_len]` — for each decoder position, a distribution over encoder positions
- Output: `[batch, tgt_len, dHead]` — decoder positions, enriched with encoder info

In self-attention, $\text{seq\_q} = \text{seq\_k}$ so the score matrix is square.
In cross-attention, it's rectangular: each decoder step scores all encoder positions.

### No Causal Mask for Cross-Attention

Cross-attention does **not** use a causal mask. The decoder is allowed to attend to
any position in the source (including "future" source positions relative to the current
target position) — the source is fully available.

Causal masking only applies to the decoder's **self-attention** (where the decoder
must not peek at future target tokens it hasn't generated yet).

### The Information Flow in a Full Transformer

The full flow looks like:

```
Source tokens                      Target tokens (generated so far)
     ↓                                     ↓
[Token Embedding + PE]           [Token Embedding + PE]
     ↓                                     ↓
[Encoder Block × N]              [Decoder Self-Attention (causal)]
     ↓                                     ↓
Encoder Output ────────────────→ [Cross-Attention: Q from decoder, K/V from encoder]
                                          ↓
                                    [FFN Block]
                                          ↓
                                   Next token logits
```

The encoder output is the same for every decoder layer — it's computed once and then
reused as K and V in every cross-attention layer of the decoder.

### KV-Caching Intuition (Preview)

Because K and V come from the encoder (which is fixed once the source is encoded),
they can be computed once and **cached**. This is the origin of the "KV cache" optimization
used in modern LLM inference — rather than recomputing K/V from the context at every
new token generation step, you store them. We'll revisit this in the optional post-course extensions.

---

## What to Implement

The key insight: cross-attention is implemented by making Q and (K, V) come from different sources.
The simplest approach is to generalize `MultiHeadAttention` to accept a separate `context` input.

| Symbol | Description |
|--------|-------------|
| `class CrossAttention` | Like `MultiHeadAttention` but `forward(x, context, mask?)` |
| `CrossAttention.forward(x, context, mask?)` | `x` → Q; `context` → K, V |
| `CrossAttention.parameters()` | All 4 weight matrices |

---

## TypeScript Hints

```typescript
/**
 * Cross-Attention: Q comes from `x` (decoder state),
 * K and V come from `context` (encoder output).
 *
 * The implementation is nearly identical to MultiHeadAttention.
 * The only difference: K and V use `context` instead of `x`.
 */
export class CrossAttention {
  dModel: number;
  numHeads: number;
  dHead: number;

  WQ: Linear;   // projects decoder state   → Q
  WK: Linear;   // projects encoder output  → K
  WV: Linear;   // projects encoder output  → V
  WO: Linear;   // output projection

  constructor(dModel: number, numHeads: number) {
    if (dModel % numHeads !== 0) {
      throw new Error(`dModel must be divisible by numHeads`);
    }
    this.dModel = dModel;
    this.numHeads = numHeads;
    this.dHead = dModel / numHeads;

    this.WQ = new Linear(dModel, dModel);
    this.WK = new Linear(dModel, dModel);
    this.WV = new Linear(dModel, dModel);
    this.WO = new Linear(dModel, dModel);
  }

  forward(x: Value, context: Value, mask?: Tensor): Value {
    // x:       [batch, tgt_len, dModel] — decoder query source
    // context: [batch, src_len, dModel] — encoder key/value source

    // Q from decoder, K/V from encoder
    const Q = this.WQ.forward(x);
    const K = this.WK.forward(context);  // ← note: context, not x
    const V = this.WV.forward(context);  // ← note: context, not x

    const QH = splitHeads(Q, this.numHeads);  // [batch, H, tgt_len, dHead]
    const KH = splitHeads(K, this.numHeads);  // [batch, H, src_len, dHead]
    const VH = splitHeads(V, this.numHeads);  // [batch, H, src_len, dHead]

    // Attention: QH @ KH^T is [batch, H, tgt_len, src_len] — rectangular!
    const attnOut = scaledDotProductAttention(QH, KH, VH, mask);

    const merged = mergeHeads(attnOut);  // [batch, tgt_len, dModel]
    return this.WO.forward(merged);
  }

  parameters(): Value[] {
    return [
      ...this.WQ.parameters(),
      ...this.WK.parameters(),
      ...this.WV.parameters(),
      ...this.WO.parameters(),
    ];
  }
}
```

---

## Self-Check Questions

1. Cross-attention scores have shape `[batch, H, tgt_len, src_len]`. What does entry
   `[0, 2, 5, 3]` represent? (Describe it in English — batch 0, head 2, decoder position 5, encoder position 3.)
2. Why is there no causal mask in cross-attention? What would happen if you applied one?
3. In the translation task "Il fait chaud" → "It is hot", describe what you'd expect
   a well-trained cross-attention head to look like. Which target token attends to which source token?
4. Cross-attention and self-attention have the same number of parameters (4 linear layers).
   Are the K and V projections in cross-attention initialized differently than in self-attention?
5. If the encoder outputs `[2, 20, 512]` (batch=2, src_len=20, dModel=512) and the decoder
   has `[2, 5, 512]` (tgt_len=5), what is the shape of the cross-attention score matrix?

---

## End of Part 5

The full attention mechanism is now in place:
- Scaled dot-product attention with causal masking (Ch 22)
- Multi-head attention with parallel heads (Ch 23)
- Cross-attention bridging encoder and decoder (Ch 24)

---

## → Next Part

**Part 6 — Transformer (Ch 25–30):** assemble encoder and decoder blocks, build the full
encoder-decoder transformer, then build a GPT-style decoder-only model.
