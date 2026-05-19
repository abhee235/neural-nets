# Chapter 22: Self-Attention

> **Part 5 of 6 — Attention Mechanism**
> `src/ch-22-self-attention/`

---

## What You're Building

Scaled dot-product self-attention: the core mechanism of the transformer. Given a sequence
of token vectors, each token learns to "attend to" other tokens in the sequence and pull in
relevant information. Implements Q, K, V projections, the attention score formula, and
optional causal masking.

Before this chapter, complete Ch 21. Attention masking is simple only when binary masks,
additive masks, padding masks, and causal masks have precise shapes and meanings.

---

## Why This Matters

This is the most important chapter in the entire course. Attention is *why* transformers
outperform all previous architectures: it lets every token directly attend to every other
token in the sequence, capturing long-range dependencies in a single operation.

Without attention, a language model only sees a fixed context window (RNNs have this problem).
With attention, a token at position 512 can directly attend to token at position 0.

---

## Concepts

### The Intuition

Imagine you're reading the sentence: "The bank by the river was steep."
When you process the word "bank", you need to figure out its meaning (financial or riverbank?).
You look at other words — "river" — and decide it's a riverbank. You've "attended" to "river."

Self-attention formalizes this:
- Each token asks: **"What am I looking for?"** → the Query (Q)
- Each token says: **"Here's what I have to offer"** → the Key (K)
- Each token says: **"Here's the content I'll share if selected"** → the Value (V)

Tokens with similar Queries and Keys attend strongly to each other.

### The Formula

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Step by step:

**1. Project inputs to Q, K, V:**

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

Where $X \in \mathbb{R}^{n \times d_{\text{model}}}$ is the input, and $W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_k}$ are learnable projection matrices.

**2. Compute attention scores:**

$$\text{scores} = \frac{QK^T}{\sqrt{d_k}}$$

$Q$: `[batch, seq, d_k]`, $K^T$: `[batch, d_k, seq]` → scores: `[batch, seq, seq]`

Each entry `scores[b, i, j]` measures how much token $i$ attends to token $j$.
The $\sqrt{d_k}$ scaling prevents the dot products from growing too large (which would push
softmax into a saturated, near-zero-gradient region).

**3. Apply causal mask (decoder only):**

For autoregressive models (predicting next token), token $i$ must not see tokens $j > i$:
$$\text{maskedScores}[i, j] = \begin{cases} \text{scores}[i, j] & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

After softmax, $-\infty$ becomes $0$: masked positions get zero attention weight.

**4. Softmax to get attention weights:**

$$A = \text{softmax}(\text{scores}) \quad \in \mathbb{R}^{n \times n}$$

Row $i$ of $A$ sums to 1. It's a probability distribution: how much does token $i$ attend
to each other token?

**5. Weighted sum of Values:**

$$\text{output} = A \cdot V \quad \in \mathbb{R}^{n \times d_k}$$

Each output token is a weighted combination of Value vectors. The weights come from attention.

### The Scaling Factor $\sqrt{d_k}$

Why divide by $\sqrt{d_k}$?

When $d_k$ dimensions are summed in the dot product $q \cdot k$, the expected value grows
as $O(d_k)$ for random unit vectors. With $d_k = 64$, the dot products are ~8× larger than
with $d_k = 1$. This pushes softmax into the saturation region where gradients are tiny.

Dividing by $\sqrt{d_k}$ normalizes the dot products back to unit scale regardless of $d_k$.

### Causal Mask

The causal (or look-ahead) mask is an upper-triangular matrix of $-\infty$:

```
[[  0, -∞, -∞, -∞],   ← token 0 can only attend to itself
 [  0,  0, -∞, -∞],   ← token 1 can attend to 0 and 1
 [  0,  0,  0, -∞],   ← token 2 can attend to 0, 1, and 2
 [  0,  0,  0,  0]]   ← token 3 can attend to all
```

After adding this to the scores and applying softmax, each token only attends to itself
and tokens before it — never to future tokens.

---

## What to Implement

| Symbol | Description |
|--------|-------------|
| `causalMask(seqLen)` | Return upper-triangular `[seqLen, seqLen]` mask of 0/-Infinity |
| `scaledDotProductAttention(Q, K, V, mask?)` | The core attention formula |
| `class SelfAttention` | Full self-attention head: W_Q, W_K, W_V projections + attention |
| `SelfAttention.forward(x, mask?)` | Input `[batch, seq, dModel]`. Output `[batch, seq, dHead]`. |
| `SelfAttention.parameters()` | Return all three projection matrices |

---

## TypeScript Hints

```typescript
/**
 * Causal mask for autoregressive (decoder-style) attention.
 * Returns a [seqLen, seqLen] tensor where position [i, j] is:
 *   0     if j <= i (token i CAN attend to token j)
 *   -Inf  if j > i  (token i CANNOT see future token j)
 */
export function causalMask(seqLen: number): Tensor {
  const data: number[] = [];
  for (let i = 0; i < seqLen; i++) {
    for (let j = 0; j < seqLen; j++) {
      // Upper triangle (j > i) is masked with -Infinity
      data.push(j > i ? -Infinity : 0);
    }
  }
  return createTensor(data, [seqLen, seqLen]);
}

/**
 * Scaled dot-product attention.
 * Q: [batch, seq_q, dHead]
 * K: [batch, seq_k, dHead]
 * V: [batch, seq_k, dHead]
 * mask: [seq_q, seq_k] or [batch, seq_q, seq_k] — optional, additive mask
 * returns: [batch, seq_q, dHead]
 */
export function scaledDotProductAttention(
  Q: Value, K: Value, V: Value, mask?: Tensor
): Value {
  const dHead = Q.data.shape[Q.data.ndim - 1]!;

  // scores = Q @ K^T / sqrt(dHead)
  // Q: [batch, seq_q, dHead], K^T: [batch, dHead, seq_k]
  const KT = Q /* ... transpose K */;
  const scores = Q.matMul(KT).mulScalar(1 / Math.sqrt(dHead));

  // Add mask (causal or padding mask)
  const maskedScores = mask ? scores.add(new Value(mask)) : scores;

  // Attention weights: softmax over the last axis (over seq_k)
  const attnWeights = maskedScores.softmax(/* axis= */ -1);

  // Weighted sum of values
  return attnWeights.matMul(V);
}
```

---

## Self-Check Questions

1. For a sequence of length 4, write out the causal mask matrix.
2. Why does attention compute $QK^T$ (dot products between Q and K^T) rather than,
   say, comparing Q and K with a learned similarity function?
3. After softmax, row $i$ of the attention matrix sums to 1. What does this mean
   physically? What if all attention weights were equal?
4. Without the $\sqrt{d_k}$ scaling, softmax of large dot products approaches a
   one-hot distribution. What effect does this have on gradients?
5. For `Q, K, V` of shape `[2, 6, 64]` (batch=2, seq=6, dHead=64):
   what is the shape of the attention scores before softmax?
   What is the shape of the final attention output?

---

## → Next Chapter

**Ch 23: Multi-Head Attention** — run H attention heads in parallel and combine them,
allowing the model to attend to different relationships simultaneously.
