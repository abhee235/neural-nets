# Chapter 11: Activation Functions

> **Part 3 of 6 — Neural Net Primitives**
> `src/ch-11-activation-functions/`

---

## What You're Building

The standard activation functions used in neural networks: `relu`, `sigmoid`, `tanh`, `softmax`,
and `gelu`. Each is implemented using the tensor primitives from Part 1, with autograd-compatible
backward passes. These are the non-linearities that give neural networks their power.

---

## Why This Matters

Without activation functions, a stack of linear layers collapses into a single linear function —
no matter how many layers you add, the model can only learn linear relationships. Activations
introduce non-linearity, allowing the network to learn curves, XOR, language, and everything else.

In the transformer, GELU is used inside the Feed-Forward Block (Ch 25). Softmax is the
final step of attention (Ch 22) and of the language model head (Ch 29).

---

## Concepts

### Why Non-linearity?

Consider two linear layers: $y = W_2(W_1 x) = (W_2 W_1) x = W_{\text{combined}} x$.
They are equivalent to one linear layer. No matter how many you stack, the composition
is still linear. A linear model can only separate linearly separable classes.

Activation functions break this: $y = W_2 \cdot \sigma(W_1 x)$ is no longer linear in $x$,
so the model can represent arbitrary functions (given enough width).

### ReLU — Rectified Linear Unit

$$\text{ReLU}(x) = \max(0, x) = \begin{cases} x & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$

**Derivative:**
$$\text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$

**Why it's popular:** simple, fast, gradients don't vanish for positive inputs.
**Pitfall:** "dead ReLU" — neurons whose input is always negative receive zero gradient
and never learn. He initialization (Ch 13) mitigates this.

### Sigmoid

$$\sigma(x) = \frac{1}{1 + e^{-x}} \in (0, 1)$$

**Derivative:**
$$\sigma'(x) = \sigma(x)(1 - \sigma(x))$$

**Use case:** binary classification output, LSTM gates.
**Pitfall:** vanishing gradients — for large $|x|$, $\sigma'(x) \approx 0$. Deep networks
with sigmoid saturate and training stalls. This is why ReLU replaced sigmoid in hidden layers.

### Tanh

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} \in (-1, 1)$$

**Derivative:**
$$\tanh'(x) = 1 - \tanh^2(x)$$

Note: $\tanh(x) = 2\sigma(2x) - 1$. Tanh is a rescaled sigmoid centered at 0 instead of 0.5.
Zero-centered outputs help training slightly compared to sigmoid.

### Softmax

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

Softmax converts a vector of arbitrary real numbers ("logits") into a probability distribution
(all values in $(0, 1)$, summing to exactly 1).

**Numerically stable implementation** (always subtract max first):

$$\text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$$

This is mathematically identical (max cancels) but prevents `exp(1000) = Infinity`.

**Backward pass of softmax:** The Jacobian is $J_{ij} = s_i(\delta_{ij} - s_j)$ where $s = \text{softmax}(x)$.
In practice, softmax is almost always followed immediately by cross-entropy loss.
Their combined gradient is beautifully simple: $\text{grad} = s - y_{\text{true}}$
(just subtract the one-hot label from the softmax output). Implement this combined form in Ch 12.

### GELU — Gaussian Error Linear Unit

$$\text{GELU}(x) = x \cdot \Phi(x)$$

where $\Phi(x)$ is the CDF of the standard normal distribution. In practice, the approximation:

$$\text{GELU}(x) \approx 0.5x \left(1 + \tanh\!\left(\sqrt{\frac{2}{\pi}} \left(x + 0.044715x^3\right)\right)\right)$$

**Why transformers use GELU:** It's smoother than ReLU (no hard zero for $x < 0$, just a
soft gate). Empirically, transformers trained with GELU outperform those with ReLU slightly.

---

## What to Implement

| Function | Notes |
|----------|-------|
| `relu(t)` | Uses `maximum(t, zeros(t.shape))` from Ch 06 |
| `sigmoid(t)` | Uses `exp`, `add`, `div` from Ch 03/06 |
| `tanh(t)` | Uses `tanh` from Ch 06 |
| `softmax(t, axis?)` | Numerically stable; default axis = last axis |
| `gelu(t)` | Use the tanh approximation |
| `leakyRelu(t, alpha?)` | Variant of ReLU: negative side has small slope `alpha` (default 0.01) |

---

## TypeScript Hints

```typescript
// Softmax along the last axis (most common use: classification, attention)
export function softmax(t: Tensor, axis = t.ndim - 1): Tensor {
  // Stability: subtract max along the axis before exp
  const maxVals = max(t, axis, /* keepDims= */ true);
  const shifted = sub(t, maxVals);   // broadcast: (t - max)
  const exps = exp(shifted);
  const sumExps = sum(exps, axis, /* keepDims= */ true);
  return div(exps, sumExps);         // normalize to sum to 1
}

// GELU approximation (used in GPT-2 and most modern transformers)
export function gelu(t: Tensor): Tensor {
  // 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
  const SQRT_2_OVER_PI = Math.sqrt(2 / Math.PI);  // ≈ 0.7978845608
  const x3 = pow(t, 3);
  const inner = addScalar(mulScalar(x3, 0.044715), 0);  // 0.044715 * x³
  const tanhArg = mulScalar(add(t, inner), SQRT_2_OVER_PI);
  return mulScalar(mul(t, addScalar(tanh(tanhArg), 1)), 0.5);
}
```

---

## Self-Check Questions

1. `relu([-3, -1, 0, 2, 5])` — what is the output?
2. `sigmoid(0)` = ? `sigmoid(Infinity)` ≈ ? `sigmoid(-Infinity)` ≈ ?
3. Compute `softmax([1, 2, 3])` by hand (subtract max first). Does it sum to 1?
4. Why is `softmax([1, 2, 3])` === `softmax([0, 1, 2])`? Show algebraically.
5. A model outputs logits `[2, -1, 0.5]` for a 3-class problem. Apply softmax.
   Which class is predicted? What is its probability?

---

## → Next Chapter

**Ch 12: Loss Functions** — MSE, Cross-Entropy, and Binary Cross-Entropy, including
the numerically stable combined softmax + cross-entropy backward pass.
