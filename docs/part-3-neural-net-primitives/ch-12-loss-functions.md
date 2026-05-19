# Chapter 12: Loss Functions

> **Part 3 of 6 — Neural Net Primitives**
> `src/ch-12-loss-functions/`

---

## What You're Building

The standard loss functions: Mean Squared Error (MSE), Cross-Entropy, and Binary
Cross-Entropy — each with a correct, numerically stable implementation and autograd support.

---

## Why This Matters

The loss function is what training optimizes. It's the single number that tells the model
"how wrong you are." Different tasks need different losses:
- **Regression** (predict a number): MSE
- **Multi-class classification** (which of N classes?): Cross-Entropy
- **Binary classification** (yes/no): Binary Cross-Entropy

In the transformer (Ch 29), you use Cross-Entropy between the predicted token distribution
and the true next token at every position. Getting the loss right — especially numerically
stable log computation — is essential.

---

## Concepts

### Mean Squared Error (MSE)

For regression: predicting continuous values like house prices or temperature.

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i^{\text{pred}} - y_i^{\text{true}})^2$$

**Gradient w.r.t. predictions:**
$$\frac{\partial \text{MSE}}{\partial y_i^{\text{pred}}} = \frac{2}{n}(y_i^{\text{pred}} - y_i^{\text{true}})$$

Intuition: the gradient pushes predictions toward targets. Large errors get larger gradients.

### Cross-Entropy Loss

For multi-class classification: given probabilities $p \in (0, 1)^C$ (output of softmax)
and a one-hot true label $y \in \{0, 1\}^C$:

$$\mathcal{L}_{\text{CE}} = -\sum_{c=1}^{C} y_c \log(p_c) = -\log(p_{\text{true class}})$$

Since $y$ is one-hot, all terms except the true class are zero. The loss is simply the negative
log-probability assigned to the correct class.

**Intuition:** A probability of 1.0 for the correct class → loss = 0 (perfect).
A probability of 0.01 for the correct class → loss = $-\log(0.01) \approx 4.6$ (terrible).
The model is penalized by how surprised it was to see the true label.

**Numerical stability:** Never compute `log(softmax(logits))` separately. It's numerically
unstable because softmax can produce very small values, and `log` of a small number has large
error. Instead, use the **log-sum-exp trick**:

$$\log(\text{softmax}(x_i)) = x_i - \log\!\left(\sum_j e^{x_j}\right) = x_i - \text{logsumexp}(x)$$

$$\text{logsumexp}(x) = \max(x) + \log\!\left(\sum_j e^{x_j - \max(x)}\right)$$

The combined function `crossEntropyFromLogits(logits, trueLabels)` (operating on raw logits
before softmax) is the canonical way to compute cross-entropy loss in ML frameworks.

**Gradient of combined softmax + cross-entropy:**
This is famously simple. For logits $z$, softmax probabilities $p = \text{softmax}(z)$,
and one-hot label $y$:

$$\frac{\partial \mathcal{L}}{\partial z_i} = p_i - y_i$$

That's it. The gradient is just "predicted probability minus actual label."
Positive gradient → prediction was too high. Negative → too low.

### Binary Cross-Entropy (BCE)

For binary classification (one output neuron with sigmoid activation):

$$\mathcal{L}_{\text{BCE}} = -\left[ y \log(p) + (1 - y) \log(1 - p) \right]$$

Where $p = \sigma(z) \in (0, 1)$ is the predicted probability of class 1.

**Numerically stable form** (from logits directly):

$$\mathcal{L}_{\text{BCE-logits}}(z, y) = \max(z, 0) - z \cdot y + \log(1 + e^{-|z|})$$

This avoids `exp(z)` overflow for large $z$.

---

## What to Implement

| Function | Description |
|----------|-------------|
| `mseLoss(pred, target)` | Mean squared error. Both `pred` and `target` same shape. Returns scalar. |
| `crossEntropyLoss(probs, labels)` | CE from softmax probabilities + one-hot labels. Clips probs for stability. |
| `crossEntropyFromLogits(logits, labels)` | Numerically stable CE directly from raw logits using log-sum-exp. |
| `binaryCrossEntropyLoss(probs, labels)` | BCE from sigmoid probabilities + binary labels {0, 1}. |
| `binaryCrossEntropyFromLogits(logits, labels)` | Numerically stable BCE from raw logits. |
| `logsumexp(t, axis?)` | The log-sum-exp reduction for numerical stability. |

---

## TypeScript Hints

```typescript
// Numerically stable log-sum-exp
export function logsumexp(t: Tensor, axis = t.ndim - 1): Tensor {
  const maxVals = max(t, axis, /* keepDims= */ true);
  const shifted = sub(t, maxVals);     // broadcast subtract max
  const logSumShifted = log(sum(exp(shifted), axis, /* keepDims= */ true));
  return squeeze(add(maxVals, logSumShifted), axis);
}

// Cross-entropy from logits — the canonical form
export function crossEntropyFromLogits(logits: Tensor, oneHotLabels: Tensor): Tensor {
  // log(softmax(z))_i = z_i - logsumexp(z)
  const logProbs = sub(logits, unsqueeze(logsumexp(logits), logits.ndim - 1));
  // -sum(y * log_probs) — only the true class contributes
  const perSampleLoss = neg(sum(mul(oneHotLabels, logProbs), /* axis= */ -1));
  return mean(perSampleLoss);  // average over batch
}

// MSE — simple but always clip and validate shapes
export function mseLoss(pred: Tensor, target: Tensor): Tensor {
  assertShape(pred, target.shape);  // from Ch 01
  const diff = sub(pred, target);
  return mean(mul(diff, diff));     // mean of squared differences
}
```

---

## Self-Check Questions

1. MSE of predictions `[1, 2, 4]` vs targets `[1, 3, 3]`. Compute by hand.
2. Cross-entropy loss when the model assigns probability 0.9 to the correct class vs 0.1.
   Which is higher? By how much?
3. Why is `log(softmax(logits))` numerically unstable for very large or very small logits?
   Give a concrete example where it fails.
4. For logits `[3, 1, 0.5]` and true class 0, compute the cross-entropy loss using
   the log-sum-exp trick. Verify it matches `−log(softmax([3,1,0.5])[0])`.
5. The gradient of softmax+CE is `p - y`. For logits `[3, 1, 0.5]` and true class 0,
   what is the gradient vector? What does it mean for each logit?

---

## → Next Chapter

**Ch 13: Linear Layer** — the fundamental building block of neural networks: $y = xW + b$
with proper weight initialization and gradient support.
