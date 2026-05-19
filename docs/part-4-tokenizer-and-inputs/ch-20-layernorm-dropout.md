# Chapter 20: LayerNorm & Dropout

> **Part 4 of 6 — Tokenizer & Inputs**
> Source: [`src/nn/layernorm.ts`](../../src/nn/layernorm.ts) · [`src/nn/dropout.ts`](../../src/nn/dropout.ts)
> Tests: [`src/nn/transformer.test.ts`](../../src/nn/transformer.test.ts)
> Exercise: [`exercises/ch-20-layernorm-dropout.ts`](../../exercises/ch-20-layernorm-dropout.ts)

---

## Learning Goals

By the end of this chapter you can:

- Implement LayerNorm: per-token mean/variance with learnable `γ`, `β`.
- Implement inverted Dropout: zero a fraction `p` and scale the rest by `1/(1−p)`.
- Switch Dropout between training (active) and evaluation (no-op).
- Place LayerNorm in pre-norm position inside a transformer block.
- Identify which axes LayerNorm reduces over (the last one) versus BatchNorm.

---

## Intuition First

Activations inside a deep network drift in scale over training; LayerNorm forces every token's feature vector back to mean 0 and variance 1, then lets the network re-stretch with learnable `γ`/`β`. It is the thermostat of the network.

Dropout is the opposite: it deliberately injects noise during training by zeroing random units. The network learns redundant pathways; at evaluation time we use the full network and trust the redundancy.

---

## Mental Model

```text
  LayerNorm (per token):
      μ = mean(x, axis=-1)
      σ² = var(x, axis=-1)
      x̂ = (x − μ) / √(σ² + ε)
      y  = γ · x̂ + β

  Dropout (inverted, training):
      mask = bernoulli(1 − p)
      y    = (x · mask) / (1 − p)
  Dropout (eval):
      y    = x
```

---

## Concepts

### Layer Normalization

**The Problem:** As data flows through many layers, the distribution of activations shifts.
Early layers update their weights, changing what they output, which shifts the distribution
that later layers receive as input. Each layer has to constantly re-adapt to its input
distribution — this is called **internal covariate shift**. It slows training.

**BatchNorm** (used in CNNs): normalizes each feature across the batch dimension.
Problem: doesn't work well for variable-length sequences (different tokens in different
batch positions), and requires large batches.

**LayerNorm** (used in transformers): normalizes each token's feature vector independently.
For each position `[batch, seq, :]`, normalize over the `dModel` dimension:

$$\mu = \frac{1}{d} \sum_{i=1}^{d} x_i \qquad \sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2$$

$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \varepsilon}}$$

$$y_i = \gamma \hat{x}_i + \beta$$

Where:
- $\mu, \sigma^2$ are computed independently for each token (each row of `[batch, seq, dModel]`)
- $\varepsilon = 10^{-5}$ prevents division by zero
- $\gamma \in \mathbb{R}^d$ (scale parameter, initialized to 1) — learnable
- $\beta \in \mathbb{R}^d$ (shift parameter, initialized to 0) — learnable

After normalization, $\hat{x}$ has mean 0 and std 1. Then the learnable $\gamma, \beta$
allow the network to rescale and shift: the normalization is undoable if the network wants
to. This preserves the network's representational capacity while improving optimization.

### Pre-Norm vs Post-Norm

The original transformer (2017) used **Post-Norm**: apply LayerNorm after the residual:
```
x = LayerNorm(x + sublayer(x))
```

Modern transformers (GPT-2 and all successors) use **Pre-Norm**: apply LayerNorm before:
```
x = x + sublayer(LayerNorm(x))
```

Pre-Norm is more stable at the start of training and doesn't require warm-up learning rate
schedules. We implement pre-norm.

### Dropout

During training, randomly set each activation to 0 with probability $p$ (the "drop rate").
Scale the remaining activations by $\frac{1}{1-p}$ to preserve the expected value.

$$\text{Dropout}(x_i) = \begin{cases} 0 & \text{with probability } p \\ \frac{x_i}{1-p} & \text{with probability } 1-p \end{cases}$$

This is **inverted dropout** — the scaling happens during training (not inference).
At inference time, no dropout is applied and no scaling is needed.

**Why does dropout help?**
It forces the network to not rely on any single neuron. Each training step uses a
different random subset of neurons. The network learns redundant representations —
multiple pathways to the same answer. This prevents memorization (overfitting).

Typical dropout rates: 0.1 for transformers (drop 10% of activations).

**Important:** Dropout must be disabled at inference time. Your `Dropout` class needs
a `training` flag.

---

## What to Implement

| Symbol | Description |
|--------|-------------|
| `class LayerNorm` | Constructor: `(dModel, eps?)`. Learnable `gamma`, `beta`. |
| `LayerNorm.forward(x)` | Normalize last dimension; apply gamma and beta. Returns `Value`. |
| `LayerNorm.parameters()` | Returns `[this.gamma, this.beta]`. |
| `class Dropout` | Constructor: `(rate)`. Has `training` flag. |
| `Dropout.forward(x)` | Apply inverted dropout during training; identity during eval. |
| `Dropout.train()` | Enable training mode. |
| `Dropout.eval()` | Disable dropout (eval mode). |

---

## TypeScript Hints

```typescript
export class LayerNorm {
  dModel: number;
  eps: number;
  gamma: Value;   // scale: shape [dModel], initialized to ones
  beta: Value;    // shift: shape [dModel], initialized to zeros

  constructor(dModel: number, eps = 1e-5) {
    this.dModel = dModel;
    this.eps = eps;
    this.gamma = new Value(ones([dModel]));   // ones from Ch 02
    this.beta  = new Value(zeros([dModel])); // zeros from Ch 02
  }

  // x: [batch, seq, dModel] (or any shape where last dim = dModel)
  forward(x: Value): Value {
    // Normalize over the last axis (dModel dimension)
    // Each token vector is normalized independently
    const lastAxis = x.data.ndim - 1;
    const mu    = mean(x.data, lastAxis, /* keepDims= */ true);   // Ch 05
    const diff  = sub(x.data, mu);                                 // Ch 03
    const var_  = mean(mul(diff, diff), lastAxis, /* keepDims= */ true);
    const std_  = sqrt(addScalar(var_, this.eps));                 // Ch 06

    // Normalized: xHat = (x - mu) / sqrt(var + eps)
    const xHat = div(diff, std_);

    // Scale and shift: y = gamma * xHat + beta  (broadcast gamma/beta over batch+seq)
    const scaled = mul(xHat, this.gamma.data);
    const shifted = add(scaled, this.beta.data);

    // Wrap in Value for autograd
    const out = new Value(shifted, [x, this.gamma, this.beta], "layernorm");
    // ... set _backward
    return out;
  }

  parameters(): Value[] {
    return [this.gamma, this.beta];
  }
}

export class Dropout {
  rate: number;
  isTraining: boolean = true;

  constructor(rate: number) {
    this.rate = rate;
  }

  forward(x: Value): Value {
    if (!this.isTraining || this.rate === 0) return x;  // identity in eval mode

    // Create a random binary mask (1 = keep, 0 = drop), scaled by 1/(1-rate)
    const scale = 1 / (1 - this.rate);
    const maskData = x.data.data.map(() => Math.random() > this.rate ? scale : 0);
    const mask = createTensor(maskData, x.data.shape);

    // Element-wise multiply: zero out dropped activations, scale up the rest
    return x.mul(new Value(mask));
  }

  train(): void { this.isTraining = true; }
  eval():  void { this.isTraining = false; }
}
```

---

## Common Pitfalls

- Normalising over the batch axis (that's BatchNorm) instead of the feature axis.
- Forgetting `ε` inside `√` — variance can be exactly 0 for a constant token.
- Scaling dropout output by `(1 − p)` instead of dividing — back-to-front; inverted dropout *divides*.
- Leaving dropout active during validation; loss looks much worse than it really is.
- Skipping `γ`/`β` parameters; the network needs to be able to undo the normalisation.

---

## How to Verify

Run the tests and the exercise. Both should pass cleanly with no warnings:

```bash
bun test src/nn/transformer.test.ts
```
```bash
bun run exercises/ch-20-layernorm-dropout.ts
```

---

## Self-Check Questions

1. LayerNorm on `x = [1, 3, 5, 7]` (single token, `dModel=4`): compute $\mu$, $\sigma^2$,
   and the normalized output $\hat{x}$ by hand. Does $\hat{x}$ have mean 0 and std 1?
2. Why does LayerNorm use the last axis (dModel) while BatchNorm uses the batch axis?
3. Dropout with `rate=0.3` on 10 neurons: on average, how many survive? What is the
   scale factor applied to survivors? Why?
4. If you forget to call `model.eval()` before inference, what goes wrong with dropout?
   Does the model give consistent predictions on the same input?
5. A transformer has 12 layers. Each has 2 LayerNorm operations (one before attention,
   one before FFN). How many `gamma` and `beta` parameter vectors are there in total?

---

## End of Part 4

All the language model input machinery is in place:
- Text → token IDs (Ch 16, 16)
- Token IDs → embeddings (Ch 18)
- Embeddings → embeddings + position (Ch 19)
- Normalization + regularization (Ch 20)

One piece remains before attention: mask semantics. Padding masks, causal masks, and
additive attention masks must be precise before the model can safely ignore padding and
future tokens.

---

## Further Reading

- [Ba, Kiros, Hinton — Layer Normalization (2016)](https://arxiv.org/abs/1607.06450) — the LayerNorm paper.
- [Srivastava et al. — Dropout: A Simple Way to Prevent Overfitting](https://jmlr.org/papers/v15/srivastava14a.html) — the Dropout paper.
- [Xiong et al. — On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — pre-norm vs. post-norm; the analysis behind modern transformers.
- [Distill — Norm matters](https://distill.pub/2019/visualizing-memorization/) — visualisations of how normalisation shapes training dynamics.

---

## Next Chapter

**[Masks](ch-21-mask-cookbook.md)** — express "don't look at padding" and "don't peek at the future" with a single tensor primitive.
