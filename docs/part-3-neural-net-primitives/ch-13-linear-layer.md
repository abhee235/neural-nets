# Chapter 13: Linear Layer

> **Part 3 of 6 — Neural Net Primitives**
> Source: [`src/nn/linear.ts`](../../src/nn/linear.ts)
> Tests: [`src/nn/linear.test.ts`](../../src/nn/linear.test.ts)
> Exercise: [`exercises/ch-13-linear-layer.ts`](../../exercises/ch-13-linear-layer.ts)

---

## Learning Goals

By the end of this chapter you can:

- Implement `Linear(inDim, outDim)` as `y = x @ W + b`.
- Initialise weights with Xavier (for tanh) and He (for ReLU) scaling.
- Initialise biases to zero and explain why that is safe.
- Confirm the forward shape is `[..., outDim]` for any leading shape on `x`.
- Read your parameter count and predict it for the full transformer.

---

## Intuition First

A linear layer is a learned matrix transform. Geometrically it can rotate, stretch, and project the input; statistically it learns which combinations of inputs predict the output. Initialisation matters: weights too small and the signal dies; too large and it explodes. Xavier and He are calibrated so the variance of activations stays roughly constant through many layers.

---

## Mental Model

```text
  x : [..., in]  ──►  x @ W  ──►  + b  ──►  y : [..., out]
                       W: [in, out]      b: [out]

  Xavier (tanh):   W ~ U(±√(6 / (in + out)))
  He (ReLU):       W ~ N(0, √(2 / in))
  bias:            zeros
```

---

## Concepts

### The Forward Pass

$$y = xW + b$$

Where:
- $x \in \mathbb{R}^{n \times d_{\text{in}}}$ — input (batch of $n$ samples, each with $d_{\text{in}}$ features)
- $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$ — weight matrix (learnable)
- $b \in \mathbb{R}^{d_{\text{out}}}$ — bias vector (learnable)
- $y \in \mathbb{R}^{n \times d_{\text{out}}}$ — output

The bias $b$ is broadcast across the batch dimension (shape `[dOut]` added to `[n, dOut]`).

### Why This Is the "Neuron" Analogy

A single neuron computes a weighted sum of its inputs plus a bias:

$$y_j = \sum_{i} x_i w_{ij} + b_j$$

The weight matrix $W$ contains the weights for all $d_{\text{out}}$ neurons simultaneously.
Each column of $W$ is one neuron's weight vector. The entire layer is a batch of neurons.

### Weight Initialization: Why It Matters

If weights are initialized too large, activations explode; too small, they vanish.
After many layers, the variance of activations (and gradients) must stay roughly constant.

**Xavier / Glorot Initialization** (for `tanh` and `sigmoid`):

Initialize weights from a uniform distribution:
$$W \sim U\left(-\sqrt{\frac{6}{d_{\text{in}} + d_{\text{out}}}}, +\sqrt{\frac{6}{d_{\text{in}} + d_{\text{out}}}}\right)$$

Or from a normal distribution:
$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{d_{\text{in}} + d_{\text{out}}}}\right)$$

**He / Kaiming Initialization** (for `relu`):

$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{d_{\text{in}}}}\right)$$

He init accounts for the fact that ReLU zeroes out half the activations, halving the variance.
Using Xavier with ReLU causes activations to shrink through deep networks.

**Bias initialization:** Always initialize to 0. The asymmetry from random weights is enough.

### The Backward Pass (for Verification)

For $y = xW + b$, the gradients are:

$$\frac{\partial L}{\partial W} = x^T \cdot \frac{\partial L}{\partial y}$$

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot W^T$$

$$\frac{\partial L}{\partial b} = \sum_{\text{batch}} \frac{\partial L}{\partial y}$$

Your autograd engine from Ch 08b computes these automatically via `matMul` and `add` backward
passes. But verify them with numerical gradient checks (Ch 07).

### Structuring a Layer

Use a class to bundle weights and forward logic:

```typescript
class Linear {
  W: Value;     // shape [inFeatures, outFeatures]
  b: Value;     // shape [outFeatures]

  constructor(inFeatures: number, outFeatures: number, init: "xavier" | "he" = "xavier")
  forward(x: Value): Value   // returns xW + b
  parameters(): Value[]      // returns [W, b] for the optimizer
}
```

The `parameters()` method is important: the optimizer needs to know which `Value` nodes are
the learnable parameters. Every layer should implement this interface.

---

## What to Implement

| Symbol | Description |
|--------|-------------|
| `xavierInit(inF, outF)` | Sample from Xavier uniform distribution; returns Tensor |
| `heInit(inF, outF)` | Sample from He normal distribution; returns Tensor |
| `class Linear` | Linear layer: weight + bias, forward pass, parameters list |
| `Linear.forward(x)` | Compute $xW + b$; returns a `Value` for autograd |
| `Linear.parameters()` | Return `[this.W, this.b]` |

---

## TypeScript Hints

```typescript
export class Linear {
  W: Value;
  b: Value;
  inFeatures: number;
  outFeatures: number;

  constructor(inFeatures: number, outFeatures: number, init: "xavier" | "he" = "xavier") {
    this.inFeatures = inFeatures;
    this.outFeatures = outFeatures;

    // Initialize weights: Xavier for sigmoid/tanh, He for ReLU
    const wData = init === "he"
      ? heInit(inFeatures, outFeatures)
      : xavierInit(inFeatures, outFeatures);

    this.W = new Value(wData);
    // Biases start at zero — the random weights provide enough asymmetry
    this.b = new Value(zeros([outFeatures]));
  }

  forward(x: Value): Value {
    // y = x @ W + b
    // x: [batch, inFeatures], W: [inFeatures, outFeatures] → y: [batch, outFeatures]
    return x.matMul(this.W).add(this.b);   // broadcast b across batch
  }

  parameters(): Value[] {
    return [this.W, this.b];
  }
}

// Xavier uniform initialization
function xavierInit(inF: number, outF: number): Tensor {
  const limit = Math.sqrt(6 / (inF + outF));
  // rand gives [0,1), scale and shift to [-limit, limit]
  const r = rand([inF, outF]);
  return addScalar(mulScalar(r, 2 * limit), -limit);
}
```

---

## Common Pitfalls

- Using zero initialisation for `W` — all neurons compute the same function and gradients are identical.
- Using a huge std-dev so activations saturate or explode after a few layers.
- Forgetting to broadcast the bias across the leading axes of `x`.
- Hard-coding the weight shape as `[out, in]` then transposing in forward; pick one convention.
- Mis-counting parameters; total = `in * out + out`.

---

## How to Verify

Run the tests and the exercise. Both should pass cleanly with no warnings:

```bash
bun test src/nn/linear.test.ts
```
```bash
bun run exercises/ch-13-linear-layer.ts
```

---

## Self-Check Questions

1. A `Linear(128, 64)` layer: what are the shapes of `W` and `b`? How many parameters total?
2. For input `x` of shape `[32, 128]`, what is the output shape of `Linear(128, 64).forward(x)`?
3. Why are biases initialized to 0 but weights are not? What would happen if all weights were 0?
4. Verify with autograd: create `Linear(2, 1)`, compute $y = xW + b$ for $x = [[1, 2]]$,
   compute MSE loss against target $[[5]]$, call `backward()`, and check that `W.grad` and
   `b.grad` are correct analytically.
5. In the transformer, the Q projection takes `[batch, seq, 512]` and produces `[batch, seq, 64]`.
   What is `inFeatures` and `outFeatures` for this `Linear` layer?

---

## Further Reading

- [Glorot & Bengio — Understanding the difficulty of training deep networks (2010)](http://proceedings.mlr.press/v9/glorot10a.html) — the Xavier paper; derives the initialisation.
- [He et al. — Delving Deep into Rectifiers (2015)](https://arxiv.org/abs/1502.01852) — He initialisation tuned for ReLU networks.
- [Stanford CS231n — weight initialisation](https://cs231n.github.io/neural-networks-2/#init) — intuitive variance analysis.
- [Goodfellow, Bengio, Courville — Deep Learning](https://www.deeplearningbook.org/) — the standard graduate textbook; chapters map cleanly to this course.

---

## Next Chapter

**[Optimizers](ch-14-optimizers.md)** — go beyond plain SGD with momentum, Adam, and AdamW.
