# Chapter 13: Linear Layer

> **Part 3 of 6 — Neural Net Primitives**
> `src/ch-13-linear-layer/`

---

## What You're Building

The `Linear` layer (also called `Dense` or `Fully Connected`): a function $y = xW + b$
with learnable weight matrix $W$ and bias vector $b$. This is the most-used building block
in all of deep learning — every attention projection, every FFN layer in the transformer is
a `Linear` layer.

---

## Why This Matters

In the transformer chapters (Ch 25–30), `Linear` layers appear everywhere:
- Q, K, V projections: `Linear(dModel, dHead)`
- Output projection: `Linear(dModel, dModel)`
- FFN first layer: `Linear(dModel, dFf)` (e.g., `Linear(512, 2048)`)
- FFN second layer: `Linear(dFf, dModel)`
- Final language model head: `Linear(dModel, vocabSize)`

Building it once and correctly here means you just reuse it for all 20+ linear projections
in a full transformer.

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

## → Next Chapter

**Ch 14: Optimizers** — SGD with momentum and Adam, the two workhorses of neural network training.
