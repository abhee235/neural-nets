# Chapter 14: Optimizers

> **Part 3 of 6 — Neural Net Primitives**
> Source: [`src/optim/sgd.ts`](../../src/optim/sgd.ts) · [`src/optim/adam.ts`](../../src/optim/adam.ts)
> Tests: [`src/optim/sgd.test.ts`](../../src/optim/sgd.test.ts) · [`src/optim/adam.test.ts`](../../src/optim/adam.test.ts)
> Exercise: [`exercises/ch-14-optimizers.ts`](../../exercises/ch-14-optimizers.ts)

---

## Learning Goals

By the end of this chapter you can:

- Implement plain SGD: `θ ← θ − η ∇L`.
- Implement SGD with momentum: `v ← μ v + ∇L; θ ← θ − η v`.
- Implement Adam with bias-corrected first and second moments.
- Implement AdamW (Adam with decoupled weight decay).
- Pick the right optimizer for a task and tune `lr`/`β₁`/`β₂` sanely.

---

## Intuition First

Plain SGD treats every parameter the same. Real loss surfaces have steep, narrow valleys; SGD bounces off the walls. **Momentum** smooths updates by averaging recent gradients. **Adam** goes further: it gives each parameter its own learning rate based on its recent gradient magnitudes (parameters with big gradients get smaller steps, parameters with tiny gradients get bigger ones).

---

## Mental Model

```text
  SGD:        θ ← θ − η · g

  Momentum:   v ← μ v + g
              θ ← θ − η · v

  Adam:       m ← β₁ m + (1−β₁) g            ← first moment (mean)
              v ← β₂ v + (1−β₂) g²           ← second moment (uncentred variance)
              m̂ ← m / (1 − β₁ᵗ),  v̂ ← v / (1 − β₂ᵗ)
              θ ← θ − η · m̂ / (√v̂ + ε)
```

---

## Concepts

### SGD — Stochastic Gradient Descent

The simplest optimizer. Each step: $\theta \leftarrow \theta - \eta \nabla L$

**Problem:** gradients are noisy (computed on a mini-batch, not the full dataset).
The update direction oscillates, especially in directions where the curvature is high.
This causes slow convergence or oscillation.

### SGD with Momentum

Momentum smooths the update direction by accumulating a "velocity" vector:

$$v_t = \beta v_{t-1} + \nabla L_t$$
$$\theta_t = \theta_{t-1} - \eta v_t$$

Where $\beta \in [0, 1)$ is the momentum coefficient (typically 0.9).

**Intuition:** Think of a ball rolling downhill. The velocity builds up in the consistent
downhill direction and dampens the oscillating side-to-side components.
After many steps, the ball moves mostly downhill with little oscillation.

With momentum, the **effective learning rate** in consistent gradient directions grows up to:
$$\eta_{\text{eff}} = \frac{\eta}{1 - \beta}$$

So with $\beta = 0.9$ and $\eta = 0.01$: effective rate = 0.1. That's 10× larger in the
consistent direction, which is exactly where you want to move fast.

### Adam — Adaptive Moment Estimation

Adam tracks two moving averages (moments) of the gradient:

**First moment** (mean of gradients, like momentum):
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla L_t$$

**Second moment** (uncentered variance of gradients):
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla L_t)^2$$

**Bias correction** (accounts for zero initialization of $m_0 = v_0 = 0$):
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t} \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

**Parameter update:**
$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \varepsilon} \hat{m}_t$$

**Typical hyperparameters:** $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$, $\eta = 10^{-3}$.

**Why Adam is better than SGD:**

1. **Adaptive learning rates per parameter:** Parameters with large, consistent gradients
   get smaller effective learning rates (they don't need big steps). Parameters with small
   gradients get larger effective learning rates (they need more signal).

2. **Invariant to gradient scale:** Multiplying the loss by 10 doesn't change Adam's
   updates (because $\sqrt{v}$ scales up proportionally). SGD updates would all be 10× larger.

3. **Fast convergence:** The bias correction means Adam moves sensibly even in the first few steps.

**The division by $\sqrt{\hat{v}} + \varepsilon$** is the key: it normalizes each parameter's
update by the RMS of its recent gradients. If a parameter's gradient has been consistently large,
$\sqrt{\hat{v}}$ is large, and the effective step is small (don't overshoot). If the gradient
has been small, $\sqrt{\hat{v}}$ is small, and the effective step is larger (use the signal).

### Weight Decay (L2 Regularization)

Often added to Adam as "AdamW":
$$\theta_t = \theta_{t-1} - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon} + \lambda \theta_{t-1} \right)$$

The $\lambda \theta$ term penalizes large weights, acting as regularization. Modern transformer
training almost always uses AdamW.

---

## What to Implement

| Symbol | Description |
|--------|-------------|
| `class SGD` | Basic SGD. `step(params)`, `zeroGrad(params)`. |
| `class SGDMomentum` | SGD with momentum. Stores velocity state per parameter. |
| `class Adam` | Adam optimizer. Stores $m$ and $v$ state. Implements bias correction. |
| `class AdamW` | Adam with weight decay (adds $\lambda \theta$ to gradient before update). |

All optimizers should implement the same interface:
```typescript
interface Optimizer {
  step(params: Value[]): void;
  zeroGrad(params: Value[]): void;
}
```

---

## TypeScript Hints

```typescript
export class Adam implements Optimizer {
  lr: number;
  beta1: number;
  beta2: number;
  epsilon: number;
  t: number = 0;                    // timestep (for bias correction)
  m: Map<Value, number[]> = new Map();   // first moment per parameter
  v: Map<Value, number[]> = new Map();   // second moment per parameter

  constructor(lr = 1e-3, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8) {
    this.lr = lr;
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.epsilon = epsilon;
  }

  step(params: Value[]): void {
    this.t += 1;

    for (const p of params) {
      const g = p.grad;  // gradient (scalar for now, tensor array later)

      // Initialize moment vectors for this parameter if first time
      if (!this.m.has(p)) {
        this.m.set(p, new Array(/* size */).fill(0));
        this.v.set(p, new Array(/* size */).fill(0));
      }

      const m = this.m.get(p)!;
      const v = this.v.get(p)!;

      // Update biased first and second moments
      // m_t = β1 * m_{t-1} + (1 - β1) * g_t
      // v_t = β2 * v_{t-1} + (1 - β2) * g_t²
      // (implement element-wise over the parameter array)

      // Bias-corrected estimates
      const mHat = m[0]! / (1 - Math.pow(this.beta1, this.t));
      const vHat = v[0]! / (1 - Math.pow(this.beta2, this.t));

      // Update: θ ← θ - η * m̂ / (√v̂ + ε)
      p.data -= this.lr * mHat / (Math.sqrt(vHat) + this.epsilon);
    }
  }

  zeroGrad(params: Value[]): void {
    for (const p of params) p.grad = 0;
  }
}
```

---

## Common Pitfalls

- Mutating the parameter tensor in place inside the autograd graph; do updates outside.
- Forgetting bias correction in Adam — early steps will be tiny because `m`, `v` start at 0.
- Initialising `v` (Adam's second moment) at 0 and then doing `1/√v` on step 1 without `eps`.
- Using Adam's high learning rate (1e-3) with SGD — SGD wants 1e-2 or higher.
- Letting state (`m`, `v`) drift between training runs; reset it whenever you reset the model.

---

## How to Verify

Run the tests and the exercise. Both should pass cleanly with no warnings:

```bash
bun test src/optim/sgd.test.ts src/optim/adam.test.ts
```
```bash
bun run exercises/ch-14-optimizers.ts
```

---

## Self-Check Questions

1. With momentum $\beta = 0.9$, after how many steps does the contribution of the initial
   gradient $g_0$ drop below 1% of its original weight?
2. Adam's bias correction: why are $m_0 = v_0 = 0$ biased estimates? What would happen
   without the correction in the first few steps?
3. If every gradient is exactly 1.0 for 1000 steps, what does Adam's $v_t$ converge to?
   What is the effective learning rate?
4. Compare: for a parameter with alternating gradients `+1, -1, +1, -1, ...`, which
   optimizer converges faster — SGD, SGD+Momentum, or Adam? Why?
5. Why doesn't Adam's learning rate scale with the magnitude of the loss? Is this a feature
   or a limitation?

---

## Further Reading

- [Kingma & Ba — Adam: A Method for Stochastic Optimization (2014)](https://arxiv.org/abs/1412.6980) — the original Adam paper.
- [Loshchilov & Hutter — Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101) — why AdamW differs from Adam + L2 and when it matters.
- [Sebastian Ruder — gradient descent algorithms](https://ruder.io/optimizing-gradient-descent/) — compares every optimizer in one place.
- [Distill — Why Momentum Really Works](https://distill.pub/2017/momentum/) — interactive treatment of the momentum dynamics.

---

## Next Chapter

**[Training Loop](ch-15-training-loop.md)** — assemble layers, loss, optimizer, and data into a working training routine.
