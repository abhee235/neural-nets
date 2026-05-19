# Chapter 09: Gradient Descent

> **Part 2 of 6 — Autodiff Engine**
> `src/ch-09-gradient-descent/`

---

## What You're Building

The training loop primitive: a `Parameter` wrapper for learnable tensors, a `SGD` optimizer,
and a demonstration that gradient descent actually minimizes a loss function. This chapter
closes the loop from Ch 07–08: define a loss, call backward, update parameters, repeat.

---

## Why This Matters

Every training run in existence — from a 2-weight toy network to GPT-4 — is gradient descent
at its core. This chapter makes it concrete: you will watch the loss decrease step by step
and develop intuition for learning rate, convergence, and what "training" actually means.

---

## Concepts

### The Parameter Update Rule

Given a loss $L$ and a parameter $\theta$, gradient descent moves $\theta$ in the direction
that decreases $L$:

$$\theta_{\text{new}} = \theta_{\text{old}} - \eta \cdot \frac{\partial L}{\partial \theta}$$

Where:
- $\eta$ (eta) is the **learning rate** — controls the step size
- $\partial L / \partial \theta$ is the gradient — computed by `backward()`
- The **minus sign** is critical: we move *against* the gradient because the gradient points uphill

### The Training Loop

```
for each iteration:
  1. Forward pass  — compute predictions from current parameters
  2. Compute loss  — how wrong are the predictions?
  3. Zero gradients — reset .grad to 0 on all parameters
  4. Backward pass — compute ∂L/∂θ for all parameters via backward()
  5. Update params — θ ← θ - η * θ.grad
```

Step 3 (zero gradients) is easy to forget and causes a subtle bug: gradients accumulate
across iterations, making the effective learning rate grow every step.

### Learning Rate: The Critical Hyperparameter

| Learning rate | Effect |
|---------------|--------|
| Too large | Loss oscillates or diverges (overshoots the minimum) |
| Too small | Training is very slow (takes thousands of steps) |
| Just right | Loss decreases smoothly and converges |

A typical starting point: `1e-3` for Adam, `1e-2` to `0.1` for SGD.

**Experiment:** train the same model with `lr = 0.001`, `lr = 0.01`, `lr = 0.1`, `lr = 1.0`.
Plot the loss curve. Seeing the divergence for large lr is a key intuition.

### Loss Landscape Intuition

The loss $L$ as a function of all parameters forms a high-dimensional surface.
Gradient descent is walking downhill on this surface, step by step.

- **Global minimum**: the lowest possible loss — often unreachable in practice
- **Local minimum**: a valley that isn't the lowest — gradient descent can get stuck here
- **Saddle point**: flat region where gradient is near zero but it's not a minimum
- **Plateau**: large flat region — gradient is tiny, training stalls

For well-designed neural networks (with proper normalization, etc.), the landscape is
surprisingly well-behaved. Most local minima are close in quality to the global minimum.

### Parameter Wrapper

To cleanly track which tensors are learnable (have gradients) vs which are fixed inputs,
wrap learnable tensors in a `Parameter` type:

```typescript
interface Parameter {
  value: Value;       // the underlying Value (with .grad)
  name?: string;      // for debugging: "W1", "b1", etc.
}
```

A model is just a collection of `Parameter`s plus a forward function.

### What SGD Looks Like in Code

```typescript
// After forward + backward:
for (const param of parameters) {
  // θ ← θ - η * ∂L/∂θ
  param.value.data -= learningRate * param.value.grad;
  // Zero gradient for next iteration
  param.value.grad = 0;
}
```

---

## What to Implement

| Symbol | Description |
|--------|-------------|
| `class SGD` | Stochastic gradient descent optimizer |
| `SGD.step(params)` | Apply one gradient step to all parameters |
| `SGD.zeroGrad(params)` | Zero all gradients |
| `lossHistory` demo | Fit a simple function (e.g. $y = 2x + 3$) using gradient descent and print the loss curve |

---

## TypeScript Hints

```typescript
class SGD {
  learningRate: number;

  constructor(learningRate: number) {
    this.learningRate = learningRate;
  }

  // Apply the update rule: θ ← θ - η * ∂L/∂θ
  step(params: Value[]): void {
    for (const p of params) {
      p.data -= this.learningRate * p.grad;
    }
  }

  // Must be called before each backward() to prevent gradient accumulation
  zeroGrad(params: Value[]): void {
    for (const p of params) {
      p.grad = 0;
    }
  }
}
```

**Demo to build:** Fit $f(x) = wx + b$ to points $(1, 5), (2, 7), (3, 9)$.
The true function is $y = 2x + 3$. Initialize $w = 0, b = 0$ and run 100 gradient steps.
Print the loss every 10 steps. By the end, $w$ should be near $2$ and $b$ near $3$.

---

## Self-Check Questions

1. For the linear fit demo, compute the gradient of MSE loss w.r.t. $w$ by hand at step 0.
   Does it match what your autograd computes?
2. What happens if you forget to zero gradients? Run the demo both ways and compare.
3. Try learning rate 10.0 for the linear fit. What happens to the loss?
4. The gradient of $L = w^2$ is $2w$. If $w = 3$ and $\eta = 0.1$, what is $w$ after one step?
   After 10 steps?
5. Why does SGD need to see all parameters at the `step` call? Why not just iterate the graph?

---

## Checkpoint

You now have a working autograd engine: a computation graph (Ch 08a), automatic backward pass
(Ch 08b), and gradient-based learning (Ch 09).

Test it: build a 2-variable quadratic $L = (w_1 - 3)^2 + (w_2 + 1)^2$, run gradient descent,
and watch $w_1 \to 3$ and $w_2 \to -1$.

---

## → Next Chapter

**Ch 10: Tensor Autograd Bridge** — extend scalar autograd into tensor autograd with
broadcasting, reductions, matmul, reshape, transpose, and gradient checks. This is the
real bridge into neural networks and transformers.
