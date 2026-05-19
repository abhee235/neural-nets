# Chapter 07: Calculus for ML

> **Part 2 of 6 — Autodiff Engine**
> `src/ch-07-calculus-for-ml/`

---

## What You're Building

A small utility module for calculus concepts used in training: numerical gradient checking
and gradient utilities. More importantly, this chapter is the **conceptual foundation** for
Ch 08 (autograd). You will implement `numericalGradient` — a function that estimates
derivatives numerically — which you will then use to **verify** your autograd engine.

---

## Why This Matters

Neural network training is gradient descent: repeatedly compute "how much does the loss
change if I nudge each weight?" and move the weights in the direction that decreases the loss.
This "how much" is the **gradient**. Without gradients, training is impossible.

Autograd (Ch 08–09) computes gradients analytically and automatically. But how do you know
your autograd is correct? You compare it to a numerical approximation — the finite difference
estimate from this chapter. Every ML framework uses this as a test.

---

## Concepts

### The Derivative

The derivative of a function $f$ at point $x$ measures the instantaneous rate of change:

$$f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$$

Geometrically: the slope of the tangent line to $f$ at $x$.
In ML: if $f$ is the loss and $x$ is a weight, $f'(x)$ tells you how much the loss increases
per unit increase in that weight.

### Partial Derivatives and the Gradient

When a function takes many inputs $f(x_1, x_2, \ldots, x_n)$, the **partial derivative**
$\partial f / \partial x_i$ is the derivative with respect to $x_i$ while holding all others fixed.

The **gradient** $\nabla f$ is the vector of all partial derivatives:

$$\nabla f = \left(\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}\right)$$

The gradient points in the direction of steepest increase. To minimize $f$, move in the
**negative** gradient direction — this is gradient descent.

### The Chain Rule

The most important rule in all of deep learning.

If $y = f(g(x))$, then:

$$\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx} = f'(g(x)) \cdot g'(x)$$

For a neural network, the loss is a composition of many functions:
$L = \text{loss}(\text{softmax}(\text{linear}(\text{relu}(\text{linear}(x)))))$.

The chain rule lets you compute $\partial L / \partial w$ for any weight $w$ deep in the network
by multiplying together the local derivatives of each function. This multiplication is what
backpropagation (Ch 08b) automates.

### Numerical Gradient Checking (Finite Differences)

The **centered finite difference** approximation of the derivative:

$$f'(x) \approx \frac{f(x + h) - f(x - h)}{2h}$$

The **centered** version is more accurate than the one-sided version because the $O(h^2)$ error
terms cancel out (versus $O(h)$ for one-sided). In practice, $h = 10^{-5}$ gives good estimates.

**To check a gradient for a function with a tensor input:**

For each element $x_i$ of the input tensor, perturb it by $+h$ and $-h$, evaluate $f$ both
times, and compute the finite difference. This is the numerical gradient for that element.
Compare it to the analytical gradient from autograd.

If the **relative error** is below a threshold (typically $10^{-5}$), the gradient is correct:

$$\text{relative error} = \frac{|g_{\text{analytical}} - g_{\text{numerical}}|}{|g_{\text{analytical}}| + |g_{\text{numerical}}| + \varepsilon}$$

### The Jacobian

When the function output is also a vector/tensor (not just a scalar), the full derivative is
a **Jacobian matrix**: $J_{ij} = \partial f_i / \partial x_j$.

In deep learning, loss is always a scalar, so we deal with gradients (not Jacobians) most of
the time. However, understanding the Jacobian helps explain why matrix multiplication has the
gradient it does (Ch 08b).

---

## What to Implement

| Function | Description |
|----------|-------------|
| `numericalGradient(f, x, h?)` | Estimate $\nabla f$ at tensor `x` using centered finite differences |
| `relativeError(a, b)` | Compute relative error between two tensors element-wise |
| `checkGradient(f, x, analyticalGrad, threshold?)` | Compare numerical vs analytical gradient, return max relative error |

---

## TypeScript Hints

```typescript
/**
 * Estimates the gradient of f with respect to x using centered finite differences.
 * f must return a scalar tensor (shape []).
 * Perturbs each element of x by ±h and measures the effect on f.
 */
function numericalGradient(
  f: (x: Tensor) => Tensor,
  x: Tensor,
  h = 1e-5
): Tensor {
  const grad = zeros(x.shape);   // will fill in gradients element by element

  for (let i = 0; i < x.size; i++) {
    // Perturb element i upward
    const xPlusH = { ...x, data: [...x.data] };
    xPlusH.data[i] = (x.data[i] ?? 0) + h;

    // Perturb element i downward
    const xMinusH = { ...x, data: [...x.data] };
    xMinusH.data[i] = (x.data[i] ?? 0) - h;

    // Centered finite difference: (f(x+h) - f(x-h)) / 2h
    const fPlus  = f(xPlusH).data[0]  ?? 0;
    const fMinus = f(xMinusH).data[0] ?? 0;
    grad.data[i] = (fPlus - fMinus) / (2 * h);
  }

  return grad;
}
```

---

## Self-Check Questions

1. Compute the numerical gradient of $f(x) = x^3$ at $x = 2$ using $h = 0.001$.
   What is the analytical answer? How close is the numerical estimate?
2. Why is the centered difference $\frac{f(x+h) - f(x-h)}{2h}$ more accurate than
   the one-sided $\frac{f(x+h) - f(x)}{h}$?
3. Apply the chain rule to find $\frac{d}{dx} \sin(x^2)$.
4. For $f(x, y) = x^2 y + y^3$, what is $\nabla f$? Evaluate it at $(1, 2)$.
5. If the numerical gradient check fails (relative error > 1e-4), what are three possible causes?

---

## → Next Chapter

**Ch 08a: Autograd — Forward Pass** — build the computation graph that records every
operation on a Value, enabling automatic gradient computation in Ch 08b.
