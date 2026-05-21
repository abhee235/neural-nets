# Chapter 10: Tensor Autograd

> **Part 2 of 6 — Autodiff Engine**
> Source: [`src/autograd/grad.ts`](../../src/autograd/grad.ts) · [`src/autograd/engine.ts`](../../src/autograd/engine.ts)
> Tests: [`src/autograd/grad.test.ts`](../../src/autograd/grad.test.ts)
> Exercise: [`exercises/ch-10-tensor-autograd.ts`](../../exercises/ch-10-tensor-autograd.ts)

---

## Learning Goals

By the end of this chapter you can:

- Promote `Value` from scalar to tensor: data and grad are both `Tensor`s with matching shapes.
- Implement backward passes for tensor `add`, `mul`, `matMul`, `reshape`, and `transpose`.
- Implement `sumToShape` to un-broadcast a gradient back to its original tensor shape.
- Verify every tensor op against finite differences before depending on it.
- Confirm that all 19 tests pass; this is the foundation every later chapter relies on.

---

## Intuition First

Tensor autograd is scalar autograd with two extra rules:

1. **Matrix backward.** If `C = A @ B`, then `∂L/∂A = ∂L/∂C @ Bᵀ` and `∂L/∂B = Aᵀ @ ∂L/∂C`. These two formulas are the workhorses of every backward pass in this course.
2. **Un-broadcasting.** If a forward op broadcast a small tensor up to a large shape, the backward op must sum the gradient back down to the small shape. Otherwise gradients have the wrong shape and the optimizer blows up.

---

## Mental Model

```text
  forward:  C = A @ B           (shapes: A:[m,k], B:[k,n] → C:[m,n])

  backward (given dL/dC of shape [m,n]):
      dL/dA = dL/dC @ Bᵀ       → [m, k]
      dL/dB = Aᵀ @ dL/dC      → [k, n]

  broadcast rule:
      forward stretched [d] → [batch, seq, d]
      backward must  Σ over batch and seq to get [d] again
```

---

## Concepts

### From Number Gradients to Tensor Gradients

In scalar autograd:

$$x.data \in \mathbb{R}, \qquad x.grad \in \mathbb{R}$$

In tensor autograd:

$$X.data \in \mathbb{R}^{s_1 \times s_2 \times \cdots \times s_n}, \qquad X.grad \in \mathbb{R}^{s_1 \times s_2 \times \cdots \times s_n}$$

The shape of `grad` must always match the shape of `data` for that `Value`.

Plain English: each number in a tensor gets its own derivative. A tensor with 512 values has 512 gradients.

### Gradient Accumulation Still Uses `+=`

If a tensor is reused in two places, both paths contribute to its gradient:

$$\frac{\partial L}{\partial X} = \frac{\partial L_1}{\partial X} + \frac{\partial L_2}{\partial X}$$

So tensor gradients must be accumulated element-wise:

```typescript
x.grad = add(x.grad, contribution);
```

Do not assign over an existing gradient unless you are intentionally zeroing it before a new backward pass.

### Broadcasting Backward

Broadcasting copies values conceptually. Backward must sum gradients back over the copied dimensions.

Example:

```text
x shape: [3, 1]
y shape: [3, 4]
z = x + y
z shape: [3, 4]
```

`x` was stretched across 4 columns. During backward, each original `x[i, 0]` receives the sum of the 4 gradients that used it:

$$\frac{\partial L}{\partial x_{i,0}} = \sum_{j=0}^{3} \frac{\partial L}{\partial z_{i,j}}$$

This operation is often called `unbroadcast` or `sumToShape`.

### Reduction Backward

If `sum(x, axis)` collapses a dimension, backward broadcasts the upstream gradient back to the original shape.

If:

```text
x shape: [2, 3]
y = sum(x, axis=1, keepDims=true)
y shape: [2, 1]
```

Then each element in a row receives the same upstream gradient:

$$\frac{\partial L}{\partial x_{i,j}} = \frac{\partial L}{\partial y_{i,0}}$$

For `mean`, divide by the number of reduced elements:

$$\frac{\partial}{\partial x_i}\text{mean}(x) = \frac{1}{n}$$

### MatMul Backward

If:

$$Z = AB$$

Then:

$$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial Z} B^T$$

$$\frac{\partial L}{\partial B} = A^T \frac{\partial L}{\partial Z}$$

Shape check:

```text
A:      [m, k]
B:      [k, n]
Z:      [m, n]
dZ:     [m, n]
dA:     [m, n] @ [n, k] = [m, k]
dB:     [k, m] @ [m, n] = [k, n]
```

For batched matmul, the same rule applies independently for every batch/head dimension.

### Reshape Backward

Reshape changes the view of the same numbers. Backward reshapes the upstream gradient back to the original shape.

If:

```text
x shape: [2, 3, 4]
y = reshape(x, [6, 4])
```

Then:

```text
dx = reshape(dy, [2, 3, 4])
```

### Transpose Backward

Transpose permutes axes. Backward applies the inverse permutation.

If:

```text
y = transpose(x, [1, 0, 2])
```

Then:

```text
dx = transpose(dy, [1, 0, 2])
```

For a general permutation, compute the inverse permutation rather than assuming the same permutation works.

---

## What to Implement

| Symbol | Description |
|--------|-------------|
| `class Value` | Tensor-aware autograd node with `data: Tensor`, `grad: Tensor`, `_backward`, `_inputs`, `_op`. |
| `zerosLike(t)` | Create a zero tensor with the same shape as `t`. |
| `onesLike(t)` | Create a one tensor with the same shape as `t`. |
| `addGrad(target, contribution)` | Element-wise gradient accumulation with shape checks. |
| `sumToShape(grad, shape)` | Reverse broadcasting by summing over broadcasted dimensions. |
| `Value.add(other)` | Forward add plus broadcast-aware backward. |
| `Value.mul(other)` | Forward multiply plus broadcast-aware backward. |
| `Value.sum(axis?, keepDims?)` | Reduction forward plus broadcast backward. |
| `Value.mean(axis?, keepDims?)` | Mean forward plus scaled reduction backward. |
| `Value.matMul(other)` | Matmul forward plus matrix-gradient backward. |
| `Value.reshape(shape)` | Reshape forward plus inverse reshape backward. |
| `Value.transpose(axes)` | Transpose forward plus inverse permutation backward. |
| `checkTensorGradient(fn, inputs)` | Numerical gradient check for tensor-valued parameters. |

---

## TypeScript Hints

```typescript
export class Value {
  data: Tensor;
  grad: Tensor;
  _backward: () => void;
  _inputs: Value[];
  _op: string;

  constructor(data: Tensor, inputs: Value[] = [], op = "") {
    this.data = data;
    this.grad = zeros(data.shape);
    this._inputs = inputs;
    this._op = op;
    this._backward = () => {};
  }

  add(other: Value): Value {
    const out = new Value(add(this.data, other.data), [this, other], "add");
    out._backward = () => {
      addGrad(this, sumToShape(out.grad, this.data.shape));
      addGrad(other, sumToShape(out.grad, other.data.shape));
    };
    return out;
  }

  matMul(other: Value): Value {
    const out = new Value(matMul(this.data, other.data), [this, other], "matmul");
    out._backward = () => {
      const dA = matMul(out.grad, transposeLastTwo(other.data));
      const dB = matMul(transposeLastTwo(this.data), out.grad);
      addGrad(this, sumToShape(dA, this.data.shape));
      addGrad(other, sumToShape(dB, other.data.shape));
    };
    return out;
  }
}
```

---

## Common Pitfalls

- Forgetting to transpose in matmul backward — the shape error will be obvious if your shape checks are strict.
- Skipping `sumToShape` after a broadcasted op; the gradient comes back the wrong shape.
- Reusing a tensor's data buffer in two ops without cloning — backward will corrupt forward data.
- Implementing `reshape` backward as a fresh reshape; it should be `reshape(grad, original_shape)`.
- Trusting your op without a finite-difference check; this chapter *needs* checks because every later layer depends on it.

---

## How to Verify

Run the tests and the exercise. Both should pass cleanly with no warnings:

```bash
bun test src/autograd/grad.test.ts
```
```bash
bun run exercises/ch-10-tensor-autograd.ts
```

---

## Self-Check Questions

1. If `x.shape = [3, 1]` and `y.shape = [3, 4]`, what is the shape of `x.grad` after `z = x + y` and `loss = sum(z)`?
2. Why does reduction backward broadcast gradients, while broadcasting backward reduces gradients?
3. Derive the shape of `dA` and `dB` for `A @ B` where `A = [5, 7]` and `B = [7, 11]`.
4. Why is `reshape` backward simpler than `matMul` backward?
5. What bug would appear if you forgot to use `sumToShape` after a broadcasted multiply?

---

## Further Reading

- [PyTorch internals — autograd](https://pytorch.org/blog/overview-of-pytorch-autograd-engine/) — production autograd; same ideas, more bookkeeping.
- [Bornschein — Matrix calculus you need for deep learning](https://arxiv.org/abs/1802.01528) — derivations of matmul, sum, mean, and softmax backward.
- [Justin Domke — Reverse-mode AD](https://people.cs.umass.edu/~domke/courses/sml/08autodiff_nnets.pdf) — clean lecture notes generalising 08b to tensors.
- [Goodfellow, Bengio, Courville — Deep Learning](https://www.deeplearningbook.org/) — the standard graduate textbook; chapters map cleanly to this course.

---

## Next Chapter

**[Activations](../part-3-neural-net-primitives/ch-11-activation-functions.md)** — with tensor autograd in place, we can build the nonlinear layers that make networks expressive.
