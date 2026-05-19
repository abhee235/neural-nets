# Chapter 10: Tensor Autograd Bridge

> **Part 2 of 6 — Autodiff Engine**
> `src/ch-10-tensor-autograd-bridge/`

---

## What You're Building

The missing bridge between scalar autograd and neural-network autograd: a `Value` object that wraps a full `Tensor`, stores a tensor-shaped gradient, and knows how to backpropagate through broadcasting, reductions, matrix multiplication, reshape, and transpose.

This is the chapter that turns a tiny scalar micrograd-style engine into the foundation of a tiny PyTorch-style learning library.

---

## Why This Matters

Scalar autograd teaches the idea: build a computation graph, sort it topologically, and run local backward functions in reverse.

Tensor autograd is where the real engineering starts.

Every transformer component depends on tensor gradients:

- Linear layers need matmul backward.
- LayerNorm needs reduction and broadcasting backward.
- Softmax and cross-entropy need stable tensor operations.
- Attention needs batched matmul, transpose, masking, and softmax.
- Embeddings need scatter-add style gradient accumulation.

If this bridge is weak, the final transformer will fail in confusing ways. If this bridge is solid, the rest of the course becomes much easier.

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

## Required Tests

- `sumToShape([2, 3, 4] -> [1, 3, 1])` sums the correct axes.
- `add` backward passes through unchanged for equal shapes.
- `add` backward sums over broadcasted dimensions.
- `mean` backward gives each input element `1 / n` for a full-tensor mean.
- `matMul` backward matches finite differences for both operands.
- `reshape` backward preserves all gradient values in the original order.
- `transpose` backward applies the inverse axis permutation.

---

## Common Pitfalls

- Forgetting to initialize `grad` as a tensor of zeros.
- Returning a gradient with the wrong shape after broadcasting.
- Using assignment instead of accumulation in `_backward`.
- Implementing 2D matmul backward but forgetting batched matmul for attention.
- Running gradient checks with too large or too small an epsilon.

---

## Self-Check Questions

1. If `x.shape = [3, 1]` and `y.shape = [3, 4]`, what is the shape of `x.grad` after `z = x + y` and `loss = sum(z)`?
2. Why does reduction backward broadcast gradients, while broadcasting backward reduces gradients?
3. Derive the shape of `dA` and `dB` for `A @ B` where `A = [5, 7]` and `B = [7, 11]`.
4. Why is `reshape` backward simpler than `matMul` backward?
5. What bug would appear if you forgot to use `sumToShape` after a broadcasted multiply?

---

## → Next Chapter

**Ch 11: Activation Functions** — now that tensor autograd is solid, nonlinearities can become differentiable tensor operations.
