# Chapter 08b: Autograd — Backward Pass

> **Part 2 of 6 — Autodiff Engine**
> Source: [`src/autograd/value.ts`](../../src/autograd/value.ts)
> Tests: [`src/autograd/value.test.ts`](../../src/autograd/value.test.ts)
> Exercise: [`exercises/ch-08-autograd.ts`](../../exercises/ch-08-autograd.ts)

---

## Learning Goals

By the end of this chapter you can:

- Implement `backward()` as a reverse topological traversal of the computation graph.
- Accumulate gradients with `+=` so that a `Value` used in two places sums its contributions.
- Derive and memorise the local-gradient table for `add`, `mul`, `pow`, `exp`, `log`, `tanh`.
- Verify each operator's gradient with finite differences from Ch 07.
- Identify when *not* zeroing gradients between steps causes runaway updates.

---

## Intuition First

Backprop is one rule applied repeatedly: **the gradient flowing into a node is the sum of the gradients flowing out of it, each multiplied by the local derivative on that edge.** Topological order guarantees we never visit a node before all its downstream users have contributed.

---

## Mental Model

```text
  forward:        a ─► (*) ─► c ─► (+) ─► L
                   ╲                ╱
                    b              d

  backward starts at L.grad = 1, then for each node in reverse-topo order:
      for each parent p:
          p.grad += local_∂node/∂p · node.grad
```

---

## Concepts

### The Key Insight: Local Gradients

For each operation $z = f(x, y)$, we need to know:
- $\partial z / \partial x$ — how does $z$ change when $x$ changes? (local gradient w.r.t. $x$)
- $\partial z / \partial y$ — how does $z$ change when $y$ changes? (local gradient w.r.t. $y$)

Then by the **chain rule**, if the loss is $L$ and we already know $\partial L / \partial z$
(the gradient that flowed into this node from above), we can compute:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial x}$$

This is called **backpropagating the gradient**: the incoming gradient ($\partial L / \partial z$)
multiplied by the local gradient ($\partial z / \partial x$) gives the outgoing gradient.

### Local Gradients for Each Operation

| Operation | $z = f(x, y)$ | $\partial z / \partial x$ | $\partial z / \partial y$ |
|-----------|--------------|--------------------------|--------------------------|
| add | $z = x + y$ | $1$ | $1$ |
| mul | $z = x \cdot y$ | $y$ | $x$ |
| pow | $z = x^n$ | $n \cdot x^{n-1}$ | — |
| neg | $z = -x$ | $-1$ | — |
| exp | $z = e^x$ | $e^x = z$ | — |
| log | $z = \ln x$ | $1/x$ | — |
| tanh | $z = \tanh x$ | $1 - \tanh^2(x) = 1 - z^2$ | — |

The `_backward` function for each operation **adds** to the inputs' `.grad` using these formulas:

```typescript
// For add: z = x + y
// ∂L/∂x += ∂L/∂z * 1 = outGrad
// ∂L/∂y += ∂L/∂z * 1 = outGrad
out._backward = () => {
  x.grad += out.grad * 1;
  y.grad += out.grad * 1;
};

// For mul: z = x * y
// ∂L/∂x += ∂L/∂z * y
// ∂L/∂y += ∂L/∂z * x
out._backward = () => {
  x.grad += out.grad * y.data;
  y.grad += out.grad * x.data;
};
```

**Why `+=` and not `=`?** A node can be used multiple times in the same graph (e.g., $x^2 + x$).
Each use contributes a separate gradient that must be **summed** — this is called gradient accumulation.

### Topological Sort and Reverse-Mode AD

The `backward()` method must call `_backward` functions in the **right order**: each node
must receive its full accumulated gradient before it backpropagates to its inputs.

The correct order is: **reverse topological order** — process the loss node first, then work
backwards through the graph, never visiting a node before all of its dependents are processed.

**Topological sort algorithm (DFS postorder):**

```
function topoSort(node):
    if node not visited:
        mark node visited
        for each input of node:
            topoSort(input)
        append node to result   ← node goes AFTER all its inputs
return reversed result          ← reverse to get from output back to inputs
```

**The `backward()` method:**

```typescript
backward(): void {
  // The loss node's gradient w.r.t. itself is always 1 (∂L/∂L = 1)
  this.grad = 1;

  // Walk in reverse topological order and call each _backward
  const sorted = topoSort(this);
  for (const node of sorted.reverse()) {
    node._backward();
  }
}
```

### Zeroing Gradients

Between training iterations, gradients must be **reset to 0**.
If you don't zero them, gradients accumulate across calls and corrupt training.

```typescript
zeroGrad(): void {
  this.grad = 0;
}
```

This is why PyTorch has `optimizer.zero_grad()` — it's essential before every backward pass.

### Extending to Tensor Autograd

Once scalar autograd works, extend `Value` to wrap a `Tensor` instead of a `number`.
The same structure applies, but:
- `grad` becomes a `Tensor` of the same shape as `data`
- Local gradient formulas are now tensor operations (using the functions from Part 1)
- `matMul` backward: if $Z = AB$, then $\partial L / \partial A = (\partial L / \partial Z) \cdot B^T$
  and $\partial L / \partial B = A^T \cdot (\partial L / \partial Z)$

---

## What to Implement

| Symbol | Description |
|--------|-------------|
| `topoSort(node)` | Return all nodes reachable from `node` in topological order |
| `Value.backward()` | Set `this.grad = 1`, run `_backward` in reverse topo order |
| `Value.zeroGrad()` | Reset `.grad` to 0 recursively |
| `Value.add._backward` | Distribute gradient through addition |
| `Value.mul._backward` | Distribute gradient through multiplication |
| `Value.pow._backward` | Distribute gradient through power |
| `Value.exp._backward` | Distribute gradient through exp |
| `Value.log._backward` | Distribute gradient through log |
| `Value.tanh._backward` | Distribute gradient through tanh |

---

## TypeScript Hints

```typescript
// Topological sort using iterative DFS (avoids call-stack overflow for deep graphs)
function topoSort(root: Value): Value[] {
  const visited = new Set<Value>();
  const result: Value[] = [];

  function dfs(node: Value): void {
    if (visited.has(node)) return;
    visited.add(node);
    for (const input of node._inputs) {
      dfs(input);
    }
    result.push(node);  // postorder: node goes after its inputs
  }

  dfs(root);
  return result;  // do NOT reverse here; backward() will reverse
}

// Example: full backward for exp
exp(): Value {
  const expVal = Math.exp(this.data);
  const out = new Value(expVal, [this], "exp");
  out._backward = () => {
    // d/dx e^x = e^x = out.data
    this.grad += out.grad * out.data;
  };
  return out;
}
```

---

## Common Pitfalls

- Assigning `p.grad = …` instead of `p.grad += …` — a node reused twice will lose half its gradient.
- Forgetting to call `zeroGrad()` between training steps; gradients accumulate forever.
- Visiting nodes in forward order; you must run in **reverse** topological order.
- Re-running `backward()` on a graph that has already been backwarded — fresh forward each time.
- Implementing `tanh` backward as `1 - x²` instead of `1 - out²` (use the cached output).

---

## How to Verify

Run the tests and the exercise. Both should pass cleanly with no warnings:

```bash
bun test src/autograd/value.test.ts
```
```bash
bun run exercises/ch-08-autograd.ts
```

---

## Self-Check Questions

1. For $z = x \cdot y$ with $x = 3, y = 4$: if $\partial L / \partial z = 2$,
   what is $\partial L / \partial x$ and $\partial L / \partial y$?
2. For $L = (x + y)^2$ with $x = 1, y = 2$: compute $\partial L / \partial x$
   analytically, then verify with `backward()`.
3. Why do we use `+=` instead of `=` when accumulating gradients?
4. What would happen if you called `backward()` without running `topoSort`?
   (Try processing in the wrong order and see what breaks.)
5. For matrix multiply $Z = AB$, the gradient formula is $\partial L / \partial A = G \cdot B^T$
   where $G = \partial L / \partial Z$. Verify this makes dimensional sense:
   if $A$ is $(M, K)$, $B$ is $(K, N)$, and $G$ is $(M, N)$, what shape is $G \cdot B^T$?

---

## Further Reading

- [Karpathy — micrograd walkthrough video](https://www.youtube.com/watch?v=VMj-3S1tku0) — step-by-step build of the same backward pass we are coding.
- [Goodfellow et al. — Deep Learning, ch. 6.5 (Back-Propagation)](https://www.deeplearningbook.org/contents/mlp.html) — the formal treatment of reverse-mode AD.
- [Justin Domke — Why is automatic differentiation efficient?](https://justindomke.wordpress.com/2009/02/17/automatic-differentiation-the-most-criminally-underused-tool-in-the-potential-machine-learning-toolbox/) — explains why reverse mode is `O(forward)` for scalar loss.
- [Chris Olah — Calculus on Computational Graphs](https://colah.github.io/posts/2015-08-Backprop/) — complements the 08a reading; focus on multivariate examples.

---

## Next Chapter

**[Gradient Descent](ch-09-gradient-descent.md)** — now that gradients flow, use them to actually move parameters downhill.
