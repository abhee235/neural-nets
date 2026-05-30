# Chapter 08a: Autograd — Forward Graph

> **Part 2 of 6 — Autodiff Engine**
> Source: [`src/autograd/value.ts`](../../src/autograd/value.ts)
> Tests: [`src/autograd/value.test.ts`](../../src/autograd/value.test.ts)
> Exercise: [`exercises/ch-08-autograd.ts`](../../exercises/ch-08-autograd.ts)

---

## Learning Goals

By the end of this chapter you can:

- Wrap a single number in a `Value` that knows its parents, its operation, and its gradient.
- Build a computation graph by overloading `add`, `mul`, `pow`, `exp`, `log`, and `tanh`.
- Explain why every `Value` stores a `_backward` closure instead of a giant switch statement.
- Spot the difference between the forward graph (built eagerly) and the backward pass (run on demand).
- Replicate Karpathy's micrograd API in strict TypeScript.

---

## Intuition First

Autograd is bookkeeping. Every time you do `c = a * b`, we secretly record:

- `c` came from `a` and `b`.
- The operation was `*`.
- The local derivatives are `∂c/∂a = b` and `∂c/∂b = a`.

When you later call `c.backward()`, we walk the recorded graph backwards and multiply local derivatives together using the chain rule. The forward pass *builds* the graph; the backward pass *walks* it.

---

## Mental Model

```text
    a ───┐
         ├── (*) ──► c ───┐
    b ───┘                ├── (+) ──► L
                     d ───┘

  forward:  store data, parents, op on each node
  backward: visit nodes in reverse topological order and accumulate `grad`
```

---

## Concepts

### What is a Computation Graph?

Every mathematical expression can be drawn as a directed acyclic graph (DAG) where:
- **Nodes** are values (intermediate results)
- **Edges** point from inputs to outputs (outputs depend on inputs)

Example: $z = (x + y) \cdot w$

```
x ──┐
    ├──[+]──→ s ──[×]──→ z
y ──┘            ↑
                 w
```

Each node stores:
1. Its **data** (the forward value)
2. Its **gradient** (initially 0, filled in during backward pass)
3. Its **_backward function** (how to distribute the gradient to its inputs — filled in Ch 08b)
4. Its **_inputs** (which nodes produced this one)
5. An **operation name** for debugging ("+", "×", "matmul", etc.)

### The `Value` Class

```typescript
class Value {
  data: Tensor;               // the forward result
  grad: Tensor;               // gradient (∂loss/∂this), zero initially
  _backward: () => void;      // filled in by each operation
  _inputs: Value[];           // which Values produced this one
  _op: string;                // operation label for debugging
}
```

### Scalar-First Approach

Autograd for scalars is much easier to understand and debug than tensor autograd.
Start with a `Value` that wraps a **single number** (not a tensor). Once scalar autograd
is working and tested, extend it to full tensors.

For scalar `Value`:
```typescript
class Value {
  data: number;
  grad: number = 0;     // ∂L/∂this — will be filled by backward()
  _backward: () => void = () => {};
  _inputs: Value[];
}
```

### Forward Pass: Recording Operations

Every operation on a `Value` must:
1. Compute the **result** (the normal forward math)
2. Create a **new `Value`** wrapping the result
3. Record the **input Values** that produced it
4. Store a `_backward` function (stub for now — implemented in Ch 08b)

```typescript
add(other: Value): Value {
  const result = new Value(this.data + other.data);
  result._inputs = [this, other];
  result._op = "+";
  result._backward = () => {
    // Ch 08b will fill this in
  };
  return result;
}
```

The key insight: **the graph is built implicitly by running the forward pass**.
You don't construct it manually — it emerges from the operations.

### Operations to Support

For scalar `Value` (Ch 08a):

| Method | Math |
|--------|------|
| `add(other)` | $a + b$ |
| `mul(other)` | $a \times b$ |
| `sub(other)` | $a - b$ (= `add(other.neg())`) |
| `div(other)` | $a / b$ (= `mul(other.pow(-1))`) |
| `pow(n)` | $a^n$ |
| `neg()` | $-a$ |
| `exp()` | $e^a$ |
| `log()` | $\ln(a)$ |
| `tanh()` | $\tanh(a)$ |

Each returns a new `Value` with `_inputs` and `_op` set.

### Visualizing the Graph

Add a `toGraph()` or `print()` helper that walks `_inputs` recursively and prints the DAG.
This is invaluable for debugging — you can see exactly what operations were recorded.

```
Value(z=6.0, op=×)
├── Value(s=3.0, op=+)
│   ├── Value(x=1.0)
│   └── Value(y=2.0)
└── Value(w=2.0)
```

---

## What to Implement

| Symbol | Description |
|--------|-------------|
| `class Value` | Core class: `data`, `grad`, `_inputs`, `_op`, `_backward` |
| `Value.add(other)` | Addition, records graph |
| `Value.mul(other)` | Multiplication, records graph |
| `Value.pow(n)` | Power, records graph |
| `Value.neg()` | Negation |
| `Value.sub(other)` | Subtraction (uses add + neg) |
| `Value.div(other)` | Division (uses mul + pow) |
| `Value.exp()` | Exponential |
| `Value.log()` | Natural log |
| `Value.tanh()` | Hyperbolic tangent |
| `printGraph(v)` | Pretty-print the computation graph (for debugging) |

---

## TypeScript Hints

```typescript
class Value {
  data: number;
  grad: number = 0;
  _backward: () => void = () => {};  // no-op until backward() is called
  _inputs: Value[] = [];
  _op: string = "";

  constructor(data: number, inputs: Value[] = [], op = "") {
    this.data = data;
    this._inputs = inputs;
    this._op = op;
  }

  add(other: Value): Value {
    const out = new Value(this.data + other.data, [this, other], "+");
    // _backward will be set in Ch 08b
    return out;
  }

  mul(other: Value): Value {
    const out = new Value(this.data * other.data, [this, other], "*");
    return out;
  }

  pow(n: number): Value {
    const out = new Value(Math.pow(this.data, n), [this], `**${n}`);
    return out;
  }
}
```

---

## Common Pitfalls

- Storing parents as a `Set` and losing iteration order — use an array.
- Capturing `this` inside `_backward` and confusing it with a different node; bind the locals you need.
- Forgetting that `Value` is *scalar* in 08a — tensor autograd waits until Ch 10.
- Reusing the same `Value` across two unrelated computations; build a fresh graph for each forward pass.
- Returning a primitive `number` from an operator when you should return a `Value`.

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

1. Draw the computation graph for $L = (a \cdot b + c)^2$.
   Label each node with its operation and value when $a=2, b=3, c=1$.
2. How many `Value` nodes does the expression `(x + y) * (y - z)` create?
   (Hint: each operation creates one new node.)
3. Why must `_backward` be stored on each node, not computed at call time?
4. What would the graph look like for a 2-layer neural network's forward pass?
   (Just the structure, not the actual numbers.)
5. Why is the computation graph a DAG (acyclic)? What would a cycle mean physically?

---

## Further Reading

- [Karpathy — micrograd](https://github.com/karpathy/micrograd) — the ~100-line Python autograd this chapter is modelled on.
- [Karpathy — Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) — video series that builds micrograd and nanoGPT from scratch.
- [Baydin et al. — Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/abs/1502.05767) — the survey paper; covers forward, reverse, and tape-based AD.
- [Chris Olah — Calculus on Computational Graphs](https://colah.github.io/posts/2015-08-Backprop/) — best intuition piece on graph-based backprop.

---

## Next Chapter

**[Autograd Backward](ch-08b-autograd-backward.md)** — wire up the `backward()` pass that walks the graph in reverse and accumulates gradients.
