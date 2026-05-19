# Chapter 08a: Autograd — Forward Pass & Computation Graph

> **Part 2 of 6 — Autodiff Engine**
> `src/ch-08a-autograd-forward/`

---

## What You're Building

A `Value` class that wraps a tensor and records the operation that created it, forming a
**computation graph** during the forward pass. This is the data structure that makes automatic
differentiation possible. It is the core of PyTorch's autograd engine, Micrograd, and every
modern ML framework.

---

## Why This Matters

Without a computation graph, computing gradients means manually deriving $\partial L / \partial w$
for every weight in every architecture — new architecture, new derivation, hours of calculus.

With a computation graph, you write the forward pass once (just normal math), and the backward
pass is computed automatically. The graph records every operation, and backpropagation
(Ch 08b) walks the graph in reverse to compute all gradients at once.

This is the single most intellectually important chapter in the course.

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

## → Next Chapter

**Ch 08b: Autograd — Backward Pass** — implement the `_backward` functions for each
operation and write the `backward()` method that walks the graph in reverse topological
order to accumulate gradients into every node's `.grad`.
