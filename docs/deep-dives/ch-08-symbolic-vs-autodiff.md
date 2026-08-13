# Deep Dive: Why Can't We Just Use Ordinary Calculus?

> Optional reading for Chapter 08 — pure perspective, no code, readable any time after 08a.
>
> **TL;DR — Question:** calculus already differentiates anything — so why invent autodiff?
> **Answer:** symbolic differentiation produces a *formula*, and formulas explode at network
> scale (expression swell). Autodiff produces the derivative's *value at a point* — which is
> the only thing training ever needs, computed exactly, at the cost of one extra pass.
> **Read when:** you (or a math-background friend) ask "isn't this all just the chain rule
> we already know?" — yes, and here's why that isn't enough.

---

The question that led to autodiff's invention. And the honest answer up front: **for one function on paper, you can, and you should.** Autodiff exists because *computers* can't do what *you* do — not efficiently, and not at the scale a neural network demands. Let's watch ordinary calculus break down in three steps.

## Case 1 — a simple function

$$f(x) = x^2 + \sin(x)$$

You write $f'(x) = 2x + \cos(x)$ instantly. Done. No autodiff needed, and none wanted.

## Case 2 — a bigger function

$$f(x) = \sin\!\Big(\exp\big(x^2 + \log(x^3 + \sqrt{x})\big)\Big)$$

Still doable by hand. Chain rule three layers deep, a little bookkeeping, manageable on one sheet of paper.

## Case 3 — a neural network

Now the function is:

```
input → matMul → add bias → ReLU → matMul → add bias → GELU
      → … (repeat for 96 layers) … → softmax → loss
```

This is *still just a mathematical function* — one formula from input to loss. But written out symbolically, that formula spans **millions of terms**. Printing it would take thousands of pages. Differentiating it symbolically, per weight, for 124 million weights? In principle, yes. In practice, look at what happens next.

## Expression swell — how symbolic differentiation blows up

Take the derivative of $\sin(x^2)$: it's $2x\cos(x^2)$. Fine — the derivative is about as big as the function.

Now nest it:

$$f(x) = \sin(\sin(\sin(\cdots\sin(x)\cdots)))) \qquad \text{(1000 nested sines)}$$

Its symbolic derivative is

$$\cos(\sin(\sin(\cdots))) \times \cos(\sin(\cdots)) \times \cdots \times \cos(x)$$

— a product of 1000 factors, where **each factor contains a full copy of a nested subexpression**. The derivative *expression* grows far faster than the original computation. This is called **expression swell**, and it's the disease that kills symbolic differentiation at scale.

Now watch autograd handle the same monster: the forward pass computes 1000 sines, storing 1000 numbers. The backward pass walks back through 1000 nodes, multiplying by $\cos(\text{stored value})$ at each — 1000 cheap steps, **one number per node, never an expression**. No swell. The nesting that made the symbolic form explode is exactly what the graph handles for free.

## The real difference — a formula versus a number

Here is the cleanest way to see it. Symbolic differentiation and autodiff *answer different questions*:

| | Input | Output |
|---|---|---|
| **Symbolic** | the formula $\sin(x^2)$ | a **new formula**: $2x\cos(x^2)$ |
| **Automatic** | the recorded steps, *and a point* $x = 2$ | a **number**: $4\cos(4) \approx -2.61$ |

Symbolic differentiation produces a reusable artifact — a formula you can study, simplify, publish. Autodiff produces the derivative's *value at one specific point*, and nothing else.

And that's not a weakness — because **the value is all machine learning ever needs**. During training you stand at the current weights and ask one question: *"at these exact values, which way is downhill?"* You never need the gradient's formula; you need today's number. Then the weights change, and tomorrow you need tomorrow's number — computed just as cheaply.

This is also why PyTorch never shows you a derivative formula. With 500 million parameters, the symbolic gradient would be an expression of 500 million variables — unprintable, unusable, and pointless, since the *next* training step would need it re-evaluated anyway.

## An analogy

You're driving Delhi → Mumbai and want to know your fuel consumption.

- **The symbolic approach:** first derive a complete general equation — Fuel(distance, speed, temperature, wind, traffic, road slope, tyre pressure, …) — valid for *every possible trip*. Then plug in today's values.
- **The autodiff approach:** today's values are already known — the trip is *happening*. Just measure today's consumption directly.

If you needed to publish a paper on fuel economy, you'd want the general equation. But you just want to drive — and training a network is ten million consecutive trips, each needing only *that trip's* number.

## The three kinds of differentiation, side by side

| Kind | What it produces | Exact? | Scales to millions of inputs? |
|---|---|:---:|:---:|
| **Symbolic** | a new formula | ✅ | ❌ expression swell |
| **Numerical** | approximate value at a point | ❌ truncation error | ❌ `O(N)` passes |
| **Automatic** | **exact** value at a point | ✅ | ✅ one backward pass |

(These are exactly the routes of the [three routes deep dive](ch-08-three-ways-to-a-gradient.md): "by hand" is symbolic, "nudge it" is numerical, autograd is automatic. The cost column has [its own deep dive](ch-08-why-reverse-mode.md).)

Autodiff takes the exactness of symbolic (it applies *real* derivative rules, not approximations) and the evaluate-at-a-point cheapness of numerical (it only ever touches numbers, never expressions). It is not a compromise between the two — it's the best property of each, at the same time.

So, the final word: **if you're doing mathematics, do use ordinary calculus** — symbolic differentiation is the natural tool for understanding one function deeply. But training a network means re-computing one enormous gradient at ever-changing points, millions of times — and for that, recording the computation once and replaying the chain rule backward is not a shortcut around calculus. It **is** calculus, industrialized.

---

## Pen-and-paper exercises

1. **Expression swell, quantified.** For 1000 nested sines, the symbolic derivative is a product of 1000 cosine factors, each wrapping a nested subexpression up to ~1000 sines deep — roughly how many `sin`/`cos` evaluations would naively evaluating that whole expression cost? Compare with autograd's bill: 1000 stored forward values + 1000 backward multiplications. (You don't need an exact count — the point is the *order* of the difference.)
2. Write out the *symbolic* derivative of `sin(sin(sin(x)))` in full. Now imagine the 10-deep version. At what depth would you stop wanting to write it — and what, exactly, does autograd store instead at that depth?
