# Chapter 11: Activation Functions

> **Part 3 of 6 — Neural Net Primitives**
> Source: [`src/nn/activations.ts`](../../src/nn/activations.ts)
> Tests: [`src/nn/activations.test.ts`](../../src/nn/activations.test.ts)
> Exercise: [`exercises/ch-11-activations.ts`](../../exercises/ch-11-activations.ts)

---

## Where we left off (and why this chapter exists)

Part 2 is finished. You have a graph that records itself, a backward pass that fills every gradient in one sweep, an optimizer that moves parameters, and — since Chapter 10 — an engine that does all of that on tensors.

Count what `TensorValue` can actually do, though. Seven operations:

```
add    mul    matMul    sum    mean    reshape    transpose
```

That is enough to build `(X @ W) + b` — a linear layer, which Chapter 10's closing checkpoint had you assemble. So stack two of them and you have a deeper network?

No. Chapter 09's deep dive already proved why not, with numbers:

```
layer1: 3x + 2     layer2: -4h + 5     →     -12x - 3
```

Two linear layers collapse into one. Ten thousand of them still collapse into one. **Depth buys you nothing until something bends the line between the layers** — and none of your seven operations can bend it.

That is what this chapter adds. Four functions:

| | the idea behind it | where it lives in the transformer |
|---|---|---|
| `relu` | a hinge — one bend, placed where the weights want it | the classic hidden-layer nonlinearity |
| `gelu` | a gate with the certainty removed | inside every FFN block in GPT (Ch 25) |
| `sigmoid` | the step function with its corner rounded off | binary outputs, gates |
| `softmax` | a differentiable `argmax` | the last step of every attention head (Ch 22) |

But do not start from the formulas. Sections 1 to 3 are about the *problem* these were invented to solve — a problem concrete enough to state in four points, and serious enough that it stalled the field for fifteen years. The formulas make far more sense as answers than as definitions.

> **🗺️ How to read this chapter**
> Read a bit, build a bit — the same rhythm as Ch 09 and Ch 10.
>
> | | Sections | Then |
> |---|---|---|
> | **Read** | 1 → 6 (the *why*, then the pattern) | **Build** `relu` (section 7) |
> | **Read** | 8 | **Build** `sigmoid` (section 9) |
> | **Read** | 10 → 11 | **Build** `gelu` (section 12) |
> | **Read** | 13 | **Build** `softmax` (section 14) |
> | **Read** | 15 | Verify everything |
>
> Good news on the mechanics: **Chapter 10 was the hard one.** There, shapes changed under you — broadcasting grew them, reductions shrank them, and half the chapter was putting them back. Here, three of the four activations do not change shape at all. Same shape in, same shape out, no `sumToShape`, no `unsqueeze`, no broadcasting. One pattern, three times. Only `softmax` is different, and it gets its own section.

---

## Learning Goals

By the end of this chapter you can:

- Prove in four lines that no linear model can solve XOR — and explain what that impossibility cost the field.
- State the three-part brief every activation function exists to satisfy: decide, keep a usable slope, and survive being stacked a hundred deep.
- Say why a polynomial — which bends *and* is smooth — still fails, and why that failure is the same collapse as a linear layer's, one level up.
- Explain each of the four as an *idea* rather than a formula — a hinge, a rounded step, a probabilistic gate, a soft `argmax`.
- Explain why none of `TensorValue`'s seven operations can produce a nonlinearity, and what has to be added instead.
- Add a new differentiable primitive to the engine — the tensor version of what you did in Ch 08.
- Write the elementwise backward pattern from memory, and say why it needs no shape bookkeeping.
- Derive and implement the gradients of `relu`, `sigmoid`, `gelu` and `softmax`.
- Measure why sigmoid stalls deep networks and ReLU does not — with the actual numbers.

---

## Words we'll use in this chapter

| Word | Plain meaning |
|------|---------------|
| **activation** | Confusingly, two things. The *noun* is a unit's output — inherited from "a neuron is activated when it fires". The *activation function* is what produces it. Section 2 has the etymology. |
| **nonlinearity** | The modern synonym for "activation function", and the more honest one. |
| **primitive** | An operation the engine knows how to differentiate — one that owns a `_backward`. |
| **elementwise** | Each output cell depends on exactly one input cell. `relu`, `sigmoid`, `gelu`. |
| **local derivative** | For an elementwise `f`, the value `f'(x)` at one cell. Ch 08's "local gradient". |
| **saturation** | An input region where a function flattens out, so its derivative goes to ~0. |
| **logits** | Raw, unbounded scores — what `softmax` turns into probabilities. |
| **step function** | Output 1 above a threshold, 0 below. The original activation, and untrainable — section 2. |
| **hinge** | One `relu` unit's shape: flat, then a bend, then a straight line. |
| **piecewise-linear** | Made of straight segments joined at bends. What a `relu` network computes. |
| **`argmax`** | "Which entry is largest?" — a hard, gradient-free choice. `softmax` is its soft version. |

---

## 1. The problem that made these necessary

Before any formula, the problem. It is small enough to hold in your hand, and it once stopped the entire field for over a decade.

**XOR.** Four points, two classes. Output 1 when the inputs differ, 0 when they match:

```
    x2
     │
   1 │  ●(0,1)=1        ○(1,1)=0
     │
   0 │  ○(0,0)=0        ●(1,0)=1
     └──────────────────────────── x1
        0                1
```

Try to separate the filled dots from the hollow ones with **one straight line**. Go on — the failure is instructive. They sit on opposite diagonals; no line has them on opposite sides.

<p align="center">
  <img src="../assets/ch-11/xor-and-the-hinge.svg" alt="Two panels. The left shows XOR's four points on the x1-x2 square: hollow class-0 circles at (0,0) and (1,1), filled green class-1 circles at (0,1) and (1,0), so the two classes sit on opposite diagonals. A dashed red candidate line sweeps through three different positions and each one leaves a point on the wrong side, captioned that every line fails, with a note on Minsky and Papert 1969 and the fifteen-year winter that followed. The right panel plots f(x) = 2·relu(x+2) − 3·relu(x) + 1.5·relu(x−2) as a piecewise-linear curve from x = -4 to 4, flat at zero until -2, rising steeply to a peak of 4 at x = 0, falling to 2 at x = 2, then rising gently to 3 at x = 4. Three red dots mark the bends at x = -2, 0 and 2, one per relu unit, and a blue marker travels along the curve. Captions note that the slope is constant between bends and changes only at them, that three units give three bends and four straight pieces, and that enough hinges trace any curve. A footer observes that a line cannot treat one case differently from the others but a bend can, as in the XOR solution where h2 stays asleep for three inputs and wakes only at (1,1)." />
</p>

*Figure 1: the impossibility on the left, and the thing that defeats it on the right.*

That is not a lack of imagination. It is provable in four lines. A linear model is `y = w₁x₁ + w₂x₂ + b`, so the four points demand:

```
(0,0) → 0    ⇒   b = 0
(0,1) → 1    ⇒   w₂ + b = 1     ⇒  w₂ = 1
(1,0) → 1    ⇒   w₁ + b = 1     ⇒  w₁ = 1
(1,1) → 0    ⇒   w₁ + w₂ + b = 0
                  but that is 1 + 1 + 0 = 2, and we need 0
```

**Contradiction.** No weights exist. Not "hard to find" — they do not exist, for any linear model, ever. And Chapter 09 already showed that stacking linear layers changes nothing, because they collapse: a hundred layers is still one line, still helpless against four points.

> **This is real history, not a toy.** Minsky and Papert made exactly this argument in *Perceptrons* (1969). It was correct, it was devastating, and funding for neural networks largely dried up for the next fifteen years — the first "AI winter". The field's comeback needed an answer to this one picture.

### The answer, in two units

Here is that answer, with weights chosen by hand so you can check every number:

```
h₁ = relu(x₁ + x₂)          h₂ = relu(x₁ + x₂ − 1)          y = h₁ − 2h₂
```

| input | `h₁` | `h₂` | `y` | want |
|---|---|---|---|---|
| (0,0) | `relu(0) = 0` | `relu(-1) = 0` | 0 | 0 ✓ |
| (0,1) | `relu(1) = 1` | `relu(0) = 0` | 1 | 1 ✓ |
| (1,0) | `relu(1) = 1` | `relu(0) = 0` | 1 | 1 ✓ |
| (1,1) | `relu(2) = 2` | `relu(1) = 1` | 0 | 0 ✓ |

Two `relu` units, and the impossible becomes arithmetic you can verify in your head.

Look closely at *why* it works. `h₂` contributes **nothing** for the first three inputs — `relu` zeroes it — and only wakes up at `(1,1)`, where it fires and subtracts. The network has effectively said: *"treat this last case differently from the others."* That is something no line can do, and it is the whole trick.

So the real question of this chapter is not the vague "how do we bend a line". It is sharper:

> **How can a unit make a *decision* — behave one way here and another way there?**

---

## 2. The obvious answer, and why it fails

If you want a unit to decide, the obvious thing is a **threshold**. Fire when the input clears a bar, stay silent otherwise:

```
step(x) = 1 if x > 0
          0 otherwise
```

This is not an arbitrary choice — it is where the whole field started. A biological neuron either fires or it does not, and McCulloch and Pitts' 1943 model of a neuron, and Rosenblatt's perceptron, both used exactly this. It decides. It is nonlinear. It solves XOR.

And it is completely untrainable. Here is what that actually means.

### Watch one weight try to learn

Take a single unit — one weight, one input, a threshold:

```
h = step(w · x + b)          with   w = 0.5,   x = 2,   b = 0

    w · x + b  =  0.5 × 2 + 0  =  1.0
    h = step(1.0) = 1                        the unit fires
```

Training means adjusting `w`. So ask the Chapter 07 question about it: **if I nudge `w`, what happens to `h`?**

```
   w = 0.50   →   w·x + b = 1.00   →   h = 1
   w = 0.51   →   w·x + b = 1.02   →   h = 1        no change
   w = 0.60   →   w·x + b = 1.20   →   h = 1        no change
   w = 1.00   →   w·x + b = 2.00   →   h = 1        still no change
```

Read the right-hand column. You **doubled** the weight and the unit's output did not budge. The middle column moved every time; the output ignored it, because `step` only cares which side of zero the number is on, never how far.

"How much does `h` move per unit of `w`" is the definition of the gradient — and here it is plainly zero:

$$\frac{\partial h}{\partial w} = 0$$

Now put that into the update rule you wrote in Chapter 09:

$$w \leftarrow w - \eta \cdot 0 \;=\; w$$

**The weight does not move.** Not slowly — not at all. Run a thousand steps and it sits exactly where it started, along with every other weight feeding this unit. The loss curve would be a perfectly horizontal line.

### "But surely it responds *somewhere*?"

It does — at exactly one point. Keep pushing `w` down until the pre-activation crosses zero:

```
   w =  0.01  →   w·x + b =  0.02  →   h = 1
   w = -0.01  →   w·x + b = -0.02  →   h = 0        it flipped
```

But look at *how* it flipped: all at once, at one exact value, with nothing in between. That is not a slope — it is a cliff. The derivative there isn't a big number; it is **undefined**, because the function jumps rather than rises.

So the step function offers you:

- a gradient of **zero** everywhere it is flat — no information about which way to move, and
- **no gradient at all** at the single point where anything happens.

There is nowhere for learning to get a grip. The perceptron could only be trained by a special hand-built rule, and that rule could never be extended through multiple layers — which is precisely why XOR stayed unsolved for so long even though, as section 1 showed, two hidden units are enough.

### The tension every activation resolves

Put the two candidates side by side and the problem becomes exact:

| | makes a decision? | has a usable slope? |
|---|---|---|
| a straight line | ❌ no — same behaviour everywhere | ✅ yes |
| a step | ✅ yes | ❌ no — flat, then a cliff |
| **what we need** | ✅ | ✅ |

That bottom row looks like the whole design brief. It is not — and the fastest way to find the missing part is to try the obvious candidate that satisfies both columns.

### Why not just use a curve? (`x²`, `x³`, a polynomial)

A reasonable objection at this point: a parabola **bends** — so it decides — and it is smooth, so it has a slope everywhere. It passes both tests. Why not use that and be done?

This is not a naive idea; it was tried. "Higher-order units" and "sigma-pi units" were an active line of research in the 1980s and 90s. They lost, and the reasons only appear when you **stack** them.

**They do not survive composition.** Put `x²` in ten successive layers and follow one value through:

```
   x = 2      →  4 → 16 → 256 → 65536 → 4.3e9 → … → 1.3e154 → Infinity
   x = 1.1    →  … → 2.4e42
   x = 1.0    →  1 → 1 → 1 → 1 → 1 → 1 → 1 → 1 → 1 → 1
   x = 0.9    →  … → 1.4e-47
   x = 0.5    →  … → 5.6e-309        (about to reach zero)
```

Anything above 1 **overflows a 64-bit float in ten layers**, starting from an input as ordinary as 2. Anything below 1 collapses to nothing. Only `x = 1` *exactly* holds still — a razor's edge, not an operating range. Now the same ten layers of `relu`:

```
   x = 2      →  2 → 2 → 2 → 2 → 2 → 2 → 2 → 2 → 2 → 2
```

`relu(x) = x` on the active side, so the scale is preserved **exactly**, forever. That is the property that turns out to matter most, and it is invisible until you stack.

**The gradients explode too.** `d/dx x² = 2x`, which grows without bound as the input grows. Ten layers at `x = 10` puts `10¹³` into the first layer's gradient — the exploding-gradient twin of section 10's vanishing problem. `relu`'s local derivative is `1`, so ten layers give `1`.

**And squaring destroys information:** `x²(-3)` and `x²(3)` are both `9`. A unit built on it cannot tell those inputs apart; half the meaning is gone before the next layer sees anything.

**The deepest reason, though, is one you have already met.** Chapter 09 proved:

```
linear ∘ linear  =  linear                 ← the collapse
```

The same closure holds one level up:

```
degree-2 ∘ degree-2      =  degree 4
ten layers of degree 2   =  degree 1024    ← still ONE polynomial
```

Polynomials are **closed under composition as well**. Stacking them does not produce a new *kind* of function — only a higher degree, which a single layer could have produced directly. It is the identical failure to the linear case, hiding one level up where it is harder to notice.

And a high-degree polynomial is precisely what you do not want. Fit a smooth bump with a degree-20 polynomial through 21 *exact* points:

```
   x = 0.90     true = 0.0471     poly =   0.0471
   x = 0.95     true = 0.0424     poly = -39.9524      ← between the points
   x = 0.99     true = 0.0392     poly = -42.4705
```

Exact at every node, wildly wrong between them — *Runge's phenomenon*, and raising the degree makes it **worse**. So the only thing depth buys with polynomials is the thing that ruins them.

**The difference is global versus local.** A polynomial is one rigid formula governing the entire line: change a single coefficient and the curve moves everywhere at once. A `relu` hinge is local — one bend, one place, linear either side — so each unit can take charge of its own region without disturbing the others. Composition therefore means opposite things: stacking polynomials multiplies **degree**, stacking hinges multiplies **regions**. Section 6 is about the second.

### The real design brief

So there is a third column, and it is the one that only reveals itself under depth:

| | makes a decision? | has a usable slope? | survives being stacked? |
|---|---|---|---|
| a straight line | ❌ | ✅ | ✅ (but pointless — it collapses) |
| a step | ✅ | ❌ | — |
| a polynomial | ✅ | ✅ | ❌ explodes or vanishes; degree blows up |
| **what we need** | ✅ | ✅ | ✅ |

That bottom row is the actual brief. **Every activation function ever invented is an attempt to fill it:** keep enough of the step's decisiveness to be nonlinear, recover a slope that learning can use, and stay numerically well-behaved when a hundred copies are stacked on top of each other.

The four in this chapter are four different bargains struck with that one problem.

---

> **Aside — where the name "activation function" comes from, and why it fits badly now**
>
> The vocabulary was borrowed from neuroscience, not invented. When a real neuron fires, it is said to be *activated*; McCulloch and Pitts' paper is titled *"A Logical Calculus of the Ideas Immanent in Nervous **Activity**"*. So an artificial unit's output was called its **activation** — a noun, meaning how switched-on the unit is — and the function producing it became the **activation function**. The terminology still carries that layering:
>
> ```
>      x·w + b        →      f(x·w + b)
>   ───────────           ──────────────
>   "pre-activation"      "the activation"
>   (also: logits)         ← the unit's output
>
>                       f = the ACTIVATION FUNCTION
>                           — named for what it produces
> ```
>
> No single person coined it; it drifted in from biology with the metaphor and hardened into standard terminology through 1980s connectionism.
>
> **And the name has aged badly** — which this section is in a good position to show, because it fits the step function and nothing else. A step unit really does activate, like a neuron. But `relu(5) = 5` is not a neuron activating; it is a hinge passing a magnitude. `gelu` scales by a probability. `softmax` is not even per-unit. The biological metaphor was left behind somewhere around 2010; the name it deposited stayed.
>
> Which is why modern papers often just write **"nonlinearity"** — *"we use a GELU nonlinearity"*. Same object, more honest name: it says what the function *is*, rather than naming a biological process it no longer models. Two names, two eras.

---

## 3. Four answers to one question

The four functions in this chapter are four different ways to resolve that tension. Seeing them as four answers — rather than four formulas — is most of the chapter:

| | the idea | what it gives up |
|---|---|---|
| **`sigmoid`** | Round off the step's corner into a smooth S. Now it has a slope everywhere. | The slope is tiny except near the threshold — section 10 |
| **`relu`** | Keep the hard decision, but make the "on" branch a plain `y = x` instead of flattening out. | Off means *completely* off — those units can die |
| **`gelu`** | Make the decision *probabilistic* rather than a hard cut. | Costs more arithmetic |
| **`softmax`** | *(a different question)* Make "pick the winner" differentiable. | Not a per-unit gate at all |

Three of these are about a single unit deciding whether to pass its input on. `softmax` answers a separate question that appears at the *output* of a network, and it gets its own section.

Notice that `sigmoid` and `relu` sit at opposite ends of the same trade-off — one keeps the smoothness and loses the gradient's size, the other keeps the gradient's size and loses the smoothness. Almost everything in the literature since is a negotiation between those two, and `gelu` is the negotiation transformers settled on.

---

## 4. What your engine cannot do yet

One practical obstacle before building any of them. You can write this:

```typescript
x.matMul(W).add(b)          // a linear layer — Ch 10 gave you this
```

but there is **no** combination of `add`, `mul`, `matMul`, `sum`, `mean`, `reshape` and `transpose` — however long — that produces this:

```
relu(x)        sigmoid(x)        tanh(x)        exp(x)
```

Not "it would be slow" or "it would be awkward". It is impossible. Adding and multiplying can only ever build **polynomials**, and `exp` is not a polynomial — no finite number of adds and multiplies produces it. Same for `tanh`, `log`, and the hard corner in `relu`.

This is the same wall as section 1, seen from inside the code: your engine is made of linear parts, and linear parts compose into linear things. A nonlinearity cannot be *composed*. It has to be **added to the engine as a new primitive** — a node that knows its own forward value and its own derivative.

And you have done exactly this before. In Chapter 08, `Value` started with `add` and `mul`, and then you added `exp`, `log`, `tanh` and `relu` — each one a new primitive with its own `_backward` closure. This chapter is that same act, with tensors in the nodes instead of numbers.

> **A note on where the *forward* comes from.** You are not writing `exp` from scratch. Chapter 06 already built the tensor math — `exp`, `tanh`, `sigmoid`, `pow` — and Chapter 05 built `softmax`. Those give you the forward value for free. **The work in this chapter is the backward.** That is the part the engine cannot infer and you must supply.

---

## 5. Adding a primitive: the pattern you already know

Here is Chapter 08's `relu`, from your own [`value.ts`](../../src/autograd/value.ts), unchanged:

```typescript
relu(): Value {
  const out = new Value(Math.max(0, this.data), [this], "relu");
  out._backward = () => {
    this.grad += (this.data > 0 ? 1 : 0) * out.grad;
  };
  return out;
}
```

Three things happen: compute the forward value, record the parent, and attach a closure that multiplies the upstream gradient by the local derivative. That is a primitive.

The tensor version is the same three things, with two substitutions:

```
Math.max(0, this.data)              →  the same rule applied to every cell
(this.data > 0 ? 1 : 0) * out.grad  →  the same product, every cell at once
```

And here is why this chapter is gentler than the last one. Look at what an **elementwise** operation does to shape:

```
input   [ -2  -1   0   1   2 ]     shape [5]
                ↓  relu, cell by cell
output  [  0   0   0   1   2 ]     shape [5]     ← identical shape
```

Cell `i` of the output depends on cell `i` of the input and nothing else. No broadcasting, no reduction — so backward has **no shape to repair**. No `sumToShape`, no `unsqueeze`. The whole backward is one elementwise multiply:

$$\texttt{x.grad}  \mathrel{+}=  f'(x) \odot \texttt{out.grad}$$

where `⊙` means "multiply cell by cell" — Chapter 03's `mul`. That single line is the backward of `relu`, `sigmoid`, `gelu`, and every elementwise activation you will ever write.

<p align="center">
  <img src="../assets/ch-11/elementwise-backward.svg" alt="A five-cell tensor flowing forward and backward through an elementwise activation. Forward, the input row holding -2, -1, 0, 1, 2 passes cell by cell through relu, each cell mapping only to the cell directly below it, producing 0, 0, 0, 1, 2 with the same shape [5]. Below, the local derivative row shows relu prime at each input: 0, 0, 0, 1, 1. Backward, an upstream gradient row of all ones is multiplied cell by cell by that local derivative row, giving the input gradient 0, 0, 0, 1, 1 — the same shape as the input. A caption notes that because each output cell depends on exactly one input cell, the shape never changes and backward needs no sumToShape, no unsqueeze, and no broadcasting: it is a single elementwise multiply." />
</p>

*Figure 2: the elementwise pattern. One cell in, one cell out, shapes untouched.*

---

## 6. `relu` — a hinge, and what a network builds out of them

`relu` is the answer that keeps the hard decision and simply refuses to flatten out on the "on" side:

$$\text{relu}(x) = \max(0, x)$$

It looks almost insultingly simple — the joke is that a decade of research produced `max(0, x)`. But look at what it *is* geometrically, because that is where the insight lives.

**One `relu` unit is a hinge.** It is flat, and then at one point it bends and becomes a straight line. One bend, at a location the weights choose.

Now stack a few and watch what happens. Take three units and add them with different weights:

$$f(x) = 2\,\text{relu}(x+2) \;-\; 3\,\text{relu}(x) \;+\; 1.5\,\text{relu}(x-2)$$

```
   x   :   -4    -3    -2    -1     0     1     2     3     4
  f(x) :  0.0   0.0   0.0   2.0   4.0   3.0   2.0   2.5   3.0
  slope:    0.0   0.0   2.0   2.0  -1.0  -1.0   0.5   0.5
                        └── bend ──┘└── bend ──┘└── bend ──┘
                            at -2       at 0        at 2
```

The slope is **constant between the bends and changes only at them** — one bend per unit, exactly where that unit's threshold sits. The result is a *piecewise-linear* function: straight segments hinged together.

That is what a ReLU network is. Each unit contributes one hinge; together they carve the input space into regions, and **inside each region the whole network is just a linear function**. Different region, different linear function. A network with many ReLU units is a committee of linear models, each one taking charge of its own patch of input space.

This is also the intuition behind the *universal approximation* result you may have heard quoted. Any smooth curve can be traced as closely as you like by enough short straight segments — so enough hinges can approximate any function. Add units, get more pieces, fit finer detail. You do not need exotic mathematics; you need enough bends.

And it is exactly what happened in XOR back in section 1: `h₂`'s hinge sat precisely so that only `(1,1)` landed on its active side. One bend, placed to isolate one case.

### The behaviour you already met

Chapter 09 introduced `relu`'s gate on a scalar. Here is the same thing on a row.

Take five inputs spanning the interesting region — this row is the running example for the whole chapter:

```
x        = [ -2  -1   0   1   2 ]     shape [5]

relu(x)  = [  0   0   0   1   2 ]     negatives flattened to zero
```

**The derivative** is a switch, not a scale:

$$\text{relu}'(x) = \begin{cases} 1 & x > 0 \\ 0 & x \le 0 \end{cases}$$

```
relu'(x) = [  0   0   0   1   1 ]
```

Read that row as a **gate**: where the input was positive the gradient passes through completely untouched (`×1`); where it was negative the gate is shut and nothing passes (`×0`).

**One backward pass, by hand.** Say the upstream gradient is all ones:

```
out.grad = [  1   1   1   1   1 ]

x.grad  =  relu'(x)  ⊙  out.grad
        =  [ 0  0  0  1  1 ]  ⊙  [ 1  1  1  1  1 ]
        =  [ 0  0  0  1  1 ]
```

Those last two cells passed their gradient intact. The first three received **exactly zero** — and Chapter 09 already told you what that means for a parameter: *a parameter with zero gradient does not move.* If a unit's input stays negative, its gradient stays zero, and it never learns again. That is the **dying ReLU**, and it is the same mechanism as the frozen parameter in Ch 09's deep dive, where `∂f/∂a = 0` left `a` motionless for the entire run.

> **What about `x = 0` exactly?** The derivative genuinely does not exist there — the function has a corner. Every library picks a convention and moves on; we use `0`, matching what you chose in Ch 08. This is also why the test is named "gradient check passes (x ≠ 0)": a centered finite difference straddling the corner averages the two one-sided slopes to `0.5` and would disagree with *any* convention.

---

## 7. Build it — `relu`

Open [`activations.ts`](../../src/nn/activations.ts). Each function carries the same paired guidance as `grad.ts`: your Ch 08 scalar version quoted, then what changes for tensors.

**Milestone 1 — `relu`.**

The recipe, identical for every elementwise activation in this chapter:

1. Compute the forward tensor — apply the rule to every cell.
2. Wrap it: `const out = new TensorValue(forwardTensor)`.
3. Wire the parent by hand: `out._inputs = [x]`.
4. Attach `out._backward`: accumulate `mul(localDerivative, out.grad!)` into `x`.
5. `return out`.

For the forward, `applyFn` from Ch 03 maps a plain function over every cell — `applyFn(x.data, v => Math.max(0, v))`. Build the local-derivative tensor the same way.

✅ *Checkpoint:* on the row above, forward gives `[0, 0, 0, 1, 2]` and backward (with upstream ones) gives `[0, 0, 0, 1, 1]`.

> **One decision to make first.** Accumulating into `x.grad` needs the null-aware pattern from Ch 10 — *first contribution assigns, later ones add*. You already wrote that as `accumulate` in [`grad.ts`](../../src/autograd/grad.ts), but it is a private function there, so this file cannot see it. Either **export it** (one keyword, and every activation reuses it — the recommended route) or write the two lines here. Make the choice once, now, rather than four times.
>
> No `sumToShape` in this chapter: the contribution already has the input's shape, because nothing broadcast.

---

## 8. `sigmoid` — the step with its corner rounded off

This is the other answer to section 2's problem, and historically it came first by decades.

The step function was right in spirit and untrainable in practice. So: *keep the shape, round off the corner.* Take the hard jump from 0 to 1 and smooth it into a curve that rises gradually through the same territory:

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

```
        step                        sigmoid
                                              ╭────────
    ────────┐   ╭────────                 ╭───╯
            │   │                      ╭──╯
            └───╯                 ─────╯

  jumps at 0, flat elsewhere      rises smoothly through 0
  slope: 0 or undefined           slope: never zero, never undefined
```

That is the entire idea. `sigmoid` is a **differentiable step** — it still says "roughly off" for very negative inputs and "roughly on" for very positive ones, but it gets there by a route that has a slope the whole way. And it happens to be interpretable: an output in `(0, 1)` reads naturally as a probability, or as a neuron's firing *rate* rather than a binary spike, which is why it dominated for so long. (The curve itself long predates neural networks — Verhulst introduced it in the 1830s for population growth.)

On the running row:

```
x           = [ -2       -1       0       1       2      ]
sigmoid(x)  = [  0.1192   0.2689  0.5000  0.7311  0.8808 ]
```

Note `σ(0) = 0.5` exactly, and that the output never reaches 0 or 1 — it only approaches them.

**The derivative has a lovely form:**

$$\sigma'(x) = \sigma(x)\,\bigl(1 - \sigma(x)\bigr)$$

```
sigmoid'(x) = [  0.1050   0.1966  0.2500  0.1966  0.1050 ]
```

Look at what that formula needs: `σ(x)` — which is the value you **already computed in the forward pass**. You do not recompute the exponential; you read `out.data`. That is exactly the trick from Ch 08's `exp` and `tanh`, where the derivative reused `out.data` rather than the input, and for the same two reasons: it is cheaper, and it cannot drift from the value the forward pass actually used.

> **Where this rule was proved:** Ch 08b's [local gradient table](../part-2-autodiff/ch-08b-autograd-backward.md#local-gradient-table) lists it alongside `exp` and `tanh`. If `σ' = σ(1−σ)` looks like it came from nowhere, the one-line derivation is: write `σ = (1 + e^{-x})^{-1}`, differentiate with the chain rule to get `e^{-x}/(1+e^{-x})^2`, then notice that equals `σ · (1 − σ)`.

---

## 9. Build it — `sigmoid`

**Milestone 2 — `sigmoid`.**

Same five-step recipe as `relu`. Two differences:

- The forward is Chapter 06's `sigmoid(x.data)` — already written, already numerically careful.
- The local derivative is built from `out.data`, not `x.data`.

✅ *Checkpoint:* `sigmoid(0) = 0.5` exactly; the row above reproduces to four decimals; and every output is strictly inside `(0, 1)` for any finite input.

> **Pitfall — `out.data`, not `x.data`.** Writing `σ(x.data)·(1−σ(x.data))` gives the right answer while recomputing the whole exponential. Writing `x.data·(1−x.data)` gives a *wrong* answer that happens to be right at `x = 0` — the same trap as Ch 08's `tanh` backward, which is why its test is checked at `x = 1` and not at the origin.

---

## 10. What saturation costs — the vanishing gradient, measured

Look again at that derivative row, and notice its shape:

```
sigmoid'(x) = [  0.1050   0.1966   0.2500   0.1966   0.1050 ]
                                     ↑
                           the maximum, at x = 0
```

**Sigmoid's derivative never exceeds 0.25.** Anywhere. And it collapses fast as you move away from the origin:

```
sigmoid'(0)  = 0.2500
sigmoid'(2)  = 0.1050
sigmoid'(4)  = 0.0177
sigmoid'(6)  = 0.0025
sigmoid'(10) = 0.0000454
```

That flattening is **saturation** — the function has run out of room, so nudging the input barely moves the output. And notice *why* it is unavoidable: `sigmoid` squeezes the entire infinite real line into a box of height 1. A curve with nowhere left to rise must go flat, and flat means no gradient. **The smoothing that made the step trainable is the same property that starves it.** Rounding the corner cost something, and this is the bill.

Now recall Ch 09's deep dive, which showed that a gradient reaching an early layer is a **product** of every local derivative along the way. Put a sigmoid in each of ten layers and, in the *best possible case*, that product is:

$$0.25^{10} \approx 9.5 \times 10^{-7}$$

A gradient a million times smaller than the one at the output. The early layers effectively stop learning, and `step()` is behaving perfectly — it is faithfully multiplying a learning rate by a number near zero.

Now compare `relu`:

$$1^{10} = 1$$

On its active side, `relu`'s derivative is exactly 1 — so the product does not decay at all. **That single fact is why ReLU replaced sigmoid in hidden layers**, and it is the whole reason ReLU exists.

> **This was a real crisis, and the diagnosis took years.** Through the 1990s and 2000s, deep networks were known not to train — add layers and performance got *worse*, which felt like a law of nature rather than a bug. Glorot and Bengio's 2010 paper put a name and a number on it: gradients were dying on the way back, exactly by the multiplication above. Once the cause was clear the cure was almost embarrassing — stop using a saturating function — and ReLU's adoption is a large part of why deep learning worked from roughly 2012 onward. `max(0, x)` unlocked depth.

<p align="center">
  <img src="../assets/ch-11/saturation-and-decay.svg" alt="Two panels. The left plots the derivative curves of sigmoid and relu against x from -6 to 6: sigmoid's derivative is a bell peaking at 0.25 at x=0 and falling to nearly zero by x = plus or minus 6, marked saturation at both ends; relu's derivative is a step, exactly 0 for negative x and exactly 1 for positive x. A dashed line marks the 0.25 ceiling that sigmoid's derivative can never exceed. The right panel shows what happens when those factors multiply through depth: a bar chart of the gradient reaching the first layer after n layers, sigmoid at its best case 0.25 to the power n falling from 0.25 to 9.5e-7 across ten layers, while relu stays flat at 1 the whole way. A caption states that on its active side relu's derivative is exactly one, so the product does not decay, and that this is why relu replaced sigmoid in hidden layers." />
</p>

*Figure 3: sigmoid's ceiling is 0.25; relu's is 1. Ten layers turn that into a factor of a million.*

The trade-off is the one section 3 set out: relu's gate can shut permanently (its dying units), while sigmoid never fully dies but starves everything upstream. The next section is the activation that tries to have both.

---

## 11. `gelu` — a gate with the certainty removed

`relu` decides with a rule: *is `x` above zero?* `gelu` asks a stranger question — and where it comes from is more interesting than the formula.

**Suppose the threshold itself were random.** Instead of always comparing against 0, compare `x` against a random draw from a standard normal distribution, and keep `x` only if it wins:

```
keep x  with probability  P(Z ≤ x),  where Z ~ Normal(0, 1)
drop it otherwise
```

A large `x` almost always beats the draw and survives. A very negative `x` almost never does. An `x` near zero is a coin flip. That is a *stochastic gate* — closely related to dropout, which also randomly keeps or discards activations.

Random behaviour is awkward to train, so take the **expected value** instead of actually flipping the coin. The expectation of "keep `x` with probability `p`, else 0" is simply `x · p`:

$$\text{gelu}(x) = x \cdot \Phi(x) \qquad\text{where } \Phi(x) = P(Z \le x)$$

That is the whole derivation, and it explains the shape for free — no curve-fitting, no arbitrary smoothing:

```
   x   |  P(Z ≤ x)  |  gelu = x·P  |  relu
  -----+------------+--------------+--------
   -2  |   0.0228   |   -0.0455    |  0.0000
   -1  |   0.1587   |   -0.1587    |  0.0000
    0  |   0.5000   |    0.0000    |  0.0000
    1  |   0.8413   |    0.8413    |  1.0000
    2  |   0.9772   |    1.9545    |  2.0000
```

Read the middle column as a survival probability. At `x = 2` the input survives 98% of the time, so almost all of it passes. At `x = 0` it is exactly a coin flip — which is why, as you will see, `gelu`'s *derivative* at zero is `0.5`: the gate is precisely half open. At `x = -1` it survives 16% of the time, so a *small* amount gets through rather than nothing at all.

`relu` is what you get if you make that gate deterministic: threshold at zero, survive with probability 1 or 0. **`gelu` is `relu` with the certainty removed.**

In practice everyone uses a `tanh` approximation of `Φ`, which matches to about `1e-3` and is much faster:

$$\text{gelu}(x) \approx 0.5\,x\left(1 + \tanh\!\left(\sqrt{\tfrac{2}{\pi}}\left(x + 0.044715\,x^3\right)\right)\right)$$

On the running row:

```
x         = [ -2       -1       0       1       2      ]
relu(x)   = [  0.0000   0.0000  0.0000  1.0000  2.0000 ]
gelu(x)   = [ -0.0454  -0.1588  0.0000  0.8412  1.9546 ]
```

Read the two rows against each other, because the comparison *is* the explanation:

- At `x = 2`, gelu gives `1.9546` — almost the full value, like relu's `2`.
- At `x = -1`, relu gives a hard `0`; gelu gives `-0.1588` — small, but **not zero**, so a gradient still flows.
- At `x = 0` both give `0`, but relu has a corner there while gelu is smooth.

That "small but not zero" is the point. A gelu unit sitting at a negative input is discouraged, not executed — it keeps a path back to the optimizer and can recover. This is why GPT-2 and essentially every modern transformer use it inside the feedforward block (Ch 25).

**Its derivative** is genuinely messier than the others — differentiate the approximation with the product and chain rules. Two options, and both are legitimate:

1. **Compose it.** Build the forward out of primitives you already have — but `TensorValue` has no `tanh` or `pow`, so this needs those added first.
2. **Write it directly.** Compute the derivative into a tensor with `applyFn`, exactly as with `relu` and `sigmoid`.

Option 2 fits this chapter's pattern and is what the guidance in `activations.ts` walks through.

```
gelu'(x) = [ -0.0861  -0.0830  0.5000  1.0830  1.0861 ]
```

Two features worth noticing, because they surprise people: at `x = 0` the derivative is `0.5`, not 0 or 1 — the gate is *half* open. And at `x = -2` and `x = -1` the derivative is **negative**, which means `gelu` is not monotonic; it dips slightly below zero before flattening. That is deliberate, and it is visible in the curve.

<p align="center">
  <img src="../assets/ch-11/four-activations.svg" alt="Four activation curves plotted on shared axes from x = -3 to 3, each with its derivative drawn beneath it as a lighter line, and an animated marker sweeping left to right across all four in step. relu is two straight segments, flat at zero for negative x then rising at 45 degrees, with a step derivative of 0 then 1 and a marked corner at the origin. gelu closely follows relu for large positive x but curves smoothly through the origin and dips slightly negative around x = -1 before flattening, its derivative peaking just above 1 and going slightly negative on the left. sigmoid is an S-curve from 0 to 1 crossing 0.5 at the origin, with a bell-shaped derivative peaking at 0.25. tanh is an S-curve from -1 to 1 through the origin, with a bell-shaped derivative peaking at 1.0, four times sigmoid's peak. Values on the shared row x = -2, -1, 0, 1, 2 are labelled beneath each curve." />
</p>

*Figure 4: the four curves, each with its derivative. The marker sweeps all four together so the same `x` can be compared across them.*

---

## 12. Build it — `gelu`

**Milestone 3 — `gelu`.**

The same five-step recipe. The forward is the `tanh` approximation above; the constant `√(2/π) ≈ 0.7978845608` is worth naming.

✅ *Checkpoint:* `gelu(0) = 0` exactly, and the row reproduces as `[-0.0454, -0.1588, 0.0000, 0.8412, 1.9546]` — the same values the exercise prints.

> **Verify this one numerically, not by eye.** Its derivative is the messiest algebra in the chapter and the easiest place to drop a term. `checkTensorGradient` from Ch 10 will catch a wrong constant instantly; reading the formula again will not.

---

## 13. `softmax` — a differentiable `argmax`

The first three answered section 2's question: *how does one unit decide whether to pass its input on?* `softmax` answers a completely different one, and it shows up at the **end** of a network rather than between layers.

**The question:** a classifier produces one score per class — say `[1, 2, 3]` for three classes. You want to know which class it picked. The obvious operation is `argmax`: winner takes all.

```
argmax([1, 2, 3])  =  [0, 0, 1]        class 3 wins, everyone else gets nothing
```

And now you already know what is wrong with it, because **`argmax` is section 2's step function all over again**. Nudge the losing scores and the output does not move at all — zero gradient. Nudge them past the winner and it snaps discontinuously. It decides, and it cannot be trained through.

So apply the same medicine: **soften it.** Keep "the biggest score gets the most weight", but let every score keep a share that varies smoothly:

$$\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

The name is not decoration — **"softmax" means "soft argmax"**, and that is precisely what it is.

You can watch it become `argmax` by turning down a temperature knob. Divide the scores by `T` before exponentiating:

```
logits [1, 2, 3]        argmax would be [0, 0, 1]

  T = 3      0.2302   0.3213   0.4484      nearly uniform — barely committed
  T = 1      0.0900   0.2447   0.6652      ← our softmax
  T = 0.5    0.0159   0.1173   0.8668      sharper
  T = 0.1    0.0000   0.0000   1.0000      argmax, to four decimals
```

As `T → 0` softmax *becomes* `argmax`; as `T` grows it flattens toward "no opinion". Our version is `T = 1` — soft enough to have a gradient everywhere, sharp enough to name a winner. (That knob is real: you will meet it again as the temperature setting when sampling from a language model in Ch 30, where it controls how adventurous the generated text is.)

The exponential is what guarantees the outputs behave like probabilities: `eˣ` is always positive, so no score can come out negative, and dividing by the total forces them to sum to 1.

### The consequence: outputs are tied together

That denominator sums over **every** element along the axis. So changing one input changes *all* the outputs — they are coupled by the requirement that they sum to 1. Push one class's score up and the others must give way; there is a fixed amount of probability to share.

This is why `softmax` is the one activation in this chapter that is **not elementwise**, and it is the reason its backward pass looks different from the other three.

```
x          = [ 1         2         3        ]
softmax(x) = [ 0.090031  0.244728  0.665241 ]     sums to 1.0
```

That is the point of it: `softmax` turns arbitrary scores ("logits") into a probability distribution. The largest input gets the largest share, but everything gets something.

### The two properties that matter

**1. Shifting the input changes nothing.** Subtract any constant from every element and the output is identical:

```
softmax([1, 2, 3])  = [ 0.090031  0.244728  0.665241 ]
softmax([0, 1, 2])  = [ 0.090031  0.244728  0.665241 ]     ← the same
```

Algebraically, `e^{x_i - c}` puts a factor of `e^{-c}` in every numerator *and* in the denominator, where it cancels. Chapter 05 proved this in [why subtract the max](../deep-dives/ch-05-why-subtract-the-max.md).

**2. That invariance is what makes it safe.** `exp(1000)` is `Infinity`, and `Infinity / Infinity` is `NaN` — one large logit would destroy the whole distribution. Since shifting is free, always shift by the maximum first, making the largest exponent `e^0 = 1`:

```
softmax([1000, 1001, 1002])  =  [ 0.090031  0.244728  0.665241 ]
```

Identical to `softmax([1, 2, 3])`, because those inputs differ by a constant. Chapter 05's `softmax` already does this subtraction for you — the forward pass is one call.

### Its backward is a Jacobian, not a scalar

For the elementwise activations, each input had one local derivative. Here, changing `x_j` moves *every* output, so the local derivatives form a matrix:

$$\frac{\partial s_i}{\partial x_j} = s_i(\delta_{ij} - s_j)$$

which the backward pass collapses to a form that is much friendlier than it looks:

$$\texttt{x.grad}_j = s_j\left(\texttt{out.grad}_j - \sum_k \texttt{out.grad}_k\, s_k\right)$$

In words: *take the upstream gradient, subtract its `s`-weighted average, then scale by `s`.* That is one weighted sum along the axis plus two elementwise operations — no matrix is ever built.

> **In practice you will rarely call this.** Softmax is almost always followed immediately by cross-entropy loss, and the two together have a famously simple combined gradient: `s − y_true` — the softmax output minus the one-hot label. Chapter 12 implements that fused form, and it is both faster and more numerically stable than composing the two. You are implementing the standalone version here because attention (Ch 22) uses softmax *without* a loss attached.

---

## 14. Build it — `softmax`

**Milestone 4 — `softmax`.**

Forward: Chapter 05's `softmax(x.data, axis)`, which already subtracts the max. Default the axis to the last one (`x.data.ndim - 1`) — that is what classification and attention both want.

Backward: the formula above. The weighted sum `Σ_k out.grad_k · s_k` is a `sum` along the same axis with `keepDims = true`, so it broadcasts back cleanly against the full tensor — the Ch 10 machinery, doing exactly what it was built for.

✅ *Checkpoints:* the output sums to 1 along the axis; `softmax(x)` equals `softmax(x + c)`; and the gradient check passes.

---

## 15. Verify

**Milestone 5.** Run `checkTensorGradient` on all four, exactly as in Ch 10.

```bash
bun test src/nn/activations.test.ts
bun run exercises/ch-11-activations.ts
```

Two notes specific to this chapter:

- **Check `relu` away from zero.** The corner has no derivative, and a centered difference across it averages the one-sided slopes to `0.5`, disagreeing with every convention. Test at `±2`, not `0`.
- **`gelu` is where the check earns its keep.** Its derivative has the most terms and the least intuition, so a dropped constant produces plausible-looking numbers. Finite differences do not care how plausible they look.

---

## What to Implement

| Symbol | Description |
|---|---|
| `relu(x)` | `max(0, x)`; backward is a 0/1 gate |
| `sigmoid(x)` | `1/(1+e⁻ˣ)`; backward is `σ(1−σ)`, built from `out.data` |
| `gelu(x)` | the `tanh` approximation; backward written directly |
| `softmax(x, axis?)` | Ch 05's stable forward; backward is the weighted-subtraction form |

---

## Common Pitfalls

- **Building the derivative from `x.data` when it should come from `out.data`** — `sigmoid` and `tanh` both reuse the output. The `x.data` version is right at `x = 0` and wrong everywhere else.
- **Testing `relu`'s gradient at exactly 0.** No derivative exists there; a finite difference reports `0.5`.
- **Forgetting that `softmax` needs an axis.** Default to the last one, and make it explicit — attention will pass others.
- **Composing `softmax` and cross-entropy separately in Ch 12.** Use the fused gradient; it is simpler and more stable.
- **Trusting `gelu` by inspection.** Run the finite-difference check.
- **Expecting `sigmoid` in a deep hidden stack to train.** Section 7's arithmetic says it will not.

---

## Self-Check Questions

1. `relu([-3, -1, 0, 2, 5])` — what is the output, and what is the gradient row if the upstream gradient is all ones?
2. Why can't `exp` be built from `add` and `mul`? What does that force you to do instead?
3. `sigmoid'` peaks at `0.25`. What is the best-case gradient reaching layer 1 of a 20-layer sigmoid stack? What is it for relu?
4. Show algebraically that `softmax(x)` = `softmax(x + c)` for any constant `c`.
5. `x²` bends and has a slope everywhere, so it passes the first two tests in section 2's brief. Give two separate reasons it still fails — one about magnitudes under stacking, one about what `degree-2 ∘ degree-2` produces.
6. Why does `sigmoid`'s backward use `out.data` rather than `x.data`? Name the two reasons.
7. Which of the four activations is not elementwise, and what does that change about its backward pass?
8. A `gelu` unit sits at `x = -1`. Compare its gradient with a `relu` unit at the same input. Which one can recover, and why?

---

## Further Reading

- [Nair & Hinton — Rectified Linear Units (2010)](https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf) — the paper that made ReLU standard.
- [Hendrycks & Gimpel — Gaussian Error Linear Units](https://arxiv.org/abs/1606.08415) — GELU, as used in GPT.
- [Stanford CS231n — activations](https://cs231n.github.io/neural-networks-1/#actfun) — a compact side-by-side comparison.
- [Deep dive: why subtract the max](../deep-dives/ch-05-why-subtract-the-max.md) — the shift-invariance proof behind stable softmax.

---

## Checkpoint

Your engine can now bend a line. That is the last structural thing it was missing.

Prove it before moving on: build `relu((X @ W1) + b1)` and feed it into a second linear layer. Two layers with a nonlinearity between them — which, unlike Chapter 09's collapsing pair, genuinely cannot be rewritten as one. Call `backward()` and confirm gradients reach `W1`: they travel back through the second layer, through the activation's gate, and into the first.

That is a two-layer neural network, built entirely from parts you wrote. Chapter 13 gives it a proper `Linear` class and a sensible initialisation; nothing about the mathematics changes.

---

## Next Chapter

**[Loss Functions](ch-12-loss-functions.md)** — every `backward()` so far has started from a `sum()` you added by hand. Chapter 12 replaces it with a real objective: mean squared error for regression, cross-entropy for classification — including the fused softmax-plus-cross-entropy gradient this chapter deferred.
