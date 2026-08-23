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

| | what it does | where it lives in the transformer |
|---|---|---|
| `relu` | zeroes negatives | the classic hidden-layer nonlinearity |
| `gelu` | a smooth `relu` | inside every FFN block in GPT (Ch 25) |
| `sigmoid` | squashes to `(0, 1)` | binary outputs, gates |
| `softmax` | turns scores into probabilities | the last step of every attention head (Ch 22) |

> **🗺️ How to read this chapter**
> Read a bit, build a bit — the same rhythm as Ch 09 and Ch 10.
>
> | | Sections | Then |
> |---|---|---|
> | **Read** | 1 → 3 | **Build** `relu` (section 4) |
> | **Read** | 5 | **Build** `sigmoid` (section 6) |
> | **Read** | 7 → 8 | **Build** `gelu` (section 9) |
> | **Read** | 10 | **Build** `softmax` (section 11) |
> | **Read** | 12 | Verify everything |
>
> Good news first: **Chapter 10 was the hard one.** There, shapes changed under you — broadcasting grew them, reductions shrank them, and half the chapter was putting them back. Here, three of the four activations do not change shape at all. Same shape in, same shape out, no `sumToShape`, no `unsqueeze`, no broadcasting. One pattern, three times. Only `softmax` is different, and it gets its own section.

---

## Learning Goals

By the end of this chapter you can:

- Explain why none of `TensorValue`'s seven operations can produce a nonlinearity, and what has to be added instead.
- Add a new differentiable primitive to the engine — the tensor version of what you did in Ch 08.
- Write the elementwise backward pattern from memory, and say why it needs no shape bookkeeping.
- Derive and implement the gradients of `relu`, `sigmoid`, `gelu` and `softmax`.
- Measure why sigmoid stalls deep networks and ReLU does not — with the actual numbers.
- Explain why `softmax` is the one activation that is not elementwise, and what that costs.

---

## Words we'll use in this chapter

| Word | Plain meaning |
|------|---------------|
| **activation** | A function applied between layers to bend the signal. The subject of this chapter. |
| **primitive** | An operation the engine knows how to differentiate — one that owns a `_backward`. |
| **elementwise** | Each output cell depends on exactly one input cell. `relu`, `sigmoid`, `gelu`. |
| **local derivative** | For an elementwise `f`, the value `f'(x)` at one cell. Ch 08's "local gradient". |
| **saturation** | An input region where a function flattens out, so its derivative goes to ~0. |
| **logits** | Raw, unbounded scores — what `softmax` turns into probabilities. |

---

## 1. What your engine cannot do yet

Here is a fact worth sitting with. You can write this:

```typescript
x.matMul(W).add(b)          // a linear layer — Ch 10 gave you this
```

but there is **no** combination of `add`, `mul`, `matMul`, `sum`, `mean`, `reshape` and `transpose` — however long — that produces this:

```
relu(x)        sigmoid(x)        tanh(x)        exp(x)
```

Not "it would be slow" or "it would be awkward". It is impossible. Adding and multiplying can only ever build **polynomials**, and `exp` is not a polynomial — no finite number of adds and multiplies produces it. Same for `tanh`, `log`, and the hard corner in `relu`.

So a nonlinearity cannot be *composed*. It has to be **added to the engine as a new primitive** — a new node type that knows its own forward value and its own derivative.

And you have done exactly this before. In Chapter 08, `Value` started with `add` and `mul`, and then you added `exp`, `log`, `tanh` and `relu` — each one a new primitive with its own `_backward` closure. This chapter is that same act, with tensors in the nodes instead of numbers.

> **A note on where the *forward* comes from.** You are not writing `exp` from scratch. Chapter 06 already built the tensor math — `exp`, `tanh`, `sigmoid`, `pow` — and Chapter 05 built `softmax`. Those give you the forward value for free. **The work in this chapter is the backward.** That is the part the engine cannot infer and you must supply.

---

## 2. Adding a primitive: the pattern you already know

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

*Figure 1: the elementwise pattern. One cell in, one cell out, shapes untouched.*

---

## 3. `relu`, by hand

The simplest activation, and the one whose behaviour you already met in Chapter 09.

$$\text{relu}(x) = \max(0, x)$$

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

## 4. Build it — `relu`

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

## 5. `sigmoid` — and a derivative that reuses the output

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

Sigmoid squashes any real number into `(0, 1)`. On the running row:

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

## 6. Build it — `sigmoid`

**Milestone 2 — `sigmoid`.**

Same five-step recipe as `relu`. Two differences:

- The forward is Chapter 06's `sigmoid(x.data)` — already written, already numerically careful.
- The local derivative is built from `out.data`, not `x.data`.

✅ *Checkpoint:* `sigmoid(0) = 0.5` exactly; the row above reproduces to four decimals; and every output is strictly inside `(0, 1)` for any finite input.

> **Pitfall — `out.data`, not `x.data`.** Writing `σ(x.data)·(1−σ(x.data))` gives the right answer while recomputing the whole exponential. Writing `x.data·(1−x.data)` gives a *wrong* answer that happens to be right at `x = 0` — the same trap as Ch 08's `tanh` backward, which is why its test is checked at `x = 1` and not at the origin.

---

## 7. What saturation costs — the vanishing gradient, measured

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

That flattening is **saturation** — the function has run out of room, so nudging the input barely moves the output.

Now recall Ch 09's deep dive, which showed that a gradient reaching an early layer is a **product** of every local derivative along the way. Put a sigmoid in each of ten layers and, in the *best possible case*, that product is:

$$0.25^{10} \approx 9.5 \times 10^{-7}$$

A gradient a million times smaller than the one at the output. The early layers effectively stop learning, and `step()` is behaving perfectly — it is faithfully multiplying a learning rate by a number near zero.

Now compare `relu`:

$$1^{10} = 1$$

On its active side, `relu`'s derivative is exactly 1 — so the product does not decay at all. **That single fact is why ReLU replaced sigmoid in hidden layers**, and it is the whole reason ReLU exists.

<p align="center">
  <img src="../assets/ch-11/saturation-and-decay.svg" alt="Two panels. The left plots the derivative curves of sigmoid and relu against x from -6 to 6: sigmoid's derivative is a bell peaking at 0.25 at x=0 and falling to nearly zero by x = plus or minus 6, marked saturation at both ends; relu's derivative is a step, exactly 0 for negative x and exactly 1 for positive x. A dashed line marks the 0.25 ceiling that sigmoid's derivative can never exceed. The right panel shows what happens when those factors multiply through depth: a bar chart of the gradient reaching the first layer after n layers, sigmoid at its best case 0.25 to the power n falling from 0.25 to 9.5e-7 across ten layers, while relu stays flat at 1 the whole way. A caption states that on its active side relu's derivative is exactly one, so the product does not decay, and that this is why relu replaced sigmoid in hidden layers." />
</p>

*Figure 2: sigmoid's ceiling is 0.25; relu's is 1. Ten layers turn that into a factor of a million.*

The trade-off is the one you already know: relu's gate can shut permanently (section 3's dying units), while sigmoid never fully dies but starves everything upstream. The next section is the activation that tries to have both.

---

## 8. `gelu` — the smooth gate transformers use

`relu` makes a hard decision at zero: pass completely, or block completely. `gelu` makes a **soft** one — it scales each input by roughly "how likely this value is to be worth keeping":

$$\text{gelu}(x) = x \cdot \Phi(x)$$

where `Φ` is the normal distribution's CDF — an S-curve from 0 to 1. In practice everyone uses the `tanh` approximation, which matches to about `1e-3` and is much faster:

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

*Figure 3: the four curves, each with its derivative. The marker sweeps all four together so the same `x` can be compared across them.*

---

## 9. Build it — `gelu`

**Milestone 3 — `gelu`.**

The same five-step recipe. The forward is the `tanh` approximation above; the constant `√(2/π) ≈ 0.7978845608` is worth naming.

✅ *Checkpoint:* `gelu(0) = 0` exactly, and the row reproduces as `[-0.0454, -0.1588, 0.0000, 0.8412, 1.9546]` — the same values the exercise prints.

> **Verify this one numerically, not by eye.** Its derivative is the messiest algebra in the chapter and the easiest place to drop a term. `checkTensorGradient` from Ch 10 will catch a wrong constant instantly; reading the formula again will not.

---

## 10. `softmax` — the one that is not elementwise

Everything so far shared one property: output cell `i` depended only on input cell `i`. `softmax` breaks it.

$$\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

That denominator sums over **every** element along the axis. So changing one input changes *all* the outputs — they are tied together by the requirement that they sum to 1.

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

## 11. Build it — `softmax`

**Milestone 4 — `softmax`.**

Forward: Chapter 05's `softmax(x.data, axis)`, which already subtracts the max. Default the axis to the last one (`x.data.ndim - 1`) — that is what classification and attention both want.

Backward: the formula above. The weighted sum `Σ_k out.grad_k · s_k` is a `sum` along the same axis with `keepDims = true`, so it broadcasts back cleanly against the full tensor — the Ch 10 machinery, doing exactly what it was built for.

✅ *Checkpoints:* the output sums to 1 along the axis; `softmax(x)` equals `softmax(x + c)`; and the gradient check passes.

---

## 12. Verify

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
5. Why does `sigmoid`'s backward use `out.data` rather than `x.data`? Name the two reasons.
6. Which of the four activations is not elementwise, and what does that change about its backward pass?
7. A `gelu` unit sits at `x = -1`. Compare its gradient with a `relu` unit at the same input. Which one can recover, and why?

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
