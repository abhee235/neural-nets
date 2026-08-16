# Chapter 09: Gradient Descent

> **Part 2 of 6 — Autodiff Engine**
> Source: [`src/optim/sgd.ts`](../../src/optim/sgd.ts)
> Tests: [`src/optim/sgd.test.ts`](../../src/optim/sgd.test.ts)
> Exercise: [`exercises/ch-09-gradient-descent.ts`](../../exercises/ch-09-gradient-descent.ts)

---

## Where we left off (and why this chapter exists)

Chapter 08 built an engine that answers one question: *"how does the loss respond to each parameter?"* Call `backward()` and every node in the graph ends up holding its own `∂L/∂θ`.

And then… nothing happens. The gradients sit there. `value.test.ts` checks that they are *correct*, but nothing in the library has ever changed a parameter. The model cannot learn, because knowing which way is downhill is not the same as walking.

This chapter is the walk. It is startlingly small — one subtraction per parameter — and it is the same rule that trains the GPT in Chapter 30. The only thing that changes between here and there is how many parameters are in the list.

So why give it a chapter? Because this is the first code that **closes the loop**: forward, loss, backward, update, repeat. Every warning you have been given since Ch 08 has been theoretical until now. Here they become bugs you can watch happen. The rule takes ten minutes; the loop around it is what takes the chapter.

> **🗺️ How to read this chapter**
> This one alternates between reading and building, so don't try to read it end to end first.
>
> | | Sections | Then |
> |---|---|---|
> | **Read** | 1 → 6 | **Build** `SGD` (§7) |
> | **Read** | 8 → 10 | **Build** `SGDMomentum` (§11) |
> | **Read** | 12 → 13 | Run the exercise |
>
> Sections 12 and 13 are context, not prerequisites — you can finish every line of code without them. Nothing in this chapter requires the [deep dive](../deep-dives/ch-09-how-big-a-step.md); save it for after your tests pass.

---

## Learning Goals

By the end of this chapter you can:

- Apply `θ ← θ − η∇L` to one parameter and to a list of parameters, by hand and in code.
- Read a gradient as a sentence about what happens to the loss, not just as a number.
- Write the five-stage training loop from memory, and say what breaks if you reorder it.
- Explain why the parameter update must happen *outside* the computation graph.
- Say what a neural network is in terms of the operations you have already built, and why Ch 08's test functions were never meant to be models.
- Predict — not just observe — whether a given learning rate will converge, oscillate, or diverge.
- Explain what momentum remembers between steps, and why that memory shapes the API.
- Diagnose the three classic failures: a loss that climbs, a loss that never moves, and a model that trains but never improves.

---

## Words we'll use in this chapter

Skim this now and come back to it — each word is explained again where it first matters.

| Word | Plain meaning |
|------|---------------|
| **optimizer** | The object that owns the parameters and knows how to move them. |
| **parameter** | A `Value` the model is allowed to change — a weight. An input, by contrast, is fixed. |
| **learning rate** (`η`, "eta") | Step size. How far to move per unit of gradient. |
| **step** | One application of the update rule to every parameter. |
| **training loop** | forward → loss → backward → step → zero, repeated. |
| **convergence** | The loss settling near a minimum and staying there. |
| **divergence** | The loss growing without bound. Almost always too large a learning rate. |
| **velocity** | Momentum's memory of how it has been moving recently. |
| **hyperparameter** | A number *you* choose (η, β) rather than one the model learns. |

---

## 1. Why we need this at all

Imagine you are standing partway up a mountain, and you want to reach the lowest point in the valley. There is thick fog — you cannot see the valley, or the peak, or anything more than a step away.

But you can feel the ground under your feet. And that is enough to tell you which direction slopes downward.

So you do the only sensible thing: face downhill, take a step, feel the ground again, take another step. Repeat until the ground feels flat in every direction. You have arrived somewhere low without ever seeing where you were going.

That is gradient descent, completely. Here is the mapping:

| On the mountain | In the model |
|---|---|
| Your altitude | The loss — how wrong the model currently is |
| Your position | The parameters — the numbers we are allowed to change |
| The slope under your feet | The gradient, which `backward()` computed in Ch 08 |
| The downhill direction | The **negative** gradient |
| How long a stride you take | The learning rate |
| One step | One call to `step()` |
| The fog | The model never sees the whole loss surface — only the slope at the single point where it currently stands |

The fog is the part people skip, and it is worth keeping. The model has no map. At every step it knows one thing only: the slope right where it is standing. Everything gradient descent achieves, it achieves from that.

---

## 2. One parameter, one equation

Let's build the rule up in three layers. Same idea each time, written more precisely.

**Layer 1 — in words.**

> The new value of a parameter is its old value, minus an adjustment.

**Layer 2 — with a variable.** Call the parameter `w`, the way you would in code:

$$w_{\text{new}} = w_{\text{old}} - \text{learningRate} \times \text{gradient}$$

That is already the complete algorithm. If you understand this line, you understand gradient descent — everything after this is either notation or refinement.

> **"But a network has thousands of parameters, buried under layers of other parameters. One line cannot possibly cover that."**
>
> It does, and this is the right moment to be suspicious of it. The resolution is that **all the difficulty of being deep lives inside `backward()`, not inside the update.** A parameter sitting four operations away from the loss has a gradient that is a *product of four local derivatives* — but the chain rule assembles that product during the backward pass, and by the time the update runs, that parameter is holding one number just like every other.
>
> And if the question underneath yours is *"none of these examples look like a neural network — is any of this actually going anywhere?"*, that one is answered head-on in **§6, "So where is the neural network?"**, with a real neuron trained by this chapter's loop.
>
> If you would rather see the scaling claim demonstrated than take it on trust, [**one rule, many layers**](../deep-dives/ch-09-one-rule-many-layers.md) runs gradient descent on the graph you already know — Chapter 08's own `f = b·sin(a) + b²`. One of its parameters gets its gradient through a chain (`∂f/∂a = b·cos(a)`, passing through the `sin`), the other as a sum over two paths (`∂f/∂b = sin(a) + 2b`, the `1 + 6 = 7` you derived by hand). Those are exactly the two complications a deep network presents — and the same single update line moves both. Worth reading once this chapter's `SGD` works.

**Layer 3 — the notation you'll meet everywhere else.** Papers and libraries write the same equation like this:

$$\theta_{\text{new}} = \theta_{\text{old}} - \eta \cdot \frac{\partial L}{\partial \theta}$$

Nothing changed except the symbols:

- $\theta$ ("theta") — a parameter. Papers use it because a model has thousands and `w₁, w₂, w₃…` gets unwieldy. In code it is `param.data`.
- $\eta$ ("eta") — the learning rate. In code, `learningRate`.
- $\partial L/\partial \theta$ — the gradient of the loss with respect to that parameter. In code, `param.grad`, put there by `backward()`.

> **Three symbols, one idea.** The derivative gets written three ways in this course. For our purposes they all mean the same thing, and it is worth getting that straight now rather than being tripped by it later:
>
> | Symbol | Read it as | Used when |
> |---|---|---|
> | $\dfrac{dL}{dw}$ | "the derivative of L with respect to w" | there is only one variable — ordinary calculus |
> | $\dfrac{\partial L}{\partial w}$ | "the **partial** derivative of L with respect to w" | there are several variables. It means: differentiate with respect to `w` and hold everything else still |
> | $\nabla L$ | "grad L" | shorthand for *all* the partials at once, one per parameter, bundled together |
>
> A model has thousands of parameters, so `∂` is the honest symbol and `∇L` is just a way of saying "all of them" without listing them. **In code there is no distinction at all**: every parameter carries its own `.grad`, and `∇L` is simply those numbers across the whole list. If partial derivatives are new to you, nothing here needs more than: *the slope in one parameter's direction, with the others frozen.*

So the whole of vanilla gradient descent, in the code you are about to write, is:

```
param.data  =  param.data  -  learningRate * param.grad
```

### Why the minus sign

This is the one part of the equation that carries an idea rather than a name. The reasoning is three steps:

1. The gradient tells us which direction makes the loss **increase**.
2. We want the loss to **decrease**.
3. So we move in the **opposite** direction.

That is the minus sign. Change it to a plus and the code still runs, the loss climbs steadily, and you have implemented gradient *ascent* — a perfectly good algorithm for making a model as wrong as possible. It is an easy bug to stare straight through, because nothing crashes.

### Reading a gradient as a sentence

A gradient is a number, but it is worth training yourself to hear it as a sentence. Take the loss we'll use throughout this chapter:

$$L(w) = (w-5)^2 \qquad\text{so}\qquad \frac{dL}{dw} = 2(w-5)$$

At `w = 0` the gradient is `2(0−5) = −10`. Read that as:

> **"If `w` increases a little, the loss goes down — by about 10 units of loss per unit of `w`."**

So we want `w` to increase. And notice that the rule does that on its own, with nobody checking a sign:

```
w  =  0  -  0.1 × (-10)  =  0 + 1  =  1
```

Subtracting a negative number moves `w` *up*. You never reason about direction yourself — the sign of the gradient handles it, in both directions, always. That is why one formula works for every parameter in a network without any per-parameter logic.

---

## 3. Three updates, by hand

Before writing any code, do the algorithm on paper. Same loss, `L(w) = (w−5)²`, starting at `w = 0`, with a learning rate of `0.1`. The minimum is obviously at `w = 5` — which is the point: you know the answer, so you can tell whether the method is working.

```
start:   w = 0

step 1:  gradient = 2(0 − 5)   = -10
         w = 0    - 0.1 × (-10) = 1.0

step 2:  gradient = 2(1.0 − 5) = -8
         w = 1.0  - 0.1 × (-8)  = 1.8

step 3:  gradient = 2(1.8 − 5) = -6.4
         w = 1.8  - 0.1 × (-6.4) = 2.44
```

Work through those three yourself with a pen. It takes two minutes and it is the single best investment in this chapter — when your code disagrees with these numbers later, you will know exactly which of the three quantities to print.

Two things to notice in the trace:

**It is walking toward 5.** 0 → 1.0 → 1.8 → 2.44. Slowly, but every step in the right direction.

**The steps are shrinking:** 1.0, then 0.8, then 0.64. Nobody wrote any code to slow it down near the bottom. That falls out of the rule itself, which is the subject of the next section.

---

## 4. What the learning rate does

The update is **proportional to the gradient**. That single fact gives the algorithm a useful property for free.

Far from the minimum the slope is steep, the gradient is large, and the steps are big. Close to the minimum the slope flattens, the gradient shrinks, and the steps become small automatically. The method takes big strides when it is confident and creeps when it is close — with no braking logic anywhere.

In the trace above each step was exactly `0.8×` the one before it. That is not a coincidence, and you can see why without any new mathematics. Look at how far `w` is from the minimum at each point:

```
w        :  0     1.0    1.8    2.44
5 − w    :  5     4      3.2    2.56     ← each one 0.8× the last
gradient : -10   -8     -6.4   -5.12     ← gradient = 2 × (w − 5), so also 0.8×
step     :  1.0   0.8    0.64            ← step = 0.1 × gradient, so also 0.8×
```

The distance shrinks by a factor, so the gradient shrinks by the same factor, so the step does too. Everything rides on one number. Where that number comes from — and why it happens to be `0.8` at `η = 0.1` — is §13.

The learning rate is the one thing *you* choose. It scales every step:

- **Small `η`** — short strides. Safe, but you may still be walking at nightfall.
- **Large `η`** — long strides. Fast, until a stride carries you clean across the valley and partway up the opposite slope.

There is no universally correct value. There is, however, a value that is correct *for a given loss surface*, and more can be said about it than "try some numbers" — see §13.

---

## 5. When the stride is wrong

<p align="center">
  <img src="../assets/ch-09/learning-rate-regimes.svg" alt="Three panels, each showing the same bowl-shaped loss L(w)=(w−5)² with a ball starting at w=0 and the minimum at w=5. Left panel, eta 0.05: the ball inches down the slope, reaching only w=2.05 after six steps. Middle panel, eta 0.2: the ball takes a large first step then progressively smaller ones, settling near the minimum at w=4.61. Right panel, eta 1.1: a single step carries the ball straight past the minimum and up the far wall of the bowl to a point higher than it started, with a dashed arrow showing it continuing to −2.2 then 13.6 and escaping. A caption notes the error shrinks by exactly the factor |1−2eta| each step, so the method converges only when eta is below 1, and eta=0.5 reaches the answer in a single step." />
</p>

*Figure 1: same bowl, same starting point, three learning rates.*

The left and middle panels differ only in patience — both are heading to the same place, one just needs far more steps.

The right panel is a different kind of failure. One step at `η = 1.1` carries the ball past the minimum and **higher up the far wall than where it started**. The next step is therefore bigger, and lands higher still. The loss grows without bound. This is divergence, and it is not fixed by running longer — running longer makes it worse.

So the three outcomes to recognise are:

| Symptom | Cause |
|---|---|
| Loss falls, but barely | Learning rate too small |
| Loss falls smoothly, flattens out | Learning rate about right |
| Loss grows, often to `Infinity` or `NaN` | Learning rate too large |

The middle panel is what a healthy training run looks like: big improvements early, then progressively smaller ones, then a flat tail.

One footnote for later: on this particular bowl, the boundary between "converges" and "diverges" is not a matter of taste — it can be derived exactly, and it turns out to be `η = 1`. That derivation is §13.

---

## 6. The five-step training loop

This is the most important section in the chapter. If you remember one thing from Ch 09, remember this shape — you will write it again in Ch 15, and it is still there, unchanged, in Ch 30.

**Every training iteration is five actions:**

1. **Make a prediction.**
2. **Measure how wrong it is.**
3. **Work out which parameters caused the error.**
4. **Change those parameters.**
5. **Clear the old gradients.**

And in code:

| The action | The call | Built in |
|---|---|---|
| predict | forward pass | Ch 08a |
| measure the error | the loss | Ch 12 (for now, by hand) |
| assign blame | `loss.backward()` | Ch 08b |
| change the parameters | `opt.step()` | **this chapter** |
| clean up | `opt.zeroGrad()` | **this chapter** |

<p align="center">
  <img src="../assets/ch-09/training-loop.svg" alt="A ring of five stages with a highlight travelling around it: forward (build the graph, Ch 08a), loss (one number: how wrong), backward() (fill every .grad, Ch 08b), opt.step() (theta moves against the gradient, Ch 09), and opt.zeroGrad() (or the next step is wrong). The centre reads 'repeat until the loss stops falling'. A callout warns that stages 3 and 4 must not be separated, because zeroing gradients between them erases the gradients before step() can read them. A footer notes a fresh graph is built every iteration, so backward() is never re-run on the previous step's graph." />
</p>

*Figure 2: the loop. Stages 1–3 are Chapter 08; stages 4–5 are what you are about to build.*

Two practical notes now, and the reasoning behind them in §8.

**Step 5 exists because `backward()` adds to gradients rather than replacing them.** So after you have used this step's gradients, clear them, and the next iteration starts from a clean slate. That is all you need to believe for now.

**The loss must be a single number.** `backward()` starts from the node you call it on and works backward from there, so there has to be one node that *is* the loss. This is why every loss function in Ch 12 ends in a sum or a mean.

### Wait — where is the prediction in `(w−5)²`?

If you just read "make a prediction, measure how wrong it is" and thought *"but the example has no prediction, and no data either"* — good. That is the right question, and it deserves an answer before you write code.

`L(w) = (w−5)²` has no dataset, no prediction and no target. That is deliberate. It is a bare function of one variable, chosen so that **nothing stands between you and the update rule**. You already know the answer is `w = 5`, so if your optimizer walks there, it works; and if it doesn't, the bug is in the four lines you just wrote, not somewhere in a model.

But the loop really is the same loop. Here is an actual prediction problem, about as small as one can get — a model `ŷ = w · x`, with a single training example `x = 2`, `y = 10`:

| Step | On the bowl | On the tiny model |
|---|---|---|
| predict | *(nothing to predict)* | `ŷ = w × 2` |
| measure how wrong | `L = (w − 5)²` | `L = (ŷ − 10)²` |
| assign blame | `L.backward()` | `L.backward()` |
| change | `w.data -= η × w.grad` | identical |
| clean up | `w.zeroGrad()` | identical |

Now substitute the prediction into the loss and watch what happens:

$$L = (\hat y - 10)^2 = (2w - 10)^2 = 4(w-5)^2$$

It **is** the bowl. The same shape, with its minimum at exactly the same `w = 5`, just four times as steep. So `(w−5)²` was never a toy stand-in for a learning problem — it is a learning problem with the prediction already substituted in and the arithmetic already done.

This is worth holding onto, because it is what "the loss" always means:

> **A single number that is large when the model is wrong and small when it is right.**

Where that number comes from — squared error here, cross-entropy in Ch 12 — gradient descent genuinely does not care. It needs one number, and the gradients leading back from it. That indifference is why the same five lines train a one-parameter bowl and a GPT.

(Do notice the "four times as steep", though. Steepness changes which learning rates are safe — the model version diverges above `η = 0.25` where the plain bowl tolerates up to `1.0`. §13 is where that gets pinned down.)

### So where is the neural network?

A fair follow-up, and one worth answering directly, because it applies to Chapter 08 too. `(w−5)²` is not a neural network. Neither was `f = b·sin(a) + b²`, the expression you differentiated over and over in Ch 08. So what were those for, and when does an actual network turn up?

**Those were test functions, not models.** They were chosen to be small enough to differentiate by hand, so you could check your engine against ordinary calculus and *know* it was right. That is their whole job. Once `backward()` is trustworthy, they have done it and can be thrown away.

**The engine is the thing you were building.** `Value` has no idea what a neural network is. It knows seven operations — `add`, `mul`, `pow`, `exp`, `log`, `tanh`, `relu` — and how to differentiate any expression assembled from them. That generality is not an accident; it is the entire design.

**And a neural network is exactly such an expression.** Here is a single neuron, the smallest real unit of one:

$$\hat y = \text{relu}(w_1 x_1 + w_2 x_2 + b)$$

Written with the methods you already implemented:

```typescript
const pre  = w1.mul(x1).add(w2.mul(x2)).add(bias);
const yHat = pre.relu();
```

`mul`, `add`, `relu`. No new class, no new node type, no new mathematics. A **layer** is several of those side by side. A **network** is layers fed into each other. A **loss** is `(ŷ − y)²` — one more `add` and one `pow`. The finished thing is one large `Value` expression, and `backward()` differentiates it for precisely the reason it differentiated `b·sin(a) + b²`: it does not care what the expression *means*.

So, to answer the question as asked — *will that equation be used anywhere?*

> **The equation: no. The machinery you built to solve it: everywhere.**

Here is the same point as a table:

| | Ch 08's test function | A real neural network |
|---|---|---|
| built from | `mul`, `add`, `sin` | `mul`, `add`, `relu` |
| parameters | `a`, `b` | `w₁, w₂, b, …` thousands to billions |
| graph size | about 6 nodes | thousands to billions of nodes |
| written by | you, by hand | layer code, in a loop |
| differentiated by | `backward()` | `backward()` — unchanged |
| optimised by | the update rule | the update rule — unchanged |

The bottom two rows are the payoff. Nothing in Chapters 08 or 09 gets rewritten when a real network arrives; it just gets called with a bigger expression.

And this is not a promise — it already works. Take the three-parameter neuron above, four training examples of `y = 2x₁ + 3x₂`, and run the exact five-stage loop from this chapter:

```
step    0    loss 19.987500    w1=0.1000  w2=0.1000  b=0.0000
step   10    loss  0.464676    w1=1.7882  w2=1.4672  b=1.3197
step  100    loss  0.065725    w1=1.8600  w2=2.4708  b=0.5953
step  500    loss  0.000020    w1=1.9973  w2=2.9908  b=0.0106
step 2000    loss  0.000000    w1=2.0000  w2=3.0000  b=0.0000
```

It recovers `w₁ = 2`, `w₂ = 3`, `b = 0` — the function the data came from. That is a neural network being trained, by the `Value` class from Ch 08 and the `SGD` you are about to write, with nothing else involved. (The loss for those four examples is a graph of 48 `Value` nodes. A GPT is the same picture with more zeros.)

What genuinely changes later is narrower than it looks:

- **Ch 10** swaps one number per node for one *tensor* per node, because a million scalar nodes is unbearably slow. Same graph, same chain rule, same update.
- **Ch 13** writes a `Linear` class so you stop typing `w1.mul(x1).add(w2.mul(x2))` by hand.
- **Ch 12** provides losses with better numerical behaviour than a raw squared error.

None of those is a new idea. They are ergonomics and speed on top of what you have.

If you want to see this for yourself before Ch 13, the STRETCH at the bottom of [`exercises/ch-09-gradient-descent.ts`](../../exercises/ch-09-gradient-descent.ts) is exactly that neuron.

---

## 7. Build it — `SGD`

You now know enough to write the class. Open [`sgd.ts`](../../src/optim/sgd.ts); the per-method comments there carry the recipe and the traps.

The API is: you hand the optimizer its parameters **once**, at construction, and afterwards `step()` and `zeroGrad()` take no arguments.

```
const opt = new SGD([w, b], 0.1);
…
loss.backward();
opt.step();
opt.zeroGrad();
```

This mirrors PyTorch's `torch.optim.SGD(model.parameters(), lr=0.1)`, so what you learn here transfers directly. (§11 shows why owning the list is not merely cosmetic.)

### One full iteration, line by line

The five stages have been described in words and drawn as a ring. Here they are as actual calls, with the value of everything after each line. These are the numbers from step 1 of the §3 trace:

```
const w   = new Value(0);                  // the parameter, starting at 0
const opt = new SGD([w], 0.1);             // the optimizer now owns w

// ── 1 & 2 — predict, and measure how wrong ────────────────────────
const loss = w.add(new Value(-5)).pow(2);  // (w − 5)²
//   w.data    = 0
//   loss.data = (0 − 5)² = 25

// ── 3 — assign blame ──────────────────────────────────────────────
loss.backward();
//   w.grad = -10        ← "increase w and the loss falls, ~10 per unit"

// ── 4 — change the parameter ──────────────────────────────────────
opt.step();
//   w.data = 0 − 0.1 × (-10) = 1.0

// ── 5 — clean up ──────────────────────────────────────────────────
opt.zeroGrad();
//   w.grad = 0          ← ready for the next iteration
```

Run that same block again and it produces `w = 1.8`. A third time, `2.44`. The training loop is literally these five lines wrapped in a `for` — there is nothing else to it.

Two details in there are easy to miss and both matter:

**The loss is built *inside* the loop, not before it.** `w.add(new Value(-5)).pow(2)` creates brand-new nodes every iteration. That is what "a fresh graph each step" means in practice — you are not reusing last iteration's graph, you are constructing a new one around the parameter's new value.

**`w` is the only thing that survives between iterations.** The graph is thrown away and rebuilt; the parameter persists. That is exactly the split from §9: the graph is scratch work, the parameter is state.

### The milestones

**Milestone 1 — the constructor.** Store the parameter list and the learning rate.
✅ *Checkpoint:* `new SGD([w], 0.1).params[0]` is the *same object* as `w`, not a copy. If you copy the numbers out, every method below will run correctly on the copies and the model will never change.

**Milestone 2 — `step()`.** One loop over the parameters, one subtraction each, written straight onto `.data`.
✅ *Checkpoint:* reproduce the hand trace from §3 exactly — `1.0`, `1.8`, `2.44`. If it disagrees, print `w.data`, `w.grad` and `learningRate`; one of the three is wrong and the trace tells you which.

**Milestone 3 — `zeroGrad()`.** Loop over the parameters and set each `.grad` back to 0. One line of work.
✅ *Checkpoint:* `bun test src/optim/sgd` — the `SGD` block should now go green.

**Milestone 4 — the loop.** Assemble all five stages and minimise `(w−5)²` for 100 iterations from `w = 0`.
✅ *Checkpoint:* `w ≈ 5.0`. Then sweep the learning rate and check Figure 1's predictions: `0.05` crawls, `0.2` converges cleanly, `0.9` oscillates but still converges, `1.1` diverges.

---

## 8. Why `zeroGrad` matters more than it looks

Now the reasoning behind step 5.

Ch 08 built `backward()` to **accumulate** with `+=` rather than assign. That was not an arbitrary choice — it is what makes a parameter used in two places get the sum of both contributions, which is what the chain rule requires.

The side effect is that gradients from an old step are still sitting there when the next `backward()` runs, and the new contributions land **on top of** the old ones. Ch 08 measured this on a three-node graph: calling `backward()` repeatedly without clearing gave

```
-3,  then  -9,  then  -18
```

Notice it does not simply double — and the reason is worth spelling out, because it explains why this gets rapidly worse in a bigger network. That graph was `L = (a·b) + d` with `a = 2`, `b = −3`, so `a.grad` should be `−3` every time. Watch what actually happens across three calls:

```
                  interior node c        the parameter a
1st backward():   c.grad = 1             a.grad =  0 + (1 × -3)  =  -3   ✓ correct
2nd backward():   c.grad = 1 + 1 = 2     a.grad = -3 + (2 × -3)  =  -9
3rd backward():   c.grad = 2 + 1 = 3     a.grad = -9 + (3 × -3)  = -18
```

`backward()` **sets** the root's gradient to 1 each time, so the root itself never drifts. But `c` is an *interior* node, and nothing resets it — so it climbs 1, 2, 3. And `a` doesn't just accumulate its own error; it accumulates `c`'s inflated gradient, which is why `a` grows faster than `c` does.

That is the mechanism: every layer sitting between a parameter and the loss adds another multiplier to the contamination. A three-node graph gets mildly wrong answers. A twelve-layer transformer gets catastrophically wrong ones.

Inside a training loop that inflated gradient goes straight into `step()`, so the effective step size grows every iteration and the loss climbs. **The symptom is identical to a learning rate that is too large**, which is why this one costs people so much time: they lower the learning rate, it appears to help briefly, and then it diverges anyway.

### Where in the loop it goes

Both of these are correct:

```
backward → step → zeroGrad → (next iteration)
zeroGrad → backward → step → (next iteration)
```

What is *not* correct is putting it **between `backward()` and `step()`**. That erases the gradients before `step()` has read them, so every parameter is updated by zero. Nothing crashes, nothing warns, and the loss simply sits at the same value forever. If your model refuses to move at all, this is the first thing to check.

---

## 9. Why the update happens outside the graph

Here is the update again:

```
param.data = param.data - learningRate * param.grad
```

Notice what it does *not* do. It does not call `add`, or `mul`, or any of the operations you spent Ch 08 building. It reaches past all of that and writes a plain number into `.data`.

After a whole chapter of "every operation must record itself in the graph", that feels wrong. The distinction is this:

> The forward pass is the model **computing something**. That belongs in the graph.
> The update is us **changing a stored number**. That does not.

We are not asking the model to calculate anything new. We are telling the optimizer: *set this weight to a different value.*

Two concrete things break if you use `Value` operations instead:

**The model would never change.** `w.add(...)` doesn't modify `w` — it returns a *new* node. Your model still holds a reference to the original `w`, which is untouched, so the next forward pass uses the old value. Training runs, loss never moves.

**Memory would grow without bound.** Every step would append more nodes to a graph that is never released. After a thousand iterations you are carrying a thousand steps of history; after a hundred thousand, the process dies.

PyTorch draws exactly the same line — its optimizer step runs inside `with torch.no_grad():`, which is its way of saying "the following is bookkeeping, not model computation."

---

## 10. Momentum — the intuition

Vanilla SGD asks one question at every step:

> *"Which way is downhill **right now**?"*

It has no memory. Whatever it did last step is forgotten completely.

Momentum adds a second question:

> *"And which way have I been going **recently**?"*

Think of the difference between sliding a box across a floor and rolling a ball. The box stops the instant you stop pushing. The ball has built up speed, and keeps going.

So momentum keeps a running number called the **velocity** — how it has been moving lately — and updates it like this:

```
new velocity  =  (old velocity × momentum)  +  (push from the current gradient)
```

The `momentum` coefficient (written `β`, "beta", usually `0.9`) decides how much of the previous motion survives. At `0.9`, nine-tenths of the old velocity carries over and the new gradient nudges it. At `0`, nothing carries over — and momentum collapses back into plain SGD exactly.

Written properly, that is:

$$v \leftarrow \beta v - \eta \nabla L, \qquad \theta \leftarrow \theta + v$$

Note the parameter now moves by the **velocity**, not by the gradient. The gradient's only job is to adjust the velocity.

<p align="center">
  <img src="../assets/ch-09/momentum-vs-vanilla.svg" alt="One bowl, L(w)=(w−5)², with two balls run for six steps from w=0 at learning rate 0.1. The blue vanilla SGD ball moves 0, 1.0, 1.8, 2.44, 2.95, 3.36 — each step 0.8 times the last, decaying as it goes. The purple momentum ball with beta 0.9 moves 0, 1.0, 2.7, 4.69, 6.54, 7.90: it reaches the minimum by step three and then sails past it up the far side, marked 'overshoots — inertia, not a bug'. A panel shows the velocity growing 0, 1.0, 1.7, 1.99 even as the gradient shrinks. The caption notes contributions add along a consistent slope and cancel across an oscillation." />
</p>

*Figure 3: the same six steps, with and without inertia.*

This buys two things:

**Going the same way for a while builds speed.** Watch the velocity panel: `0 → 1.0 → 1.7 → 1.99`. The gradient is *shrinking* over those steps, yet the velocity keeps growing, because each step adds to what was already there. Momentum reaches the minimum by step three while vanilla SGD is not yet halfway.

**Changing direction repeatedly cancels out.** In a narrow valley where the gradient flips sign every step, consecutive pushes point opposite ways and partly cancel, damping the bouncing instead of amplifying it.

That second property — the damping — is the reason momentum survives into every modern optimizer, including Adam in Ch 14.

And Figure 3 is deliberately honest: this run **overshoots**, sailing well past the minimum to `w = 7.9` before coming back. That is inertia doing exactly what inertia does. `β = 0.9` is too much for a bowl this simple. It is a genuine trade-off, not a bug, and tuning it is Ch 14's business.

---

## 11. Build it — `SGDMomentum`

**Milestone 5.** The class needs one more field than `SGD`: a **velocity per parameter**, all starting at zero.

This is where the API earns itself. The velocity has to survive *between* calls to `step()` — that is the entire point of it. So it must be allocated **once, in the constructor**, alongside a parameter list that stays put. An optimizer handed a fresh array on every call would have no way to know that today's third parameter is the same one as yesterday's third parameter.

✅ *Checkpoint — the hand trace,* same bowl, `η = 0.1`, `β = 0.9`:

```
step 1:  grad = -10     v = 0.9×0    - 0.1×(-10)  = 1.0    w = 1.0
step 2:  grad =  -8     v = 0.9×1.0  - 0.1×(-8)   = 1.7    w = 2.7
step 3:  grad = -4.6    v = 0.9×1.7  - 0.1×(-4.6) = 1.99   w = 4.69
```

Compare against vanilla's `1.0, 1.8, 2.44` at the same learning rate.

✅ *Checkpoint — the algebraic one that catches the classic bug:* **`momentum = 0` must behave exactly like vanilla `SGD`.** With a zero coefficient the old velocity is discarded entirely and the rule reduces to `θ -= η·grad`. If your two classes disagree there, the velocity is being mishandled — most often by allocating it inside `step()`, where it silently resets to zero every call. That bug degrades momentum into plain SGD, which still converges, so only a test that inspects the velocity itself will catch it.

---

## 12. What SGD can run into

*(Context, not a prerequisite. Your code is already done.)*

For one parameter the loss is a curve. For two, a surface. For GPT's parameters, a surface in a space with more dimensions than anyone can picture. Gradient descent walks downhill on it and stops wherever the ground is flat — which need not be the lowest point anywhere.

The four features worth naming:

- **Global minimum** — the lowest loss achievable anywhere.
- **Local minimum** — the bottom of a valley that isn't the deepest one. Descent can settle here.
- **Saddle point** — downhill in one direction, uphill in another. The gradient is near zero, so progress stalls without having arrived anywhere good.
- **Plateau** — a large flat region. Tiny gradients, tiny steps, training appears frozen.

### A common misconception, worth correcting now

You have probably been told that the big danger of gradient descent is **getting stuck in a local minimum**. It is the standard cautionary tale, and it is drawn with a picture of a 1-D curve with two dips.

For the networks in this course, it is largely not the thing that goes wrong.

The reason is dimensionality. In one dimension, a flat point is a local minimum if the curve turns upward on both sides — a coin flip. In a space with thousands of dimensions, a flat point is a local minimum only if the surface curves upward in **every single one** of those thousands of directions at once. That is vanishingly unlikely. Almost every flat point in a high-dimensional loss surface is a saddle, with at least one direction still leading down.

So the picture is not wrong, it just does not scale: local minima genuinely trap you in 1-D, and mostly do not in 10,000-D. Saddle points and plateaus are the realistic obstacles, and the local minima that do exist tend to be about as good as one another.

In practice your training runs will fail for far more mundane reasons — a learning rate that is too large, a bad initialisation, a sign error — long before they fail from a bad local minimum.

---

## 13. Going deeper (optional)

Everything above treats the learning rate as something you tune by trial. On a simple loss you can do better than that, and it is worth seeing once so that "lower the learning rate" stops feeling like superstition.

For `L(w) = (w−5)²`, track the **error** `e = w − 5` instead of `w` itself, and the whole thing collapses to one line:

$$e_{\text{new}} = e_{\text{old}}\,(1 - 2\eta)$$

The error is multiplied by the same constant every step. That immediately explains the `0.8×` ratio you saw back in §3 (at `η = 0.1`), tells you the method converges exactly when `η < 1`, and predicts that `η = 0.5` lands on the answer in a **single step**.

[**Deep dive: how big a step can you take?**](../deep-dives/ch-09-how-big-a-step.md) derives that, then extends it: why the general threshold is `1/curvature`, how much speed momentum can actually build up, and why summing a loss instead of averaging it makes your safe learning rate shrink as you add training data — which is the reason the exercise needs `0.01` for the linear fit but `0.1` for the bowl.

And the question this whole chapter has been deferring — *does a one-parameter rule really train a network?* — gets answered in [**one rule, many layers**](../deep-dives/ch-09-one-rule-many-layers.md), by descending on Chapter 08's own graph, `f = b·sin(a) + b²`. It shows where the depth actually goes: into the product that `backward()` builds, and nowhere near `step()`. Along the way it catches a genuine saddle-point stall — twenty-five iterations that barely move — which is §12's story happening in a two-parameter function you can hold in your head. And it explains why gradients shrink with depth, and why ReLU (Ch 11), LayerNorm (Ch 20) and residual connections (Ch 26) all exist to fight the same multiplication.

---

## What to Implement

| Symbol | Description |
|---|---|
| `SGD` constructor | Store the parameter list and the learning rate |
| `SGD.step()` | Apply `θ ← θ − η·∇L` to every owned parameter |
| `SGD.zeroGrad()` | Reset every owned parameter's `.grad` to 0 |
| `SGDMomentum` constructor | As above, plus `momentum` (default 0.9) and a velocity array allocated **once** |
| `SGDMomentum.step()` | Update each velocity, then move each parameter by it |
| `SGDMomentum.zeroGrad()` | As `SGD.zeroGrad` |

---

## Common Pitfalls

- **Building the update out of `Value` operations.** Returns a new node, leaves the model pointing at the old one, and grows the graph forever. Assign to `.data`.
- **Forgetting `zeroGrad`.** Gradients compound: −3, −9, −18. The loss climbs, and it looks exactly like too high a learning rate.
- **Calling `zeroGrad` between `backward()` and `step()`.** Everything runs; nothing moves.
- **Copying parameters instead of holding references.** `step()` updates the copies. Silent, and indistinguishable from a learning rate near zero.
- **Allocating momentum's velocity inside `step()`.** Silently degrades to vanilla SGD — which still converges, so most tests still pass.
- **Getting momentum's two lines backwards.** Update the velocity *first*, then move the parameter by the new velocity. The other order lags one step behind and is nearly invisible on a smooth bowl.
- **Guessing the learning rate.** Sweep on a log scale — `1e-4, 1e-3, 1e-2, 1e-1` — rather than nudging.
- **Declaring victory after one step.** Print the loss every N iterations. A number that went down once has told you nothing.

---

## How to Verify

```bash
bun test src/optim/sgd
```
```bash
bun run exercises/ch-09-gradient-descent.ts
```

### What a healthy run looks like

Print the loss every ten steps while minimising `(w−5)²` from `w = 0` at `η = 0.1`, and you should see this:

```
step   0    loss 25.000000
step  10    loss  0.288230
step  20    loss  0.003323
step  30    loss  0.000038
step  40    loss  0.000000
```

That shape — a steep drop, then a long flattening tail — is what convergence looks like, and it is worth knowing by sight. The loss falls by a constant *factor* per step, not a constant amount, so most of the visible progress happens early and the rest is refinement.

Compare with the two failure modes on the same bowl, printed at the same intervals:

```
η = 0.01  (too small)         η = 1.1  (too large)
step   0    loss 25.000000    step   0    loss 25.000000
step  10    loss 16.690199    step  10    loss 958.44
step  20    loss 11.142510    step  20    loss 36744.29
step  30    loss  7.438829    step  30    loss 1.41e+6
step  40    loss  4.966221    step  40    loss 5.40e+7
```

The left run is *working* — it is genuinely descending, just far too slowly to be useful. Give it a few thousand steps and it would arrive. The right run is not converging at all, and running it longer only makes the number bigger; eventually it reaches `Infinity`, and then `NaN` once an infinity meets a subtraction.

Learning to tell those two apart on sight saves a lot of time. **Falling slowly and rising are different problems with opposite fixes**: the first wants a larger learning rate, the second a smaller one. Reaching for the same knob in the same direction for both is the most common way to spend an afternoon getting nowhere.

### Predict before you run

The exercise is the real gate, and it is worth using properly: **predict every printed number before you run it.**

This is the habit to carry through the whole course. Seeing `w = 1.8` and thinking *"okay, it works"* teaches you nothing. Being able to say *"I expect 1.8, because the gradient is −8 and the learning rate is 0.1, so the step is +0.8"* means you understand the algorithm rather than the output.

You can do this for the whole exercise, not just single steps. The bowl is `(w−5)²` from `w = 0`, and §13's recurrence gives `wₙ = 5 − 5(1−2η)ⁿ` — so every printed value is predictable on paper. `minimise(0.01, 200)` should print `4.912`; both `minimise(0.1, 100)` and `minimise(0.9, 100)` should print `5.000`. Matching a trajectory you predicted is a much stronger signal than watching a number go down.

---

## Self-Check Questions

1. At `w = 8` on `L = (w−5)²`, the gradient is `+6`. Say in one sentence what that number means, and then say which way `step()` will move `w` and why.
2. `L = w²`, so `∇L = 2w`. With `w = 3` and `η = 0.1`, what is `w` after one step? After ten? (The multiplier is constant — you shouldn't need to iterate ten times.)
3. Why must the parameter update write to `.data` rather than go through `add`? Name both failure modes.
4. What exactly goes wrong if you call `zeroGrad()` between `backward()` and `step()`? What does the loss curve look like?
5. Your loss is climbing steadily. Name two different causes, and describe an experiment that distinguishes them.
6. In your own words: what does momentum remember, and what does `β = 0` do?
7. Momentum with `β = 0.9` overshot the minimum in Figure 3. Is that a bug? What would you change to reduce it, and what would you give up?
8. Why does the optimizer own its parameter list, rather than receiving it at each `step()` call? Answer in terms of `SGDMomentum` specifically.
9. *(After §13)* On `L = (w−5)²`, why does `η = 0.5` reach the minimum in one step? Why doesn't that trick work for a general loss?

---

## Further Reading

- [3Blue1Brown — Gradient descent, how neural networks learn](https://www.3blue1brown.com/lessons/gradient-descent) — the loss landscape, visually. The best first watch.
- [Sebastian Ruder — An overview of gradient descent optimization algorithms](https://ruder.io/optimizing-gradient-descent/) — the tour of SGD variants; several land in Ch 14.
- [Distill — Why Momentum Really Works](https://distill.pub/2017/momentum/) — interactive, and the clearest explanation of the damping effect anywhere.
- [Bottou, Curtis & Nocedal — Optimization Methods for Large-Scale Machine Learning](https://arxiv.org/abs/1606.04838) — the rigorous treatment, for later.

---

## Checkpoint

You now have a complete, working learning system: a computation graph (Ch 08a), an automatic backward pass (Ch 08b), and parameters that move (Ch 09). Everything from here scales that up — none of it replaces it.

Prove it to yourself on two parameters. Minimise

$$L = (w_1 - 3)^2 + (w_2 + 1)^2$$

and watch `w₁ → 3` and `w₂ → −1`. Both parameters descend in the same loop, from one `backward()` call, and nothing in your code knows there are two of them rather than one. That property — that the same five lines work for any number of parameters — is what makes the next twenty-one chapters possible.

---

## Next Chapter

**[Tensor Autograd Bridge](ch-10-tensor-autograd-bridge.md)** — everything so far has been one number per node, which is unusable at scale: a single transformer weight matrix would be hundreds of thousands of `Value`s. Ch 10 lifts this identical machinery onto tensors, where one node holds an entire matrix and one gradient holds an entire matrix of derivatives.
