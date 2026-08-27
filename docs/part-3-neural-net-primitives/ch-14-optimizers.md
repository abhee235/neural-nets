# Chapter 14: Optimizers

> **Part 3 of 6 — Neural Net Primitives**
>
> Sources: `src/optim/sgd.ts`, `src/optim/adam.ts`
>
> Tests: `src/optim/sgd.test.ts`, `src/optim/adam.test.ts`
>
> Exercise: `exercises/ch-14-optimizers.ts`

---

# The gradients are ready

We have reached an important point in the course.

A model can now:

```text
input
  ↓
Linear layer
  ↓
prediction
  ↓
loss
  ↓
backward()
  ↓
gradients
```

The gradients are sitting inside the parameters.

So the model knows:

> **"If I change this parameter, the loss will change in this direction."**

But there is still one very simple question:

> **What should we actually do with that information?**

In Chapter 12, we used a tiny example:

```text
weight = 3
gradient = -16
learning rate = 0.1
```

and manually performed:

```text
w  ←  w − lr · g
```

which gave:

```text
3 - 0.1(-16) = 4.6
```

That worked.

But a real model doesn't have one weight.

It might have thousands, millions, or billions.

We need a piece of machinery that can take a whole collection of parameters and their gradients and decide how to update them.

That machinery is an **optimizer**.

---

# Start with the simplest possible rule

Let's forget the word "optimizer" for a moment.

Suppose you are standing on a hill.

You don't know where the bottom is.

But you can measure the slope beneath your feet.

If the slope points upward in this direction, walk the other way.

If the slope points downward in this direction, walk that way.

That's the basic idea behind gradient descent.

For a parameter `θ`, let:

```text
g  =  ∂L / ∂θ
```

The simplest update is:

```text
θ  ←  θ − lr · g
```

where:

- `θ` is the parameter
- `g` is its gradient
- `lr` is the learning rate

The learning rate controls how large a step we take.

---

# Watch one parameter learn

Let's use a deliberately simple problem.

Suppose our loss is:

```text
L = (θ − 5)²
```

We want the parameter to eventually reach:

```text
θ = 5
```

The derivative is:

```text
dL/dθ  =  2(θ − 5)
```

Start at:

```text
θ = 0
```

Then:

```text
g = 2(0 − 5) = −10
```

Take a step with:

```text
learning rate = 0.1
```

so:

```text
θ = 0 − 0.1·(−10) = 1
```

Now the parameter is closer to 5.

Run it again:

```text
θ = 1
g = 2(1-5) = -8
θ = 1 - 0.1(-8)
  = 1.8
```

Again:

```text
θ = 1.8
g = -6.4

θ = 2.44
```

Then:

```text
2.44
→ 2.952
→ 3.3616
→ 3.68928
→ ...
→ 5
```

The parameter walks downhill toward the minimum.

That's gradient descent.

Nothing else is required.

---

# So why do we need an optimizer class?

Because we don't want to write:

```text
weight1 -= learningRate * weight1.grad
weight2 -= learningRate * weight2.grad
weight3 -= learningRate * weight3.grad
...
```

for every model.

We want something that can receive:

```text
[parameter1, parameter2, parameter3, ...]
```

and update all of them.

So the optimizer gets a simple job:

```text
parameters + gradients
        ↓
   new parameters
```

That gives us a useful separation:

```text
MODEL

owns parameters
      ↓
backward()
      ↓
fills gradients
      ↓
OPTIMIZER

reads parameters + gradients
      ↓
updates parameters
```

The model does not need to know how the optimizer works.

And the optimizer does not need to know what the model is.

---

# Our first optimizer: SGD

The simplest optimizer is just the rule we already used.

**SGD — Stochastic Gradient Descent**

For every parameter:

```text
θ  ←  θ − lr · g
```

"Stochastic" usually means that the gradient was computed from a mini-batch rather than the entire training dataset.

The important point for us is the update rule.

Suppose:

```text
parameter = 3
gradient  = -2
lr        = 0.1
```

Then:

```text
parameter
= 3 - 0.1 × (-2)
= 3.2
```

If another parameter has:

```text
parameter = 7
gradient  = 4
```

then:

```text
7 - 0.1 × 4
= 6.6
```

Every parameter follows its own gradient.

---

# Build SGD

At this point the implementation should feel almost trivial.

We need something like:

```typescript
export class SGD {
  constructor(
    private params: TensorValue[],
    private lr = 1e-2,
  ) {}

  step(): void {
    for (const p of this.params) {
      // p.data -= lr * p.grad
    }
  }

  zeroGrad(): void {
    for (const p of this.params) {
      // set gradient to zero
    }
  }
}
```

The exact data manipulation depends on your tensor representation.

**The parameters go in the constructor, not into `step()`.** You already chose this shape in Chapter 09, and it is PyTorch's: an optimizer is *bound* to a list of parameters once, then `step()` and `zeroGrad()` take no arguments. It also means the optimizer can hold per-parameter state — which momentum and Adam are about to need.

> **You have written this before.** Chapter 09 built `SGD` and `SGDMomentum` on scalar `Value` objects, because Ch 09 comes before `TensorValue` exists. That version is kept as [`src/optim/sgd-scalar.ts`](../../src/optim/sgd-scalar.ts), untouched and still passing its 16 tests, so you can read it beside what you write here.
>
> The rule is identical. Only the type of `.data` and `.grad` moves from `number` to `Tensor`, so `a - b` becomes `sub(a, b)` and `lr * g` becomes `mulScalar(g, lr)`. What the rebuild buys you is that `TensorValue[]` is exactly what `Linear.parameters()` returns — so this optimizer can train a real layer, and the scalar one cannot.

But the important idea is:

> **SGD does not calculate gradients. It uses gradients that `backward()` has already calculated.**

That distinction is worth keeping clear.

```text
backward()
    ↓
calculate gradients

optimizer.step()
    ↓
use gradients
```

---

# There's a problem with plain SGD

Let's return to the hill analogy.

Imagine the landscape is shaped like a long narrow **valley** — steep walls up and down, and a floor that slopes only gently toward the goal.

A concrete one: `L = 0.5x² + 10y²`. Move a little up or down and the loss changes a lot; move left or right and it barely changes at all.

Plain SGD follows every gradient literally. The up-and-down gradient is twenty times larger than the left-to-right one, so almost all of every step is spent crossing the valley rather than travelling along it.

Here is what that actually looks like — the same 28 steps, walked twice:

<p align="center">
  <img src="../assets/ch-14/ravine-sgd-vs-momentum.svg" alt="A narrow valley drawn as horizontal bands, palest green along the centre floor and reddening toward steep walls above and below, with a start marker at the upper left and a goal marker at the centre right. Two paths cross it. The red plain SGD path zigzags violently up and down, crossing the valley floor twenty-eight times while creeping only slowly rightward, and ends short of the goal. The blue momentum path swings across a few times at the start, settles onto the valley floor, and runs along it to finish much closer to the goal, crossing only eight times. Two animated markers walk both paths in step. Result boxes report plain SGD at learning rate 0.09 crossing the valley twenty-eight times and finishing 0.642 from the goal, against momentum at learning rate 0.03 with beta 0.7 crossing eight times and finishing 0.142 away, four and a half times closer. A note records that both runs use the same effective step size because beta amplifies by one over one minus 0.7, so 0.03 times 3.3 is about 0.09, and closes: up-and-down gradients disagree and cancel, left-to-right gradients agree and accumulate." />
</p>

*Figure 1: the same valley, the same number of steps, the same effective step size.*

SGD crossed the floor **28 times** and finished `0.642` from the goal. Momentum crossed it **8 times** and finished `0.142` away — four and a half times closer.

Look at why, because it is the whole idea in one sentence. Going up and down, consecutive gradients **disagree**, so they cancel. Going left to right, they **agree**, so they accumulate. Momentum does not know anything about valleys; it just adds up recent gradients, and that arithmetic alone separates the two directions.

This is one reason the learning rate is difficult to choose.

Make it too large:

```text
overshoot
overshoot
overshoot
```

Make it too small:

```text
move
move
move
```

and training becomes painfully slow.

We need a way to use the history of previous gradients instead of pretending every gradient is the first one we've ever seen.

That leads us to momentum.

---

# The idea of momentum

Imagine pushing a heavy ball downhill.

If you push it to the right:

```text
→
```

and then one instant later the slope points slightly left:

```text
←
```

the ball doesn't instantly reverse direction.

It has momentum.

Previous movement still matters.

We can give gradient descent the same idea.

Instead of using only the current gradient:

```text
gₜ
```

we keep a running velocity:

```text
vₜ  =  β · vₜ₋₁  +  gₜ
```

and then update:

```text
θₜ  =  θₜ₋₁  −  lr · vₜ
```

The parameter now has memory.

---

# Watch momentum on a simple sequence

Suppose the gradients over several steps are:

```text
+1
+1
+1
+1
```

With momentum:

```text
β = 0.9
v₀ = 0
```

the velocities become:

```text
step 1:
v = 0.9(0) + 1
  = 1

step 2:
v = 0.9(1) + 1
  = 1.9

step 3:
v = 0.9(1.9) + 1
  = 2.71

step 4:
v = 0.9(2.71) + 1
  = 3.439
```

The consistent direction builds up.

Now try alternating gradients:

```text
+1
-1
+1
-1
```

Same rule, same gradient *size* — only the signs differ:

```text
step 1:
v = 0.9(0) + 1
  = 1

step 2:
v = 0.9(1) - 1
  = -0.1

step 3:
v = 0.9(-0.1) + 1
  = 0.91

step 4:
v = 0.9(0.91) - 1
  = -0.181
```

The history partially cancels itself. After eight steps the consistent run has reached **5.695** and the alternating one is at **−0.300** — from identical rules and identical gradient magnitudes.

That is exactly what we want when the gradient is bouncing from side to side.

<p align="center">
  <img src="../assets/ch-14/momentum-buildup.svg" alt="Two velocity curves plotted over eight steps against a zero baseline, both produced by the same rule v equals 0.9 v plus g. The green consistent run, where every gradient is plus one, climbs steadily away from the line through 1, 1.9, 2.71, 3.439, 4.095, 4.686, 5.217 and reaches 5.695, labelled and still climbing. The red alternating run, where gradients are plus one then minus one repeatedly, oscillates tightly around zero through 1, minus 0.1, 0.91, minus 0.181, 0.837, minus 0.247, 0.778 and ends at minus 0.300, labelled going nowhere. Two markers step through both runs together. A footer states that with the same rule and the same gradient size, only the signs differing, the outcomes are 5.695 against minus 0.300 — momentum amplifies a direction the gradients agree on and cancels one they do not." />
</p>

*Figure 1: the same rule and the same gradient size, run twice. Only the signs differ.*

So momentum does two things:

```text
consistent direction
    → builds up

rapidly changing direction
    → gets smoothed out
```

---

# Momentum is still gradient descent

It is important not to think of momentum as replacing gradients.

The gradient is still telling us which way the loss wants to move.

Momentum simply changes how we respond.

Without momentum:

```text
current gradient
      ↓
step
```

With momentum:

```text
current gradient
      +
previous movement
      ↓
velocity
      ↓
step
```

The gradient remains the source of information.

The optimizer is simply using more than one moment of that information.

---

# Build SGD with momentum

Our optimizer now needs some memory.

For each parameter, we need to remember its velocity.

Conceptually:

```typescript
class SGDMomentum {
  constructor(
    private params: TensorValue[],
    private lr: number,
    private beta = 0.9,
  ) {
    // one velocity tensor per parameter, all zeros to start
  }

  step(): void {
    // update each velocity from its parameter's current gradient
    // then update the parameter using that velocity
  }
}
```

Notice the important difference from SGD.

SGD has no persistent state besides the hyperparameters.

Momentum has **state per parameter**.

So if we have:

```text
W
b
```

we need:

```text
velocity[W]
velocity[b]
```

and those vectors remain between training steps.

---

# Why does the velocity need to belong to the parameter?

Suppose your network has:

```text
W1
b1
W2
b2
```

The velocity for `W1` has nothing to do with the velocity for `W2`.

Each parameter has its own history.

That's why the optimizer needs a mapping like:

```text
parameter → optimizer state
```

This idea will become even more important with Adam.

---

# Momentum has a small mathematical consequence

If the gradient stays constant:

```text
g
g
g
g
...
```

then:

```text
vₜ  =  β · vₜ₋₁  +  g
```

approaches:

```text
v  ≈  g / (1 − β)
```

For:

```text
β = 0.9
```

that is:

```text
1 / (1 − 0.9)  =  10
```

times the gradient.

So the velocity can become much larger than an individual gradient.

This is why the learning rate and momentum coefficient interact.

Don't interpret:

> "momentum always means use a ten-times larger learning rate."

It does not.

The actual behavior depends on the gradient history.

The important idea is:

> **Repeated motion in the same direction accumulates.**

---

# Momentum solved one problem

We can now say:

```text
SGD:
    reacts to the current gradient

Momentum:
    reacts to the current gradient
    while remembering recent movement
```

This helps when gradients are noisy or oscillatory.

But there's another problem.

Imagine two parameters:

```text
parameter A → gradients around 100
parameter B → gradients around 0.001
```

Using exactly the same global learning rate for both may be awkward.

Parameter A receives huge gradients.

Parameter B receives tiny gradients.

Why should they necessarily take equally scaled steps?

Maybe each parameter should adapt to its own gradient scale.

And that leads to Adam.

---

# Adam starts by asking two questions

For every parameter, Adam keeps track of two pieces of information.

First:

> **What direction has the gradient usually been pointing?**

Second:

> **How large have the gradients usually been?**

The first becomes the **first moment**:

```text
mₜ  =  β₁ · mₜ₋₁  +  (1 − β₁) · gₜ
```

This is similar to momentum.

The second is the **second raw moment**:

```text
vₜ  =  β₂ · vₜ₋₁  +  (1 − β₂) · gₜ²
```

Notice the square.

We're no longer asking only:

```text
positive or negative?
```

We're also tracking:

```text
how large?
```

---

# Why square the gradient?

Suppose gradients are:

```text
+3
-3
+3
-3
```

Their average is near zero.

So the first moment tells us:

```text
"no consistent direction"
```

But clearly the gradients aren't tiny.

They're large.

Square them:

```text
9
9
9
9
```

Now the second moment tells us:

> "The gradients have substantial magnitude."

That's useful information.

So Adam keeps:

```text
m → direction
v → scale
```

---

# Now we can normalize the update

Suppose one parameter consistently receives large gradients.

Its second moment becomes large:

```text
vₜ large
```

so:

```text
√vₜ large
```

and dividing by it makes the step smaller.

A parameter receiving tiny gradients has a smaller denominator and can retain a larger normalized step.

This produces an update of the form:

```text
θₜ  =  θₜ₋₁  −  lr · mₜ / (√vₜ + ε)
```

Conceptually:

```text
             direction
                │
                ▼
             m_t
                │
                │ divide by
                ▼
       recent gradient scale
                │
                ▼
        normalized update
```

This is the central idea behind Adam.

> **Estimate direction, estimate magnitude, then normalize the direction by the recent magnitude.**

---

# Why can't we use `m` and `v` immediately?

There's a small problem.

Both start at zero:

```text
m₀ = 0
v₀ = 0
```

Suppose the first gradient is:

```text
g₁ = 1
```

Then:

```text
m₁  =  (1 − β₁)
```

With:

```text
β₁ = 0.9
```

that becomes:

```text
0.1
```

The gradient was `1`, but the moving average is only `0.1`.

The same thing happens to the second moment.

The averages start artificially close to zero simply because they had no history yet.

Adam corrects this initial bias:

```text
m̂ₜ  =  mₜ / (1 − β₁ᵗ)
```

```text
v̂ₜ  =  vₜ / (1 − β₂ᵗ)
```

Now the update uses:

```text
θₜ  =  θₜ₋₁  −  lr · m̂ₜ / (√v̂ₜ + ε)
```

The hats mean:

> **"We've corrected the early pull toward zero."**

And notice what kind of fix it is. The factor `1 / (1 − β₁ᵗ)` is a **×10 rescue on the first step**, `×1.54` by step 10, and `×1.005` by step 50. Nothing switches it off — as `β₁ᵗ` decays toward zero the denominator approaches 1, and the formula quietly stops doing anything.

<p align="center">
  <img src="../assets/ch-14/bias-correction-fades.svg" alt="A plot of the bias correction factor one over one minus beta-one to the power t, with beta-one equal to 0.9, across steps 1 to 50. The curve starts at ten on the first step, marked in red with a note that m-hat at step one equals 0.1 divided by 0.1 equals 1, a tenfold rescue, then falls steeply through 2.44 at step 5, 1.54 at step 10 and 1.14 at step 20 before flattening against a dashed green line marking a factor of one, reaching 1.005 by step 50 and labelled effectively off. A marker walks down the curve. A footer explains that m and v start at zero so their first averages read far too small and the correction repairs exactly that, and that nothing turns it off — as 0.9 to the t decays the denominator approaches one and the formula quietly stops mattering." />
</p>

*Figure 3: the correction is large exactly when the averages are empty, and disappears once they are not.*

---

# Let's watch the first Adam step

Take:

```text
g = 1
β₁ = 0.9
β₂ = 0.999
```

Initial state:

```text
m₀ = 0
v₀ = 0
```

After one step:

```text
m₁ = 0.9·0 + 0.1·1 = 0.1
```

and:

```text
v₁ = 0.999·0 + 0.001·1 = 0.001
```

Without bias correction, these values would look much smaller than the actual gradient information.

Correct them:

```text
m̂₁ = 0.1 / (1 − 0.9) = 1
```

```text
v̂₁ = 0.001 / (1 − 0.999) = 1
```

So the normalized update becomes approximately:

```text
1 / (√1 + ε)  ≈  1
```

The first step therefore has a sensible scale.

That is what bias correction is doing.

---

# The complete Adam picture

Adam is now easier to read:

```text
gradient g
    │
    ├──────────────► m
    │                 │
    │                 │ direction
    │                 ▼
    │               m̂
    │
    └──────────────► v
                      │
                      │ size
                      ▼
                    v̂
                      │
                      ▼
              m̂ / (√v̂ + ε)
                      │
                      ▼
                  update
```

So:

```text
m → which way?
v → how big?
```

and the optimizer combines them.

---

# Why is Adam less sensitive to gradient scale?

Suppose every gradient is multiplied by 10.

For example:

```text
g
```

becomes:

```text
10g
```

The first moment scales roughly by 10:

```text
m → 10m
```

The second moment scales roughly by 100:

```text
v → 100v
```

so:

```text
√v  →  10·√v
```

The ratio therefore stays roughly the same:

```text
10·m / (10·√v)   =   m / √v
```

That gives Adam a useful degree of scale adaptation.

It is worth seeing how complete the cancellation is. Run the same problem twice — once normally, once with every gradient multiplied by ten — and Adam's parameter trajectories come out **identical to within 2.6e-9** over eighteen steps. Plain SGD given the same treatment does not converge at all: its step is `lr · (10g)`, ten times too large, and the parameter flips between `0` and `10` forever.

<p align="center">
  <img src="../assets/ch-14/adam-scale-invariance.svg" alt="Three trajectories of the parameter theta over eighteen steps on the loss theta minus five squared, starting from zero with learning rate 0.1. Adam with normal gradients and Adam with every gradient multiplied by ten trace exactly the same gently rising curve, drawn as a thick pale line with a thin line directly on top to show they coincide, labelled exactly on top of each other. Plain SGD with gradients multiplied by ten instead produces a red sawtooth flipping between zero and ten on every step and never settling. A dashed green line marks the target value of five. Two explanation boxes close the figure: why Adam cannot tell, because ten m-hat over ten root v-hat equals m-hat over root v-hat and the scale cancels above and below; and why SGD does, because theta minus learning rate times ten g multiplies the step directly so the learning rate must be retuned." />
</p>

*Figure 2: Adam at ×1 and ×10 lie on the same curve. SGD at ×10 never settles.*

This does **not** mean changing the loss can never matter. The epsilon term, finite precision, parameter interactions, and other details can still make the behavior differ.

The main idea is simply:

> **Adam normalizes updates using the parameter's own gradient history.**

---

# We still haven't talked about large weights

So far we've been asking:

> How do we choose a good direction for each update?

There is another concern:

> **What if the parameters themselves become unnecessarily large?**

Imagine two models that make similarly good predictions.

One uses:

```text
weights around 0.2
```

The other uses:

```text
weights around 200
```

Depending on the model, very large weights can make the system less well behaved and can encourage unnecessary parameter growth.

We can add a pressure toward smaller weights.

This is called **weight decay**.

---

# Weight decay

A simple way to express weight decay is:

```text
θ  ←  θ  −  lr · ( update direction  +  λ·θ )
```

The extra term:

```text
λ · θ
```

pulls parameters toward zero.

So without decay:

```text
θ = 5
```

gets whatever update the optimizer chooses.

With decay, there is also a gentle pull:

```text
5 → slightly smaller
```

and:

```text
-5 → slightly larger
```

toward zero.

This is a form of regularization.

It discourages unnecessarily large parameters.

---

# Why AdamW has its own name

You may see:

```text
Adam + L2 regularization
```

and:

```text
AdamW
```

treated as though they are identical.

They are not exactly the same update.

AdamW uses **decoupled weight decay**.

Conceptually, Adam first computes its adaptive update, and weight decay is then applied separately:

```text
θ  ←  θ  −  lr · m̂ / (√v̂ + ε)  −  lr · λ · θ
```

The important idea is the separation:

```text
Adam update
     +
weight decay
```

rather than hiding the decay term inside the gradient that Adam normalizes.

That is why the "W" matters.

---

# Put the optimizers side by side

Now the three ideas are easy to compare.

### SGD

```text
gradient
   ↓
step
```

```text
θ  ←  θ  −  lr · g
```

### SGD + Momentum

```text
gradient
   ↓
running velocity
   ↓
step
```

```text
vₜ  =  β · vₜ₋₁  +  gₜ
```

```text
θₜ  =  θₜ₋₁  −  lr · vₜ
```

### Adam

```text
gradient
   ├──► direction history
   └──► magnitude history
             ↓
         normalize
             ↓
            step
```

```text
mₜ  =  β₁ · mₜ₋₁  +  (1 − β₁) · gₜ
```

```text
vₜ  =  β₂ · vₜ₋₁  +  (1 − β₂) · gₜ²
```

```text
m̂ₜ  =  mₜ / (1 − β₁ᵗ)
```

```text
v̂ₜ  =  vₜ / (1 − β₂ᵗ)
```

```text
θₜ  =  θₜ₋₁  −  lr · m̂ₜ / (√v̂ₜ + ε)
```

### AdamW

```text
Adam
  +
decoupled weight decay
```

---

# The optimizer is not the learner by itself

This distinction is easy to miss.

An optimizer does not decide whether the model is correct.

It doesn't calculate the loss.

It doesn't calculate the gradient.

The complete system is:

```text
                 forward
data ─────────────────────────► prediction
                                  │
                                  ▼
                                 loss
                                  │
                                  │ backward
                                  ▼
                              gradients
                                  │
                                  ▼
                              optimizer
                                  │
                                  ▼
                            new parameters
                                  │
                                  ▼
                              next step
```

Each piece has one job.

```text
Loss:
    "How wrong?"

Backpropagation:
    "How does each parameter affect that wrongness?"

Optimizer:
    "Given those gradients, how should I move the parameters?"
```

That last question is what this chapter answers.

---

# One more important piece: `zeroGrad()`

Suppose we run:

```text
loss₁.backward()
```

and a parameter gets:

```text
grad = 3
```

Then we run another backward pass.

Unless your engine explicitly replaces gradients, the next gradient may accumulate:

```text
3 + newGradient
```

Sometimes accumulation is exactly what a framework wants.

But in the normal training loop, we want each step to start fresh.

So before the next backward pass, we clear gradients:

```typescript
optimizer.zeroGrad();
```

The normal order is:

```text
zero gradients
      ↓
forward
      ↓
loss
      ↓
backward
      ↓
optimizer.step()
```

Notice that the optimizer doesn't necessarily have to own this responsibility in every design. But giving the optimizer a `zeroGrad()` helper is a convenient interface for our engine.

---

# The complete training step

We've accumulated enough pieces to write a real training step.

Conceptually:

```typescript
optimizer.zeroGrad();

const prediction = model.forward(input);
const loss = lossFunction(prediction, target);

loss.backward();

optimizer.step();
```

Read it in plain English:

```text
forget the old gradients

make a prediction

measure how wrong it is

trace the error backward

use the gradients to change the parameters
```

Then repeat.

That loop is the engine behind training.

---

# Choosing an optimizer

For this course, keep the decision simple.

Start with **SGD** when you want to understand the basic optimization mechanism or when you specifically want the behavior of plain gradient descent.

Use **momentum** when you want SGD to make better use of recent gradient direction and reduce oscillation.

Use **Adam** when you want per-parameter adaptive scaling with relatively little tuning.

Use **AdamW** when you want Adam-style adaptation together with decoupled weight decay, which is a particularly common choice for transformer training.

The optimizer is not a magic switch that makes training good.

A poor learning rate can still make any optimizer fail.

---

# The learning rate is still important

Even with Adam:

```text
too large
→ unstable updates
```

and:

```text
too small
→ painfully slow learning
```

The optimizer changes the update rule.

It does not remove the need to choose a sensible learning rate.

So think of:

```text
optimizer
+
learning rate
```

as a pair.

A useful default is not a law of nature.

It is a starting point that you verify on your actual model.

---

# Build it

Implement these optimizers in:

```text
src/optim/sgd.ts
src/optim/adam.ts
```

with tests in:

```text
src/optim/sgd.test.ts
src/optim/adam.test.ts
```

The common interface is:

```typescript
interface Optimizer {
  step(): void;        // parameters were given in the constructor
  zeroGrad(): void;
}
```

The implementations differ mainly in the state they keep.

### SGD

Needs:

```text
learning rate
```

### Momentum

Needs:

```text
learning rate
momentum coefficient
velocity per parameter
```

### Adam

Needs:

```text
learning rate
beta1
beta2
epsilon
timestep
m per parameter
v per parameter
```

### AdamW

Needs the Adam state plus:

```text
weight decay
```

---

# Implementation details that matter

## Never build optimizer updates into the autograd graph

An optimizer modifies parameters.

That modification is part of the training algorithm, not part of the differentiable forward computation.

So the conceptual boundary is:

```text
autograd graph
    ↓
calculate gradients
    ↓
optimizer
    ↓
mutate parameters
```

Do not accidentally create new graph nodes while updating the parameters.

---

## Optimizer state must follow the parameter

If you have:

```text
W
b
```

then Adam needs:

```text
m[W], v[W]
m[b], v[b]
```

not one shared vector.

The state belongs to the parameter.

---

## State should persist between steps

For Adam:

```text
step 1 → m₁, v₁
step 2 → m₂, v₂
step 3 → m₃, v₃
```

Do not recreate `m` and `v` every time `step()` is called.

If you do, Adam forgets its history and stops being Adam.

When starting a completely new training run, however, the optimizer state should also start fresh.

---

# Verify

Run:

```bash
bun test src/optim/sgd.test.ts src/optim/adam.test.ts
```

and:

```bash
bun run exercises/ch-14-optimizers.ts
```

The most useful tests are tiny, deterministic ones.

For SGD:

```text
parameter = 3
gradient  = -2
lr        = 0.1

after step:
parameter = 3.2
```

For momentum, verify that the velocity carries information from previous steps.

For Adam, verify:

```text
m update
v update
bias correction
parameter update
```

separately.

Don't rely only on a full neural-network training run.

A tiny hand-calculated optimizer test is much easier to debug.

---

# Checkpoint

Before moving on, you should be able to explain:

### Why does SGD work?

Because a gradient tells us which direction increases the loss, so subtracting it moves toward lower loss.

### What does the learning rate control?

The size of the parameter update.

### Why does momentum help?

It remembers recent gradient direction, reinforcing consistent movement and smoothing oscillations.

### What does Adam remember?

A moving average of gradient direction and a moving average of squared gradient magnitude.

### Why square the gradients?

Because positive and negative gradients can cancel when measuring direction, but their squared magnitudes remain positive.

### Why do we need bias correction?

Because Adam's moment estimates start at zero and are therefore initially biased toward zero.

### What does epsilon do?

It prevents division by zero or an excessively tiny denominator.

### What does weight decay do?

It adds a gradual pull toward smaller parameter values.

### Why is AdamW different from simply putting weight decay inside the gradient?

Because AdamW applies the decay separately from the adaptive gradient normalization.

---

# The idea to carry forward

A gradient tells a parameter:

> **"Which way would reduce the loss?"**

But it doesn't completely answer:

> **"How should I move?"**

That's the optimizer's job.

```text
Gradient
    ↓
direction

Optimizer
    ↓
decides the update
```

SGD says:

> Follow the current gradient.

Momentum says:

> Follow the gradient, but remember where you've already been moving.

Adam says:

> Track the usual direction and the usual scale for each parameter, then normalize the update.

AdamW says:

> Do that, and separately encourage the parameters to stay small.

The important progression is:

```text
gradient
   ↓
SGD
   ↓
"current gradient isn't always enough"
   ↓
momentum
   ↓
"gradient scale differs between parameters"
   ↓
Adam
   ↓
"we also want controlled parameter growth"
   ↓
AdamW
```

And now the pieces of the training system are finally complete:

```text
              ┌──────────────┐
              │    MODEL     │
              └──────┬───────┘
                     │
                     ▼
                  output
                     │
                     ▼
              ┌──────────────┐
              │     LOSS     │
              └──────┬───────┘
                     │
                  backward
                     │
                     ▼
              ┌──────────────┐
              │  GRADIENTS   │
              └──────┬───────┘
                     │
                     ▼
              ┌──────────────┐
              │  OPTIMIZER   │
              └──────┬───────┘
                     │
                     ▼
              updated weights
                     │
                     └──────► next forward pass
```

The next chapter will stop studying these pieces separately.

We're going to connect them into one program.

That is where a neural network stops being a collection of classes and starts becoming a **training system**.

---

# Next Chapter

## Chapter 15: The Training Loop

We have all the pieces:

```text
data
  ↓
model
  ↓
prediction
  ↓
loss
  ↓
backward
  ↓
optimizer
```

Next, we'll put them inside a loop and watch the loss decrease over repeated training steps.

That loop will be the first complete training program in the course.
