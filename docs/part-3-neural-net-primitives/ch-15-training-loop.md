# Chapter 15: The Training Loop

> **Part 3 of 6 — Neural Net Primitives**
>
> Exercise: `exercises/ch-15-training-loop.ts`
>
> This chapter adds no new machinery. Everything it uses, you have already built.

---

# Nothing left to build

Look at what is finished.

```text
Ch 10   TensorValue      records operations, and backward() fills every gradient
Ch 11   relu             bends the line, so depth means something
Ch 12   mseLoss          turns "how wrong?" into one number
Ch 13   Linear           owns weights and hands them over via parameters()
Ch 14   SGD, Adam        take that list and move it downhill
```

Five chapters, five pieces, and they have never once run together.

This chapter runs them. It introduces nothing new — no class, no formula, no derivation. Its whole content is the order the pieces go in, and what happens when you turn the handle.

---

# The five lines

Here is the entire thing. Every training run in this course, and every training run in machine learning, is this:

```typescript
optimizer.zeroGrad();                       // 1. forget the last gradients
const prediction = model.forward(input);    // 2. guess
const loss = lossFn(prediction, target);    // 3. score the guess
loss.backward();                            // 4. assign blame
optimizer.step();                           // 5. move the weights
```

Repeat until the loss stops falling.

That is not a simplification. GPT is trained with those five lines, wrapped in data loading and logging and distributed plumbing — but the loop at the centre is exactly this.

What is worth spending a chapter on is **why they are in that order**, because three of the five can be moved and only one arrangement is correct.

---

# Why the order is the order

## `backward()` must come after the loss

Obvious once said: `backward()` walks the graph from the loss downward. There is no graph until the forward pass has built one, and no root to start from until the loss has collapsed it to a single number (Ch 10's guard exists to say exactly this).

## `step()` must come after `backward()`

`step()` reads `.grad`. Before `backward()` those are all `null` — so a `step()` placed earlier would find nothing to do and silently move nothing. The parameters would sit still while everything else appeared to run.

## `zeroGrad()` is the one with a real choice

`backward()` **accumulates** with `+=`. Ch 08 measured what happens when nothing clears it: gradients went `-3`, then `-9`, then `-18` — not merely doubling, because interior nodes compound the contamination with depth. In a loop that inflates the effective learning rate on every iteration, and the loss climbs. It looks exactly like a learning rate set far too high, which is why it is such an annoying bug to find.

So it has to be called. The question is where, and there are two correct answers and one wrong one:

```text
  zeroGrad → forward → loss → backward → step        ✓ correct
  forward → loss → backward → step → zeroGrad        ✓ correct, same thing
  forward → loss → backward → zeroGrad → step        ✗ erases the gradients
                                                       before step() uses them
```

The third runs without error and trains nothing. `step()` finds every gradient `null`, skips every parameter, and the loss sits perfectly still while the loop spins.

The rule underneath: **`zeroGrad()` goes anywhere between one `step()` and the next `backward()`.** Both correct orderings above satisfy that; the broken one does not.

---

# First: what happens when there is more than one layer

Everything so far has been one layer. Ch 13 traced a gradient into a single `Linear`; Ch 14 handed a single layer's parameters to an optimizer. Stacking two of them raises three questions that have never been answered, and they need answering before the loop makes sense.

## But first — what counts as a layer?

The word names two different things, and both usages are standard. It is worth pinning down before counting anything.

<div align="center">
  <img src="../assets/ch-15/what-is-a-layer.svg" alt="The same three-column network drawn twice. In the upper copy the three columns of circles are ringed with dashed boxes and labelled input layer, hidden layer and output layer, with a count on the right reading: counting circles, 3 layers, noted as the older usage and the one most diagrams use. In the lower copy the circles are identical but the two bundles of connecting arrows are boxed in green instead, labelled layer 1 equals Linear of 2 and 3 holding W-one of shape 3 by 2 and b-one of shape 3, and layer 2 equals Linear of 3 and 1 holding W-two of shape 1 by 3 and b-two of shape 1, with a count on the right reading: counting weights, 2 layers, noted as what the code means and what this book means. A footer states that a circle holds a value which is recomputed for every input and thrown away so it owns nothing, while an arrow holds a weight which is kept between inputs and learned, so the arrows are what an object can be." />
</div>

Count the **columns of circles** and that network has three layers: input, hidden, output. This is the older usage, and the one most diagrams you have seen will be using.

Count the **weight matrices** and it has two: one carrying the inputs to the hidden column, one carrying the hidden column to the output. This is what the code means, and what "a two-layer network" means in practice. The input column is never counted here — nothing is learned there, it is only where the data arrives.

Both readings describe the same network. They just name it from different ends.

**Why the code counts the arrows.** A layer object has to own something, and the circles own nothing. A circle holds a value that is computed for one input and thrown away for the next — `h` was `2` for this input and will be something else for the next one. The arrows hold `W` and `b`, which persist between inputs and are what the optimizer updates. So the thing worth making an object is the transformation, not the column.

`new Linear(2, 3)` says so outright: *from a column of 2, to a column of 3*. The object **is** the arrow bundle. Its `forward()` **produces** the next column of circles, but it **is** the arrows that made them.

**So why "layer", and not "section"?** Historical accident, honestly. The word came from the neuron picture — the earliest networks were drawn and described as layers of *units*, and a layer was the units. Decades later, when this was written as code, the transformation was the thing that needed a name, and it inherited the existing word instead of getting a new one. Nobody renamed the old usage either, so both are still in use and you have to read from context. "Sections" would have been clearer. It is too late.

The rule that settles any diagram:

> **Count the weight matrices.** `n` weight matrices is an `n`-layer network, and it will have `n + 1` columns of circles.

One caution about the figure below. It uses **one unit per column**, so that every number can be checked by hand — and at width 1 a column and a single circle look identical. `h` there is a hidden *column* that happens to be one circle wide. When the real XOR model appears later in this chapter it is eight circles wide, and the distinction becomes visible again.

Take the smallest two-layer network there is — one input, one hidden unit, one output — with the weights set by hand so every number can be checked:

```text
  x ──► [ layer1: W₁=2, b₁=0 ] ──► relu ──► [ layer2: W₂=3, b₂=0 ] ──► y
```

Forward, with $x = 1$ and a target of $0$:

$$
\begin{aligned}
z_1 &= W_1 x + b_1 = 2 \cdot 1 + 0 = 2 \\[4pt]
h   &= \text{relu}(z_1) = \text{relu}(2) = 2 \\[4pt]
y   &= W_2 h + b_2 = 3 \cdot 2 + 0 = 6 \\[4pt]
L   &= (y - 0)^2 = 6^2 = 36
\end{aligned}
$$

where $z_1$ is layer 1's **pre-activation** — its output before `relu` sees it — and $h$ is the hidden value that layer 2 actually receives.

Now one call to `loss.backward()`. It walks the same path in reverse, four steps.

**Step 1 — at the loss.** Differentiating $L = (y - t)^2$ with respect to $y$:

$$\frac{\partial L}{\partial y} = 2(y - t) = 2(6 - 0) = 12$$

**Step 2 — layer 2.** Its input is the hidden value $h = 2$, so by the rule from Ch 13, each weight's blame is the upstream gradient times the input it multiplied:

$$
\begin{aligned}
\frac{\partial L}{\partial W_2} &= \frac{\partial L}{\partial y} \cdot h  = 12 \cdot 2 = 24 \\[4pt]
\frac{\partial L}{\partial b_2} &= \frac{\partial L}{\partial y} = 12 \\[4pt]
\frac{\partial L}{\partial h}   &= \frac{\partial L}{\partial y} \cdot W_2 = 12 \cdot 3 = 36
\end{aligned}
$$

That last line is the one to watch:

$$\boxed{\;\frac{\partial L}{\partial h} = 36\;}$$

**Step 3 — the `relu` gate.** The pre-activation was $z_1 = 2$, which is positive, so $\text{relu}'(z_1) = 1$ and the gradient passes through untouched (Ch 11's gate, not a scale):

$$\frac{\partial L}{\partial z_1} = \frac{\partial L}{\partial h} \cdot \text{relu}'(z_1) = 36 \cdot 1 = 36$$

Had $z_1$ been negative, this would be $36 \cdot 0 = 0$ — and layer 1 would have received nothing at all. That is the dying ReLU, seen from the inside.

**Step 4 — layer 1.** Its input is $x = 1$, and its upstream is that $36$:

$$
\begin{aligned}
\frac{\partial L}{\partial W_1} &= \frac{\partial L}{\partial z_1} \cdot x = 36 \cdot 1 = 36 \\[4pt]
\frac{\partial L}{\partial b_1} &= \frac{\partial L}{\partial z_1} = 36
\end{aligned}
$$

<p align="center">
  <img src="../assets/ch-15/two-layer-gradient-flow.svg" alt="A two-layer network drawn as circles and connections: an input node holding 1, an edge labelled W-one equals 2, a hidden node holding 2 after relu, an edge labelled W-two equals 3, an output node holding 6, and a loss box showing 36. The animation alternates between two phases. In the forward phase a green arrow runs left to right across the top with the arithmetic on each edge, 2 times 1 plus 0 equals 2, then 3 times 2 plus 0 equals 6. In the backward phase a red arrow runs right to left underneath, starting from dL by dy equals 12 at the output. A box under layer two reads that its input is h equals 2, giving dL by dW-two equals 12 times 2 equals 24 and dL by db-two equals 12. Between the layers, dL by dh equals 12 times 3 equals 36 is labelled as layer two's input gradient becoming layer one's upstream, passing through a relu gate marked open because the pre-activation 2 is positive. A box under layer one reads that its input is x equals 1, giving dL by dW-one equals 36 times 1 equals 36 and dL by db-one equals 36. The footer states that one backward call filled all four parameters, that each layer blames its own weights and passes what is left to the layer beneath, and that with n layers it is the same two jobs n times." />
</p>

*Figure 1: the same trace, drawn. Forward along the top, backward along the bottom.*

## The number that connects them

Look at `dL/dh = 36`. That is `x.grad` **of the second layer** — and Chapter 13 mentioned it in passing: *"unused here, but it is how blame reaches the layer below when layers stack."*

This is that moment. Layer 2's input is layer 1's output, so layer 2's `x.grad` is exactly layer 1's upstream gradient. The chain hands off there, and `relu` sits in between as a gate deciding whether anything gets through.

Each layer does the same two jobs it always did:

```text
  blame my own W and b     using whatever came from above
  pass blame further down   so the layer beneath can do the same
```

With `n` layers it is that, `n` times. Nothing new appears at three layers, or at ninety-six.

## Parameters are not inputs

Worth stating flatly, because the words invite the mix-up: **a parameter is any number the network learns.** Not the input, and not only the first layer's numbers.

```text
  x           an INPUT       given to us. never changes. not a parameter.
  W₁, b₁      parameters     layer 1's, learned
  W₂, b₂      parameters     layer 2's, learned
  h           an activation  computed on the way through. not stored, not learned.
```

Our tiny network has **four** parameters, two per layer. And after that single `backward()`, all four have a gradient:

```text
  [0]  W₁   value 2    grad 36
  [1]  b₁   value 0    grad 36
  [2]  W₂   value 3    grad 24
  [3]  b₂   value 0    grad 12
```

That is why the optimizer receives every layer's parameters in one list, and why it updates them **all in the same step**. There is no notion of training layer 1 first and layer 2 afterwards. One forward pass, one backward pass, and every weight in the network moves at once.

The flat list works because each tensor already carries its own gradient. The optimizer walks the list, reads `.grad`, and subtracts — it never needs to know which layer a tensor came from, or that layers exist at all. That is what Chapter 13's `parameters()` contract was for, and this is the first place it is doing real work.

---

# And how wide should the hidden layer be?

The other question the code below raises: `new Linear(2, 8)` — where did the `8` come from?

Honestly: it is a choice, not a derivation. The input width is fixed by the data (two features) and the output width by the task (one number), but **everything in between is yours to pick.** What follows is how to pick it.

The floor is set by what the problem needs. Ch 11's exercise E6 built an exact XOR solution with **two** hidden units, so two is the theoretical minimum. Here is what actually happens at each width — 20 random initialisations, SGD at `lr = 0.1`:

```text
  hidden   parameters   solved

     1          5        0/20
     2          9        8/20   ████████
     3         13       14/20   ██████████████
     4         17       14/20   ██████████████
     8         33       20/20   ████████████████████
    16         65       20/20   ████████████████████
    32        129       19/20   ███████████████████
```

**One unit cannot do it, ever** — that is Ch 11's impossibility result, and no amount of training changes it. **Two units can, but only 8 times in 20.** A solution exists at that width; gradient descent just does not reliably find it, because losing a single unit to the dying-ReLU problem leaves nothing to work with.

By eight units it is 20/20. The extra units are not adding expressive power the problem needs — they are **redundancy**. With eight hinges available, a few can die and the rest still cover the job.

That is the practical rule, and it is not glamorous:

> **Wide enough that losing a few units does not matter. Then stop.**

`8` is that for XOR. Going to 32 buys nothing (`19/20`, statistically the same as 20) while quadrupling the parameters. Chapter 30's GPT will use a hidden width of hundreds for the same reason at a different scale — enough capacity, plus slack.

---

# Build it — XOR, with the real classes

The problem is the one this part of the course opened with. Chapter 11 introduced XOR as the thing a linear model provably cannot solve, and its exercise trained a net on it with a hand-rolled loop, because `Linear`, `mseLoss` and `Adam` did not exist yet.

Now they do. Same problem, real classes:

```text
  inputs            targets
  [0, 0]      →       0
  [0, 1]      →       1
  [1, 0]      →       1
  [1, 1]      →       0
```

The model needs a hidden layer — that is the whole lesson of Ch 11 — so two `Linear` layers with a `relu` between them, exactly the shape traced above:

```typescript
const layer1 = new Linear(2, 8);   // 2 features in, 8 hidden units
const layer2 = new Linear(8, 1);   // 8 hidden units in, 1 score out

const forward = (x: TensorValue) => layer2.forward(relu(layer1.forward(x)));
```

The `2` and the `1` are fixed by the problem. The `8` is the choice from the previous section — comfortably above the two-unit minimum, so a dead unit or two costs nothing.

Notice the widths have to meet: `layer1` produces 8 numbers, so `layer2` must expect 8. That is the only constraint between adjacent layers, and getting it wrong is a shape error rather than a silent bug.

Now the parameters — all of them, from both layers, in one list:

```typescript
const params = [...layer1.parameters(), ...layer2.parameters()];
const optimizer = new SGD(params, 0.1);
```

**4 tensors, 33 numbers** — shapes `[8,2]`, `[8]`, `[1,8]`, `[1]`. Two weight matrices and two bias vectors, which is every learnable number in the network.

One `backward()` fills all four, and one `step()` moves all four, exactly as in the hand-traced example. The optimizer walks the list without knowing a layer exists.

Drawn out, that is where those 33 numbers sit:

<p align="center">
  <img src="../assets/ch-15/xor-network.svg" alt="The XOR network drawn as neurons and connections. On the left, two input circles labelled x-one and x-two, annotated as given and not learned. In the middle, a column of eight hidden circles labelled h1 through h8, each with relu, annotated as activations that are computed and not stored. On the right, a single output circle. Sixteen pale edges fan from the two inputs to the eight hidden units, and eight blue edges converge from the hidden units to the output. Three boxes along the bottom account for every learnable number: W-one of shape 8 by 2 is 16, one per edge on the left side, plus b-one of shape 8 is 8, one per hidden unit; W-two of shape 1 by 8 is 8, one per edge on the right side, plus b-two is 1; giving 33 parameters in 4 tensors, one flat list. A note at the far left reads that every edge is a weight." />
</p>

*Figure 2: every edge is a weight, every hidden unit carries a bias, and the output has one of its own. 16 + 8 + 8 + 1 = 33.*

Then the five lines, in a loop:

```typescript
for (let step = 0; step < 600; step++) {
  optimizer.zeroGrad();
  const loss = mseLoss(forward(inputs), targets);
  loss.backward();
  optimizer.step();
}
```

✅ *Checkpoint:* after 600 steps the loss is `0.000000` and the four predictions are `0, 1, 1, 0` to four decimals.

---

# What you actually watch

A training loop prints two numbers, and Chapter 12 explained why they are two rather than one. Here is that argument as a real run — `SGD`, `lr = 0.1`:

```text
  step     loss       accuracy
     0    0.725774      50%
    50    0.071295     100%
   100    0.010616     100%
   200    0.000094     100%
   600    0.000000     100%
```

Weights start random, so your exact numbers will differ. The *shape* does not, and it is the shape that matters.

**Accuracy reaches 100% early and then never moves again.** Across eight runs it first hit 100% somewhere between step 4 and step 62 — and in every one of them the loss at that moment was still around `0.09`–`0.20`, nowhere near zero. The model was already answering all four questions correctly, and training then ran for hundreds more steps.

**Accuracy changed on a handful of steps — between 2 and 6 of the 600.** The loss changed on all 600.

That is Chapter 12's figure, live: accuracy is a staircase, the loss is smooth. It is the reason you cannot train on accuracy, and the reason you report it anyway. **The loss is the thing you can descend. Accuracy is the thing you care about.** Once accuracy saturates, only one of them still has anything to say.

---

# What goes wrong

The loop is five lines and cannot really be written incorrectly once the order is right. Everything that goes wrong is in the numbers you hand it — so the useful part of this chapter is knowing what failure looks like.

Here is the same XOR setup, 3000 steps, run 30 times from different random initialisations, counting how often it actually solves the problem:

```text
  optimizer / learning rate     solved

  SGD          lr 0.05          30/30   ████████████████████
  SGD          lr 0.1           30/30   ████████████████████
  SGD          lr 0.5           11/30   ███████
  SGD          lr 1.0            0/30
  SGDMomentum  lr 0.05          23/30   ███████████████
  Adam         lr 0.01          29/30   ███████████████████
  Adam         lr 0.05          25/30   █████████████████
```

## The learning rate has a cliff, not a slope

`0.1` works every time. `0.5` works a third of the time. `1.0` never works. There is no gentle degradation — the model goes from perfect to hopeless across one factor of ten.

What is happening at `lr = 1.0` is Ch 11's dying ReLU. One oversized step drives a unit's input negative for all four inputs at once; its gradient becomes exactly `0` and it never updates again. Enough dead units and there is no hidden layer left — the network can only output a constant.

That collapse has a signature you can check for, and it is not the loss value:

```text
  lr = 1.0, 3000 steps, 12 different initialisations

  all four predictions identical:   12 / 12
  loss anywhere near 0.25:           0 / 12
```

**The tell is that every input produces the same output.** Feed the network four different things and get one number back, and the hidden layer is gone. The loss meanwhile is large and unstable — 21, 33, 217 across runs — because the surviving output bias is being driven by the same oversized step and oscillates instead of settling.

(A loss of exactly `0.25` with every prediction `0.5` is a *different* failure — that is a network that is dead but *stable*, so its output bias settles on the mean of the targets. Chapter 11's exercise E10(a) measured it from zero-initialised weights. Same "network became a constant", different cause and different arithmetic.)

## The optimizer is not a magic switch

Read the table again. **Plain SGD at a sensible learning rate is the most reliable thing on it** — 30/30, beating Adam's 25/30 at the same rate.

That is worth sitting with, because the expectation runs the other way. Adam is what transformers train with, and Chapter 14 spent a lot of pages on why. But on a small, clean, well-conditioned problem, plain gradient descent at a sane step size is hard to beat.

What Adam buys is not speed here — it is a **wider band of learning rates that work**, and per-parameter scaling that matters when different parameters see gradients of wildly different sizes. XOR has neither problem. GPT has both.

Notice `SGDMomentum` at `lr = 0.05` scoring `23/30`, worse than plain SGD at the same rate. Chapter 14 predicted this: momentum with `β = 0.9` amplifies the effective step by `1/(1−0.9) = 10`, so `lr = 0.05` behaves like `lr ≈ 0.5` — and plain SGD at `0.5` scores `11/30`. The two numbers agree about the cause.

---

# The three failures worth recognising

Everything above collapses into a short diagnostic list. When training does not work, it is almost always one of these, and each has a signature:

```text
  loss climbs steadily             zeroGrad() missing — gradients accumulate,
                                   so the effective learning rate grows

  loss does not move AT ALL        zeroGrad() called between backward() and
                                   step() — the gradients are erased unused

  every input gives the SAME       dead units, lr too high. the hidden layer
  output, loss large and erratic   is gone; only a constant is left

  every prediction is 0.5 and      dead but stable — usually zero init.
  the loss sits at 0.25            the output bias settled on the mean

  loss falls then flattens         working. the lr may be too small, or the
  far above zero                   model too small for the problem
```

The first four are bugs. The last is a decision.

---

# What to implement

Nothing new — that is the point of this chapter. The exercise file assembles what you have:

| | |
|---|---|
| the loop | five lines, in the order above |
| the model | two `Linear` layers with `relu` between them |
| the parameters | `[...layer1.parameters(), ...layer2.parameters()]` |
| accuracy | three lines: threshold at 0.5, compare, average |

There is no `MLP` class and no dataset generator, deliberately. Two layers written out are clearer than a container that hides them, and XOR is four rows you can read.

---

# Verify

```bash
bun run exercises/ch-15-training-loop.ts
```

You should see the loss reach `0.000000`, predictions of `0, 1, 1, 0`, and — if you run the learning-rate sweep — the cliff between `0.1` and `1.0`.

---

# What you should now be able to explain

1. Why must `backward()` come after the loss and not after the forward pass?
2. In the two-layer trace, `dL/dh` came out as 36. What is that number to layer 2, and what is it to layer 1?
3. Name every parameter in a `Linear(2,8) → relu → Linear(8,1)` network. Is `x` one of them? Is the hidden activation?
4. The optimizer gets one flat list of four tensors. How does it know which layer each came from — and why does it not need to?
5. Where can `zeroGrad()` go, and which single position breaks training silently?
6. What does a loop with no `zeroGrad()` look like from the outside, and why is it mistaken for a learning rate problem?
7. Accuracy reaches 100% within the first few dozen steps and training runs to 600. Was the rest wasted?
8. Why can you not train on accuracy directly?
9. `lr = 0.1` solves XOR 30 times out of 30 and `lr = 1.0` zero times out of 30. What is physically happening at the higher rate?
10. Two runs both end with the network outputting a constant. In one the loss is `0.25` and every prediction is `0.5`; in the other the loss is `217` and every prediction is `-14.2`. What is different about the two failures?
11. Plain SGD beat Adam on this problem. What does Adam actually buy, and why does XOR not benefit?

---# End of Part 3

You can now build and train a neural network from nothing.

```text
  tensors            Ch 01–06     the arithmetic
  autodiff           Ch 07–10     gradients, for free, through any graph
  activations        Ch 11        the bend that makes depth mean something
  losses             Ch 12        one number to descend
  layers             Ch 13        parameters, owned and handed over
  optimizers         Ch 14        the list, moved downhill
  the loop           Ch 15        all of it, turning
```

Everything after this point is **architecture** — new arrangements of these same parts. Attention (Ch 22) is `Linear` layers and a `softmax`. A transformer block is those, plus a normalisation and a residual connection. GPT is that block, stacked.

None of it adds a new kind of thing. The engine is finished.

---

# Next Chapter

**Ch 16: Character Tokenizer** — Part 4 begins, and the question changes from *how does a network learn* to *how does text become something a network can read*.
