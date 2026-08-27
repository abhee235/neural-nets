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

The model needs a hidden layer — that is the whole lesson of Ch 11 — so two `Linear` layers with a `relu` between them:

```typescript
const layer1 = new Linear(2, 8);
const layer2 = new Linear(8, 1);

const forward = (x: TensorValue) => layer2.forward(relu(layer1.forward(x)));
```

Collecting the parameters is where Ch 13's contract pays off. Two layers, one flat list:

```typescript
const params = [...layer1.parameters(), ...layer2.parameters()];
const optimizer = new SGD(params, 0.1);
```

That list is **4 tensors and 33 numbers** — shapes `[8,2]`, `[8]`, `[1,8]`, `[1]`. The optimizer walks it without knowing a layer exists, which is exactly what `parameters()` was for.

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
2. Where can `zeroGrad()` go, and which single position breaks training silently?
3. What does a loop with no `zeroGrad()` look like from the outside, and why is it mistaken for a learning rate problem?
4. Two layers produce four parameter tensors. How does the optimizer know what to do with them without knowing what a layer is?
5. Accuracy reached 100% at step 14 and training ran to step 600. Was that wasted?
6. Why can you not train on accuracy directly?
7. `lr = 0.1` solves XOR 30 times out of 30 and `lr = 1.0` zero times out of 30. What is physically happening at the higher rate?
8. Two runs both end with the network outputting a constant. In one the loss is `0.25` and every prediction is `0.5`; in the other the loss is `217` and every prediction is `-14.2`. What is different about the two failures?
9. Plain SGD beat Adam on this problem. What does Adam actually buy, and why does XOR not benefit?

---

# End of Part 3

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
