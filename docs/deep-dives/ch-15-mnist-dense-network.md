# Deep Dive — A three-hidden-layer network on MNIST

**Extends:** Chapter 15 (The Training Loop)
**Run it:** `bun run exercises/deep-dive-mnist-dense.ts`

Every chapter so far has trained on a toy. XOR is four rows. The circle from the Ch 15 exercise is a thousand points in two dimensions. Both were chosen because you can hold them in your head.

This is the other kind of test. **784 inputs, ten classes, three hidden layers, 111,146 parameters, and a dataset nobody solves by inspection.** The question is not whether neural networks work — it is whether *the library you built* is a real one.

The answer is yes, and the interesting part is what it gets wrong.

> **Nothing new is imported.** `Linear` (Ch 13), `relu` (Ch 11), `crossEntropyFromLogits` (Ch 12), `Adam` (Ch 14) and `argmax` (Ch 05). The only code written for this deep dive is a file reader and a batching helper, and neither is machine learning.

---

## The data, and the four decisions before any of it

MNIST is 70,000 handwritten digits, 28×28 pixels, greyscale. Before a single weight is initialised, four choices have already been made about it, and each one is a place beginners lose days.

### 1. What an image *is*, to the network

<div align="center">
  <img src="../assets/deep-dives/ch-15-what-a-digit-is.svg" alt="A real MNIST test image of the digit 7 rendered as a 28 by 28 grid of grey squares, labelled 784 pixels, with a note that it is flattened row-major into one row of 784 numbers giving shape 1 by 784. Beside it a 6 by 6 patch from the middle of the image is blown up, each cell showing the actual stored byte value between 0 and 255, with 0 meaning paper and 255 meaning ink. A panel explains scaling: stored values 0 to 255 become used values 0.0 to 1.0, because Ch 13 chose the initialisation scale assuming inputs near 1, and feeding raw bytes makes the first pre-activations about 255 times too large, straight into the flat tail of everything after." />
</div>

There is no image. There is a **row of 784 numbers**, laid out row-major exactly as Ch 01 defined it: pixel `(row, col)` lives at index `row × 28 + col`. The network never learns that pixel 400 is directly below pixel 372 — that fact is thrown away by flattening, and the network has to rediscover any of it that matters from the data alone.

That is a genuine weakness of a *dense* network, and it is why convolutions exist. Worth knowing now, so the accuracy ceiling later makes sense.

### 2. Sampling — why 200 of each digit, not the first 2,000

The full training set is 54 MB and has no business in a course repository, so the vendored subset takes **200 training and 50 test images per digit**. Two hundred *per class*, not two thousand off the top.

Taking the first 2,000 rows would be simpler and wrong. MNIST is close to balanced but not exactly, and a skewed subset makes accuracy hard to reason about: if 30% of your test set were the digit `1`, a model that only ever guessed `1` would score 30% and look like it had learned something. A **stratified** sample removes that ambiguity — every digit appears exactly as often as every other:

```text
  training images per digit
  0:200  1:200  2:200  3:200  4:200  5:200  6:200  7:200  8:200  9:200
```

so chance is exactly 10%, and any score above it is real signal. `scripts/make_mnist_subset.py` does the drawing and documents the file format.

### 3. Scaling — why every pixel is divided by 255

Pixels arrive as bytes, `0` to `255`. They are used as floats, `0.0` to `1.0`:

$$x_{\text{used}} = \frac{x_{\text{stored}}}{255}$$

This is not tidying. Chapter 13 derived the `he` initialisation scale $\sqrt{2 / \text{inputDim}}$ **on the assumption that inputs are roughly unit-sized** — that argument was about keeping the variance of a sum of 784 terms under control. Feed raw bytes instead and every term in that sum is up to 255× larger, so the first pre-activations are enormous, and every layer after the first spends its time in the saturated tail. The network does not crash. It just learns badly, for a reason that is invisible unless you know to look.

**One line, and it is load-bearing.**

### 4. One-hot targets — why `[500, 10]` and not `[500]`

The label of an image is a single digit, `7`. The loss does not want a `7`; it wants a row of ten numbers with a `1` in column 7:

```text
  label 7  →  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
```

That is the convention `crossEntropyFromLogits` was built around in Ch 12 — the mask has to be the same shape as the logits so the true-class logit can be selected with `mul` and `sum` and **stay in the graph**. Ch 12's instructive bug was reading that logit's *value* out instead, which severed it and turned the gradient `p − y` into plain `p`.

So targets are shape `[count, 10]`, matching the model's output exactly.

---

## The architecture, and where the parameters actually are

<div align="center">
  <img src="../assets/deep-dives/ch-15-mnist-architecture.svg" alt="The network drawn in the conventional style, as columns of circles joined by lines. Five columns are labelled input layer 784, hidden layer 1 with 128 units, hidden layer 2 with 64, hidden layer 3 with 32, and output layer 10. Only a few units per layer are drawn with a vertical ellipsis standing for the rest; the ten output units are shown in full and labelled 0 to 9. Every drawn unit connects to every drawn unit in the next column, and a caption notes that each line is one weight. Beneath each bundle of connections its parameter count and share is given: 100,480 at 90.4 percent, then 8,256 at 7.4 percent, then 2,080 at 1.9 percent, then 330 at 0.3 percent, with relu marked between each pair and logits after the last. A footer states that the first bundle holds 100,480 of 111,146 parameters, 90.4 percent of the network, because there are 784 times 128 lines on the left but only 32 times 10 on the right: depth is cheap, the input is expensive." />
</div>

Four `Linear` layers, `relu` between each pair, nothing after the last:

$$784 \;\rightarrow\; 128 \;\rightarrow\; 64 \;\rightarrow\; 32 \;\rightarrow\; 10$$

**Why a funnel?** Each layer has to describe its input using fewer numbers than it received. `128` is not enough room to store a 784-pixel image, so the layer is forced to keep whatever distinguishes digits and discard the rest. Stack three of those and the network is repeatedly asked to compress — and the only compressions that survive training are the ones that keep the loss down.

That is the intuition. Be honest that it is an intuition: the widths `128, 64, 32` are conventional, not derived. Ch 15's rule still applies — wide enough that losing a few units does not matter, then stop.

**Where the parameters are is not intuitive at all.** A layer's parameter count is $(\text{in} \times \text{out}) + \text{out}$:

| layer | shape | weights | biases | total | share |
|---|---|---|---|---|---|
| 1 | `784 → 128` | 100,352 | 128 | **100,480** | 90.4% |
| 2 | `128 → 64` | 8,192 | 64 | 8,256 | 7.4% |
| 3 | `64 → 32` | 2,048 | 32 | 2,080 | 1.9% |
| 4 | `32 → 10` | 320 | 10 | 330 | 0.3% |
| | | | | **111,146** | |

**The first layer is 90% of the network.** The three deeper layers together are under 10%. When people say a model has *N* parameters, that number is usually dominated by whichever layer touches the widest thing — here, the raw image.

### One image, every shape

<div align="center">
  <img src="../assets/deep-dives/ch-15-shape-trace.svg" alt="A vertical trace of one image through the whole network. Starting from shape 1 by 784, the image, flattened. Then layer 1 with W1 of shape 128 by 784 computing x times W1 transpose plus b1, 100,480 parameters, giving shape 1 by 128 labelled z1 pre-activation. Then relu, described as negatives to zero with shape unchanged, giving 1 by 128 labelled h1 hidden 1. Then layer 2 with W2 of shape 64 by 128, 8,256 parameters, giving 1 by 64 labelled z2, then relu, giving h2. Then layer 3 with W3 of shape 32 by 64, 2,080 parameters, giving 1 by 32 labelled z3, then relu giving h3. Then layer 4 with W4 of shape 10 by 32, 330 parameters, giving 1 by 10, labelled logits with no activation. A footer notes the batch dimension stays 1 here and in training it is 64, and nothing else changes." />
</div>

Every step is `y = xWᵀ + b` from Ch 13, then a gate from Ch 11. Read it once and the implementation writes itself.

Two details in that trace worth stating outright:

**The batch dimension changes nothing.** The trace shows `[1, 784]`; training uses `[64, 784]`. Every shape below simply gains the same first number. That is the whole reason batching is free.

**The last layer has no activation.** It emits ten raw **logits** — unbounded scores, not probabilities. `crossEntropyFromLogits` applies the softmax internally (Ch 12's log-sum-exp trick). Apply `softmax` yourself first and it happens twice; Ch 15 measured that failure — it does not crash, it just learns badly and stops short.

---

## Batching, shuffling, epochs

Three words that get used together and mean different things.

<div align="center">
  <img src="../assets/deep-dives/ch-15-shuffling-batching.svg" alt="An animated diagram of shuffling and batching using 24 stand-in images coloured by digit. The top row shows the images sorted, marked never do this, grouped into four dashed batches, with a note that every batch is one or two digits so each step pulls one way. Below, a shuffled row alternates between two different random orders labelled epoch 1 and epoch 2, with a note that every batch is a mixture so the step is an average. A footer gives the arithmetic: one batch is 64 images giving one forward, one backward and one optimizer step; one epoch is 31 batches, every image seen once, 2000 divided by 64 is 31 remainder 16; 30 epochs is 930 optimizer steps and the order is redrawn every epoch. It notes the leftover 16 are dropped each epoch but shuffling means they are different 16 every time." />
</div>

**A batch** is how many images go through *together* before the weights move. Here, 64. Not 1, because a single image gives a noisy, unrepresentative gradient. Not all 2,000, because then the whole dataset must be processed for a single step and training crawls. Sixty-four rows go in as `[64, 784]`, one loss comes out — Ch 12's mean over the batch — and one `optimizer.step()` follows.

**An epoch** is one pass through the whole training set:

$$\left\lfloor \frac{2000}{64} \right\rfloor = 31 \text{ batches per epoch}, \qquad 2000 - 31 \times 64 = 16 \text{ images left over}$$

Those 16 are dropped — the loop takes only full batches. That is fine *because of shuffling*: it is a different 16 every epoch.

**Shuffling** is redrawing the order before every epoch, and it matters more than it sounds. The vendored file is stratified but ordered. Without a shuffle, every batch would be almost entirely one digit, and each step would drag the weights toward "everything is a 3", then "everything is a 4". The gradient of a batch is an *average*, and an average over one class is not a useful direction. Shuffle, and every batch is a mixture.

$$30 \text{ epochs} \times 31 \text{ batches} = 930 \text{ optimizer steps}$$

930 steps, on 111,146 parameters, in about 83 seconds.

---

## The loop, unchanged

This is the point of the whole exercise. Scaling from four XOR rows to 111,146 parameters changed the *data pipeline* completely and changed the training loop **not at all**:

```typescript
optimizer.zeroGrad();                              // 1. forget
const logits = model.forward(new TensorValue(x));  // 2. guess
const loss = crossEntropyFromLogits(logits, y);    // 3. score
loss.backward();                                   // 4. blame
optimizer.step();                                  // 5. move
```

Same five lines as Ch 15. One `backward()` call still fills all eight parameter tensors, still by the rule traced by hand in Ch 15: each layer blames its own `W` and `b`, then hands what is left to the layer beneath. With four layers it is the same two jobs, four times.

---

## What happened

<div align="center">
  <img src="../assets/deep-dives/ch-15-training-curve.svg" alt="A chart of 30 training epochs. Two accuracy lines rise from around 72 to 77 percent at epoch 1: a blue train line that reaches 100 percent at epoch 15 and stays there, and a green test line that rises quickly to about 88 percent by epoch 5 then flattens near 91 percent. The area between them is shaded red and labelled this gap is memorisation. A dashed vertical line marks epoch 15 where train accuracy hits 100 percent. Final values are marked train 100.0 percent and test 91.4 percent. Below, the training loss is plotted on a log scale falling continuously from 1.6025 to 0.0018 across all 30 epochs, with a caption noting the loss keeps falling for all 30 epochs while test accuracy stopped improving around epoch 22 at 91.8 percent, and that the loss is not the goal." />
</div>

```text
  epoch     loss   train acc   test acc
      1   1.6025      77.0%      71.8%
      3   0.3522      93.8%      86.8%
      5   0.1787      97.1%      88.2%
     10   0.0420      99.6%      91.0%
     15   0.0127     100.0%      90.8%
     30   0.0018     100.0%      91.4%
```

**Train accuracy reaches 100% at epoch 15 and the test accuracy never moves again.** The loss keeps falling — by a further factor of seven — and buys nothing. The best test score, 91.8%, was at epoch 22; the last eight epochs were pure memorisation of 2,000 specific images.

This is the picture Ch 12 promised: **the loss is what you optimise, not what you care about.** Here you can watch them come apart.

### Where the mistakes are

<div align="center">
  <img src="../assets/deep-dives/ch-15-confusion.svg" alt="A ten by ten confusion matrix heatmap for 500 test images, 50 of each digit. Rows are the true digit and columns are the network's prediction; the diagonal is shaded green for correct and off-diagonal cells shaded red in proportion to their count. Per-digit recall is listed down the right side. Digit 6 is the weakest at 86 percent. A caption notes that every off-diagonal cell is a pair the network finds genuinely similar, and that a single accuracy number would have hidden all of this." />
</div>

91.4% means 43 mistakes out of 500. A single number cannot tell you whether those are spread evenly or concentrated — the matrix can. The weakest digit is **6, at 86% recall**, and the confusions are not random: they are pairs that genuinely share strokes.

### Confidently wrong

<div align="center">
  <img src="../assets/deep-dives/ch-15-confidently-wrong.svg" alt="The first misclassified test image, number 8, rendered as a 28 by 28 grid: a handwritten 5 that the network called a 6. Beside it, a bar chart of the softmax probabilities over all ten digits, with the bar for 6 filling almost the entire width. The label reads 99.81 percent confident, and wrong. A caption explains that softmax always sums to 1 so it always names a winner, and that a confident output is not a correct one." />
</div>

The first mistake in the test set is a `5` called a `6` — with **99.81% confidence**.

This is worth sitting with. `softmax` normalises ten numbers so they sum to 1. It will *always* produce a winner, and it will produce a confident-looking winner whenever one logit is somewhat larger than the others. **Confidence is not correctness, and the network has no way to say "I don't know."** Nothing in the loss ever asked it to.

---

## Was the depth worth it?

Three hidden layers, and the honest comparison is against simpler models on the same data, same optimizer, same 30 epochs. Five runs each, because `Linear`'s initialisation is random and a single run is not a measurement:

| model | parameters | test accuracy (min / median / max) | train |
|---|---|---|---|
| chance | — | 10.0% | — |
| linear only, `784 → 10` | 7,850 | 86.8 / **87.8** / 88.4 | 94.5% |
| one hidden layer, `784 → 128 → 10` | 101,770 | 89.0 / **89.8** / 90.8 | 99.8% |
| three hidden layers *(this)* | 111,146 | 91.0 / **91.2** / 91.2 | 99.7% |

**A model with no hidden layer at all — just `784 → 10`, a linear classifier — already gets 87.8%.** The first hidden layer buys two points. The next two buy one and a half more.

So depth *does* help here, and by a second measure it helps more than the medians suggest: look at the spread. The one-hidden-layer model ranges over 1.8 points depending on how its weights were initialised; the three-layer model ranges over **0.2**. Extra capacity made the result not just better but *repeatable* — a network with more units to spare is less at the mercy of a few unlucky dead ones, which is the redundancy argument from Ch 15's width section showing up again at a different scale.

But keep the size of the win in proportion. Three hidden layers and 111,146 parameters bought **3.4 points over having no hidden layer at all**, and the model is stuck at 91% while a human reads these digits at essentially 100%. That last gap is not something more layers will close, and the 99.7% train accuracy is the proof: the network has already extracted everything those 2,000 images contain. It is not short of capacity — it is short of data.

More images would help. Convolution, which builds in the fact that neighbouring pixels are related — the very fact flattening destroyed on the first page of this document — would help more.

---

## What this proves about the library

The point was never the accuracy. It was this:

- **Four stacked layers train correctly.** One `backward()` fills eight parameter tensors, and the `parameters()` contract from Ch 13 scaled without modification.
- **`crossEntropyFromLogits` handles ten classes** and a `[64, 10]` batch, log-sum-exp and all.
- **`Adam` manages 111,146 parameters** with per-parameter moment state, and the bias correction from Ch 14 is doing its job in the first steps — the loss falls from 1.60 to 0.62 in a single epoch.
- **It is fast enough to be useful.** 930 optimizer steps in 83 seconds, in pure TypeScript, with a `matMul` that is three nested loops.

Nothing in `src/` was changed to make this run. **The library is finished enough to be a real one.**

---

## What is deliberately missing

Every one of these is a later chapter, not an oversight:

| missing | why it would help | where |
|---|---|---|
| **Dropout** | the 100%/91% gap is exactly what dropout is for | Ch 20 |
| **LayerNorm** | steadier gradients through deeper stacks | Ch 20 |
| **Convolution** | encodes that neighbouring pixels are related — the fact flattening destroyed | not in this course |
| **More data** | the actual binding constraint here | — |

The course goes to attention rather than convolution, because attention is what a transformer needs. But MNIST is where you can *see* why a dense layer on raw pixels is the wrong tool, and that is worth having seen before Part 5 argues for a different one.

---

## Further reading

- `docs/part-3-neural-net-primitives/ch-15-training-loop.md` — the loop this reuses unchanged
- `docs/deep-dives/ch-09-one-rule-many-layers.md` — why one `backward()` suffices for four layers
- `docs/deep-dives/ch-05-why-subtract-the-max.md` — the numerical care inside the softmax used here
