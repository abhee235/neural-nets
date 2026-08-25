# Chapter 12: Loss Functions

> **Part 3 of 6 — Neural Net Primitives**
> Source: [`src/nn/losses.ts`](../../src/nn/losses.ts)
> Tests: [`src/nn/losses.test.ts`](../../src/nn/losses.test.ts)
> Exercise: [`exercises/ch-12-losses.ts`](../../exercises/ch-12-losses.ts)

---

## The problem this chapter solves

A model predicts tomorrow's temperature.

```
  model says:   32°C
  actual:       35°C
```

How wrong was it? You can see the answer — off by 3 — and that is the whole subject of this chapter: **turning "the model said X, the truth was Y" into a number that says how wrong the model is.**

That number is called the **loss**. Lower is better; zero is perfect.

Why does it have to be a *number*, rather than just knowing "we were off by a bit"? Because of what happens next. Training, which you built the pieces of in Chapters 08–10, is this pipeline:

```
  input
    ↓
  network  (weights live here)
    ↓
  prediction
    ↓
  compare with the truth  ──►  loss  (one number)
    ↓
  backward()
    ↓
  ∂loss/∂prediction  →  ∂loss/∂weights     ← gradients flow back THROUGH the network
    ↓
  update the weights                        ← Ch 09's step()
```

Notice the loss never touches the prediction directly. It produces **gradients**, and the gradients flow backward through the network until they reach the weights — the same chain you traced in Ch 10. Everything the model ever learns arrives through that one number at the top.

Which means a loss has **two jobs**, and the second is the one beginners miss:

| | |
|---|---|
| **Job 1 — measure** | say how bad the prediction is |
| **Job 2 — teach** | produce gradients that say how to *improve* |

A loss can do job 1 well and job 2 terribly. Two of this chapter's sections are about exactly that failure, and the sentence to carry out of the chapter is: **a loss is judged by its gradients, not by whether its value looks sensible.**

(There is also a mechanical reason the loss must be one number, specific to our engine: the `backward()` you wrote in Ch 10 refuses to start from anything but a scalar — you wrote that guard yourself. But that guard *enforces* the requirement; it is not the reason for it. The reason is that "which direction makes the model better?" only has an answer if "better" is a single number that can go down.)

Models predict two kinds of things, and each gets its own loss:

```
  REGRESSION — predicting a quantity

    input ──► model ──► 32.0
                          ↑ compare
                        35.0

  CLASSIFICATION — choosing between options

    input ──► model ──► [0.09  0.24  0.67]
                          ↑ compare
                        "the first one"
```

The first comparison is easy — two numbers, subtract them. The second is genuinely puzzling: how do you compare a *list of probabilities* against *one right answer*? Most of the chapter is about answering that well.

We build three functions:

| | what it measures | used for |
|---|---|---|
| `mseLoss` | how far apart are two lists of numbers? | regression |
| `logSumExp` | *(a helper)* | keeping the next one from overflowing |
| `crossEntropyFromLogits` | how much confidence went to the right answer? | classification, and every language model in this course |

> **🗺️ How to read this chapter**
> Same rhythm as Ch 09, 10 and 11 — read a bit, build a bit.
>
> | | Sections | Then |
> |---|---|---|
> | **Read** | 1 | **Build** `mseLoss` (section 2) |
> | **Read** | 3 → 8 | **Build** `logSumExp` (section 9) |
> | **Read** | 10 | **Build** `crossEntropyFromLogits` (section 11) |
> | **Read** | 12 → 13 | Verify everything |
>
> There is very little new engine work here. Ch 11 had you add primitives with hand-written backward passes; these losses are assembled from operations you already have, so autograd produces the gradients for free. The hard part of this chapter is choosing *what* to minimise, not differentiating it.

---

## Learning Goals

By the end of this chapter you can:

- State the two jobs of a loss function, and explain why the second one is where losses actually fail.
- Derive MSE from "how far apart are these numbers", and say what squaring is deliberately chosen to do.
- Explain why counting mistakes — the thing we actually care about — cannot be trained on.
- Write down the four properties a classification loss needs, and show that `−log` has exactly those properties.
- Say why the same loss is called *cross-entropy*.
- Show with measurements why MSE fails for classification, and why the failure is in the gradient rather than the value.
- Explain what breaks numerically in `log(softmax(z))`, and derive the `logSumExp` fix.
- Derive the gradient `p − y`, and explain why it always sums to zero.

---

## Words we'll use in this chapter

Two groups. The first you need to *think* with; the second only appears when we build.

**Ideas:**

| Word | Plain meaning |
|------|---------------|
| **loss** | One number saying how wrong the model is. Lower is better; 0 is perfect. |
| **regression** | Predicting a quantity — a temperature, a price, a length. |
| **classification** | Choosing one option from a fixed list — which word, which digit. |
| **class** | One of the options. |
| **logits** | The raw scores out of the network, before `softmax` (named properly in section 3). |
| **cross-entropy** | The classification loss built in sections 5–6. The name is explained there. |

**Implementation:**

| Word | Plain meaning |
|------|---------------|
| **one-hot** | A label written as a row of zeros with a single 1 at the true class. |
| **batch** | Several examples processed at once; losses are averaged across it. |
| **saturated** | An output pinned at its extreme, where the gradient has gone to ~0 (Ch 11 section 10). |
| **objective / criterion** | Other libraries' names for a loss. |

---

## 1. Regression: how far off are the numbers?

Three days of temperature predictions:

```
  predicted   [ 32   28   31 ]
  actual      [ 35   28   30 ]
```

How wrong is that, as one number? Subtract first — the obvious move:

```
  predicted     [ 32   28   31 ]
  actual        [ 35   28   30 ]
                ────────────────  −
  difference    [ -3    0    1 ]
```

Now three numbers must become one. Adding them gives `-3 + 0 + 1 = -2` — but wait, what would `-2` even mean? Worse: if the errors had been `-3` and `+3`, they would cancel to zero and declare the model perfect. Being too low on Monday and too high on Tuesday is not the same as being right, but signed errors hide it.

So make every difference positive before combining. Two candidates:

| | on our numbers | |
|---|---|---|
| absolute value | `3 + 0 + 1 = 4` | works, but has a corner at zero — the same kink that makes `relu` non-differentiable there |
| **squaring** | `9 + 0 + 1 = 10` | smooth everywhere, and treats errors *unequally* |

Look at what squaring did to the `-3`: it became `9`, while the `1` stayed `1`. That is not an accident to tolerate — it is the design. **Squaring deliberately makes large errors count disproportionately: an error of 10 contributes 100, not 10.** Whether you want that depends on the problem; usually you do, because one badly wrong prediction tends to matter more than several slightly-off ones. And separately, squaring gives a smooth derivative everywhere, which the corner in `|x|` does not.

```
  difference    [ -3    0    1 ]
  squared       [  9    0    1 ]
                        │
                  average → (9 + 0 + 1) / 3 = 3.333333
```

That is **mean squared error**:

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(p_i - y_i)^2$$

We average rather than total so the number does not grow just because there are more predictions — a learning rate tuned on 32 examples would otherwise be wrong for 64.

**Its gradient.** Differentiating `(p − y)²` with respect to `p` gives `2(p − y)`, and the mean contributes the `1/n`:

$$\frac{\partial \text{MSE}}{\partial p_i} = \frac{2}{n}(p_i - y_i)$$

```
  difference          [ -3     0    1        ]
  gradient (2/3)·d    [ -2     0    0.666667 ]
```

Monday's prediction was 3° too low and its gradient is negative — gradient descent subtracts it, pushing the prediction **up**. Wednesday was too high and gets pushed down. Tuesday was exactly right and is left alone. The gradient points at the target.

You will not write that gradient by hand. `mseLoss` is assembled from `add`, `mul` and `mean` — operations whose backward passes you wrote in Ch 10 — so autograd derives it.

---

## 2. Build it — `mseLoss`

**Milestone 1 — `mseLoss`.**

```typescript
export function mseLoss(predictions: TensorValue, targets: Tensor): TensorValue
```

Keep two things separate in your head here. The **machine-learning idea** is just section 1: subtract, square, average — nothing about it requires any particular library. The **engine detail** is the asymmetric signature: in *our* implementation, `predictions` is a `TensorValue` so the graph records the operations and gradients can flow back, while `targets` is a plain `Tensor` because the truth is a constant — nothing produced it and nothing in it gets updated. MSE does not require a `TensorValue`; our autograd engine does.

Three steps, all with operations you already have:

1. Subtract the targets. `TensorValue` has no `sub`, but it has `add` and `mul`, and subtracting is adding a negation.
2. Square the difference. There is no `pow` either, but squaring is multiplying something by itself, and `mul` records both parents correctly.
3. Collapse to one number with `.mean()`.

✅ *Checkpoint:* `mseLoss([32,28,31], [35,28,30])` returns `3.333333`, and after `.backward()` the prediction gradient is `[-2, 0, 0.666667]`.

> **Pitfall — putting the target in the graph.** Wrap `targets` in a `TensorValue` and `backward()` will compute a gradient *for the labels*. It is meaningless, wastes memory, and can corrupt real gradients if those tensors are reused.

### Follow one gradient all the way

You have now built a loss, and the checkpoint proves it produces gradients. But gradients *for what*? In the checkpoint the predictions were typed in by hand, so the answer was invisible. In a real network, predictions come **out of a weight** — and the entire point of the loss is what happens to that weight. Walk the whole loop once, with numbers small enough to do in your head.

The smallest possible model: one input, one weight, prediction = input × weight.

```
  x = 2      w = 3      target = 10
```

**Forward.** Two hops, both operations you have:

```
  x = 2          w = 3
     \           /
      \         /
        mul                    p = 2 × 3 = 6        too small — truth is 10
         │
       mseLoss                 L = (6 − 10)² = 16
```

**Backward.** The loss depends on `w` only *through* `p` — the graph is `w → p → L` — so the chain rule (Ch 07) multiplies two local rates:

```
  hop 1, loss → prediction:    dL/dp = 2(p − y) = 2(6 − 10) = -8
  hop 2, prediction → weight:  dp/dw = x = 2                       (the mul rule)

  dL/dw = (-8) × 2 = -16
```

After `loss.backward()`, that `-16` is sitting in `w.grad`. (The engine also computed `x.grad = -24` — the `mul` rule's other output. Nothing will ever read it: `x` is data, not a parameter. Gradients are computed for everything and *used* only where there is something to change.)

**Update.** Now a step that is **not** part of backpropagation — it belongs to Ch 09's `step()`:

```
  w ← w − lr · dL/dw  =  3 − 0.1 × (-16)  =  3 + 1.6  =  4.6
```

Read the signs, because they tell the story. The prediction was too *small*, so `dL/dp` came out negative, so `dL/dw` came out negative, so *subtracting* it made `w` go **up** — and a bigger `w` makes `p = xw` bigger, toward the target. The machinery never knew "too small"; the signs carried that information on their own.

**And the loop closes.** Run the forward pass again with the new weight:

```
  p = 2 × 4.6 = 9.2        L = (9.2 − 10)² = 0.64        was 16
```

One step, and the loss fell from 16 to 0.64. That is learning — all of it. Every training run in this course, up to and including Ch 30's GPT, is this loop repeated.

Keep the two stages separate in your head, because they are separate in the code:

```
  LOSS                "how wrong?"                        this chapter
    ↓
  BACKPROPAGATION     "which weights caused it,           Ch 08–10, already built
    ↓                  and in which direction?"
  GRADIENTS
    ↓
  OPTIMIZER           "change those weights"              Ch 09's step(), Ch 14's Adam
```

**Backpropagation calculates how much each weight contributed to the loss; the optimizer uses those gradients to change the weights.** The loss starts the first stage and never sees the second.

With a real network nothing changes except the count. A thousand weights in, `loss.backward()` fills a thousand `.grad` fields in one sweep — each one answering "if this particular weight moved a little, how would the loss move?" — and the optimizer updates every one of them. Same two hops you just did by hand, repeated down every path of the graph. Ch 13 builds the layer that owns those weights; Ch 15 writes the loop.

All six numbers above — `6, 16, -8, -16, 4.6, 0.64` — are reproduced by your own `mseLoss` and your own engine in the exercise file. Run it and check.

---

## 3. Classification: choosing between options

Now the harder kind of prediction — and the one this course is really heading for. Chapter 30's language model reads a few words and predicts the next one. Here is that task with a vocabulary of three.

The context is `"the cat ___"`, and there are exactly three words it could be:

```
    sat        ran        flew
```

The network reads the context and produces one number per word:

```
    sat        ran        flew
     1          2          3
```

Bigger means "more likely", but these are not probabilities — they do not sit between 0 and 1 and they do not add up to anything in particular. They are raw scores, and **raw scores like these are called `logits`**. You will see that word from here to the end of the course.

Chapter 11's `softmax` turns them into probabilities:

```
logits     [ 1          2          3        ]
softmax  → [ 0.090031   0.244728   0.665241 ]      adds up to 1
             sat        ran        flew
```

Now the network is making a readable claim: *"probably `flew` (67%), maybe `ran` (24%), probably not `sat` (9%)."*

Suppose the sentence was actually **"the cat sat"**. The network is wrong — you can see that. Your code cannot: nothing in Chapters 1–11 has ever compared an output against an answer. We need a loss for this kind of prediction, and MSE was built for a different shape of problem — there, the model output *was* the prediction; here the model outputs a whole distribution and the truth is a single word.

Before designing a good loss for this, it is worth watching the obvious one fail.

---

## 4. First attempt: count the mistakes

Count how many predictions are wrong. Ten predictions, seven right, loss is 3. Simple, and it measures exactly the thing we care about.

It cannot be trained on. Here is why.

Our network currently ranks `sat` last. Take `sat`'s score and slide it upward from `0.5`, leaving `ran` and `flew` at `2` and `3`. At each step, record two things: how many mistakes the model makes, and what probability `softmax` gives `sat`. Both are things you already have — nothing new is needed.

```
  score of "sat"     mistakes     p("sat")
      0.50              1         0.056612
      1.00              1         0.090031
      2.00              1         0.211942
      2.90              1         0.398130
      2.99              1         0.419881
     ────────────────────────────────────────  the model's top choice flips here
      3.01              0         0.424760
      3.50              0         0.546549
      5.00              0         0.843795
```

Read the middle column downward. From `0.50` to `2.99` the model goes from giving `sat` a 5.7% chance to a 42% chance — an enormous improvement — and the mistake count does not move at all. Then it drops by one, in a single step. Then it never moves again.

A flat quantity has a slope of zero. Gradient descent multiplies that zero by the learning rate and updates nothing, so the model never learns. Not slowly: not at all.

The difference between the two columns is the difference between two questions:

```
  Accuracy asks:   "did I get it right?"           yes/no — flat, then a jump
  A loss asks:     "how far am I from right?"      moves with every improvement
```

Job 1 without job 2. The mistake count *measures* fine and *teaches* nothing.

<p align="center">
  <img src="../assets/ch-12/why-accuracy-fails.svg" alt="Two stacked panels sharing one x-axis, which is the score given to the true word sat, sweeping from 0 to 6 while ran and flew stay at 2 and 3. The top panel, in red, plots the number of mistakes: a staircase sitting flat at 1 across the whole left half, dropping vertically to 0 at a score of 3, then flat at 0 across the right half, with both flat stretches labelled slope 0 and the drop labelled jump. A caption notes the model improves the whole way while this number never moves. The bottom panel, in green, plots the probability softmax assigns to sat over the same range: a smooth curve rising continuously from 0.0351 through 0.0900, 0.2119, 0.4223 and 0.6652 to 0.9362, never flat anywhere. Two animated markers sweep the panels in step, the red one jumping once while the green one moves continuously. On the right, two boxes compare what gradient descent sees: for the mistake count the slope is 0 everywhere it is defined and undefined at the jump, so step() updates nothing; for the probability the slope is never zero because it responds to every change made to the score, so there is always something to follow. A note says a usable loss has to be built out of the probability rather than the count, and the footer reads that the count is flat until it is too late while the thing that moves smoothly is what a loss must be built from." />
</p>

*Figure 1: the same sweep, measured two ways. One is flat and then jumps; the other moves the whole time.*

So the loss we train on will **not** be the thing we finally care about. It is a stand-in with a slope everywhere — and that is why real projects always report two numbers: a **loss** they train on and an **accuracy** they care about. They are not the same number and are not supposed to be.

If the flat-then-jump shape feels familiar, it is [Chapter 11's step function](ch-11-activation-functions.md#2-the-obvious-answer-and-why-it-fails): there an activation with no usable slope, here a loss with no usable slope. Same shape, same reason.

The right column of the sweep is the raw material for something better. The probability of the true word moves smoothly on every row — a usable loss for classification should be built out of *that*.

---

## 5. Scoring a distribution: look up one number

How do you score a whole distribution against one right answer? Here is the idea, and it is smaller than you might expect:

**Look up the probability the model gave to the word that actually happened. Score it on that number alone.**

```
softmax   [ 0.090031   0.244728   0.665241 ]
             sat        ran        flew
              ▲
              │
        the truth was "sat" — the model gave it 0.090031
```

Everything else in the row is ignored. If the model put 9% on `sat`, it is scored on `0.090031`, however the other 91% was split.

That feels like throwing information away, and it is worth seeing why it is not. The probabilities must add to 1, so any confidence given to `ran` or `flew` is confidence *taken away* from `sat`. Pushing the right word's probability up automatically pushes the wrong ones down — the loss never needs to mention them.

---

## 6. Turning that probability into a loss

The probability of the true word is high-is-good. A loss must be low-is-good, and it should meet four requirements:

```
  1.  high p  →  low loss          rewarding confidence in the truth
  2.  p = 1   →  loss = 0          a perfect answer costs nothing
  3.  low p   →  high loss         punishing confidence in the wrong thing
  4.  p → 0   →  loss → ∞          no ceiling: total confidence in the wrong
                                   thing can be punished arbitrarily hard
```

Now look at what the function `−log(p)` does:

```
        p        −log(p)
      1.0         0.0000       requirement 2 ✓
      0.9         0.1054
      0.7         0.3567
      0.5         0.6931
      0.3         1.2040
      0.1         2.3026
      0.01        4.6052
      0.001       6.9078
      → 0         → ∞          requirement 4 ✓
```

It falls as `p` rises (1 ✓, 3 ✓), is exactly zero at `p = 1`, and grows without limit as `p` approaches zero. `−log` was not chosen because logarithms are sacred — it is simply a function with exactly the four behaviours we listed. If you invented a different function with those behaviours, you could train networks with it too; `−log` is the one with the cleanest algebra, as sections 10 and 12 will show.

So the loss for our network is:

$$\mathcal{L} = -\log(p_{\text{true word}}) = -\log(0.090031) = 2.407606$$

> **✋ Stop and think.** The model predicts `[0.2, 0.3, 0.5]` and the truth is the *second* class. It then changes its prediction to `[0.2, 0.7, 0.1]`. Did the loss go up or down?
>
> Down, and by a lot: `−log(0.3) = 1.204` before, `−log(0.7) = 0.357` after. Only the middle number entered the calculation either time.

**Where the name comes from.** There is a pleasing interpretation of `−log(p)` — a *useful reading*, not the reason we chose it. It behaves like **surprise**: an event you were confident about is unsurprising when it happens (`p = 0.9` → `0.105`), and an event you had all but ruled out is shocking (`p = 0.01` → `4.6`). This loss is the surprise the truth causes your model, and a well-trained model is rarely surprised. The formal name for average surprise measured against another distribution is **cross-entropy**, and that is what this loss is called.

**A form you will see elsewhere.** Written with a one-hot label — a row of zeros with a single 1 at the true word — the same loss is often written as a sum over all words:

$$\mathcal{L} = -\sum_{c} y_c \log(p_c)$$

```
  y (one-hot)  [ 1          0          0        ]
  log(p)       [ -2.407606  -1.407606  -0.407606]
                     │          │          │
                  ×1 │       ×0 │       ×0 │
                     ▼          ▼          ▼
               -2.407606      0          0        →  sum, negated = 2.407606
```

Every term except the true word is multiplied by zero. Same "look up one number", in notation that will be convenient for the algebra later.

**One property to notice now**, because section 12's gradient depends on it. Score our row against each possible truth:

```
  truth = sat     p = 0.090031     loss = 2.407606
  truth = ran     p = 0.244728     loss = 1.407606
  truth = flew    p = 0.665241     loss = 0.407606
```

The losses differ by exactly `1.000000` — and so do the logits (`1, 2, 3`). Section 10 shows why that is not a coincidence.

---

## 7. Why not MSE here?

Both losses now exist, so the comparison is fair. The probabilities are numbers and the one-hot label is numbers, so MSE would *run*. Does it work?

It fails precisely when you need it most. Take a model that is **confidently and completely wrong**: the truth is `sat`, and it has put essentially all its probability on `flew`.

```
  probabilities   [ 4.54e-5   4.54e-5   1.00 ]
  one-hot         [ 1         0         0    ]
                    sat       ran       flew

  cross-entropy   =  10.0001      no ceiling
  MSE             =   0.6666      and that is the worst it can ever say
```

MSE cannot express how bad this is: probabilities are trapped between 0 and 1, so a squared difference cannot get large. With three words, `0.667` is the largest value MSE can ever return, no matter how wrong the model becomes.

But remember the two jobs. The value is job 1. Here is job 2 — what actually reaches the logits:

```
                     sat          ran         flew
  cross-entropy   -0.999955     0.000045    0.999909
  MSE             -0.000061    -0.000030    0.000091
```

MSE's gradient has essentially vanished — about 16,000 times smaller than cross-entropy's. The model is as wrong as it can possibly be, and MSE is barely asking for a change.

The reason is Ch 11 section 10 arriving from a new direction. MSE sits *after* `softmax`, and this `softmax` is **saturated** — pinned at `1.0`, out on the flat part of its curve where its derivative is nearly zero. MSE's gradient must travel back through that flat region, and almost nothing survives the trip. Cross-entropy avoids the trap by using `log`, which *undoes* the exponential inside `softmax` — section 12 shows the cancellation explicitly.

<p align="center">
  <img src="../assets/ch-12/mse-vs-crossentropy.svg" alt="Two panels, both sweeping left to right as the model becomes more confidently wrong, using logits [0, 0, w] with the true class first. The left panel plots the loss value: cross-entropy in green climbs steadily and without limit from about 1.1 to 10, while MSE in red rises briefly then flattens against a dashed ceiling line at 0.667 and stays there, captioned that probabilities are boxed into zero to one so a squared error cannot get large. The right panel plots the size of the gradient actually reaching the logits: cross-entropy in green rises toward 1.0 and holds at full strength, while MSE in red decays toward 0.00006 and effectively disappears, captioned that MSE sits behind a saturated softmax and is throttled by its flat region. Animated markers sweep both panels together. The footer states that MSE ranks this model correctly and does say the model is bad, but cannot produce enough gradient to fix it, and that the failure gets worse the more wrong the model is." />
</p>

*Figure 2: as the model gets more confidently wrong, cross-entropy's gradient grows toward full strength while MSE's decays to nothing.*

This is the chapter's central lesson, so once more, plainly: **a loss is not judged by whether its value looks sensible. It is judged by what its gradient does in the situations you need to escape from.** MSE ranks this model correctly — it does report that the model is bad — and still cannot fix it. It does job 1 and fails job 2.

---

## 8. Where the numbers break

The loss is `−log(softmax(z))`, and on paper it is finished. But computers do not compute with real numbers — they compute with float64, which has a largest and a smallest value. A formula that is harmless on paper can produce `Infinity`, `NaN`, and destroyed gradients when the inputs get big. This one does, in two separate ways.

**`log(0)`.**

```
  Math.log(0)  =  -Infinity
```

Softmax outputs are never mathematically zero, but in float64 they reach zero easily — Ch 11 section 10 measured exactly that. A confidently wrong model produces a true-word probability that underflows to `0`, `log` returns `-Infinity`, the loss is `Infinity`, and every gradient downstream becomes `NaN`. One step destroys the model.

**`exp` overflows before softmax even finishes.** Logits are unbounded, and trained networks produce large ones:

```
  logits  [1000, 1001, 1002]

  exp  →  [Infinity, Infinity, Infinity]
  sum  →   Infinity
  probs →  [NaN, NaN, NaN]                 Infinity / Infinity
```

Neither failure is a bug in your code. The mathematics is fine; the floating-point arithmetic is not.

The way out is a fact you already proved. Softmax is **shift-invariant**: subtracting the same constant from every logit leaves the output unchanged, because the constant factors out of numerator and denominator and cancels. That is why Ch 11's `softmax` subtracts the max, and the proof is in [the Ch 05 deep dive](../deep-dives/ch-05-why-subtract-the-max.md). The plan: never let a large number reach `exp`, and never let a small number reach `log`.

---

## 9. `logSumExp` — the same trick, one level up

We want `log(Σ eᶻ)` without ever computing `eᶻ` for a large `z`. You are about to implement this function, so the derivation below is your specification — it is short.

Pull the largest logit out first. Let `m = max(z)`. Since `e^z = e^m · e^(z−m)`:

$$\log\sum_j e^{z_j} \;=\; \log\left(e^{m}\sum_j e^{z_j - m}\right) \;=\; m + \log\sum_j e^{z_j - m}$$

The first step factors `e^m` out of every term; the second uses `log(ab) = log a + log b`. Nothing is approximated — this is exact.

Now look at what actually gets exponentiated. Every `z_j − m` is **at most zero**, because `m` is the largest of them. So every `e^(z_j − m)` lands in `(0, 1]`:

- **Nothing overflows** — the biggest value `exp` ever sees is `e⁰ = 1`.
- **Nothing hits `log(0)`** — the sum includes the largest term itself, contributing exactly `1`, so the total is always at least 1 and its log is always at least 0.

One subtraction closes both failures. On the input that broke:

```
  logSumExp([1000, 1001, 1002])  =  1002.407606
  logSumExp([1,    2,    3   ])  =     3.407606
                                    ──────────────
                        difference  =   999.000000
```

Exactly the shift, and no `NaN` anywhere. The digits after the decimal point are identical in both — `.407606` — which is shift-invariance again, and a good self-check when you implement this.

---

## 10. Build it — `logSumExp`

**Milestone 2 — `logSumExp`.**

```typescript
export function logSumExp(x: TensorValue, axis?: number): TensorValue
```

Four steps, straight from the identity:

1. Find the max along the axis, with `keepDims: true` so it broadcasts back against the full tensor.
2. Subtract it. Everything is now `≤ 0`.
3. `exp`, then `sum` along the axis, then `log`.
4. Add the max back on.

Default the axis to the last one — `axis ?? x.data.ndim - 1` — matching Ch 11's `softmax`.

One decision: **does the max need a gradient?** It is a value read out of the data, and it cancels exactly in the mathematics, so treating it as a constant is both correct and simpler.

✅ *Checkpoint:* `logSumExp([1,2,3]) = 3.407606`, `logSumExp([1000,1001,1002]) = 1002.407606`, and neither returns `NaN` or `Infinity`.

---

## 11. Cross-entropy straight from the scores

Now put it together: `−log(softmax(z)_y)` without ever building `softmax(z)`.

Start from softmax and take the log of it:

$$\log\left(\text{softmax}(z)_y\right) = \log\frac{e^{z_y}}{\sum_j e^{z_j}} = \log e^{z_y} - \log\sum_j e^{z_j} = z_y - \text{logsumexp}(z)$$

The middle step is `log(a/b) = log a − log b`; the last uses `log(eˣ) = x`. The exponential and the logarithm cancel. Negating gives the loss:

$$\mathcal{L} = \text{logsumexp}(z) - z_y$$

That is the whole computation. Notice what disappeared: no division, no `exp` of a raw logit, no `log` of a small probability. A stable reduction and a subtraction.

Check it against the long way, truth = `flew`:

```
  logSumExp([1,2,3]) − z_flew   =  3.407606 − 3      =  0.407606
  −log(softmax([1,2,3])[flew])  =  −log(0.665241)    =  0.407606     identical
```

And this explains section 6's "the losses differ by exactly 1": `logsumexp(z)` does not depend on which word is true, so changing the true word changes only the `z_y` term. With logits `1, 2, 3` the losses come out `2.407606, 1.407606, 0.407606` — differing by exactly the logit gaps.

**On a batch**, compute one loss per row and average, for the same reason MSE averages.

---

## 12. Build it — `crossEntropyFromLogits`

**Milestone 3 — `crossEntropyFromLogits`.**

```typescript
export function crossEntropyFromLogits(logits: TensorValue, targets: Tensor): TensorValue
```

Straight from the formula:

1. `logSumExp(logits)` along the class axis.
2. Pick out `z_y`, the logit of the true word in each row.
3. Subtract, then average over the batch.

Step 2 is the fiddly one, and it depends on how `targets` are written. If they are **class indices** (`0` for `sat`) you gather one entry per row. If they are **one-hot** (`[1,0,0]`) you multiply and sum along the axis, since every other term is zeroed — the picture in section 6. Both work. Pick one, write it in the JSDoc, and stay consistent; mixing the two conventions is the most common bug in this file.

✅ *Checkpoint:* logits `[1,2,3]` with truth `flew` gives `0.407606`; with truth `sat` gives `2.407606`; and logits `[1000,1001,1002]` with truth `flew` also gives `0.407606`, because of shift invariance.

> **Pitfall — feeding it probabilities.** The name says *FromLogits* and it means it. Hand it the output of `softmax` and it will apply `logSumExp` to already-normalised numbers and return a confidently wrong answer with no error. Raw scores only.

---

## 13. The gradient is `p − y`

First the answer, because it is readable on its own. The gradient of this loss with respect to the logits is:

```
  softmax     [  0.090031   0.244728   0.665241 ]      p — what you predicted
  one-hot     [  1          0          0        ]      y — what was true
                sat         ran        flew
              ─────────────────────────────────── −
  gradient    [ -0.909969   0.244728   0.665241 ]      p − y
```

**What you predicted, minus what was true.** Read it class by class:

```
  sat    truth, predicted far too small   →  negative gradient  →  score goes UP
  ran    wrong, predicted 0.24 too much   →  positive gradient  →  score goes down
  flew   wrong, predicted 0.67 too much   →  positive gradient  →  score goes down hardest
```

Every word gets a specific, correctly-signed instruction, sized by exactly how over- or under-predicted it was — on every step, no matter how wrong the model is. Compare section 4's mistake count, whose slope was zero everywhere.

Now the derivation, which is two short steps on `L = logsumexp(z) − z_y`.

**The first term.** The derivative of `logsumexp` turns out to be softmax itself:

$$\frac{\partial}{\partial z_i}\log\sum_j e^{z_j} = \frac{e^{z_i}}{\sum_j e^{z_j}} = p_i$$

That is the chain rule on `log`: the derivative of `log(S)` is `1/S`, and the derivative of `S = Σ eᶻ` with respect to `z_i` is `e^{z_i}`. Multiply them and you have `e^{z_i}/S` — the definition of softmax.

**The second term.** `−z_y` depends on `z_i` only when `i` is the true word, giving `−1` there and `0` everywhere else. That is exactly the one-hot row `y`.

$$\frac{\partial \mathcal{L}}{\partial z_i} = p_i - y_i$$

**It always sums to zero.** `p` sums to 1 and `y` sums to 1, so `p − y` sums to 0. That is a real statement, not an accident: cross-entropy never pushes every score in the same direction — it can only *redistribute*, taking confidence from the wrong words and handing it to the right one. Which is correct, since adding a constant to every logit changes nothing (section 8).

> **Looking ahead to Ch 22.** Real frameworks fuse softmax and cross-entropy into one operation exactly so they can return `p − y` directly, instead of running two backward passes and hoping `log` and `exp` cancel cleanly in floating point. That is what you are building. It is also why your Ch 11 `softmax` will mostly get used *without* a loss attached — inside attention, where there is no label to compare against.

---

## 14. Verify

```bash
bun test src/nn/losses.test.ts
bun run exercises/ch-12-losses.ts
```

Every loss here is assembled from operations that already have backward passes, so a numerical gradient check should pass on all of them. It is the check that catches a dropped `2/n` or a flipped sign.

---

## What to Implement

| Function | Notes |
|----------|-------|
| `mseLoss(predictions, targets)` | Mean of squared differences. `targets` stays outside the graph. |
| `logSumExp(x, axis?)` | Subtract the max, `exp`, `sum`, `log`, add the max back. Default to the last axis. |
| `crossEntropyFromLogits(logits, targets)` | `logSumExp(z) − z_y`, averaged over the batch. Raw logits only. |

---

## Common Pitfalls

- **Computing `softmax` and then `log` separately.** Both failures from section 8 are waiting. Use `logSumExp`.
- **Passing probabilities to `crossEntropyFromLogits`.** No error, wrong answer.
- **Summing instead of averaging over the batch.** Your learning rate silently becomes batch-size dependent.
- **Wrapping `targets` in a `TensorValue`.** The truth is a constant; a gradient for it is meaningless.
- **Mixing class indices and one-hot labels.** Pick one, document it, keep it.
- **Using MSE for classification.** Section 7 measured the cost: the gradient disappears exactly when the model is most wrong.
- **Forgetting `keepDims: true` on the max inside `logSumExp`.** The subtraction will not broadcast back.

---

## Self-Check Questions

Three levels — stop at the level that matches how deep you want to go today, but level 1 is not optional.

**Level 1 — the ideas:**

1. What are the two jobs of a loss function? Which one does the mistake count fail?
2. What is a logit, and why can you not read one as a probability?
3. The model predicts `[0.49, 0.51]` and separately `[0.01, 0.99]`. If class 2 is correct, which prediction has lower loss? If class **1** is correct, which one is punished harder, and why does that asymmetry make sense?
4. Why do real projects report both a loss and an accuracy?

**Level 2 — calculate:**

5. MSE of predictions `[32, 28, 31]` against actuals `[35, 28, 30]` — the loss, then the gradient.
6. `−log(0.1)` and `−log(0.9)` — compute both, and say which requirement from section 6 each illustrates.
7. For logits `[1,2,3]` with truth `sat`, write out `p − y`. Which score goes up, which go down?

**Level 3 — derive:**

8. Show that `log(softmax(z)_y) = z_y − logsumexp(z)`, naming the logarithm rule used at each step.
9. `logSumExp([1,2,3]) = 3.407606` and `logSumExp([1000,1001,1002]) = 1002.407606`. Without computing anything, explain why the fractional parts match.
10. Prove that `p − y` sums to zero, and explain what that means the loss can and cannot do to the logits.
11. Why does the max inside `logSumExp` not need a gradient?

---

## Further Reading

- [Chris Olah — Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/) — the best intuition piece on entropy and surprise; read it if section 6's surprise reading appealed to you.
- [Wikipedia — Cross entropy](https://en.wikipedia.org/wiki/Cross_entropy) — the information-theoretic derivation the name comes from.
- [Stanford CS231n — Losses](https://cs231n.github.io/neural-networks-2/#losses) — a survey of standard losses and when each applies.
- [Goodfellow, Bengio, Courville — *Deep Learning*](https://www.deeplearningbook.org/), chapter 6.2 — why maximum likelihood leads to cross-entropy.

---

## Checkpoint

Before moving on you should be able to:

- [ ] State the two jobs of a loss, and give one loss that does job 1 but fails job 2.
- [ ] Compute MSE and its gradient by hand on a three-element example.
- [ ] Explain why accuracy is not trainable, without reaching for a formula.
- [ ] List the four requirements on a classification loss and show `−log` meets them.
- [ ] Say what breaks in `log(softmax(z))` — both failures — and how the max subtraction closes each.
- [ ] Derive `L = logsumexp(z) − z_y` from `−log(softmax(z)_y)`.
- [ ] Read `p − y` off a concrete row and explain each sign, then derive it.
- [ ] Have all three functions passing their tests, including numerical gradient checks.

---

## Next Chapter

**[Linear Layer](ch-13-linear-layer.md)** — you now have a network that can transform inputs (Ch 11) and a number that says how wrong it is (Ch 12). Chapter 13 packages `y = xW + b` into a reusable layer with managed parameters, so the training loop in Ch 15 has something to optimise.
