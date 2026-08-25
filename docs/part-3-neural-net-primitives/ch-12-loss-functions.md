# Chapter 12: Loss Functions

> **Part 3 of 6 — Neural Net Primitives**
> Source: [`src/nn/losses.ts`](../../src/nn/losses.ts)
> Tests: [`src/nn/losses.test.ts`](../../src/nn/losses.test.ts)
> Exercise: [`exercises/ch-12-losses.ts`](../../exercises/ch-12-losses.ts)

---

## Where we left off

This course is building toward a language model: something that reads a few words and predicts the next one. Chapter 30 does it with a real vocabulary. Let us do it here with a vocabulary of three.

The context is `"the cat ___"`, and there are exactly three words it could be:

```
    sat        ran        flew
```

A network reads the context and produces one number per word. Suppose it produces these:

```
    sat        ran        flew
     1          2          3
```

Those three raw numbers are the network's opinion. Bigger means "more likely", but they are not probabilities — they do not sit between 0 and 1, and they do not add up to anything in particular. They are just scores.

**Raw scores like these have a name: `logits`.** You will see the word constantly from here to the end of the course, and it means nothing more than "the numbers that come out of the network, before we tidy them up".

Tidying them up is what Chapter 11's `softmax` was for. Feeding these three scores through it:

```
logits     [ 1          2          3        ]
softmax  → [ 0.090031   0.244728   0.665241 ]      adds up to 1
             sat        ran        flew
```

Now they are probabilities, and the network is making a readable claim: *"I think it's `flew`, 67%. Probably not `sat`, only 9%."*

Suppose the sentence was actually **"the cat sat"**.

The network is wrong. It put most of its confidence on `flew` and almost none on the right answer. You can see that. **Your code cannot.** Nothing in Chapters 1 to 11 has ever compared an output against an answer — you built machinery that computes, machinery that differentiates, and layers that transform, and not one line of it knows what "wrong" means.

That is what this chapter adds, and there is a second reason it has to. Chapter 10's `backward()` opens with a guard you wrote yourself:

```
backward() needs a scalar root, got shape [...] — collapse with .sum() or .mean() first
```

The whole engine only starts from **one number**. Not a row of three. So something has to sit at the end of the network, look at what it said and what was true, and produce a single number meaning "this is how bad that was".

That something is a **loss function**.

```
    context ──► network ──► logits ──► LOSS ──► one number
                                                    │
                                     backward() ◄───┘
                                         │
                                         ▼
                                  every gradient
```

Everything after this chapter — the linear layer (Ch 13), the optimizers (Ch 14), the training loop (Ch 15) — is machinery for pushing that one number down.

We will build three functions, in order of difficulty:

| | what it measures | used for |
|---|---|---|
| `mseLoss` | how far apart are two lists of numbers? | regression — predicting quantities |
| `logSumExp` | *(a helper)* | stopping the next one from overflowing |
| `crossEntropyFromLogits` | how much confidence went to the right word? | classification, and every language model here |

> **🗺️ How to read this chapter**
> Same rhythm as Ch 09, 10 and 11 — read a bit, build a bit.
>
> | | Sections | Then |
> |---|---|---|
> | **Read** | 1 → 3 | **Build** `mseLoss` (section 4) |
> | **Read** | 5 → 9 | **Build** `logSumExp` (section 10) |
> | **Read** | 11 | **Build** `crossEntropyFromLogits` (section 12) |
> | **Read** | 13 → 14 | Verify everything |
>
> There is very little new engine work here. Ch 11 had you add primitives with hand-written backward passes; these losses are assembled out of operations you already have, so autograd produces the gradients for free. The hard part of this chapter is choosing *what* to minimise, not differentiating it.

---

## Learning Goals

By the end of this chapter you can:

- Say what a logit is, and why it is not a probability.
- Explain why training needs the output squeezed down to exactly one number, and connect that to the guard you wrote in Ch 10.
- Show why counting mistakes — the thing we actually care about — cannot be trained on.
- Derive MSE starting from "how far apart are these two lists", and say why the gap is squared rather than made positive some other way.
- Score a predicted distribution by looking up one number, and turn that into a loss with `−log`.
- Say why the same loss is called *cross-entropy*.
- Show with measurements why MSE is the wrong loss for word prediction, and why the failure is in the *gradient* rather than in the loss value.
- Explain what breaks numerically in `log(softmax(z))`, and why the fix is the max subtraction from Ch 05.
- Derive the gradient `p − y`, and explain why it always sums to zero.

---

## Words we'll use in this chapter

Each of these is introduced properly where it is first needed. This table is for looking back at.

| Word | Plain meaning |
|------|---------------|
| **logits** | The raw scores straight out of the network, before `softmax`. Any real number. |
| **loss** | One number saying how wrong the model is. Lower is better; 0 is perfect. |
| **objective / criterion** | Other libraries' names for the same thing. |
| **regression** | Predicting a quantity — a price, a temperature, a length. |
| **classification** | Choosing one option from a fixed list — which word, which digit. |
| **class** | One of the options. Here there are three: `sat`, `ran`, `flew`. |
| **one-hot** | A label written as a row of zeros with a single 1 at the true class. |
| **cross-entropy** | The loss built in sections 5 to 6. The name is explained there, not here. |
| **saturated** | An output pinned at its extreme, where the gradient has gone to ~0 (Ch 11 section 10). |

---

## 1. The job: everything down to one number

Gradient descent, the thing you built in Ch 09, knows exactly one trick: **take a single number and make it smaller.**

So the loss function has to answer one question with one number:

```
     the model said              the truth was
   [0.090  0.245  0.665]              sat
     sat    ran    flew                │
              │                        │
              └──────── loss ──────────┘
                         │
                         ▼
                       2.4076
```

Two requirements, and neither is negotiable:

1. **It must be a single number.** Not for tidiness — `backward()` refuses anything else, and you are the one who made it refuse. There is no single value to seed the backward pass with if the output is a row.
2. **Lower must mean better.** Gradient descent only walks downhill. A loss that went *up* as the model improved would train it to get worse, and `step()` would carry that out faithfully.

The rest of the chapter is about choosing *which* number.

---

## 2. First attempt: count the mistakes

The obvious idea, and it is a reasonable one: count how many predictions are wrong.

Ten predictions, seven right, loss is 3. All ten right, loss is 0. Simple, and it measures exactly the thing we actually care about.

It cannot be trained on. Here is why.

Our network currently ranks `sat` last. Take its score and slide it upward from `0.5`, leaving `ran` and `flew` at `2` and `3`. At each step, record two things: how many mistakes the model makes, and what probability `softmax` assigns to `sat`.

Both of those you already have. Nothing new is needed here — `softmax` is Chapter 11.

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

Read the middle column downward. From `0.50` to `2.99` the model goes from giving `sat` a 5.7% chance to giving it a 42% chance — an enormous improvement — and the mistake count does not move at all. Then it drops by one, in a single step. Then it never moves again.

A flat quantity has a slope of zero. Gradient descent multiplies that zero by the learning rate and updates nothing, so the model never learns. Not slowly: not at all.

Now read the right column. It moves on every single row.

<p align="center">
  <img src="../assets/ch-12/why-accuracy-fails.svg" alt="Two stacked panels sharing one x-axis, which is the score given to the true word sat, sweeping from 0 to 6 while ran and flew stay at 2 and 3. The top panel, in red, plots the number of mistakes: a staircase sitting flat at 1 across the whole left half, dropping vertically to 0 at a score of 3, then flat at 0 across the right half, with both flat stretches labelled slope 0 and the drop labelled jump. A caption notes the model improves the whole way while this number never moves. The bottom panel, in green, plots the probability softmax assigns to sat over the same range: a smooth curve rising continuously from 0.0351 through 0.0900, 0.2119, 0.4223 and 0.6652 to 0.9362, never flat anywhere. Two animated markers sweep the panels in step, the red one jumping once while the green one moves continuously. On the right, two boxes compare what gradient descent sees: for the mistake count the slope is 0 everywhere it is defined and undefined at the jump, so step() updates nothing; for the probability the slope is never zero because it responds to every change made to the score, so there is always something to follow. A note says a usable loss has to be built out of the probability rather than the count, and the footer reads that the count is flat until it is too late while the thing that moves smoothly is what a loss must be built from." />
</p>

*Figure 1: the same sweep, measured two ways. One is flat and then jumps; the other moves the whole time.*

So the mistake count is out, and we know what to replace it with: something built from that smooth probability. Two things follow, and they shape the rest of the chapter.

- The loss we train on will **not** be the thing we actually care about. It is a stand-in, chosen because it has a slope everywhere.
- This is why real projects report two numbers: a **loss** they train on, and an **accuracy** they care about. They are not the same number and they are not supposed to be.

If the flat-then-jump shape feels familiar, it is [Chapter 11's step function](ch-11-activation-functions.md#2-the-obvious-answer-and-why-it-fails). There it was an activation with no usable slope; here it is a loss with no usable slope. Same shape, same reason, same outcome.

---

## 3. The easy case first: two lists of numbers

Before word prediction, take the simpler problem, because the ideas are the same and the arithmetic is visible.

**Regression** means predicting a quantity: a temperature, a price, a length. The model outputs numbers, and you know the numbers it should have produced. No probabilities involved.

Predictions `[1, 2, 4]`, true values `[1, 3, 3]`. How wrong is that, as one number?

Subtract, which is the obvious first move:

```
  predictions   [ 1    2    4 ]
  targets       [ 1    3    3 ]
                ─────────────── −
  differences   [ 0   -1    1 ]
```

Now those three have to become one. Adding them does not work: `0 + (−1) + 1 = 0`, which says the model is perfect. Being too low on one prediction and too high on another is not the same as being right, but the signs cancel and hide it.

So make every difference positive before adding. There are two ways:

| | on our numbers | |
|---|---|---|
| absolute value | `0 + 1 + 1` | works, but has a corner at zero — the same kink that makes `relu` non-differentiable there |
| **squaring** | `0 + 1 + 1` | smooth everywhere, and punishes large errors harder |

Both give the same answer here, because our differences happen to be 0 and ±1. They part company as soon as an error is bigger than 1: an error of 10 contributes 10 under absolute value and 100 under squaring. Squaring is the usual choice for both reasons — it is differentiable at zero, and one badly wrong prediction is usually worse than several slightly-off ones.

```
  differences   [ 0   -1    1 ]
  squared       [ 0    1    1 ]
                       │
                  average → (0 + 1 + 1) / 3 = 0.666667
```

That is **mean squared error**:

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(p_i - y_i)^2$$

We average rather than total so the number does not grow just because there are more predictions. A learning rate tuned on 32 examples would otherwise be wrong for 64.

**Its gradient.** Differentiating `(p − y)²` with respect to `p` gives `2(p − y)`, and the mean divides by `n`:

$$\frac{\partial \text{MSE}}{\partial p_i} = \frac{2}{n}(p_i - y_i)$$

```
  differences         [ 0    -1         1        ]
  gradient (2/3)·d    [ 0    -0.666667  0.666667 ]
```

The middle prediction was too low, and its gradient is negative — gradient descent subtracts it, so that prediction moves up. The third was too high and moves down. The gradient points at the target.

You will not write that gradient by hand. `mseLoss` is assembled from `add`, `mul` and `mean`, all of which already have backward passes you wrote, so autograd derives it. Section 4 is assembly, not calculus.

---

## 4. Build it — `mseLoss`

**Milestone 1 — `mseLoss`.**

```typescript
export function mseLoss(predictions: TensorValue, targets: Tensor): TensorValue
```

The signature is deliberately asymmetric. `predictions` is a `TensorValue` because gradients have to flow back through it. `targets` is a plain `Tensor` because **the truth is a constant** — nothing produced it, and there is nothing to update in it.

Three steps, all with operations you already have:

1. Subtract the targets. `TensorValue` has no `sub`, but it has `add` and `mul`, and subtracting is adding a negation.
2. Square the difference. There is no `pow` either, but squaring is multiplying something by itself, and `mul` records both parents correctly.
3. Collapse to one number with `.mean()`.

✅ *Checkpoint:* `mseLoss([1,2,4], [1,3,3])` returns `0.666667`, and after `.backward()` the prediction gradient is `[0, -0.666667, 0.666667]`.

> **Pitfall — putting the target in the graph.** If you wrap `targets` in a `TensorValue`, `backward()` will compute a gradient for the labels. It is meaningless, it wastes memory, and if those tensors are used elsewhere it can corrupt real gradients.

---

## 5. Back to words: scoring a guess

MSE handled numbers against numbers. Word prediction is different: the model produces a *distribution* — a probability for every word — and the truth is a single word.

So how do you score a whole distribution against one right answer?

Look at the row again, with the truth marked:

```
softmax   [ 0.090031   0.244728   0.665241 ]
             sat        ran        flew
              ▲
              │
        the truth was "sat", and the model gave it 0.090031
```

Here is the idea, and it is smaller than you might expect: **go and look up the probability the model assigned to the word that actually happened. Score it on that number alone.**

Everything else in the row is ignored. If the model put 9% on `sat`, it gets scored on `0.090031`, and it does not matter how the remaining 91% was split between `ran` and `flew`.

That feels like it is throwing information away, and it is worth seeing why it is not. The probabilities must add to 1. So any confidence given to `ran` or `flew` is confidence *taken away* from `sat`. Pushing the right word's probability up automatically pushes the wrong ones down — you never have to mention them.

High is good, low is bad. But section 1 requires the opposite: a loss must go **down** as things improve. So we cannot use the probability directly.

---

## 6. From a probability to a loss

We need something that turns 0.09 into a big number and 0.9 into a small one, is zero when the model is perfect, and keeps growing as the model gets worse.

`−log` does all of it:

```
  probability the model gave        −log(p)
  the word that actually occurred
        1.0   (certain, and right)     0.0000     perfect
        0.665241                       0.407606
        0.244728                       1.407606
        0.090031                       2.407606   ← our model
        0.01  (confidently wrong)      4.6052
        → 0   (certain, and wrong)     → ∞        no ceiling
```

Three things at once: it flips the direction, a perfect answer costs exactly zero, and there is no upper limit, so being certain about the wrong word can be punished arbitrarily hard.

So the loss for our network is `2.407606`. That is the number the whole chapter has been walking toward, and it is just `−log` of one entry in the row.

$$\mathcal{L} = -\log(p_{\text{true word}})$$

**Where the name comes from.** `−log(p)` is a measure of *surprise*. Something you were confident about is unsurprising when it happens; something you had ruled out is very surprising. This loss is the surprise the truth causes your model, and a well-trained model is rarely surprised. The formal name for average surprise measured against another distribution is **cross-entropy**, and that is what this loss is called.

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

Every term except the true word is multiplied by zero. It is the same "look up one number", written in a way that is convenient for algebra later.

**One property worth noticing**, because section 13's gradient depends on it. Score our row against each possible truth:

```
  truth = sat     p = 0.090031     loss = 2.407606
  truth = ran     p = 0.244728     loss = 1.407606
  truth = flew    p = 0.665241     loss = 0.407606
```

The losses differ by exactly `1.000000`, and so do the logits (`1, 2, 3`). Section 11 shows why: everything about the loss except the true word's own logit is identical in all three cases.

---

## 7. So why not use MSE here?

Both losses now exist, so the comparison is fair.

The probabilities are numbers and the one-hot label is numbers, so MSE would run. The question is whether it works, and the answer is that it fails precisely when you need it most.

Take a model that is **confidently and completely wrong**: the truth is `sat`, and it has put essentially all its probability on `flew`.

```
  probabilities   [ 4.54e-5   4.54e-5   1.00 ]
  one-hot         [ 1         0         0    ]
                    sat       ran       flew

  cross-entropy   =  10.0001      no ceiling
  MSE             =   0.6666      and that is the worst it can ever say
```

MSE cannot express how bad this is. Probabilities are trapped between 0 and 1, so a squared difference cannot get large — with three words, `0.667` is the maximum value MSE can ever return, no matter how wrong the model becomes. Cross-entropy has no ceiling.

The loss value is only half of it. Here is what reaches the logits:

```
                     sat          ran         flew
  cross-entropy   -0.999955     0.000045    0.999909
  MSE             -0.000061    -0.000030    0.000091
```

**MSE's gradient has essentially vanished** — about 16,000 times smaller than cross-entropy's. The model is as wrong as it can be, and MSE is barely asking for a change.

The reason is Ch 11 section 10 arriving from a new direction. MSE is applied *after* `softmax`, and this `softmax` is **saturated**: pinned at `1.0`, out on the flat part of the curve where its derivative is nearly zero. MSE's gradient has to travel back through that flat region, and almost nothing survives the trip.

Cross-entropy avoids it by using `log`, which undoes the exponential inside `softmax`. Section 13 shows that cancellation explicitly, and it is the reason its gradient comes out so simple.

<p align="center">
  <img src="../assets/ch-12/mse-vs-crossentropy.svg" alt="Two panels, both sweeping left to right as the model becomes more confidently wrong, using logits [0, 0, w] with the true class first. The left panel plots the loss value: cross-entropy in green climbs steadily and without limit from about 1.1 to 10, while MSE in red rises briefly then flattens against a dashed ceiling line at 0.667 and stays there, captioned that probabilities are boxed into zero to one so a squared error cannot get large. The right panel plots the size of the gradient actually reaching the logits: cross-entropy in green rises toward 1.0 and holds at full strength, while MSE in red decays toward 0.00006 and effectively disappears, captioned that MSE sits behind a saturated softmax and is throttled by its flat region. Animated markers sweep both panels together. The footer states that MSE ranks this model correctly and does say the model is bad, but cannot produce enough gradient to fix it, and that the failure gets worse the more wrong the model is." />
</p>

*Figure 2: as the model gets more confidently wrong, cross-entropy's gradient grows toward full strength while MSE's decays to nothing.*

The lesson is bigger than this chapter. **A loss is not judged by whether its value looks sensible. It is judged by what its gradient does in the situations you need to escape from.** MSE ranks this model correctly — it does report that the model is bad — and still cannot fix it.

---

## 8. Where the numbers break

The loss is `−log(softmax(z))`. Compute it in that order — softmax first, then log — and it will fail on real inputs, in two separate ways.

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

Neither of these is a bug in your code. The mathematics is fine; the floating-point arithmetic is not.

The way out is a fact you already proved. Softmax is **shift-invariant**: subtracting the same constant from every logit leaves the output unchanged, because the constant factors out of the numerator and denominator and cancels. That is why Ch 11's `softmax` subtracts the max, and the proof is in [the Ch 05 deep dive](../deep-dives/ch-05-why-subtract-the-max.md).

So: never let a large number reach `exp`, and never let a small number reach `log`.

---

## 9. `logSumExp` — the same trick, one level up

We want `log(Σ eᶻ)` without ever computing `eᶻ` for a large `z`.

Pull the largest logit out first. Let `m = max(z)`. Since `e^z = e^m · e^(z−m)`:

$$\log\sum_j e^{z_j} \;=\; \log\left(e^{m}\sum_j e^{z_j - m}\right) \;=\; m + \log\sum_j e^{z_j - m}$$

The first step factors `e^m` out of every term; the second uses `log(ab) = log a + log b`. Nothing is approximated — this is exact.

Now look at what actually gets exponentiated. Every `z_j − m` is **at most zero**, because `m` is the largest of them. So every `e^(z_j − m)` lands in `(0, 1]`:

- **Nothing overflows** — the biggest value `exp` ever sees is `e⁰ = 1`.
- **Nothing hits `log(0)`** — the sum includes the largest term itself, which contributes exactly `1`, so the total is always at least 1 and its log is always at least 0.

One subtraction closes both failures. On the input that broke:

```
  logSumExp([1000, 1001, 1002])  =  1002.407606
  logSumExp([1,    2,    3   ])  =     3.407606
                                    ──────────────
                        difference  =   999.000000
```

Exactly the shift, and no `NaN` anywhere — against `[NaN, NaN, NaN]` from the naive route on the same input.

The digits after the decimal point are identical in both: `.407606`. That is shift-invariance again, and it makes a good self-check when you implement this.

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

Now put it together. We want `−log(softmax(z)_y)` without ever building `softmax(z)`.

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

And this explains section 6's "the losses differ by exactly 1". `logsumexp(z)` does not depend on which word is true, so changing the true word changes only the `z_y` term. With logits `1, 2, 3` the losses come out `2.407606, 1.407606, 0.407606` — differing by exactly the logit gaps.

**On a batch**, compute one loss per row and average them, for the same reason MSE averages.

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

Differentiate `L = logsumexp(z) − z_y` with respect to one logit `z_i`.

**The first term.** The derivative of `logsumexp` turns out to be softmax itself:

$$\frac{\partial}{\partial z_i}\log\sum_j e^{z_j} = \frac{e^{z_i}}{\sum_j e^{z_j}} = p_i$$

That is the chain rule on `log`: the derivative of `log(S)` is `1/S`, and the derivative of `S = Σ eᶻ` with respect to `z_i` is `e^{z_i}`. Multiply them and you get `e^{z_i}/S`, which is the definition of softmax.

**The second term.** `−z_y` depends on `z_i` only when `i` is the true word, giving `−1` there and `0` everywhere else. That is exactly the one-hot row `y`.

Adding them:

$$\frac{\partial \mathcal{L}}{\partial z_i} = p_i - y_i$$

**What you predicted, minus what was true.** On our row, truth = `sat`:

```
  softmax     [  0.090031   0.244728   0.665241 ]
  one-hot     [  1          0          0        ]
                sat         ran        flew
              ─────────────────────────────────── −
  gradient    [ -0.909969   0.244728   0.665241 ]
```

Read the signs. `sat` is the truth and its gradient is negative, so gradient descent subtracts it and that score goes **up**. `ran` and `flew` are positive, so their scores go **down**. The sizes track the mistake: `flew` took the most probability it should not have had, and it is pushed down hardest.

**It always sums to zero.** `p` sums to 1 and `y` sums to 1, so `p − y` sums to 0. That is a real statement, not an accident of arithmetic: cross-entropy never pushes every score in the same direction. It can only *redistribute* — take confidence from the wrong words and hand it to the right one. Which is correct, since adding a constant to every logit changes nothing (section 8).

Compare that with section 2's mistake count, whose slope was zero. Here every word gets a specific, non-zero, correctly-signed instruction on every step, no matter how wrong the model is.

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

1. What is a logit, and why can you not read one as a probability?
2. Why does `backward()` need the loss to be a single number? Which chapter made you enforce that?
3. Counting mistakes measures exactly what we care about. Give the reason it cannot be trained on, and name the Ch 11 activation that fails for the same reason.
4. MSE of predictions `[1, 2, 4]` against targets `[1, 3, 3]` — compute the loss, then the gradient.
5. Why square the differences rather than take absolute values? Give both reasons, and an example where the two disagree.
6. The model gives `sat` a probability of 0.09. Explain why scoring it on that one number does not ignore what it said about `ran` and `flew`.
7. A model puts probability 1.0 on the wrong word. What does cross-entropy report, and what does MSE report? Which gradient can still fix the model?
8. Show that `log(softmax(z)_y) = z_y − logsumexp(z)`, naming the logarithm rule used at each step.
9. `logSumExp([1,2,3]) = 3.407606` and `logSumExp([1000,1001,1002]) = 1002.407606`. Without computing anything, explain why the fractional parts match.
10. For logits `[1,2,3]` with truth `sat`, write out `p − y`. Which score goes up, which go down, and why must the three numbers sum to zero?

---

## Further Reading

- [Chris Olah — Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/) — the best intuition piece on entropy and surprise; read it if section 6's surprise framing appealed to you.
- [Wikipedia — Cross entropy](https://en.wikipedia.org/wiki/Cross_entropy) — the information-theoretic derivation the name comes from.
- [Stanford CS231n — Losses](https://cs231n.github.io/neural-networks-2/#losses) — a survey of standard losses and when each applies.
- [Goodfellow, Bengio, Courville — *Deep Learning*](https://www.deeplearningbook.org/), chapter 6.2 — why maximum likelihood leads to cross-entropy.

---

## Checkpoint

Before moving on you should be able to:

- [ ] Say what a logit is and why `softmax` is needed to read one as a probability.
- [ ] State the two requirements every loss must satisfy, and why each is required.
- [ ] Explain why accuracy is not trainable, without reaching for a formula.
- [ ] Compute MSE and its gradient by hand on a three-element example.
- [ ] Explain the loss as "look up the probability of the true word, then `−log` it".
- [ ] Say what breaks in `log(softmax(z))` — both failures — and how the max subtraction closes each.
- [ ] Derive `L = logsumexp(z) − z_y` from `−log(softmax(z)_y)`.
- [ ] Derive `∂L/∂z = p − y`, and explain why it sums to zero.
- [ ] Have all three functions passing their tests, including numerical gradient checks.

---

## Next Chapter

**[Linear Layer](ch-13-linear-layer.md)** — you now have a network that can transform inputs (Ch 11) and a number that says how wrong it is (Ch 12). Chapter 13 packages `y = xW + b` into a reusable layer with managed parameters, so the training loop in Ch 15 has something to optimise.
