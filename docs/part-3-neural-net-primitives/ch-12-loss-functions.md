# Chapter 12: Loss Functions

> **Part 3 of 6 — Neural Net Primitives**
> Source: [`src/nn/losses.ts`](../../src/nn/losses.ts)
> Tests: [`src/nn/losses.test.ts`](../../src/nn/losses.test.ts)
> Exercise: [`exercises/ch-12-losses.ts`](../../exercises/ch-12-losses.ts)

---

## Where we left off (and why this chapter exists)

Chapter 11 ended with `softmax`, and with this row:

```
logits      [ 1          2          3        ]
softmax   → [ 0.090031   0.244728   0.665241 ]     sums to 1
```

The network has now made a claim: *"I'm 66% sure it's class 2, 24% class 1, 9% class 0."*

Suppose the true answer is **class 0**.

The model is wrong. You can see it is wrong. But your *code* has no idea — nothing in Chapters 1 to 11 has ever compared an output against an answer. You built machinery that computes (Part 1), machinery that differentiates (Part 2), and layers that transform (Ch 11). Not one line of it says **how wrong you are**.

That is the gap. And there is a second, more mechanical reason this chapter has to exist. Chapter 10's `backward()` opens with a guard you wrote yourself:

```
backward() needs a scalar root, got shape [...] — collapse with .sum() or .mean() first
```

The whole engine only starts from **one number**. Not a row of three, not a batch of probabilities — one. So something has to stand at the end of the network and crush everything down to a single value that means "how bad is this?".

That something is a **loss function**, and it is the last piece missing before you can train anything.

```
       inputs ──► layers ──► logits ──► LOSS ──► one number
                                                     │
                                      backward() ◄───┘
                                          │
                                          ▼
                                   every gradient
```

Everything downstream of this chapter — the linear layer (Ch 13), the optimizers (Ch 14), the training loop (Ch 15), and eventually GPT (Ch 30) — is machinery for pushing that one number down.

This chapter builds three functions:

| | what it measures | used for |
|---|---|---|
| `mseLoss` | how far off are the numbers? | regression — predicting quantities |
| `logSumExp` | *(a helper)* | keeping the next one from overflowing |
| `crossEntropyFromLogits` | how much probability did you put on the truth? | classification — and every language model in this course |

> **🗺️ How to read this chapter**
> Same rhythm as Ch 09, 10 and 11 — read a bit, build a bit.
>
> | | Sections | Then |
> |---|---|---|
> | **Read** | 1 → 3 (the *why*, then the simple one) | **Build** `mseLoss` (section 4) |
> | **Read** | 5 → 8 | **Build** `logSumExp` (section 9) |
> | **Read** | 10 | **Build** `crossEntropyFromLogits` (section 11) |
> | **Read** | 12 → 13 | Verify everything |
>
> Good news on the mechanics: **there is almost no new engine work here.** Ch 11 was about adding primitives with hand-written backward passes. These losses are built out of operations you already have, so autograd handles the gradients for free. The difficulty in this chapter is *choosing the right thing to minimise*, not differentiating it.

---

## Learning Goals

By the end of this chapter you can:

- Say why training needs the output collapsed to exactly one number, and connect it to the scalar guard you wrote in Ch 10.
- Explain why "count the mistakes" — the thing we actually care about — cannot be trained on, and recognise it as Ch 11's step function in disguise.
- Derive MSE from "how far apart are two numbers", and say why the distance is *squared* rather than absolute.
- Show with numbers why MSE is the wrong loss for classification, and why the failure is a *gradient* failure rather than a ranking failure.
- Read cross-entropy as "how much probability did you put on the right answer", before meeting any formula.
- Explain what breaks numerically in `log(softmax(z))`, and why the fix is the same max-subtraction trick from Ch 05.
- Implement `logSumExp` and use it to compute cross-entropy directly from logits.
- Derive the gradient `p − y`, and explain why it always sums to zero.

---

## Words we'll use in this chapter

| Word | Plain meaning |
|------|---------------|
| **loss** | One number saying how wrong the model is. Lower is better; 0 is perfect. |
| **objective / criterion** | The same thing. Other libraries use these names. |
| **regression** | Predicting a quantity — a price, a temperature, a length. |
| **classification** | Picking one option out of several — which digit, which word. |
| **logits** | Raw scores straight out of the network, before `softmax`. Any real number. |
| **one-hot** | A label written as a row of zeros with a single 1 at the true class. |
| **surprise** | `−log(p)`. Small when you expected what happened, large when you did not. |
| **overflow** | A number too big for a float, so it becomes `Infinity`. |
| **saturated** | An output pinned at its extreme, where the gradient has gone to ~0 (Ch 11 section 10). |

---

## 1. The job: everything into one number

A trained network has millions of parameters and produces thousands of outputs. Gradient descent — the thing you built in Ch 09 — knows how to do exactly one trick: **take a single number and make it smaller.**

So the loss function's job is to stand at the end and answer one question with one number:

```
    the model said        the truth was
   [0.090  0.245  0.665]     class 0
              │                  │
              └────── loss ──────┘
                       │
                       ▼
                     2.4076        ← "this is how wrong you are"
```

Two properties are non-negotiable:

1. **It must be one number.** Not because of style — because `backward()` refuses anything else. The gradient of a *row* with respect to a weight is not a single value, so there is nothing to seed the backward pass with. Ch 10 made you enforce this with an explicit throw.
2. **Lower must mean better.** Gradient descent only walks downhill. If your loss went *up* as the model improved, `step()` would faithfully train it to be worse.

Everything else in this chapter is about picking *which* number.

---

## 2. The obvious answer, and why it fails

Here is the first idea anyone has, and it is a good one: **just count the mistakes.**

If the model gets 7 out of 10 right, the loss is 3. Get everything right, loss is 0. It is simple, it is honest, and it measures *exactly the thing we actually care about*.

It is also completely untrainable. Watch what happens.

Take our row. True class is 0, and the model currently ranks it last. Now slide class 0's logit upward and watch both the error count and cross-entropy respond:

```
  logit[0]     mistakes     cross-entropy
    0.50          1            2.8715
    1.00          1            2.4076
    2.00          1            1.5514
    2.90          1            0.9210
    2.99          1            0.8678
   ─────────────────────────────────────  ← the answer flips here
    3.01          0            0.8562
    3.50          0            0.6041
```

Look at the middle column. From `0.50` to `2.99` — the model getting *dramatically* better the whole way — the error count does not move **at all**. Then it drops by a whole unit in one step. Then it never moves again.

**A flat function has a derivative of zero.** So:

```
∂(error count)/∂(logit)  =  0     everywhere it is defined
                         =  undefined at the jump
```

Gradient descent computes a gradient of zero, multiplies it by the learning rate, and updates nothing. The model never learns. Not slowly — **not at all**.

You have seen this exact failure before. This is [Chapter 11's step function](ch-11-activation-functions.md#2-the-obvious-answer-and-why-it-fails), wearing a different hat. There it was an activation that made a hard decision and had no usable slope. Here it is a *loss* that makes a hard decision and has no usable slope. Same shape, same fatal flaw, same reason.

Now look at the right-hand column. Cross-entropy moves **on every single row**. `2.8715 → 2.4076 → 1.5514 → 0.9210 → …`. It never stops responding, so there is always a direction to walk.

<p align="center">
  <img src="../assets/ch-12/why-accuracy-fails.svg" alt="Two stacked panels sharing one x-axis, which is the true class's logit sweeping from 0 to 6 for logits [a, 2, 3] with truth class 0. The top panel, in red, plots the number of mistakes: a staircase that sits flat at 1 across the whole left half, drops vertically to 0 at a = 3, then sits flat at 0 across the right half, with both flat stretches labelled slope 0 and the drop labelled jump. A caption notes the model improves the whole way while this number never moves. The bottom panel, in green, plots cross-entropy over the same range: a smooth curve falling continuously from 3.3490 at a = 0 through 2.4076, 1.5514, 0.8620 and 0.4076 to 0.0659 at a = 6, never flat anywhere. Two animated markers sweep the panels in step, the red one jumping once while the green one moves continuously. On the right, two boxes state what gradient descent sees: for mistakes, the derivative is 0 everywhere it is defined and undefined at the jump, so step() updates nothing; for cross-entropy, the derivative is p minus y, never zero while the model is still wrong, so there is always a way downhill. A note observes this is Chapter 11's step function in a different costume, and a footer reads that we do not train on what we want but on a smooth stand-in for it, measuring what we want separately." />
</p>

*Figure 1: the same sweep, scored two ways. One number is flat then jumps; the other moves the whole time.*

> **This is the central trade of the whole chapter.** The thing we care about (accuracy) cannot be optimised. So we optimise a **stand-in** that is smooth, always has a slope, and gets smaller as accuracy gets better. We do not train on what we want; we train on the differentiable shadow of what we want, and check the thing we actually want separately.
>
> That is why every ML project reports *two* numbers: a **loss** it trains on, and an **accuracy** it cares about.

---

## 3. `mseLoss` — how far apart are two numbers?

Start with the easy case: **regression**, where the model predicts a quantity and you know the right quantity.

Predictions `[1, 2, 4]`, true values `[1, 3, 3]`. How wrong is that, as one number?

Step one, the obvious move — subtract:

```
  predictions   [ 1    2    4 ]
  targets       [ 1    3    3 ]
                ─────────────── −
  differences   [ 0   -1    1 ]
```

Now collapse those three into one. You cannot just add them: `0 + (−1) + 1 = 0`, which claims the model is perfect. **The signs cancel, and cancellation is the enemy** — being too high on one and too low on another is not the same as being right.

So make everything positive first. Two ways:

| | | |
|---|---|---|
| absolute value | `\|0\| + \|−1\| + \|1\|` | works, but has a corner at 0 — the same non-differentiable kink as `relu` |
| **square it** | `0² + (−1)² + 1²` | smooth everywhere, *and* punishes big errors harder |

Squaring wins on both counts. It is differentiable at zero, and it means an error of 10 hurts 100× as much as an error of 1 — which is usually what you want, since one catastrophically wrong prediction is worse than several slightly-off ones.

```
  differences   [ 0   -1    1 ]
  squared       [ 0    1    1 ]
                       │
                  average → (0 + 1 + 1) / 3 = 0.666667
```

That is **mean squared error**:

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(p_i - y_i)^2$$

We take the *mean* rather than the sum so the number does not grow just because the batch got bigger. A learning rate tuned on 32 examples would otherwise be wrong for 64.

**Its gradient.** Differentiating `(p − y)²` with respect to `p` gives `2(p − y)` by the chain rule — outer derivative `2·(p−y)`, inner derivative `1`. Divided by `n` from the mean:

$$\frac{\partial \text{MSE}}{\partial p_i} = \frac{2}{n}(p_i - y_i)$$

```
  differences        [ 0    -1        1        ]
  gradient (2/3)·d   [ 0    -0.666667  0.666667 ]
```

Read the sign: prediction 2 was **too low**, and its gradient is **negative** — pushing it up. Prediction 3 was too high, gradient positive, pushing it down. **The gradient points at the target.**

You will not write this gradient by hand. `mseLoss` is built from `add`, `mul` and `mean` — all `TensorValue` methods with backward passes you already wrote — so autograd derives it for you. Section 4 is about assembling it, not differentiating it.

---

## 4. Build it — `mseLoss`

**Milestone 1 — `mseLoss`.**

```typescript
export function mseLoss(predictions: TensorValue, targets: Tensor): TensorValue
```

Note the asymmetry in the signature, and it is deliberate: `predictions` is a `TensorValue` because gradients must flow back through it, but `targets` is a plain `Tensor`. **The truth is a constant.** It has no gradient, it is not part of the graph, and nothing upstream produced it.

Three steps, all using operations you already have:

1. Subtract the targets from the predictions. `TensorValue` has no `sub` — but it has `add` and `mul`, and subtraction is adding a negation.
2. Square the difference. Again no `pow`, but squaring is multiplying something by itself, and `mul` records both parents correctly.
3. Collapse to one number with `.mean()`.

✅ *Checkpoint:* `mseLoss([1,2,4], [1,3,3])` returns exactly `0.666667`, and after `.backward()` the prediction gradient is `[0, -0.666667, 0.666667]`.

> **Pitfall — wrapping the target in the graph.** If you build a `TensorValue` from `targets` and let it participate, `backward()` will happily compute a gradient *for the labels*. It is meaningless, it wastes memory, and if you later feed those tensors somewhere else you can corrupt real gradients. Keep the truth outside the graph.

---

## 5. Why MSE is the wrong tool for classification

MSE is fine for quantities. The temptation is to reach for it everywhere — the probabilities are numbers, the one-hot label is numbers, so why not just square the difference?

Because it fails exactly when you need it most. Here is the measurement.

Take a model that is **confidently, catastrophically wrong**: the true answer is class 0, and it has put essentially all its probability on class 2.

```
  probabilities  [ 4.54e-5   4.54e-5   1.00 ]
  truth (one-hot)[ 1         0         0    ]

  cross-entropy loss  =  10.0001     ← enormous, as it should be
  MSE loss            =   0.6666     ← that is... it?
```

**MSE cannot express how wrong this is.** Probabilities are trapped between 0 and 1, so the squared difference can never be large no matter how badly the model fails. The worst possible MSE here is about `0.667`. Cross-entropy is *unbounded* — it can say `10`, or `100`, and it grows without limit as the model gets more confidently wrong.

But the loss value is only half the problem. Look at what reaches the logits:

```
                    class 0      class 1      class 2
  cross-entropy   -0.999955     0.000045     0.999909
  MSE             -0.000061    -0.000030     0.000091
```

**MSE's gradient has essentially vanished** — about `6e-5`, some sixteen thousand times smaller than cross-entropy's. The model is as wrong as it is possible to be, and MSE is whispering.

The reason is Ch 11 section 10, arriving from a new direction. MSE is applied *after* softmax, and this softmax is **saturated** — pinned at `1.0`, out on the flat part of the curve where its derivative is nearly zero. MSE's gradient has to travel back through that flat region and is throttled to nothing on the way.

<p align="center">
  <img src="../assets/ch-12/mse-vs-crossentropy.svg" alt="Two panels, both sweeping left to right as the model becomes more confidently wrong, using logits [0, 0, w] with truth class 0. The left panel plots the loss value: cross-entropy in green climbs steadily and without limit from about 1.1 to 10, while MSE in red rises briefly then flattens against a dashed ceiling line at 0.667 and stays there, captioned that probabilities are boxed into zero to one so a squared error cannot get large. The right panel plots the size of the gradient actually reaching the logits: cross-entropy in green rises toward 1.0 and holds at full strength, while MSE in red decays toward 0.00006 and effectively disappears, captioned that MSE sits behind a saturated softmax and is throttled by its flat region. Animated markers sweep both panels together. The footer states that MSE ranks this model correctly and does say the model is bad, but cannot produce enough gradient to fix it, and that the failure gets worse the more wrong the model is." />
</p>

*Figure 2: as the model gets more confidently wrong, cross-entropy's gradient grows toward full strength while MSE's decays to nothing.*

Cross-entropy sidesteps it entirely by using `log`, which **undoes** the exponential inside softmax. Section 12 shows the cancellation exactly, and it is the reason the final gradient is the astonishingly simple `p − y`.

> **The lesson, and it is bigger than this chapter:** a loss is not judged by whether its *value* looks sensible. It is judged by what its *gradient* does in the situations you need to escape from. MSE ranks this model correctly — it does say the model is bad — but it cannot generate enough gradient to fix it.

---

## 6. Cross-entropy — how much probability did you put on the truth?

Forget the name and forget the formula for a moment. The idea is almost insultingly simple.

The model handed you a distribution. The truth is one specific class. **Go and look up what probability the model assigned to the class that actually happened.** That number — and nothing else in the row — is what it gets scored on.

```
  softmax   [ 0.090031   0.244728   0.665241 ]
                  ▲
                  │
             truth is class 0  →  the model gave it 0.090031
```

High is good, low is bad. But a loss must go *down* as things get better, and it must be unbounded, so we do not use the probability directly. We use `−log` of it:

```
  the model gave the truth      loss = −log(p)
        p = 1.0    (certain, right)      0.0000     perfect
        p = 0.665                        0.4076
        p = 0.245                        1.4076
        p = 0.090                        2.4076
        p = 0.01   (confidently wrong)   4.6052
        p ≈ 0      (certain, wrong)      → ∞        unbounded
```

`−log` does three jobs at once: it flips the direction (high probability → low loss), it makes a perfect answer cost exactly zero, and it grows without limit as the model approaches certainty about the wrong thing.

That is the whole loss:

$$\mathcal{L} = -\log(p_{\text{true class}})$$

**The name comes from the "surprise" reading.** `−log(p)` is a measure of surprise: something you predicted confidently is unsurprising when it happens, and something you ruled out is very surprising. Cross-entropy is the average surprise the truth causes your model. A well-trained model is rarely surprised.

You will often see it written as a sum over all classes:

$$\mathcal{L} = -\sum_{c} y_c \log(p_c)$$

That looks like more work, but it is the same thing. `y` is one-hot — a single 1 and the rest zeros — so every term except the true class is multiplied by 0 and vanishes. The sum notation is convenient for the algebra; the meaning is still "look up one number".

**A property worth noticing**, because it will explain the gradient later. Score our row against each possible truth:

```
  truth = class 0    p = 0.090031    loss = 2.407606
  truth = class 1    p = 0.244728    loss = 1.407606
  truth = class 2    p = 0.665241    loss = 0.407606
```

The losses differ by **exactly 1.000000**, and so do the logits (`1, 2, 3`). That is not a coincidence — section 10 shows why the loss is `logsumexp(z) − z_y`, and the `logsumexp` part is identical whichever class is true. **Only the true class's logit distinguishes them.**

---

## 7. Where the numbers break

The formula is `−log(softmax(z))`. Implement it in the obvious order — compute softmax, then take the log — and it will explode on real inputs. Two separate ways.

**First: `log(0)`.**

```
  Math.log(0)  =  -Infinity
```

Softmax outputs are mathematically never zero, but in float64 they reach zero easily — Ch 11 section 10 measured this. A confidently wrong model produces a true-class probability that underflows, `log` returns `-Infinity`, and the loss is `Infinity`. Every gradient downstream becomes `NaN`, and the model is destroyed in one step.

**Second: `exp` overflows before softmax even finishes.** Logits are unbounded, and real networks produce large ones:

```
  logits [1000, 1001, 1002]

  exp    [Infinity, Infinity, Infinity]
  sum     Infinity
  probs  [NaN, NaN, NaN]                ← Infinity / Infinity
```

Nothing here is a bug in your code. The mathematics is fine; the float arithmetic is not.

The rescue is a fact you already proved. Softmax is **shift-invariant** — subtracting any constant from every logit leaves the output identical, because the constant factors out of numerator and denominator and cancels. That is exactly why Ch 11's softmax subtracts the max, and the proof is in [the Ch 05 deep dive](../deep-dives/ch-05-why-subtract-the-max.md).

So the fix is to never let a large number reach `exp`, and never let a small number reach `log`. That is what the next section builds.

---

## 8. `logSumExp` — the same trick, one level up

We want `log(Σ eᶻ)` without ever computing `eᶻ` for a large `z`.

Pull the largest logit out first. Let `m = max(z)`. Then, using `e^z = e^m · e^(z−m)`:

$$\log\sum_j e^{z_j} \;=\; \log\left(e^{m}\sum_j e^{z_j - m}\right) \;=\; m + \log\sum_j e^{z_j - m}$$

The first step factors `e^m` out of every term; the second uses `log(ab) = log a + log b`. Nothing has been approximated — this is an exact identity.

But now look at what actually gets exponentiated. Every `z_j − m` is **at most 0**, because `m` is the largest. So every `e^(z_j − m)` lies in `(0, 1]`:

- **No overflow** — the biggest thing `exp` ever sees is `e⁰ = 1`.
- **No `log(0)`** — the sum includes the `m` term itself, which contributes exactly `1`, so the total is always `≥ 1` and its log is always `≥ 0`.

Both failure modes are closed off by the same subtraction. Check it on the numbers that broke:

```
  logsumexp([1000, 1001, 1002])  =  1002.407606
  logsumexp([1,    2,    3   ])  =     3.407606
                                    ─────────────
                          difference =   999.000000
```

Exactly the shift, and not a `NaN` in sight. Compare that with the naive route on the same input, which produced `[NaN, NaN, NaN]`.

Notice the digits after the decimal point are identical — `.407606` in both. That is shift-invariance showing up again, and it is a good self-check when you implement this.

---

## 9. Build it — `logSumExp`

**Milestone 2 — `logSumExp`.**

```typescript
export function logSumExp(x: TensorValue, axis?: number): TensorValue
```

Four steps, following the identity above:

1. Find the max along the axis, with `keepDims: true` so it broadcasts back against the full tensor.
2. Subtract it. Everything is now `≤ 0`.
3. `exp`, then `sum` along the axis, then `log`.
4. Add the max back on.

Default the axis to the last one — `axis ?? x.data.ndim - 1` — the same convention as Ch 11's `softmax`.

The one thing to think about: **does the max need a gradient?** It is a value read out of the data, and it cancels exactly in the mathematics, so treating it as a constant is both correct and simpler. If you route it through the graph instead you will get the right answer by a longer path.

✅ *Checkpoint:* `logSumExp([1,2,3]) = 3.407606`, `logSumExp([1000,1001,1002]) = 1002.407606`, and neither produces `NaN` or `Infinity`.

---

## 10. Cross-entropy from logits, in one step

Now put it together. We want `−log(softmax(z)_y)` without ever building `softmax(z)`.

Start from softmax and take the log:

$$\log\left(\text{softmax}(z)_y\right) = \log\frac{e^{z_y}}{\sum_j e^{z_j}} = \log e^{z_y} - \log\sum_j e^{z_j} = z_y - \text{logsumexp}(z)$$

The middle step is `log(a/b) = log a − log b`; the last uses `log(eˣ) = x`. **The exponential and the logarithm cancel.** So the loss is:

$$\mathcal{L} = \text{logsumexp}(z) - z_y$$

That is the entire computation. Look at what disappeared: no division, no `exp` of a raw logit, no `log` of a small probability. Just a stable reduction and a subtraction.

Check it against the long way on our row, true class 2:

```
  logsumexp([1,2,3]) − z₂  =  3.407606 − 3  =  0.407606
  −log(softmax([1,2,3])[2]) =  −log(0.665241) =  0.407606     ✓ identical
```

And this is where section 6's "losses differ by exactly 1" comes from. `logsumexp(z)` does not depend on which class is true, so switching the true class changes only `z_y`. The losses differ by exactly the logit gaps — `3, 2, 1` giving losses `0.407606, 1.407606, 2.407606`.

**On batches**, compute one loss per row and average them, for the same reason MSE takes a mean: so the number does not depend on batch size.

---

## 11. Build it — `crossEntropyFromLogits`

**Milestone 3 — `crossEntropyFromLogits`.**

```typescript
export function crossEntropyFromLogits(logits: TensorValue, targets: Tensor): TensorValue
```

Straight from the formula:

1. `logSumExp(logits)` along the class axis.
2. Pick out `z_y` — the logit at the true class for each row.
3. Subtract, then average over the batch.

Step 2 is the fiddly one. If `targets` are **class indices** you need to gather one entry per row; if they are **one-hot** you can multiply and sum along the axis, since every term but the true one is zeroed. Both work. Decide which your `targets` are, write it in the JSDoc, and be consistent — mixing the two conventions is one of the most common bugs in this whole file.

✅ *Checkpoint:* logits `[1,2,3]` with true class `2` gives `0.407606`; true class `0` gives `2.407606`; and logits `[1000,1001,1002]` with true class `2` gives `0.407606` as well — identical, because of shift invariance.

> **Pitfall — feeding it probabilities.** The name says *FromLogits*, and it means it. If you hand this function the output of `softmax`, it will apply `logsumexp` to numbers that are already normalised and return a confidently wrong answer with no error. Raw scores in, always.

---

## 12. The gradient is `p − y`

Now the payoff. Differentiate `L = logsumexp(z) − z_y` with respect to a logit `z_i`.

**The first term.** The derivative of `logsumexp` is softmax itself:

$$\frac{\partial}{\partial z_i}\log\sum_j e^{z_j} = \frac{e^{z_i}}{\sum_j e^{z_j}} = p_i$$

That is the chain rule on `log`: the derivative of `log(S)` is `1/S`, and the derivative of `S = Σ eᶻ` with respect to `z_i` is `e^{z_i}`. Multiply them and you get `e^{z_i}/S`, which is the definition of softmax.

**The second term.** `−z_y` depends on `z_i` only when `i` is the true class, giving `−1` there and `0` elsewhere. That is precisely the one-hot vector `y`.

Add them:

$$\frac{\partial \mathcal{L}}{\partial z_i} = p_i - y_i$$

**"What you predicted, minus what was true."** On our row with true class 0:

```
  softmax        [  0.090031   0.244728   0.665241 ]
  one-hot        [  1          0          0        ]
                 ─────────────────────────────────── −
  gradient       [ -0.909969   0.244728   0.665241 ]
```

Read the signs. Class 0 is the truth and the gradient is **negative** — gradient descent subtracts it, so that logit goes **up**. Classes 1 and 2 are positive, so their logits go **down**. The size tracks the mistake: class 2 got the most probability it should not have had, and it gets pushed down hardest.

**It always sums to zero.** `p` sums to 1 and `y` sums to 1, so `p − y` sums to 0 — verified above as `-1.1e-16`, which is float64 rounding rather than a real value. This says something real: cross-entropy never pushes all the logits in the same direction. It can only *redistribute*, taking probability from the wrong classes and handing it to the right one. Which is exactly right, since adding a constant to every logit changes nothing.

Compare with the very first thing we tried, section 2's error count, whose gradient was `0` everywhere. Here every class gets a specific, non-zero, correctly-signed instruction — on every single step, no matter how wrong the model is.

> **Ch 22 note.** In practice frameworks fuse softmax and cross-entropy into one operation precisely so they can return `p − y` directly, instead of running two backward passes and hoping the `log` and `exp` cancel cleanly in floating point. That is what you are building here, and it is why your `softmax` from Ch 11 will mostly be used *without* a loss attached — inside attention (Ch 22), where there is no label to compare against.

---

## 13. Verify

```bash
bun test src/nn/losses.test.ts
bun run exercises/ch-12-losses.ts
```

Every loss here is built from operations that already have backward passes, so a numerical gradient check should pass on all of them — and it is the check that catches a dropped `2/n` or a sign flip.

---

## What to Implement

| Function | Notes |
|----------|-------|
| `mseLoss(predictions, targets)` | Mean of squared differences. `targets` stays outside the graph. |
| `logSumExp(x, axis?)` | Subtract the max, `exp`, `sum`, `log`, add the max back. Default to the last axis. |
| `crossEntropyFromLogits(logits, targets)` | `logSumExp(z) − z_y`, averaged over the batch. Raw logits only. |

---

## Common Pitfalls

- **Computing `softmax` then `log` separately.** The two failure modes of section 7 are waiting. Use `logSumExp`.
- **Passing probabilities to `crossEntropyFromLogits`.** No error, wrong answer. The name is the contract.
- **Summing instead of averaging over the batch.** Your learning rate silently becomes batch-size dependent.
- **Wrapping `targets` in a `TensorValue`.** The truth is a constant; giving it a gradient is meaningless.
- **Mixing class indices and one-hot labels.** Pick one, document it, keep it.
- **Using MSE for classification.** Section 5 measured the cost: the gradient vanishes exactly when the model is most wrong.
- **Forgetting `keepDims: true` on the max inside `logSumExp`.** The subtraction will not broadcast back correctly.

---

## Self-Check Questions

1. Why does `backward()` need the loss to be a single number? Which chapter made you enforce that, and what happens if you try to start from a row of three?
2. "Count the mistakes" measures exactly what we care about. Give the reason it cannot be trained on, and name the activation from Ch 11 that fails for the same reason.
3. MSE of predictions `[1, 2, 4]` against targets `[1, 3, 3]` — compute it by hand, then compute the gradient.
4. Why square the differences instead of taking absolute values? Give both reasons.
5. A model puts probability `1.0` on the wrong class. What does cross-entropy report, and what does MSE report? Which of the two gradients can still fix the model?
6. Show that `log(softmax(z)_y) = z_y − logsumexp(z)`, and say which logarithm rule you used at each step.
7. `logsumexp([1,2,3]) = 3.407606` and `logsumexp([1000,1001,1002]) = 1002.407606`. Explain, without computing anything, why the fractional parts match.
8. For logits `[1,2,3]` with true class 0, write out `p − y`. Which logit goes up, which go down, and why must the three numbers sum to zero?

---

## Further Reading

- [Chris Olah — Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/) — the best intuition piece on entropy and surprise; read this if section 6's "surprise" framing appealed to you.
- [Wikipedia — Cross entropy](https://en.wikipedia.org/wiki/Cross_entropy) — the information-theoretic derivation the name comes from.
- [Stanford CS231n — Losses](https://cs231n.github.io/neural-networks-2/#losses) — a survey of standard losses and when each applies.
- [Goodfellow, Bengio, Courville — *Deep Learning*](https://www.deeplearningbook.org/), chapter 6.2 — why maximum likelihood leads to cross-entropy.

---

## Checkpoint

Before moving on you should be able to:

- [ ] State the two non-negotiable properties of a loss function, and why each is required.
- [ ] Explain why accuracy is not trainable, without reaching for a formula.
- [ ] Compute MSE and its gradient by hand on a three-element example.
- [ ] Say what breaks in `log(softmax(z))` — both failure modes — and how the max-subtraction closes each.
- [ ] Derive `L = logsumexp(z) − z_y` from `−log(softmax(z)_y)`.
- [ ] Derive `∂L/∂z = p − y`, and explain why it sums to zero.
- [ ] Have all three functions passing their tests, including numerical gradient checks.

---

## Next Chapter

**[Linear Layer](ch-13-linear-layer.md)** — you now have a network that can transform inputs (Ch 11) and a number that says how wrong it is (Ch 12). Chapter 13 packages `y = xW + b` into a reusable layer with managed parameters, so the training loop in Ch 15 has something to optimise.
