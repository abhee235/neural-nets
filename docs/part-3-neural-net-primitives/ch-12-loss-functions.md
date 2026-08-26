# Chapter 12: Loss Functions

> **Part 3 of 6 — Neural Net Primitives**  
> Source: `src/nn/losses.ts`  
> Tests: `src/nn/losses.test.ts`  
> Exercise: `exercises/ch-12-losses.ts`

---

## How does a neural network know that it is wrong?

Imagine a tiny model whose only job is to predict tomorrow's temperature.

It looks at some input and says:

```text
32°C
```

The next day, the real temperature turns out to be:

```text
35°C
```

The model was wrong.

But something important is missing.

We can *see* that `32` is different from `35`, but the model cannot look at those two numbers and think:

> "Ah, I was three degrees too low. I should fix my weights."

The model needs a numerical way to describe its mistake.

And then it needs a way to turn that mistake into a direction for changing its weights.

That is what this chapter is about.

We will start with the simplest possible model, teach it how to improve, and then make the problem harder until we reach the loss used for classification and language models.

By the end, the whole training loop should feel like one continuous idea:

```text
weights
   ↓
prediction
   ↓
loss
   ↓
gradients
   ↓
updated weights
   ↓
better prediction
```

The formulas come later. First, let's make the problem real.

---

# 1. First, give the model a score for being wrong

Our temperature model made this prediction:

```text
prediction = 32
truth      = 35
```

The most obvious measurement is subtraction:

```text
32 - 35 = -3
```

The negative sign is useful: it tells us that the prediction was too low.

But suppose the model makes three predictions:

```text
predicted = [32, 28, 31]
actual    = [35, 28, 30]
```

The errors are:

```text
[-3, 0, +1]
```

We now have three numbers, but training needs a single quantity that summarizes how bad the whole prediction was.

What happens if we simply add them?

```text
-3 + 0 + 1 = -2
```

That number is not a useful measure of total error.

And consider this:

```text
[-3, +3]
```

The sum is zero, even though the model was wrong twice.

The signs can cancel each other.

So we need to remove the direction before combining the errors.

There are several ways to do that. Two obvious ones are absolute value and squaring.

```text
absolute value:
[-3, 0, +1]
    ↓
[ 3, 0, 1]

squaring:
[-3, 0, +1]
    ↓
[ 9, 0, 1]
```

Squaring has a useful property:

```text
error = 1   → 1
error = 3   → 9
error = 10  → 100
```

A large mistake counts much more than a small one.

It also gives us a smooth function with a simple derivative, which will matter in a moment.

So we can define our first loss:

$$
L = \frac{1}{n}\sum_i (p_i-y_i)^2
$$

This is the **mean squared error**, or **MSE**.

For our three temperatures:

$$
L = \frac{9+0+1}{3}=3.333333
$$

We now have a number that says:

> "For these predictions, the model's average squared error was 3.333333."

That is useful.

But there is a much more important question.

---

# 2. A loss is not enough. The model must know what to change.

Suppose the model has one weight:

```text
prediction = input × weight
```

Let:

```text
input   = 2
weight  = 3
truth   = 10
```

The model predicts:

$$
p = 2\times3 = 6
$$

Its loss is:

$$
L=(6-10)^2=16
$$

Great. We know the model is wrong.

Now try to answer this:

> **Should the weight go up or down?**

The loss itself does not tell us.

What we need is the rate at which the loss changes when the prediction changes.

For MSE:

$$
L=(p-y)^2
$$

Differentiate with respect to the prediction:

$$
\frac{\partial L}{\partial p}=2(p-y)
$$

For our values:

$$
\frac{\partial L}{\partial p}=2(6-10)=-8
$$

The negative sign matters.

It means that increasing the prediction would decrease the loss.

But our prediction came from the weight, so we keep moving backward through the computation:

$$
p=xw
$$

Therefore:

$$
\frac{\partial p}{\partial w}=x=2
$$

Now use the chain rule:

$$
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial p}
\frac{\partial p}{\partial w}
$$

So:

$$
\frac{\partial L}{\partial w}
=(-8)(2)=-16
$$

Our weight now has a gradient:

```text
w.grad = -16
```

That is what backpropagation is giving us: a direction and a size for the change we need.

Now the optimizer can use it.

With learning rate `0.1`:

$$
w_{new}=w-\eta\frac{\partial L}{\partial w}
$$

so:

```text
w = 3 - 0.1 × (-16)
  = 4.6
```

Run the model again:

```text
prediction = 2 × 4.6 = 9.2
```

And the new loss is:

```text
(9.2 - 10)² = 0.64
```

We started at:

```text
loss = 16
```

and after one update:

```text
loss = 0.64
```

That is learning.

Nothing mysterious happened. The loop was simply:

```text
weight
  ↓
prediction
  ↓
loss
  ↓
gradient
  ↓
weight update
  ↓
better prediction
```

Keep one distinction clear because it will matter throughout this chapter:

> **The loss measures the problem. Backpropagation calculates the gradients. The optimizer changes the weights.**

---

# 3. Why does our prediction need to be a `TensorValue`?

The previous example was easy because we did the calculus ourselves.

Your engine is supposed to do it automatically.

When you write:

```typescript
const prediction = x.mul(w);
```

your engine needs to remember more than the number `6`.

It needs to remember that `6` came from this operation:

```text
x × w
```

So conceptually, your graph looks like this:

```text
x       w
 \     /
  \   /
   mul
    │
    ▼
prediction
    │
    ▼
   loss
```

When you run:

```typescript
loss.backward();
```

the gradient can travel back through that graph.

This is why the prediction in our loss function is a `TensorValue`:

```typescript
predictions: TensorValue
```

It is a value **plus its connection to the computation graph**.

The target is different.

The target is the truth:

```text
[35, 28, 30]
```

Nothing produced that target inside the model, and there is nothing to optimize in it. So in our implementation it stays a plain `Tensor`.

This distinction is about **our autograd engine**. MSE itself does not require a special `TensorValue` type.

---

# 4. Now build the first loss

We know exactly what MSE means:

```text
subtract
↓
square
↓
average
```

So our implementation should look like the mathematics.

```typescript
export function mseLoss(
  predictions: TensorValue,
  targets: Tensor
): TensorValue {
  // negate the raw tensor first — the truth is a constant, no graph needed
  const negTargets = new TensorValue(mulScalar(targets, -1));

  const difference = predictions.add(negTargets);

  const squaredError = difference.mul(difference);

  return squaredError.mean();
}
```

One trap is worth pointing out, because it is the first thing everyone writes:

```typescript
targets.mul(-1)     // ✗ does not compile
```

`targets` is a plain `Tensor`, and a `Tensor` has no methods — it is only data. The scalar multiply lives in `tensor/ops.ts` as `mulScalar(tensor, -1)`. Keep the two levels apart: plain tensor functions for constants, `TensorValue` methods for anything the gradient must flow through.

Beyond that there is nothing special hidden in this function.

It is simply composing operations your engine already understands.

Because `add`, `mul`, and `mean` already know how to propagate gradients, autograd can derive the backward pass automatically.

For:

```text
predictions = [32, 28, 31]
targets     = [35, 28, 30]
```

you should get:

```text
loss = 3.333333
```

and after:

```typescript
loss.backward();
```

the prediction gradient should be:

```text
[-2, 0, 0.666667]
```

We now have a working loss.

But only for one kind of problem.

---

# 5. What if the answer isn't a number?

Temperature prediction is convenient because there is a natural numeric answer:

```text
35°C
```

But neural networks do many other jobs.

Suppose the model sees:

```text
"the cat ___"
```

and must choose the next word:

```text
sat
ran
flew
```

There is no target like `35`.

The answer is one of several **classes**.

This is **classification**.

Our model might produce these raw scores:

```text
sat    1
ran    2
flew   3
```

These numbers are not probabilities.

They can be negative, positive, small, or large. They are simply scores produced by the network.

We call them **logits**.

Chapter 11 gave us `softmax`, which can turn them into probabilities:

```text
logits:
[1, 2, 3]

softmax:
[0.090031, 0.244728, 0.665241]
```

So the model is saying:

```text
sat    9%
ran   24%
flew  67%
```

Suppose the true answer was `sat`.

The model is wrong.

Now the question is:

> **How do we turn this kind of mistake into a useful loss?**

---

# 6. Let's try the obvious answer: count mistakes

For classification, the simplest possible score is accuracy.

If the model gets:

```text
10 examples
7 correct
3 wrong
```

we could define:

```text
loss = 3
```

It measures exactly what we care about.

So why not train on it?

Let's watch what happens if we gradually improve the model's confidence in the correct word.

Keep the scores for `ran` and `flew` fixed, and increase the score for `sat`:

```text
sat score     p(sat)     mistakes

0.50          0.056612      1
1.00          0.090031      1
2.00          0.211942      1
2.90          0.398130      1
2.99          0.419881      1
3.01          0.424760      0
3.50          0.546549      0
5.00          0.843795      0
```

Look at what happened.

The model improved from roughly `5.7%` confidence in the correct answer to roughly `42%`.

But the mistake count stayed at `1` the entire time.

Then, when the correct class finally became the top choice, the loss suddenly changed.

This is a terrible training signal.

A flat region has zero slope.

Zero slope gives zero gradient.

Zero gradient means the weights do not move.

So accuracy is very useful for answering:

> **"Did the model get it right?"**

But it is not useful for answering:

> **"How should the model improve while it is still wrong?"**

That distinction is the reason training uses a loss and evaluation uses metrics such as accuracy.

Here is the whole sweep as one picture — the middle column and the `p(sat)` column of the table, drawn against each other:

<p align="center">
  <img src="../assets/ch-12/why-accuracy-fails.svg" alt="Two stacked panels sharing one x-axis, which is the score given to the true word sat, sweeping from 0 to 6 while ran and flew stay at 2 and 3. The top panel, in red, plots the number of mistakes: a staircase sitting flat at 1 across the whole left half, dropping vertically to 0 at a score of 3, then flat at 0 across the right half, with both flat stretches labelled slope 0 and the drop labelled jump. A caption notes the model improves the whole way while this number never moves. The bottom panel, in green, plots the probability softmax assigns to sat over the same range: a smooth curve rising continuously from 0.0351 through 0.0900, 0.2119, 0.4223 and 0.6652 to 0.9362, never flat anywhere. Two animated markers sweep the panels in step, the red one jumping once while the green one moves continuously. On the right, two boxes compare what gradient descent sees: for the mistake count the slope is 0 everywhere it is defined and undefined at the jump, so step() updates nothing; for the probability the slope is never zero because it responds to every change made to the score, so there is always something to follow. A note says a usable loss has to be built out of the probability rather than the count, and the footer reads that the count is flat until it is too late while the thing that moves smoothly is what a loss must be built from." />
</p>

*Figure 1: the same sweep, measured two ways. The mistake count is flat and then jumps; the probability moves the whole time.*

And notice the green curve. The probability of the true word responds to every single improvement. Whatever loss we design next should be built out of *that*.

---

# 7. Maybe the model's confidence is the clue

Look again at the prediction:

```text
sat    0.090031
ran    0.244728
flew   0.665241
```

The truth is `sat`.

Which number tells us how confident the model was in the correct answer?

This one:

```text
0.090031
```

Maybe that's all we need.

If the model gives the correct answer:

```text
90%
```

that should be good.

If it gives the correct answer:

```text
10%
```

that should be bad.

And if it gives the correct answer:

```text
0.1%
```

that should be extremely bad.

Notice what we have done.

We are no longer asking for a yes/no answer.

We are measuring **how much confidence the model placed on the truth**.

That's exactly the kind of smooth signal gradient descent needs.

There is only one problem:

```text
high probability = good
low probability  = bad
```

But our loss should behave the other way around:

```text
low loss  = good
high loss = bad
```

So we need a transformation.

---

# 8. We need a function that turns confidence into punishment

Let's write down what we want.

If `p` is the probability assigned to the true class, we want:

```text
p = 1       → loss = 0
p is high   → small loss
p is low    → large loss
p → 0       → enormous loss
```

One function has exactly this shape:

$$
L=-\log(p)
$$

Let's inspect it:

```text
p        -log(p)

1.0       0.0000
0.9       0.1054
0.7       0.3567
0.5       0.6931
0.3       1.2040
0.1       2.3026
0.01      4.6052
0.001     6.9078
```

As the probability of the truth increases, the loss falls.

As the probability approaches zero, the loss grows without bound.

For our original example:

```text
p(true class) = 0.090031
```

so:

$$
L=-\log(0.090031)=2.407606
$$

This is the classification loss we need.

It is called **cross-entropy loss**.

You do not need to think of that name as a new mystery.

At this point, the important idea is simply:

> **Cross-entropy asks how much probability the model gave to the answer that actually happened.**

---

# 9. See what the loss is really saying

Suppose the true class is the second class.

Model A says:

```text
[0.2, 0.3, 0.5]
```

Model B says:

```text
[0.2, 0.7, 0.1]
```

We only need to look at the probability assigned to the truth:

```text
A → 0.3
B → 0.7
```

So B should have the lower loss.

And it does:

$$
-\log(0.3)=1.204
$$

while:

$$
-\log(0.7)=0.357
$$

The model improved, and our loss improved with it.

This is exactly what the failed accuracy loss could not do.

A useful loss changes whenever the model makes useful progress, even before the final answer becomes correct.

---

# 10. Why not just use MSE for classification?

At first glance, MSE might seem fine.

Probabilities are numbers.

A one-hot target is also numbers.

For example:

```text
prediction = [0.2, 0.7, 0.1]
target     = [0,   1,   0]
```

So why not simply calculate MSE?

Because a loss has two jobs:

1. measure how bad the prediction is;
2. provide a useful gradient for improving it.

Let's look at a model that is **confidently wrong**.

The truth is `sat`, but the model puts almost all of its probability on `flew`:

```text
prediction = [0.000045, 0.000045, 0.999910]
target     = [1,        0,        0]
```

Cross-entropy gives a large loss:

```text
≈ 10
```

MSE gives:

```text
≈ 0.6667
```

That already tells us something: MSE cannot make this mistake look arbitrarily bad because probabilities are limited to the interval `[0,1]`.

But the more important difference appears in the gradient.

Don't take that on faith — measure it. Here is the gradient each loss actually delivers to the three logits, in this exact situation:

```text
                 sat          ran         flew

cross-entropy   -0.999955     0.000045    0.999909
MSE             -0.000061    -0.000030    0.000091
```

Cross-entropy sends a full-strength signal: push `sat` up hard, push `flew` down hard.

MSE's gradient is about **16,000 times smaller**. The model is as wrong as it can possibly be, and MSE is barely asking for a change.

The reason: MSE is applied after `softmax`, so its gradient has to travel backward *through* softmax. And this softmax is saturated — pinned at `1` for the wrong class, out on the flat part of its curve where its derivative is nearly zero (Chapter 11, section 10). Almost nothing survives the trip.

Cross-entropy escapes because its `log` undoes the exponential inside softmax — section 13 shows that cancellation exactly.

<p align="center">
  <img src="../assets/ch-12/mse-vs-crossentropy.svg" alt="Two panels, both sweeping left to right as the model becomes more confidently wrong, using logits [0, 0, w] with the true class first. The left panel plots the loss value: cross-entropy in green climbs steadily and without limit from about 1.1 to 10, while MSE in red rises briefly then flattens against a dashed ceiling line at 0.667 and stays there, captioned that probabilities are boxed into zero to one so a squared error cannot get large. The right panel plots the size of the gradient actually reaching the logits: cross-entropy in green rises toward 1.0 and holds at full strength, while MSE in red decays toward 0.00006 and effectively disappears, captioned that MSE sits behind a saturated softmax and is throttled by its flat region. Animated markers sweep both panels together. The footer states that MSE ranks this model correctly and does say the model is bad, but cannot produce enough gradient to fix it, and that the failure gets worse the more wrong the model is." />
</p>

*Figure 2: as the model gets more confidently wrong, cross-entropy's gradient grows toward full strength while MSE's decays to nothing.*

So MSE can correctly report:

> "This model is very wrong."

and still fail to teach the model how to escape from that state.

That is why we care about the **gradient of the loss**, not only the loss value.

---

# 11. We have the right formula. Now let's break it with a computer.

At this point our classification loss is:

$$
L=-\log(\operatorname{softmax}(z)_y)
$$

Mathematically, we're done.

But computers don't work with perfect real numbers. They work with floating-point numbers.

Let's try a large set of logits:

```text
[1000, 1001, 1002]
```

Softmax requires exponentials:

```text
exp(1000)
exp(1001)
exp(1002)
```

Those values are far beyond the range we want to calculate directly, so floating-point arithmetic can produce:

```text
Infinity
```

Then softmax can run into:

```text
Infinity / Infinity
```

which gives:

```text
NaN
```

There is a second problem.

If a true-class probability becomes so small that floating-point arithmetic rounds it to zero, then:

```text
log(0) = -Infinity
```

and our loss becomes:

```text
Infinity
```

which can poison the backward pass with `NaN`s.

So we have discovered a new problem:

> **The mathematics is correct, but the obvious way of calculating it is numerically unsafe.**

We need the same mathematics in a safer form.

---

# 12. The trick: move the logits before taking `exp`

There is a useful property of softmax that you already met in Chapter 11.

If we subtract the same constant from every logit, the final probabilities do not change — the constant factors out of the numerator and the denominator and cancels. (The full proof is in [the Ch 05 deep dive](../deep-dives/ch-05-why-subtract-the-max.md); it is three lines.)

So instead of:

```text
[1000, 1001, 1002]
```

we can subtract the largest value:

```text
[1000, 1001, 1002]
        - 1002
        ↓
[-2, -1, 0]
```

Now the largest value entering `exp` is zero:

```text
exp(0) = 1
```

and every other exponential is less than or equal to `1`.

That simple shift prevents overflow.

It also quietly closes the `log(0)` problem from the previous section. The sum always contains the max's own term, `exp(0) = 1`, so the total is at least `1` — and the log of something ≥ 1 can never be `-Infinity`. One subtraction, both failures gone.

The same idea can be folded directly into the logarithm of a sum of exponentials.

We want:

$$
\log\left(\sum_j e^{z_j}\right)
$$

Let:

$$
m=\max(z)
$$

Then:

$$
\log\left(\sum_j e^{z_j}\right)
=
\log\left(e^m\sum_j e^{z_j-m}\right)
$$

and therefore:

$$
=
 m + \log\left(\sum_j e^{z_j-m}\right)
$$

This is **log-sum-exp**.

The important part is not the name.

The important part is what it does:

> **It lets us calculate the same quantity without feeding huge values into `exp`.**

For example:

```text
logSumExp([1,2,3])
    = 3.407606

logSumExp([1000,1001,1002])
    = 1002.407606
```

The second calculation stays finite.

---

# 13. Now simplify cross-entropy itself

We currently have:

$$
L=-\log(\operatorname{softmax}(z)_y)
$$

Softmax gives:

$$
\operatorname{softmax}(z)_y
=
\frac{e^{z_y}}{\sum_j e^{z_j}}
$$

Take the log:

$$
\log\left(\frac{e^{z_y}}{\sum_j e^{z_j}}\right)
$$

Using:

$$
\log(a/b)=\log(a)-\log(b)
$$

we get:

$$
z_y-
\log\left(\sum_j e^{z_j}\right)
$$

The second term is exactly `logSumExp`.

So the loss can be written as:

$$
\boxed{L=\operatorname{logSumExp}(z)-z_y}
$$

This is the stable form we actually want to implement.

Notice what disappeared:

```text
softmax
      ↓
division
      ↓
log of a tiny probability
```

We can calculate the loss directly from the logits.

That is why the function is named:

```text
crossEntropyFromLogits
```

It wants the raw scores from the network, not the output of `softmax`.

---

# 14. Build the stable classification loss

For one example, the algorithm is now small:

```text
logits
  ↓
logSumExp(logits)
  ↓
subtract the logit of the true class
  ↓
loss
```

For a batch, calculate one loss per example and take their mean.

The only step that needs a decision is "the logit of the true class", because it depends on how `targets` is written. There is no `gather` operation in our engine, so you have two honest options:

```text
targets as class indices    [0]          pick one entry per row in plain code —
                                         picking is not differentiable and does
                                         not need to be

targets as one-hot rows     [1, 0, 0]    multiply the logits by the mask and sum
                                         along the class axis — every wrong-class
                                         term is × 0, so only the true logit
                                         survives
```

The one-hot route uses only `mul` and `sum`, which your engine already differentiates. Either way, pick one convention, write it in the JSDoc, and stay consistent — mixing the two is the most common bug in this file.

(One care point: the mask is a constant, but the true logit is **not** — the `−1` at the true class in the gradient comes from differentiating `−z_y`, so `z_y` must stay in the graph. To negate an in-graph value, `.mul()` it by a constant tensor of `−1`s; the constant is the `−1`, the logit keeps its history.)

A useful test is:

```text
logits = [1, 2, 3]
truth  = flew
```

Then:

$$
3.407606-3=0.407606
$$

which matches:

$$
-\log(0.665241)=0.407606
$$

The stable form and the intuitive probability form give exactly the same result.

---

# 15. Now look at the gradient

We have reached the part that ties the whole chapter together.

For classification:

$$
L=\operatorname{logSumExp}(z)-z_y
$$

Differentiate with respect to one logit `z_i`.

The derivative of `logSumExp` is one chain-rule step (Ch 07): the derivative of `log(S)` is `1/S`, and the derivative of `S = Σ eᶻ` with respect to `z_i` is `e^{z_i}`. Multiplied together:

$$
\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

which is exactly the softmax probability:

$$
p_i
$$

The second term, `-z_y`, contributes `-1` for the true class and `0` for the other classes.

That is exactly a one-hot target `y_i`.

So:

$$
\boxed{
\frac{\partial L}{\partial z_i}=p_i-y_i
}
$$

This looks compact, but it is very intuitive.

Suppose:

```text
prediction = [0.09, 0.24, 0.67]
truth      = [1,    0,    0   ]
```

Then:

```text
p - y

[-0.91, 0.24, 0.67]
```

Read it as instructions:

```text
true class:
    probability is too small
    → negative gradient
    → increase its score

wrong class:
    probability is positive
    → positive gradient
    → decrease its score
```

The most overconfident wrong class gets the largest push downward.

The gradient is literally:

> **what the model predicted minus what should have happened.**

That is why this gradient is so useful.

---

# 16. Why do the gradients add up to zero?

There is one more property worth seeing.

Softmax probabilities always sum to one:

$$
\sum_i p_i=1
$$

The one-hot target also sums to one:

$$
\sum_i y_i=1
$$

Therefore:

$$
\sum_i(p_i-y_i)=0
$$

So the classification loss does not push every logit up or every logit down.

It redistributes the score.

It takes confidence away from the wrong classes and gives it to the correct class.

That is exactly what we want.

---

# 17. The whole chapter in one picture

We started with a very small question:

> **How does a model know that it is wrong?**

Now we can answer it for both major cases we have seen.

### Regression

```text
prediction
    ↓
MSE
    ↓
gradient
    ↓
update weights
```

### Classification

```text
logits
    ↓
softmax probabilities
    ↓
probability of the true class
    ↓
-log(p)
    ↓
stable logSumExp form
    ↓
gradient = p - y
    ↓
update weights
```

And the training loop is still the same one we saw at the beginning:

```text
weights
   ↓
network
   ↓
prediction
   ↓
loss
   ↓
backward()
   ↓
gradients
   ↓
optimizer
   ↓
new weights
   ↓
better prediction
```

The loss is the bridge between:

```text
"What did the model predict?"
```

and:

```text
"How should its parameters change?"
```

That is why loss functions matter.

---

# 18. What you should now be able to explain

Don't try to memorize the formulas first.

See whether you can explain the story.

1. A model makes a prediction. Why do we need a loss?
2. Why is the loss not itself the weight update?
3. What does the gradient of the loss tell us?
4. Why can the prediction be a `TensorValue` while the target is a plain `Tensor` in our engine?
5. Why does simply counting classification mistakes give a bad training signal?
6. Why is the probability of the true class useful?
7. Why does `-log(p)` make a useful loss?
8. Why can MSE behave badly for classification?
9. Why does naive `log(softmax(z))` break on large or tiny values?
10. Why does subtracting the maximum make the computation safe?
11. How do we get from `-log(softmax(z)_y)` to `logSumExp(z) - z_y`?
12. Why does the gradient become `p - y`?

If you can answer those questions in your own words, you understand the chapter.

The formulas are there to make the ideas precise, not to replace the ideas.

---

# 19. What to implement

Now the code should feel like the natural conclusion of the chapter.

### `mseLoss`

```text
subtract
→ square
→ mean
```

### `logSumExp`

```text
max
→ subtract max
→ exp
→ sum
→ log
→ add max
```

### `crossEntropyFromLogits`

```text
logSumExp(logits)
→ subtract true-class logit
→ mean over batch
```

And all three can be built from the operations your engine already knows how to differentiate.

---

# 20. Verify your work

Run:

```bash
bun test src/nn/losses.test.ts
bun run exercises/ch-12-losses.ts
```

Your tests should check at least:

```text
MSE value
MSE gradient

logSumExp on ordinary values
logSumExp on very large values

cross-entropy value
cross-entropy with different target classes
cross-entropy on shifted logits

numerical gradient checks
```

A particularly useful sanity check is that adding the same constant to every logit changes neither the softmax probabilities nor the cross-entropy loss.

For example:

```text
[1, 2, 3]
```

and:

```text
[1001, 1002, 1003]
```

have the same class probabilities and therefore the same cross-entropy for the same target.

---

# The idea to carry forward

A neural network does not learn because we tell it:

> "That answer was wrong."

It learns because we turn the mistake into a number that has a useful gradient.

That gradient travels backward through the computation graph until it reaches the parameters.

The optimizer then changes those parameters.

And the next prediction gets another chance.

```text
mistake
  ↓
loss
  ↓
gradient
  ↓
weight update
  ↓
new prediction
```

That is the basic mechanism behind training.

Everything we build later is a more powerful version of this same loop.

---

## Next chapter: Linear Layer

We now have a loss that can tell us when the model is wrong and gradients that can tell its parameters how to change.

Next, we will build the part of the network that actually owns those parameters:

$$
y=xW+b
$$

and turn it into a reusable linear layer.
