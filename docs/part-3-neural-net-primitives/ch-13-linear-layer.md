# Chapter 13: The Linear Layer

> **Part 3 of 6 — Neural Net Primitives**
>
> Source: `src/nn/linear.ts`
>
> Tests: `src/nn/linear.test.ts`
>
> Exercise: `exercises/ch-13-linear-layer.ts`

---

# We need a bigger model

In the last chapter, we finally made a model learn.

We had something extremely small:

```text
input → weight → prediction → loss → gradient → update
```

The model had one input, one weight, and one prediction.

That tiny model was enough to teach us the most important idea:

> A loss produces a gradient, and the gradient tells the weights how to change.

But it is not enough to build a useful neural network.

Suppose we want to predict the next word.

The model cannot describe the entire context with one number.

And it cannot produce just one score.

For the context:

```text
"the cat ___"
```

we might want three scores:

```text
sat
ran
flew
```

So now we have a new problem:

> **How do we turn many input numbers into many output numbers?**

That's what this chapter is about.

---

# 1. Start with one output

Forget neural networks for a moment.

Suppose we have two numbers describing something:

```text
x = [1, 2]
```

We want to produce one score.

We already know how to do that from Chapter 12.

Give each input a weight:

```text
weight for x₁
weight for x₂
```

and combine them:

```text
score = w₁·x₁ + w₂·x₂
```

For example:

```text
x = [1, 2]

w₁ = 3
w₂ = 4
```

Then:

```text
score = 3·1 + 4·2 = 11
```

There is nothing new here.

It is just the one-weight model from Chapter 12, with another input added.

But we usually want one more thing.

What if the score should be `2` even when the inputs are both zero?

With:

```text
score = w₁·x₁ + w₂·x₂
```

that is impossible.

When:

```text
x₁ = 0
x₂ = 0
```

the score is always zero.

So we give the unit its own adjustable starting point:

```text
score = w₁·x₁ + w₂·x₂ + b
```

The extra number `b` is the **bias**.

Think of it as:

> **the score the unit starts with before the input has anything to say.**

For example:

```text
x = [1, 2]

w = [3, 4]

b = 2
```

gives:

```text
score = 3·1 + 4·2 + 2 = 13
```

So one output unit needs:

```text
weights
+
bias
```

---

## Where those two words come from

We have now used two words without saying why. Neither was invented for neural networks — both were borrowed, and knowing where from makes them obvious.

**Weight** comes from statistics, where a *weighted sum* is one in which the items do not count equally. The image is a balance scale: a heavier item tips it further. A weight says **how much this input counts**.

That is not a loose analogy. It is precisely what the number does — take our unit, `score = w₁x₁ + 4x₂ + 2` with `x = [1, 2]`, and slide `w₁`:

```text
   w₁     score    what it means
   -3       7      feature 1 counts AGAINST the score
    0      10      feature 1 ignored entirely — as if it were not there
    1      11      feature 1 counts, but little
    3      13      our value
    6      16      feature 1 dominates
```

A weight of `0` deletes an input from the calculation. A negative weight makes it count against. Everything in between is a dial marked *how much*.

**There is a second thread**, and the two met. Real neurons connect through synapses, and a synapse has a **strength** — how strongly one neuron's firing pushes the next. Donald Hebb proposed in 1949 that learning *is* the changing of those strengths. When Frank Rosenblatt built the perceptron in 1958, the number standing for connection strength was already called a *weight* in the weighted-sum mathematics he was using, and the two readings turned out to be the same idea:

> **how much this input counts**  =  **how strong this connection is**

Which makes the sentence "the model learns" concrete: learning is nothing but adjusting how much each input counts.

**Bias** is also statistical, where it means a *systematic offset* — something that shifts every result regardless of the data. In statistics that is usually a flaw; here it is deliberate, and the shared idea survives: a constant that does not depend on the input. The unit's standing opinion before the input says anything.

And the two together have an umbrella name you will meet in the code:

```text
weights + biases  =  parameters
```

The numbers a model fits to data — which is exactly what `parameters()` will hand to the optimizer in section 14.

---

# 2. But we need more than one output

Our language model does not need one score.

It needs one score for every possible next word.

For our tiny vocabulary:

```text
sat
ran
flew
```

we need three output scores.

So let's simply build three units.

### Unit 1: `sat`

```text
s_sat  = w₁₁·x₁ + w₁₂·x₂ + b₁
```

### Unit 2: `ran`

```text
s_ran  = w₂₁·x₁ + w₂₂·x₂ + b₂
```

### Unit 3: `flew`

```text
s_flew = w₃₁·x₁ + w₃₂·x₂ + b₃
```

Notice something.

All three units receive the **same input**:

```text
x₁
x₂
```

But each unit has its own:

```text
weights
bias
```

So the pattern is: **two inputs, three outputs, every output connected to every input** — six weights in total, plus three biases.

<p align="center">
  <img src="../assets/ch-13/linear-anatomy.svg" alt="A network diagram with two input nodes holding 1 and 2 on the left and three output units sat, ran, flew on the right, every input connected to every output with the weight written on each edge, plus a bias arrow entering each output. The three output units highlight one at a time in an animation, each highlight showing that unit's full calculation: sat computes 1 times 1 plus 0 times 2 plus 0 equals 1, ran computes 0 times 1 plus 1 times 2 plus 0 equals 2, flew computes 0.5 times 1 plus 0.5 times 2 plus 1.5 equals 3. A caption notes that each output unit owns one row of weights and one bias, and that the three results are exactly Chapter 12's logits 1, 2, 3. A footer reads: one unit is Chapter 12's one-weight model, widened; a layer is several of them sharing the same input." />
</p>

*Figure 1: two inputs, three units, six weights, three biases. Each unit lights up in turn with its own calculation — using the weights we pick in section 3.*

Every output looks at every input.

That is the important pattern:

> **Each output owns one weight for every input.**

---

# 3. Let's make the exact logits from Chapter 12

Now we can answer a question that was hidden in the previous chapter.

Where did the logits

```text
[1, 2, 3]
```

come from?

We can build them.

Let's choose:

```text
             x₁     x₂     bias

sat        [ 1       0 ]     0
ran        [ 0       1 ]     0
flew       [ 0.5     0.5 ]   1.5
```

Our input is:

```text
x = [1, 2]
```

Calculate each output separately.

### `sat`

```text
1·1 + 0·2 + 0 = 1
```

### `ran`

```text
0·1 + 1·2 + 0 = 2
```

### `flew`

```text
0.5·1 + 0.5·2 + 1.5 = 3
```

So the layer produces:

```text
[1, 2, 3]
```

Those are exactly the logits we used in Chapter 12.

They didn't appear from nowhere.

They came from a set of weights and biases acting on the input.

That is the job of a linear layer.

And look at what those three rows are actually saying, in the "how much each input counts" reading:

```text
sat   [1,   0  ]   count feature 1, ignore feature 2 completely
ran   [0,   1  ]   the exact opposite
flew  [0.5, 0.5]   count both equally
```

`flew`'s weights are worth a second look. They are `0.5` and `0.5`, which sum to `1` — so before the bias is added, that unit computes `0.5·1 + 0.5·2 = 1.5`, and the plain average of `1` and `2` is also `1.5`. **It is literally taking the mean of the two features.** A weighted sum whose weights sum to one is a weighted average — the statistical object the word "weight" came from, sitting in the middle of our example.

---

# 4. We've been writing the same calculation three times

Look at the three equations again:

```text
sat  = 1*x₁ + 0*x₂ + 0

ran  = 0*x₁ + 1*x₂ + 0

flew = 0.5*x₁ + 0.5*x₂ + 1.5
```

There is a lot of repetition.

Each output is doing the same kind of work:

> multiply each input by its weight, add the results, then add a bias.

Instead of storing the weights separately, let's put them into a matrix.

```text
W =

[ 1     0   ]
[ 0     1   ]
[ 0.5   0.5 ]
```

And put the biases into a vector:

```text
b = [0, 0, 1.5]
```

Now all three calculations can be written together:

```text
y = x @ Wᵀ + b
```

For our example:

```text
x      = [1, 2]

Wᵀ     = [1   0   0.5
          0   1   0.5]

x @ Wᵀ = [1, 2, 1.5]

+ b    = [1, 2, 3]
```

One matrix multiplication replaced three separate equations.

That's the real reason the linear layer exists:

> **A linear layer is a compact way to perform many weighted-sum calculations at once.**

---

# 5. Why is `W` shaped `[outputDim, inputDim]`?

This shape becomes much less mysterious if we look at the matrix as a collection of output units.

Our matrix is:

```text
W =

        x₁    x₂

sat     1     0
ran     0     1
flew    0.5   0.5
```

There are:

```text
3 rows
2 columns
```

So:

```text
W.shape = [3, 2]
```

which means:

```text
[outputDim, inputDim]
```

Each row belongs to one output unit.

So:

```text
row 0 → everything the "sat" unit knows
row 1 → everything the "ran" unit knows
row 2 → everything the "flew" unit knows
```

This is the convention we'll use throughout the course.

---

# 6. What does the transpose have to do with anything?

We use:

```text
y = x @ Wᵀ + b
```

rather than:

```text
y = x @ W + b
```

because of how we chose to store `W`.

For one example:

```text
x.shape = [1, 2]
W.shape = [3, 2]
```

We cannot multiply:

```text
[1, 2] @ [3, 2]
```

because the inner dimensions don't match.

But:

```text
Wᵀ.shape = [2, 3]
```

so:

```text
[1, 2] @ [2, 3]
```

gives:

```text
[1, 3]
```

Exactly what we wanted: three output scores.

So the transpose is mostly bookkeeping:

> **We store one row per output, so the matrix has to be transposed during the multiplication.**

## But why `x` on the left? Textbooks write `Wx + b`

They do, and this is worth settling, because you will read `y = Wx + b` in almost every paper.

First, the literal version of the question. Can we write `Wᵀ @ x`? Try it:

```text
Wᵀ.shape = [2, 3]
x.shape  = [1, 2]

Wᵀ @ x   →  matMul inner dims mismatch: A has 3 columns, B has 1 rows
```

Not a valid multiply at all. The inner dimensions have to touch, and `3` and `1` do not.

What *does* work is `W @ xᵀ` — with `x` standing up as a **column**:

```text
xᵀ.shape = [2, 1]           x as a column

W @ xᵀ  =  [3, 1]  =  [1, 2, 1.5]          ← a column of three scores
x @ Wᵀ  =  [1, 3]  =  [1, 2, 1.5]          ← a row of three scores
```

**Same three numbers.** One is standing up, the other lying down. They are transposes of each other, which is exactly the identity:

```text
(W x)ᵀ  =  xᵀ Wᵀ
```

So both forms are correct, and neither is more "true" than the other. The real question hiding underneath is:

> **Is one example a row, or a column?**

Textbooks pick column, because a mathematician writing `Wx` is thinking about *one* vector at a time, and a vector is conventionally a column.

We pick row. Here is why — put four examples through both and watch where the batch dimension lands:

```text
examples as ROWS                        examples as COLUMNS

X   = [4, 2]                            Xᵀ  = [2, 4]
X @ Wᵀ  →  [4, 3]                       W @ Xᵀ  →  [3, 4]

row i is example i's scores             column j is example j's scores
batch stays FIRST                       batch ends up LAST
```

Both compute the same thing. But with examples as columns, the batch dimension gets pushed to the end, and one example is no longer a row of your data — it is a vertical slice through it:

```text
rows-form   X[0]      →  [1, 2]      one contiguous piece of memory
cols-form   Xᵀ[:, 0]  →  a strided gather down a column
```

Data arrives as rows. Our tensors are stored **row-major** (Chapter 01), and every shape in this course puts batch first — `[batch, inputDim]` here, `[batch, seqLen, dModel]` from Chapter 18 onward. The row form is the one that agrees with all of that.

So: `Wx + b` in the papers, `x @ Wᵀ + b` in the code, same mathematics, and the difference is only whether you are holding one example upright or a whole batch flat.

---

# 7. And batches work automatically

Now suppose we have ten examples.

Instead of:

```text
x.shape = [1, 2]
```

we have:

```text
x.shape = [10, 2]
```

The same calculation:

```text
y = x @ Wᵀ + b
```

produces:

```text
[10, 3]
```

Ten examples in.

Three scores for each example out.

Nothing about the layer had to change.

That's one of the reasons we built `matMul` in Chapter 04 instead of writing loops everywhere.

---

# 8. Now the important question: how does learning reach these weights?

Chapter 12 taught us how a loss produces a gradient.

Our layer produces:

```text
input
  ↓
linear layer
  ↓
logits
  ↓
cross-entropy
```

For logits:

```text
[1, 2, 3]
```

and truth `sat`, Chapter 12 gives:

```text
p - y =
[-0.909969, 0.244728, 0.665241]
```

Now watch that gradient travel backward through the linear layer.

---

# 9. The bias gets the gradient directly

A unit computes:

```text
yᵢ = wᵢ · x + bᵢ
```

The bias is added directly to the output, so:

```text
∂yᵢ / ∂bᵢ  =  1
```

Therefore the gradient arriving at the output passes straight into the corresponding bias:

```text
b.grad =
[-0.909969, 0.244728, 0.665241]
```

So:

> **For a linear layer, the bias gradient is the output gradient.**

---

# 10. What happens to a weight?

Take one weight:

```text
y = w₁·x₁ + w₂·x₂ + b
```

The effect of `w₁` on the output is:

```text
∂y / ∂w₁  =  x₁
```

and similarly:

```text
∂y / ∂w₂  =  x₂
```

The chain rule gives:

```text
∂L / ∂wᵢ   =   (∂L / ∂y)  ·  (∂y / ∂wᵢ)
```

so:

```text
∂L / ∂wᵢ   =   output gradient   ×   xᵢ
```

For an entire row:

```text
W.grad row i   =   (output gradient)ᵢ   ×   x
```

For our `sat` unit:

```text
output gradient = -0.909969
x               = [1, 2]
```

so:

```text
sat W.grad =
[-0.909969, -1.819938]
```

For `ran`:

```text
[0.244728, 0.489456]
```

For `flew`:

```text
[0.665241, 1.330482]
```

Thus:

```text
W.grad =

[-0.909969  -1.819938]
[ 0.244728   0.489456]
[ 0.665241   1.330482]
```

---

# 11. Read what the gradients are saying

The `flew` unit had too much probability:

```text
0.665
```

so its gradient is positive.

Gradient descent subtracts a positive gradient, so its parameters move down.

The `sat` unit had too little probability:

```text
0.090
```

so its gradient is negative.

Gradient descent subtracts a negative gradient, so its parameters move up.

And the size of each weight gradient is scaled by the input.

Feature 2 was:

```text
2
```

while feature 1 was:

```text
1
```

So feature 2 receives twice the gradient magnitude.

That makes sense: changing the weight attached to a larger input changes the output more.

---

# 12. What about `x.grad`?

Backpropagation also calculates:

```text
x.grad
```

At first this seems unnecessary because `x` is data.

But imagine two layers:

```text
input
  ↓
Linear 1
  ↓
Linear 2
  ↓
loss
```

The output of Linear 1 becomes the input of Linear 2.

When Linear 2 sends a gradient backward, that gradient becomes the gradient entering Linear 1.

So a layer passes gradients in two directions:

```text
gradient from above
       │
       ▼
   ┌─────────┐
   │ Linear  │
   └─────────┘
      │   │
      │   └────────► parameter gradients
      │
      └────────────► gradient for layer below
```

That's how an entire network can participate in one backward pass.

---

# 13. We've reached a new problem: where do the parameters live?

So far we have been writing:

```text
W
b
```

as loose variables.

That won't work for a real network.

A network may contain many layers, and each layer needs to remember its own parameters across forward passes and training steps.

A plain function gives us:

```text
input → output
```

A layer also owns state:

```text
input
  ↓
output

inside the layer:
W
b
```

So the linear layer becomes a class.

Conceptually:

```typescript
class Linear {
  weight: TensorValue;
  bias: TensorValue | null;

  forward(x: TensorValue): TensorValue;
  parameters(): TensorValue[];
}
```

Three responsibilities:

```text
constructor → create parameters
forward()   → use parameters
parameters()→ expose parameters
```

---

# 14. Why does the optimizer need `parameters()`?

Imagine:

```text
Linear 1
Linear 2
Linear 3
```

The optimizer should not need to know what is inside each layer.

Instead, each layer says:

```text
Linear 1 → [W₁, b₁]
Linear 2 → [W₂, b₂]
Linear 3 → [W₃, b₃]
```

The model can collect these into:

```text
[W₁, b₁, W₂, b₂, W₃, b₃]
```

and the optimizer just updates the list.

So `parameters()` creates a simple contract:

> **A layer owns its trainable tensors and exposes them without exposing its internal implementation.**

That is what lets Chapter 09's `step()` scale from one weight to an entire network.

---

# 15. There is one more problem: what should the weights start as?

We've built the layer.

But before its first forward pass, we have to create its weights.

The easiest choice seems to be:

```text
W = 0
```

Let's see what happens. Every output unit starts identically:

```text
sat  → same weights
ran  → same weights
flew → same weights
```

Their outputs are identical. So are their gradients going to be identical?

Run it and the answer is surprising. One `Linear(2, 3)` with `W = 0`, `b = 0`, trained on our example with cross-entropy:

```text
step 0   loss 1.098612   rows [0.000, 0.000]  [0.000, 0.000]  [ 0.000,  0.000]
step 1   loss 0.094923   rows [0.333, 0.667]  [-0.167, -0.333] [-0.167, -0.333]
step 2   loss 0.064145   rows [0.379, 0.757]  [-0.189, -0.379] [-0.189, -0.379]
```

**It learns fine.** The `sat` row separated from the other two immediately.

Why? Because the gradient of a row is `(p − y)ᵢ × x`, and `p − y` is *different for each output* — the truth is `sat`, so that row gets a negative gradient while the others get positive ones. The **labels** broke the symmetry, even though the weights could not.

So zero init is harmless for a single output layer. The disaster is one layer deeper. Put a hidden layer in front — `Linear(2,4) → relu → Linear(4,3)`, all weights zero:

```text
step 0   loss 1.098612   W1 all zero? true   W1.grad all zero? true
step 1   loss 1.098612   W1 all zero? true   W1.grad all zero? true
step 2   loss 1.098612   W1 all zero? true   W1.grad all zero? true
```

**The loss does not move at all.** The hidden layer's gradient has to arrive *through* the second layer's weights — and those are zero, so what arrives is exactly zero. The hidden layer is not learning slowly; it is receiving nothing. This is the same parked-at-a-zero-gradient failure as Chapter 12's mistake count, and Chapter 11's exercise E10(a) measured it too.

So the rule is real, but the reason is worth stating precisely:

> **Zero weights make a layer unreachable from above.** Any layer with a zero-weight layer between it and the loss receives no gradient at all.

We need to break the symmetry — not because outputs would otherwise be identical, but because gradients must be able to travel.

---

# 16. Random weights solve one problem — but create another

Give each unit slightly different weights:

```text
sat  → different
ran  → different
flew → different
```

Now they can learn different things.

Good.

But how large should the random numbers be?

Suppose a layer has 100 inputs.

Each output computes something like:

```text
w₁·x₁ + w₂·x₂ + … + w₁₀₀·x₁₀₀
```

If the weights are too large, signals can explode as they travel through layers.

If they're too small, signals can vanish.

And the same problem eventually affects gradients.

So initialization has two jobs:

```text
1. break symmetry
2. keep signal size under control
```

---

# 17. Why does `√(1 / inputDim)` appear?

Suppose an output sums `n` roughly independent terms.

The size of the sum grows roughly like:

```text
√n
```

So with 100 inputs, the sum naturally tends to grow by roughly:

```text
√100 = 10
```

To compensate, scale each weight by:

```text
1 / √n
```

giving:

```text
√( 1 / inputDim )
```

This is the basic idea behind Xavier-style initialization.

It isn't a magic constant.

It is a way of keeping the output scale roughly stable as the number of inputs changes.

And it is measurable. Send a signal through ten layers of width 100 at three different scales, and record how big it is after each one:

```text
  randn as-is        1   10   100   1e3   1e4   1e5   1e6   1e7   1e8   1e9   1e10
  randn × 0.01       1   0.1  0.01  1e-3  1e-4  1e-5  1e-6  1e-7  1e-8  1e-9  1e-11
  randn × √(1/100)   1   0.9  1.0   0.9   0.9   0.9   0.9   0.9   0.9   0.9   1.0
```

Ten times bigger every layer, ten times smaller every layer, or steady.

<p align="center">
  <img src="../assets/ch-13/init-signal.svg" alt="A plot of signal size on a log scale against layer depth from 0 to 10, showing three lines through layers of width 100. The red line for raw randn weights climbs a straight diagonal, multiplying by ten each layer and reaching ten billion by layer ten, labelled explodes. A second red line for weights scaled by 0.01 falls the mirror-image diagonal to one ten-billionth, labelled vanishes. The green line for weights scaled by the square root of one over one hundred stays flat at one across all ten layers, labelled steady. A caption gives the reason: each output sums one hundred products, and a sum of n independent terms grows the signal by the square root of n, so dividing the weights by the square root of n cancels the growth exactly. All values are measured, not sketched." />
</p>

*Figure 2: the same three scales, plotted. The explosion and the vanishing are the same exponential failure from Chapter 11 section 10 — arriving before training even starts.*

---

# 18. Three initialization choices

You will encounter several names for variants of this idea.

### Xavier

```text
√( 1 / inputDim )
```

A straightforward scale for keeping the signal stable — exactly the `√n` argument above.

(You will also see Xavier written as `√(2/(inputDim + outputDim))`. That is Glorot's original form, which averages two constraints: keeping the signal steady going *forward*, and keeping the gradient steady coming *backward* — and the backward pass sums over `outputDim`, not `inputDim`. We use the simpler forward-only version in this course. Whichever you implement, keep the doc, the tests and the exercise agreeing on one.)

### He

```text
√( 2 / inputDim )
```

The factor of two compensates for the fact that ReLU-style activations remove roughly half the signal.

### Normal

A fixed scale such as:

```text
0.02
```

used by some transformer architectures.

The names matter less than the principle:

> **Randomize the weights, but control their scale.**

The bias can safely start at zero because the random weights already make the units different.

---

# 19. Now we can build the layer

We finally know what the class needs.

The constructor creates:

```text
weight: [outputDim, inputDim]
bias:   [outputDim] or null
```

`forward()` computes:

```text
y = x @ Wᵀ + b
```

and `parameters()` returns the tensors the optimizer should update.

Conceptually:

```typescript
class Linear {
  constructor(
    inputDim: number,
    outputDim: number,
    bias = true,
    init = "he"
  ) {
    // create parameters
  }

  forward(x: TensorValue): TensorValue {
    // x @ Wᵀ + b
  }

  parameters(): TensorValue[] {
    // return trainable tensors
  }
}
```

---

# 20. The forward pass is simple

Once the parameter management exists, the actual computation is small:

```typescript
const output = x.matMul(this.weight.transpose());

if (this.bias !== null) {
  return output.add(this.bias);
}

return output;
```

There is no new backward pass to write.

Why?

Because every operation here already knows how to propagate gradients:

```text
transpose
matMul
add
```

Your autograd system handles the chain.

This is an important milestone in the project.

> **We are starting to build bigger abstractions by composing the small differentiable operations we already built.**

---

# 21. Put the pieces together

We started this chapter with a mystery:

```text
Where did the logits come from?
```

Now we know.

```text
input
  ↓
Linear
  ↓
logits
  ↓
cross-entropy
  ↓
loss
  ↓
backward()
  ↓
W.grad, b.grad
  ↓
optimizer
  ↓
updated Linear
  ↓
better logits
```

The exact logits from Chapter 12:

```text
[1, 2, 3]
```

were just the output of:

```text
x @ Wᵀ + b
```

And the gradient from Chapter 12:

```text
p - y
```

arrives directly at the bias and then gets multiplied by the input to determine each weight's gradient.

So Chapter 12 and Chapter 13 are now connected:

```text
Chapter 12:
How do we measure and differentiate a prediction?

Chapter 13:
Where does the prediction come from?
```

---

# 22. Build it

Implement these three pieces in `src/nn/linear.ts`.

## Constructor

Create:

```text
weight: [outputDim, inputDim]
bias:   [outputDim] or null
```

Use the requested initialization strategy.

Default to He initialization.

Bias starts at zero.

## `forward(x)`

Compute:

```text
y = x @ Wᵀ + b
```

using your existing graph operations.

The batch dimension should work automatically.

## `parameters()`

Return the exact trainable `TensorValue` objects:

```text
[weight, bias]
```

or, when bias is disabled:

```text
[weight]
```

Do not create copies.

---

# 23. Checkpoint

Before moving on, you should be able to explain these without memorizing the code:

**Why do we need a linear layer?**

Because we need a systematic way to turn many inputs into many outputs.

**What does one output unit do?**

It computes:

```text
w · x + b
```

**Why is `w` called a weight, and `b` a bias?**

Because a weight says how much an input counts — the statistical weighted sum, and the strength of a synapse, which turn out to be the same idea. A bias is a constant offset that does not depend on the input.

**What does a row of `W` mean?**

All the weights belonging to one output unit.

**Why is `W` shaped `[outputDim, inputDim]`?**

One row per output and one column per input.

**Why does the forward pass use `Wᵀ`?**

Because of that storage layout: transposing makes the matrix dimensions line up for `x @ Wᵀ`.

**Papers write `Wx + b`. Why do we write `x @ Wᵀ + b`?**

Same mathematics — the two are transposes, `(Wx)ᵀ = xᵀWᵀ`. The difference is whether one example is a column or a row. We use rows so the batch dimension stays first, matching row-major storage and every other shape in the course.

**Why is `b.grad` equal to the output gradient?**

Because the bias is added directly to the output.

**Why is a weight gradient proportional to its input?**

Because:

```text
∂output / ∂weight  =  input
```

**Why do we need `parameters()`?**

So the optimizer can update parameters without knowing how each layer is implemented.

**Why can't all weights start at zero?**

Because all output units would remain identical.

**Why can't the random scale be arbitrary?**

Because signals and gradients can explode or vanish through deep networks.

---

# The idea to carry forward

A linear layer is not a mysterious new piece of mathematics.

It is the tiny model from Chapter 12, expanded.

```text
Chapter 12:

p = x × w


One output unit:

score = x · w + b


Many output units:

y = x @ Wᵀ + b


A neural network:

Linear
   ↓
activation
   ↓
Linear
   ↓
activation
   ↓
...
```

The second idea is architectural:

> **A layer owns its trainable parameters and exposes them through `parameters()`.**

That small contract is what allows everything after this chapter to treat a network as a collection of tensors that can be optimized.

---

# Verify

Run:

```bash
bun test src/nn/linear.test.ts
bun run exercises/ch-13-linear-layer.ts
```

The most important checks are:

```text
Linear(2, 3)
x = [1, 2]

→ [1, 2, 3]
```

for the hand-set example.

After backpropagation:

```text
b.grad = p - y

W.grad[i] = (p - y)[i] × x
```

And the numerical gradient check should pass.

There is no new backward formula to hand-write.

That is the payoff of the computation graph you built in the earlier chapters.

---

# Next Chapter

## Chapter 14: Optimizers

We now have a complete path from data to parameter gradients:

```text
input
  ↓
Linear
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

But who actually changes them?

Chapter 09's simple `step()` was enough when we had one weight.

Now we have many parameters, and we want something smarter than taking the same-sized step every time.

That is the job of the optimizer.
