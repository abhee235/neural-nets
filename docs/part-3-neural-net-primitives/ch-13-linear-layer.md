# Chapter 13: The Linear Layer

> **Part 3 of 6 — Neural Net Primitives**
> Source: `src/nn/linear.ts`
> Tests: `src/nn/linear.test.ts`
> Exercise: `exercises/ch-13-linear-layer.ts`

---

## Where did the logits come from?

Chapter 12 trained on this row:

```text
logits     [ 1    2    3 ]
             sat  ran  flew
```

We built the loss that scores it, the gradient `p − y` that corrects it — and never asked where those three numbers came from.

They came out of nowhere. We typed them.

In a real network, something has to *produce* the scores: a piece of machinery that takes in whatever describes the situation and puts out one number per option. This chapter builds that machinery.

It is the last missing piece of the loop:

```text
input ──► ??? ──► logits ──► loss ──► gradients ──► update ──► better logits
           ▲
           └── this chapter
```

And you have already met it. You have been writing it by hand since Chapter 11:

```typescript
x.matMul(W).add(b)
```

This chapter gives that line a name, a home, and answers the three questions it raises: what shape should `W` be, who keeps track of `W` and `b`, and what numbers should they start as.

---

# 1. One weight grows up

Chapter 12's model had one input and one weight:

```text
prediction = x × w
```

One number in, one number out. Real problems are wider on both ends.

**More inputs.** The context `"the cat"` cannot be one number. Suppose we describe it with two features:

```text
x = [ 1,  2 ]
```

(Where do features like these come from? Chapter 18 builds embeddings, which produce them. For now, two numbers describe the context.)

**More outputs.** We need one score per word — three scores, not one.

So instead of one weight, each output gets its **own row of weights, one per input, plus its own bias**:

```text
score = (weight for feature 1) × x₁ + (weight for feature 2) × x₂ + bias
```

Each output unit is Chapter 12's one-weight model, widened to take every input — and there is one such unit per word.

---

# 2. Build the exact logits from Chapter 12

Let's pick the weights by hand and watch the scores come out.

```text
              feature 1   feature 2     bias
  sat unit    [ 1          0    ]        0
  ran unit    [ 0          1    ]        0
  flew unit   [ 0.5        0.5  ]        1.5
```

Feed in `x = [1, 2]`:

```text
  y_sat  = 1·1   + 0·2   + 0    = 1
  y_ran  = 0·1   + 1·2   + 0    = 2
  y_flew = 0.5·1 + 0.5·2 + 1.5  = 3
```

```text
y = [ 1    2    3 ]
```

**These are Chapter 12's logits.** The row that the whole last chapter scored, differentiated, and trained against — produced by three rows of weights and three biases acting on two features. Nothing else.

<p align="center">
  <img src="../assets/ch-13/linear-anatomy.svg" alt="A network diagram with two input nodes holding 1 and 2 on the left and three output units sat, ran, flew on the right, every input connected to every output with the weight written on each edge, plus a bias arrow entering each output. The three output units highlight one at a time in an animation, each highlight showing that unit's full calculation: sat computes 1 times 1 plus 0 times 2 plus 0 equals 1, ran computes 0 times 1 plus 1 times 2 plus 0 equals 2, flew computes 0.5 times 1 plus 0.5 times 2 plus 1.5 equals 3. A caption notes that each output unit owns one row of weights and one bias, and that the three results are exactly Chapter 12's logits 1, 2, 3. A footer reads: one unit is Chapter 12's one-weight model, widened; a layer is several of them sharing the same input." />
</p>

*Figure 1: the layer, unit by unit. Each output owns a row of weights and a bias; the three results are Ch 12's logits.*

Notice what the bias did for `flew`: its weights only gather `0.5 + 1.0 = 1.5` from the features, and the bias contributes the other `1.5`. **A bias is a score the unit starts with before looking at any input** — without it, an input of all zeros could only ever produce a score of zero.

---

# 3. The matrix, and why its shape is [outputDim, inputDim]

Stack the three weight rows into one matrix and the three biases into one vector:

```text
      W  =  [ 1     0   ]        b  =  [ 0    0    1.5 ]
            [ 0     1   ]
            [ 0.5   0.5 ]        W shape: [3, 2]  =  [outputDim, inputDim]
```

The convention to memorise: **row `i` of `W` is everything output unit `i` knows.** The `sat` unit is row 0. That is why the shape is `[outputDim, inputDim]` — one row per output, one column per input — and it is the convention PyTorch and every published model checkpoint use.

The whole layer is then one line of Chapter 04:

```text
y = x @ Wᵀ + b
```

Why the transpose? `matMul` needs inner dimensions to touch: `x` is `[1, 2]` and `W` is `[3, 2]`, so `x @ W` does not even have compatible shapes. `Wᵀ` is `[2, 3]`, and `[1,2] @ [2,3] → [1,3]` — three scores. The transpose is bookkeeping, not mathematics: we chose row-per-output storage, so the multiply needs the matrix flipped.

**And the batch dimension comes free.** Feed ten contexts at once as `x` of shape `[10, 2]`, and the same line gives `[10, 3]` — ten rows of three scores, one matMul. Nothing in the layer changes. This is the reason Chapter 04 insisted on matMul over loops.

---

# 4. What happens on the way back

Everything in `y = x @ Wᵀ + b` is an operation your engine already differentiates — `matMul`, `transpose`, `add`, with broadcasting handled by Chapter 10's machinery. So run the full pipeline on our hand-set layer and read the gradients. Truth is `sat`:

```text
  loss = crossEntropyFromLogits(y, [1,0,0])  =  2.407606     Ch 12's number
```

After `backward()`:

```text
  b.grad  =  [ -0.909969   0.244728   0.665241 ]
```

**The bias gradient is exactly `p − y`.** Chapter 12's gradient arrives at the biases untouched, because a bias feeds straight into its score.

```text
  W.grad  =  [ -0.909969   -1.819938 ]      ← sat row
             [  0.244728    0.489456 ]      ← ran row
             [  0.665241    1.330482 ]      ← flew row
```

Look at each row against `p − y = [-0.909969, 0.244728, 0.665241]` and `x = [1, 2]`:

```text
  row i of W.grad  =  (p − y)ᵢ  ×  x
```

Every weight's blame is **its unit's error times the input it was multiplying**. The `sat` row gets pulled up (negative gradient) — twice as hard on feature 2, because feature 2 was twice as large, so changing its weight moves the score twice as much. The `flew` row gets pushed down hardest, scaled the same way.

One more gradient exists: `x.grad`. Nothing uses it here — the features are data. But when layers stack, the `x` of one layer is the *output* of the layer below, and `x.grad` is exactly what keeps the chain going downward. The layer passes blame through itself in both directions: into its own parameters, and back to whoever produced its input.

---

# 5. A layer owns its parameters

Until now, every piece we built was a function: numbers in, numbers out, nothing remembered. A layer is different. `W` and `b` have to *live* somewhere — created once, used on every forward pass, updated on every training step.

So the layer is the course's first **class**:

```typescript
class Linear {
  weight: TensorValue;        // [outputDim, inputDim]
  bias:   TensorValue | null;

  forward(x)      // y = x @ Wᵀ + b
  parameters()    // → [weight, bias]
}
```

The method that looks least important is the one to understand: `parameters()`.

Chapter 14's optimizers will train networks with dozens of layers. They cannot know — and must not care — what is inside each one. The deal is:

> **Every layer hands over its trainable tensors as a flat list. The optimizer walks the list and updates each one. Neither side knows anything else about the other.**

That one contract is how `step()` from Chapter 09 will scale from one weight to GPT without changing: collect every `parameters()` list, and after each `backward()`, update every tensor in it.

`bias: TensorValue | null` is a real option, not decoration — attention's Q/K/V projections (Ch 22) are usually built without biases, and `parameters()` must return a 1-element list in that case, not a list with a hole in it.

---

# 6. What should the weights start as?

The last question, and the one with a famous wrong answer.

**Try zeros.** Every unit computes `0·x₁ + 0·x₂ + 0 = 0`, so every unit gets the identical gradient, so every unit takes the identical step — and they stay identical copies of each other forever. Three units doing the work of one. You measured this yourself in Chapter 11's exercise E10(a): a zero-initialised net sat at loss `0.2500` with its weights still exactly zero after 3000 steps. **The randomness in initialisation is load-bearing: it is what makes the units different, so the gradients can make them differently useful.**

**So, random — but how big?** Here is what happens to a signal passing through 10 layers of width 100, with three choices of scale (measured, standard deviations per layer):

```text
  randn as-is       1  10  100  1e3  1e4  1e5  1e6  1e7  1e8  1e9  1e10
  randn × 0.01      1  0.1  0.01  1e-3  1e-4  1e-5  1e-6  1e-7  1e-8  1e-9  1e-11
  randn × √(1/100)  1  0.9  1.0  0.9  0.9  0.9  0.9  0.9  0.9  0.9  1.0
```

Ten times bigger per layer, or ten times smaller per layer — or steady. The same exponential explosion and vanishing as Chapter 11 section 10, arriving before training even begins.

<p align="center">
  <img src="../assets/ch-13/init-signal.svg" alt="A plot of signal size on a log scale against layer depth from 0 to 10, showing three lines through layers of width 100. The red line for raw randn weights climbs a straight diagonal, multiplying by ten each layer and reaching ten billion by layer ten, labelled explodes. A second red line for weights scaled by 0.01 falls the mirror-image diagonal to one ten-billionth, labelled vanishes. The green line for weights scaled by the square root of one over one hundred stays flat at one across all ten layers, labelled steady. A caption gives the reason: each output sums one hundred products, and a sum of n independent terms grows the signal by the square root of n, so dividing the weights by the square root of n cancels the growth exactly. All values are measured, not sketched." />
</p>

*Figure 2: three initial scales, ten layers. The measured standard deviations — ×10 per layer, ÷10 per layer, or steady.*

**Why `√(1/n)` is the right scale.** Each output sums `n` products (here `n = 100` inputs). A sum of `n` independent same-sized terms doesn't grow `n` times bigger — the random signs cancel most of it — it grows `√n` times bigger. So divide the weights by `√n` and the growth cancels exactly. That is the whole derivation.

Three named variants, all this one idea with a small adjustment:

```text
  "xavier"    randn × √(1/inputDim)      the plain argument above
  "he"        randn × √(2/inputDim)      ×2 because relu zeroes half the signal
                                         (use before relu/gelu layers)
  "normal"    randn × 0.02               GPT-2's flat choice — works because its
                                         layers are similar sizes and LayerNorm
                                         (Ch 20) re-steadies the signal anyway
```

**And the bias starts at zero** — safely. The symmetry problem does not apply: the weights are already random, so the units are already different. Zero bias just means "no initial opinion", which is right.

---

# 7. Build it

Everything above, in three members. The stubs are in `src/nn/linear.ts`, and the guidance comments there walk each one with this chapter's example.

**The constructor.** Create `weight` as `randn([outputDim, inputDim])` scaled by the chosen init (the three formulas above — `mulScalar` on the raw tensor, then wrap once). Create `bias` as zeros of shape `[outputDim]`, or `null` when `bias: false`. Default to `"he"` — this course's layers mostly feed relu/gelu.

✅ *Checkpoint:* a `Linear(100, 100)` has weights with standard deviation near `√(2/100) ≈ 0.1414`, and a `Linear(2, 3, false)` has `bias === null`.

**`forward`.** One line: matMul against the transposed weight, add the bias. Both `transpose` and `add` are graph methods — the transpose must be *inside* the graph so `W.grad` comes back in `W`'s own `[outputDim, inputDim]` shape, and `add` broadcasts `[outputDim]` across the batch, with Chapter 10's `sumToShape` collecting the bias gradient per unit.

✅ *Checkpoint:* the hand-set layer of section 2 maps `[1, 2] → [1, 2, 3]`, and after the Ch 12 loss, `b.grad = p − y` and each `W.grad` row is `(p − y)ᵢ · x`.

**`parameters`.** Return `[weight, bias]`, or `[weight]` when there is no bias. No copies — the optimizer must receive the *same* `TensorValue` objects the forward pass uses, or it will faithfully update tensors nothing reads.

✅ *Checkpoint:* `linear.parameters()[0] === linear.weight` is `true`.

---

# 8. Where this layer lives in the transformer

It would be hard to overstate how much of the final model is exactly this class:

```text
  every attention head:      Q, K, V projections + output    4 Linear layers   (Ch 22)
  every feed-forward block:  up-projection + down-projection 2 Linear layers   (Ch 25)
  the final head:            hidden → vocabulary             1 Linear layer    (Ch 30)
```

A 12-block GPT-2 carries `12 × (4 + 2) + 1 = 73` Linear layers, and they hold the overwhelming majority of its weights. Attention (Ch 22) is famous for the *pattern* in which it applies them — but the thing applied is this chapter's class, unchanged.

---

# 9. Verify

```bash
bun test src/nn/linear.test.ts
bun run exercises/ch-13-linear-layer.ts
```

The forward is pure composition, so the numerical gradient check should pass with no new backward code written — that is the payoff of Chapters 10 and 12 arriving together.

---

## What to Implement

| Member | Notes |
|--------|-------|
| `constructor(inputDim, outputDim, bias?, init?)` | `randn` scaled by he / xavier / normal; zero bias or `null`. |
| `forward(x)` | `x @ Wᵀ + b`, all graph methods. Batch dimension free. |
| `parameters()` | The flat list, same objects, no hole when bias is off. |

---

## What you should now be able to explain

1. Chapter 12 trained on logits `[1, 2, 3]`. What produced them, concretely?
2. What does one output unit consist of, and how is it related to Chapter 12's one-weight model?
3. Why is `W` stored as `[outputDim, inputDim]`, and what is row `i`?
4. Why does `forward` transpose, and why must the transpose happen inside the graph?
5. The bias gradient came out as exactly `p − y`. Why, and what is each `W.grad` row?
6. What is `x.grad` for, given that nothing uses it in this chapter?
7. What contract does `parameters()` promise, and who relies on it?
8. Why can a zero-initialised layer never learn, and what measurement from Chapter 11 shows it?
9. A signal passes through 10 random layers of width 100. What happens at scale 1, at 0.01, and at `√(1/100)` — and why does `√n` appear?
10. Why is zero the right starting bias when zero is the wrong starting weight?

---

# The idea to carry forward

A linear layer is Chapter 12's one-weight model, copied once per output and widened to every input — plus a place to live.

```text
one weight            p = x·w                     Ch 12
one unit              score = row · x + bias
one layer             y = x @ Wᵀ + b              this chapter
a network             layers, alternated with bends (Ch 11)
```

The layer owns its parameters and hands them over on request. That contract — `parameters()`, a flat list, same objects — is what lets everything after this chapter treat "a model" as "a list of tensors to update".

---

## Next Chapter

**Ch 14: Optimizers** — `step()` from Chapter 09 walked one weight downhill. Adam walks all of them, each at its own pace, using nothing but the `parameters()` lists this chapter's class provides.
