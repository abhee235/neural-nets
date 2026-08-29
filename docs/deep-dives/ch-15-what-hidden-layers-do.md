# Deep Dive — What a hidden layer actually does

**Extends:** Chapter 11 (Activation Functions) and Chapter 15 (The Training Loop)
**Companion:** [the MNIST deep dive](ch-15-mnist-dense-network.md), which uses three of them

There is a question that is easy to skip past and hard to un-ask once you notice it.

You already know that `relu` bends a straight line, and that bending is what lets a network learn something a straight line cannot. So if `relu` is doing the bending — **what is the layer for?** And once you have one hidden layer, why would anyone add a second, or a third? And why do the widths go `2 → 8 → 1` in one place and `784 → 128 → 64 → 32 → 10` in another — expanding in one and shrinking in the other?

This dive answers those four, in order, and every claim is either measured with the library you built or cited to where it was proved.

---

## 1. `relu` on its own bends nothing useful

Start with the thing that is easy to get backwards.

`relu(x)` has exactly one corner, and it is always at zero. Feed it `x` and it clips negatives. There is nothing to tune, nothing to learn, and one corner at a fixed place is no use for describing an arbitrary shape.

What a unit actually computes is not `relu(x)`. It is:

$$\text{relu}(w x + b)$$

and those two extra numbers change everything.

<div align="center">
  <img src="../assets/deep-dives/hidden-layers-one-bend.svg" alt="Six small plots in two rows showing what one relu unit can do. The top row keeps the weight at 1 and varies the bias: relu of 1x plus 0 has its corner at x equals 0, relu of 1x plus 1 slides the corner to minus 1, and relu of 1x minus 1 slides it to plus 1, with a dashed vertical line and a dot marking each corner. The bottom row varies the other two numbers: relu of 2x is twice as steep, relu of minus 1x plus 1 has the bend facing the other way, and minus 1.5 times relu of 1x is flipped upside down and scaled by the next layer's weight. A footer summarises that w places it, b positions it, and the output weight decides how much it counts." />
</div>

Read the top row. The bias `b` slides the corner along the axis — the corner always sits where the inside of the `relu` crosses zero, which is at

$$x = -\frac{b}{w}$$

Read the bottom row. The weight `w` sets how steeply the line leaves the corner, and its **sign** decides which way the bend faces. Then the next layer's weight scales the whole thing and can flip it upside down.

> **`relu` supplies the corner. The `Linear` aims it.** Neither is useful alone, and that is why they always appear together.

So the atom of a neural network is not "a relu" and not "a linear" — it is **one unit, which is one bend, placed anywhere you like, at any steepness, facing either way.**

---

## 2. A hidden layer is a bag of bends

One bend is not much. A curve needs several, and this is exactly what a hidden layer provides: **a hidden layer of width `n` gives you `n` bends to work with.** That is the entire meaning of "width".

Here is a network with six hidden units fitting $\sin(3x)$, with every unit's contribution drawn separately.

<div align="center">
  <img src="../assets/deep-dives/hidden-layers-bag-of-bends.svg" alt="Two stacked charts. The top chart shows six coloured lines, each perfectly straight until it bends once, labelled as each hidden unit contributing one bend, with dashed vertical marks showing where each unit's kink sits. A note says the Linear chooses where the bend goes and how steep it is while relu supplies the bend itself, and that unit 1's kink is at x equals 3.91, outside the data, so it never bends there and its output weight is minus 0.04, meaning it does nothing. An arrow reads: the next layer adds them up. The bottom chart shows the weighted sum of those bends as a solid line lying almost exactly on a dashed target sine curve, captioned that five working hinges joined end to end become a sine wave." />
</div>

Every one of those six lines is straight, then bent, then straight. None of them looks remotely like a sine wave. **Added together with the right weights, they are one** — that is the lower panel.

This is also why a hidden layer is never the last layer. Something has to *combine* the bends, and that something is the next `Linear`. The pattern is always:

```text
  Linear  →  relu     makes the bends
  Linear             adds them up
```

Look at where this particular network chose to put its corners:

```text
  corners at:   -0.70   -0.70   -0.42    0.42    0.70
  unused:        3.91   ← outside the data entirely
```

Five corners spread across the range, and one unit that landed its corner at `x = 3.91`, far outside the data it was ever shown. Within the range it never bends at all, and its output weight settled at `-0.04`. **Six units, five doing the work.** That is the same redundancy Chapter 15 measured in the XOR width table — you provision more units than the minimum precisely because some are wasted.

---

## 3. How many units? — measured

If a unit is a bend, then "how wide should the hidden layer be" becomes a concrete question: **how many bends does this shape need?**

Here is the same $\sin(3x)$ target fitted at increasing widths. One hidden layer throughout, median of five runs each, so a lucky initialisation cannot flatter a result.

<div align="center">
  <img src="../assets/deep-dives/hidden-layers-width-sweep.svg" alt="Five small charts side by side showing the same dashed target sine curve fitted by a solid red line at widths 1, 2, 4, 8 and 16 units. At 1 unit the fit is a single bend and badly wrong with loss 0.5191 and 4 parameters. At 2 units it has two bends and loss 0.1357. At 4 units loss 0.1191. At 8 units the fit visually matches with loss 0.0010 and 25 parameters. At 16 units loss 0.0001 with 49 parameters. Below, a log-log chart plots median error against width for widths 1 through 32, falling in a roughly straight line, captioned that every doubling of width divides the error by roughly the same factor." />
</div>

| hidden units | parameters | median loss | best of 5 | worst of 5 |
|---|---|---|---|---|
| 1 | 4 | 0.519070 | 0.13568 | 0.51907 |
| 2 | 7 | 0.135680 | 0.02531 | 0.51907 |
| 3 | 10 | 0.119141 | 0.00296 | 0.13568 |
| 4 | 13 | 0.119141 | 0.00297 | 0.12067 |
| 6 | 19 | 0.001811 | 0.00077 | 0.00480 |
| 8 | 25 | 0.001038 | 0.00024 | 0.00286 |
| 16 | 49 | 0.000129 | 0.00004 | 0.00019 |
| 32 | 97 | 0.000007 | 0.00000 | 0.00015 |

Two things in that table are worth more than the headline.

**One unit cannot do it, at any learning rate, ever.** One bend is one bend. This is the same impossibility Chapter 11 proved for XOR, in a different costume.

**Look at the spread at widths 3 and 4.** The best run reached `0.003` and the worst `0.12` — a factor of forty, from nothing but the random starting weights. At width 16 the best and worst differ by a factor of five. **Narrow layers are not just less capable, they are less reliable**, because losing one unit to a dead `relu` costs a third of your capacity at width 3 and a sixteenth at width 16.

That is the real argument for "wide enough that losing a few does not matter, then stop."

---

## 4. How many layers? — measured, and it cuts both ways

Now the harder question, and the one where the honest answer is not the popular one.

To compare fairly you must hold the **parameter count** fixed — otherwise you are just measuring who got more numbers. So: roughly 50 parameters each, arranged as one wide layer, two narrow ones, or three narrower ones.

<div align="center">
  <img src="../assets/deep-dives/hidden-layers-depth.svg" alt="Two panels comparing depth on two different problems, with longer bars meaning better in both. The left panel, a simple one-dimensional sine curve at a matched budget of about 50 parameters, shows one hidden layer of 16 units with 49 parameters reaching loss 0.000058, two hidden layers of 5 and 5 with 46 parameters reaching loss 0.023003, and three hidden layers of 4, 4 and 4 with 53 parameters reaching loss 0.002499; the caption says one wide layer wins by a large margin and that splitting the same budget into narrow layers makes it harder to train rather than more capable. The right panel, MNIST digits with 2000 images and medians of five runs, shows no hidden layer with 7,850 parameters at 87.8 percent test accuracy, one hidden layer with 101,770 parameters at 89.8 percent, and three hidden layers with 111,146 parameters at 91.2 percent; the caption says here depth does help by 1.4 points because an image is built from parts made of parts. A footer reads that depth pays when the data has structure to compose, and on a plain curve it costs you." />
</div>

**On the simple curve, one wide layer wins by a wide margin** — `0.000058` against `0.023` for two narrow layers, at the same budget. Deeper was not better. Deeper was forty times *worse*, and much more erratic.

**On MNIST, depth helps** — `91.2%` against `89.8%` for a single hidden layer, and against `87.8%` for no hidden layer at all.

Same question, opposite answers. So what actually decides it?

### The difference is whether the data has structure to compose

A sine wave is just a shape. There are no "parts" to it — no sub-pattern that appears in several places and can be reused. All you need is enough bends laid side by side, and one wide layer supplies those most directly.

An image is not like that. A digit is made of strokes; strokes are made of edges; edges are made of pixels. That is a hierarchy, and layers can mirror it: layer 1's units look at raw pixels, layer 2's units look at *combinations of layer 1's findings*, layer 3's at combinations of those. **A deeper network builds bends out of bends**, and when the data really is built that way, the layers have something to latch onto.

When it is not, the extra layers are just a narrower path for the signal to squeeze through.

---

## 5. What these experiments can and cannot tell you

Everything above came from **one 1-D function and one 2,000-image subset of MNIST**. That is nowhere near enough to establish a general law, and it would be dishonest to present it as one. Two small experiments cannot rule out that some other function, or some other dataset, behaves differently — and the space of possible architectures and problems is effectively endless.

So here is where each claim actually stands.

| claim | status |
|---|---|
| one unit contributes exactly one bend, at $x = -b/w$ | **arithmetic** — read straight off `relu(wx + b)`, not an experiment |
| a hidden layer's width is how many bends are available | **arithmetic**, same reason |
| parameters = (inputs × outputs) + outputs | **arithmetic** |
| more units → lower error on this curve | **measured here**, 5 runs per width |
| narrow layers are more erratic run to run | **measured here**, and matches published work |
| one wide layer beat deep-and-narrow *on this curve at this budget* | **measured here** — a single problem, not a law |
| depth helped *on MNIST at this size* | **measured here** — a single dataset, not a law |

And here is what is actually established in the literature, which is what you should lean on for the general statements:

**One hidden layer is enough, in principle.** The universal approximation theorem (Cybenko, 1989; Hornik, Stinchcombe & White, 1989) says a single hidden layer with enough units can approximate any continuous function to any accuracy. **But it gives no bound on how many units** — and for some functions the required width grows exponentially with the input dimension. "Possible" and "practical" are different claims.

**Depth buys representational power that width cannot cheaply match.** For ReLU networks, the number of distinct linear pieces the network can carve — which is exactly the "number of bends" idea, generalised to many dimensions — grows *polynomially* with width but *exponentially* with depth ([Montúfar et al., 2014](https://arxiv.org/abs/1402.1869)). This is the rigorous version of "layers build on layers", and it is why the field went deep.

**But deep and narrow networks are genuinely hard to train.** There is published work specifically on deep, narrow ReLU networks converging to a useless constant — the mean or median of the target — rather than fitting it, even though ReLU is supposed to protect against vanishing gradients ([Lu et al., 2018](https://arxiv.org/abs/1808.04947)). Better initialisation reduces the probability but does not eliminate it. **My `[5, 5]` and `[4, 4, 4]` results are that phenomenon**, at small scale, which is why they were both worse *and* wildly inconsistent.

Put those three together and the apparent contradiction dissolves:

> Depth **can represent** far more than width for the same parameters. Depth is also **harder to optimise**, and narrowness makes it much harder. On a problem with no compositional structure, you pay the optimisation cost and collect none of the representational benefit.

That is not a rule you could have derived from my two experiments. It is what my two experiments are consistent with.

---

## 6. Expand, or contract?

Now the shape question, which has a much simpler answer than it looks.

**The first and last widths are not yours to choose.** The input width is whatever the data is — 2 numbers for XOR, 784 pixels for MNIST. The output width is whatever the task needs — 1 number for a yes/no, 10 for ten digits. Only the middle is a decision.

<div align="center">
  <img src="../assets/deep-dives/hidden-layers-shapes.svg" alt="Four network shapes drawn as rows of bars whose heights represent layer width on a compressed scale. XOR from Chapter 15 goes 2 to 8 to 1, labelled expands then contracts. The circle exercise goes 2 to 16 to 1, also expands then contracts. MNIST goes 784 to 128 to 64 to 32 to 10, labelled contracts all the way. The Transformer feed-forward block from Chapter 25 goes 512 to 2048 to 512, expands then contracts. A footer reads: widen when you need more bends than you have inputs, since XOR has 2 inputs and needs about 8 folds; narrow when the input carries far more numbers than the answer needs, since MNIST has 784 pixels and 10 digits." />
</div>

Once you see a hidden unit as a bend, both directions stop being mysterious:

**Widen when you need more bends than you have inputs.** XOR has two input numbers, but the boundary it needs cannot be drawn with two folds. The number of bends required has *nothing to do with* the number of inputs — so `2 → 8` is not "adding information", it is providing enough folds to carve the space.

**Narrow when the input carries far more numbers than the answer needs.** MNIST hands you 784 pixels, most of them blank border, to produce one answer out of ten. Squeezing `784 → 128 → 64 → 32` forces each layer to describe its input in fewer numbers than it received, so it must keep what separates digits and drop the rest.

And the two are not exclusive — the transformer's feed-forward block (Chapter 25) does both, `512 → 2048 → 512`, widening to give itself room to work and then coming back to the width the rest of the model expects.

> There is no rule that says "expand" or "contract". There is a rule that says **the ends are fixed, and the middle should be wide enough for the bends you need.**

---

## 7. Counting parameters

You should be able to look at any architecture and say how big it is. The arithmetic is small.

<div align="center">
  <img src="../assets/deep-dives/hidden-layers-counting.svg" alt="A diagram of parameter counting. A small case is shown first: Linear of 3 and 2, with W drawn as a 2 by 3 grid of six labelled cells, one row per output and one column per input, plus b drawn as a column of two cells, one per output, totalling 6 plus 2 equals 8 parameters. A highlighted formula box reads: parameters equals inputs times outputs plus outputs. Below, a table applies it to four real layers: XOR hidden Linear 2 to 8 is 2 times 8 plus 8 equals 24; XOR output Linear 8 to 1 is 8 times 1 plus 1 equals 9; MNIST layer 1 Linear 784 to 128 is 784 times 128 plus 128 equals 100,480; MNIST layer 4 Linear 32 to 10 is 32 times 10 plus 10 equals 330. A footer notes the product dominates: a layer costs what the two widths it sits between multiply to." />
</div>

A `Linear` owns two things. `W` has one row per output and one column per input, so it holds `inputs × outputs` numbers. `b` holds one number per output. Therefore:

$$\text{parameters} = (\text{inputs} \times \text{outputs}) + \text{outputs}$$

Applied to the MNIST network from the companion dive:

| layer | shape | arithmetic | total | share |
|---|---|---|---|---|
| 1 | `784 → 128` | 784×128 + 128 | **100,480** | 90.4% |
| 2 | `128 → 64` | 128×64 + 64 | 8,256 | 7.4% |
| 3 | `64 → 32` | 64×32 + 32 | 2,080 | 1.9% |
| 4 | `32 → 10` | 32×10 + 10 | 330 | 0.3% |
| | | | **111,146** | |

**The multiplication dominates, and it has a consequence people find surprising: depth is cheap.** Layers 2, 3 and 4 together are under 10% of that network. Adding a fourth hidden layer of 32 units would cost about a thousand parameters — under 1%. Doubling the *first* layer to 256 units would cost a hundred thousand.

So when you are deciding what to try: **the width of the layer nearest your input is the expensive decision. Extra depth is close to free.**

---

## 8. A practical order to try things in

Not a law — a sensible order, given everything above.

1. **Start with one hidden layer.** It is enough in principle, and on anything without obvious compositional structure it is often enough in practice.
2. **Make it comfortably wide.** Aim past the minimum, because the minimum is fragile — that is what the spread at widths 3 and 4 was showing.
3. **Add depth when the data has parts made of parts** — images, audio, language. Not because deeper is better, but because there is something for the extra layers to compose.
4. **Do not go narrow to afford being deep.** That trade is what produced the worst result in this entire dive, and it has a name and a paper behind it.
5. **Count the parameters before you train.** If the first layer holds 90% of them, then that is the number that matters, and shuffling the later widths is rearranging the furniture.

---

## Further reading

**In this course**

- Chapter 11 — where the bend comes from, and why a step function cannot be trained
- Chapter 13 — `y = xWᵀ + b`, and where `W`'s shape comes from
- Chapter 15 — the width table for XOR, measured the same way as section 3 here
- [The MNIST deep dive](ch-15-mnist-dense-network.md) — three hidden layers doing real work

**The papers behind section 5**

- Cybenko, G. (1989) and Hornik, Stinchcombe & White (1989) — the universal approximation theorem: one hidden layer suffices, with no bound on how wide.
- [Montúfar, Pascanu, Cho & Bengio (2014), *On the Number of Linear Regions of Deep Neural Networks*](https://arxiv.org/abs/1402.1869) — linear regions grow polynomially in width, exponentially in depth.
- [Lu, Shin, Su & Karniadakis (2018), *Collapse of Deep and Narrow Neural Nets*](https://arxiv.org/abs/1808.04947) — deep narrow ReLU networks converging to the mean of the target instead of fitting it.
