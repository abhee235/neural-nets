# Chapter 10: Tensor Autograd Bridge

> **Part 2 of 6 — Autodiff Engine**
> Source: [`src/autograd/grad.ts`](../../src/autograd/grad.ts) · [`src/autograd/engine.ts`](../../src/autograd/engine.ts)
> Tests: [`src/autograd/grad.test.ts`](../../src/autograd/grad.test.ts)
> Exercise: [`exercises/ch-10-tensor-autograd.ts`](../../exercises/ch-10-tensor-autograd.ts)

---

## Where we left off (and why this chapter exists)

By the end of Chapter 09 you had a complete, working learning system. A graph that records itself, a backward pass that fills every gradient in one sweep, and an optimizer that moves parameters downhill. It trains a neuron. It would train a network.

It would just never finish.

Every node in that graph holds **one number**. A single weight matrix at GPT-2's width is 768 × 768 = **589,824 numbers** — so as scalar `Value` nodes, that one matrix is 589,824 separate objects, each with its own `_inputs` array and `_backward` closure, before a single forward pass has run. And that is one matrix out of the roughly 150 in the smallest GPT-2.

So this chapter is not a performance tweak. Scalar autograd is *correct* and *unusable*, and without this bridge the course stops at Chapter 09.

The fix is one idea:

> **A node holds a whole tensor instead of a single number.**

That is genuinely all. The graph does not change. The topological order does not change. The reverse walk, the `+=` accumulation, the seed, `zeroGrad`, and your `SGD` do not change. Only the contents of a node change — and then a small number of consequences follow from that, which is what the chapter is actually about.

> **🗺️ How to read this chapter**
> Alternating, like Ch 09 — read a bit, build a bit.
>
> | | Sections | Then |
> |---|---|---|
> | **Read** | 1 → 5 | **Build** `sumToShape`, `reshape`, `transpose`, `zeroGrad` (section 6) |
> | **Read** | 7 | **Build** `add`, `mul`, `matMul` (section 8) |
> | **Read** | 9 | **Build** `sum`, `mean`, `backward` (section 10) |
> | **Read** | 11 | Verify everything, then sections 12–13 |
>
> [`grad.ts`](../../src/autograd/grad.ts) presents every method as a **pair** — the scalar code you already wrote in Ch 08, then what changes for tensors. Read it alongside this doc; the two are designed to be used together.

---

## Learning Goals

By the end of this chapter you can:

- Explain why scalar autograd cannot train a real network, with numbers.
- State the shape invariant that governs this entire chapter, and name the two problems it creates.
- Explain why broadcasting backward is a *sum* and reduction backward is a *broadcast*, and why those are the same fact.
- Implement `sumToShape`, handling both added rank and stretched size-1 axes.
- Derive `matMul`'s backward from shape constraints alone, without memorising it.
- Verify every tensor operation against finite differences — and say why that matters more here than anywhere else in the course.

---

## Words we'll use in this chapter

| Word | Plain meaning |
|------|---------------|
| **`TensorValue`** | The Ch 10 node. Like `Value`, but `data` and `grad` are Tensors. |
| **shape invariant** | `grad.shape` must always equal `data.shape`. The rule everything follows from. |
| **broadcasting** | Ch 03's rule for stretching a small tensor across a bigger one. |
| **un-broadcasting** | Undoing that in the backward pass, by summing. Our `sumToShape`. |
| **reduction** | An op that collapses an axis — `sum`, `mean` (Ch 05). |
| **upstream gradient** | `out.grad` — what reached this node from the loss. Same meaning as Ch 08. |
| **rank** | Number of axes. `[2,3]` has rank 2. |

---

## 1. Why one number per node stops working

Chapter 08's design puts one `Value` object around every single number. That was the right choice for learning: you could print any node, hand-check any gradient, and see the whole graph on one page.

Now count what it costs on something real.

<p align="center">
  <img src="../assets/ch-10/one-node-many-numbers.svg" alt="Two panels comparing how one 768 by 768 weight matrix is held. On the left, labelled Ch 08 one number per node, a dense field of small red dots fills a box, captioned 589,824 objects — one per number, each with its own _inputs array and _backward closure — and noting that this is one weight matrix before a single forward pass runs. On the right, labelled Ch 10 one tensor per node, a single green box reads TensorValue with fields data: Tensor [768,768] and grad: Tensor or null, described as one contiguous Float64Array of 4.5 MiB, captioned 1 object, visited once by topoSort. A caption states that scalar autograd is correct and unusable, and that this chapter changes what is in a node and nothing else." />
</p>

*Figure 1: the same matrix, held two ways.*

| | scalar `Value` | `TensorValue` |
|---|---|---|
| one 768×768 weight matrix | **589,824 objects** | **1 object** |
| storage for its numbers | 589,824 boxed values, each in its own object | one `Float64Array`, ~4.5 MB, contiguous |
| its gradient | 589,824 more numbers, scattered | one `Float64Array` of the same shape |
| one matmul against it | ~590k multiply nodes + ~590k add nodes | 1 node |

The object count is the part that kills it. Every one of those nodes carries an array and a closure, gets visited individually by `topoSort`, and is chased through memory one pointer at a time. The arithmetic is not the problem — the bookkeeping around the arithmetic is a few hundred times larger than the arithmetic itself.

A tensor collapses all of that into a single node over a single flat buffer. The numbers sit next to each other in memory, and one operation touches all of them.

---

## 2. One example, by hand — before any theory

Everything in this chapter comes out of one small example. Do it on paper before reading any further; it takes three minutes, and the rest of the chapter is commentary on it.

Two rows of activations, and a bias added to both:

```
X = [ 1  2  3 ]            b = [ 10  20  30 ]
    [ 4  5  6 ]

    shape [2,3]                shape [1,3]
```

**Forward.** The bias has only one row, so it gets copied into both rows of `X` — that is broadcasting, from Ch 03:

```
Z = X + b = [ 11  22  33 ]        shape [2,3]
            [ 14  25  36 ]

L = sum(Z) = 141                  shape []  — a single number
```

**Backward.** Seed `L.grad = 1` and walk back one node at a time.

**Step 1 — through `sum`.** `L` is the sum of all six entries of `Z`, so nudging any one of them moves `L` by exactly that much: `∂L/∂Zᵢⱼ = 1` for every entry.

```
Z.grad = [ 1  1  1 ]              shape [2,3]
         [ 1  1  1 ]
```

**Step 2 — through `add`, into `X`.** Addition is still the router from Ch 08: it passes the gradient through unchanged. `X` has the same shape as `Z`, so nothing else is needed.

```
X.grad = [ 1  1  1 ]              shape [2,3]   ✓ matches X
         [ 1  1  1 ]
```

**Step 3 — through `add`, into `b`. This step is the entire chapter.**

`b` has shape `[1,3]`, but the gradient arriving is `[2,3]`. Those do not match, and they cannot — `b` only has three numbers, and six gradients turned up.

So ask the Chapter 08 question instead. **How many times was `b[0]` — the number 10 — actually used?** Twice: once in row 0, once in row 1. And Ch 08 already told you what happens to a value used in several places — *its gradient is the sum of the contributions.*

```
b.grad[0] = Z.grad[0][0] + Z.grad[1][0] = 1 + 1 = 2
b.grad[1] = Z.grad[0][1] + Z.grad[1][1] = 1 + 1 = 2
b.grad[2] = Z.grad[0][2] + Z.grad[1][2] = 1 + 1 = 2

b.grad = [ 2  2  2 ]              shape [1,3]   ✓ matches b
```

That is it. That summing-down-the-copies is `sumToShape`, the one genuinely new idea in the first half of this chapter — and you just did it by hand, with no new mathematics, using a rule you already had.

> Those exact numbers are the first thing to check once `add` and `sum` work. If `b.grad` comes back `[2,2,2]` with shape `[1,3]`, the hard part is done.

---

## 3. What changed, and what didn't

This is the shortest way to see the chapter:

```
Ch 08:   class Value       { data: number,  grad: number }
Ch 10:   class TensorValue  { data: Tensor,  grad: Tensor | null }
```

Everything else on that class — `_inputs`, `_backward`, the operations, the backward sweep — keeps the same structure. The [`grad.ts`](../../src/autograd/grad.ts) stub shows each method next to the scalar version you wrote, so you can see the delta directly rather than re-deriving it.

**What carries over untouched:**

- Topological order, and the reverse walk. `topoSort` never reads `.data` or `.grad` — it only follows `_inputs`. Order is a property of the graph, not of what is in the nodes.
- Gradient accumulation. A node used twice still sums its contributions; `+=` on a number becomes an element-wise tensor `add`.
- The seed. Still "the derivative of the output with respect to itself" — still 1, just one per element.
- Not zeroing inside `backward()`. Still the caller's job.
- **Your `SGD`.** It reads `.data` and `.grad` and subtracts. It will need tensor arithmetic in Ch 14, but conceptually it never learns that anything happened.

**What is genuinely new** — and it is only two things:

1. `matMul` backward, which has no scalar counterpart at all.
2. `sumToShape`, because scalars do not broadcast.

Everything else in the chapter is one of those two ideas wearing a different hat.

### Why `grad` is nullable now

In Ch 08, `grad` started at `0`. Here it starts at `null`, and that is a deliberate change rather than an oversight.

Once shape is involved, "a gradient of zero" and "no gradient yet" stop being the same statement. A zero gradient has to be an entire tensor of the right shape — real memory, allocated for every parameter, to represent *nothing has happened yet*. PyTorch draws the same line: `.grad` is `None` until the first backward pass.

The cost is that every accumulation site has to ask "first contribution, or another one?". That is two lines, identical everywhere, and worth extracting into one small helper the first time you write it.

---

### The invariant behind all of it

Notice what section 2 kept checking after every step: *does the gradient's shape match its tensor's shape?* That is the rule the whole chapter runs on.

A gradient answers a question about one number — *if I nudge this, how does the loss respond?* — so there has to be exactly one gradient per data element:

$$\boxed{\texttt{node.grad.shape} \;=\; \texttt{node.data.shape}}$$

Always. A `[4,3]` tensor has twelve numbers and therefore twelve gradients, arranged `[4,3]`.

Simple enough — except that **forward operations are allowed to change shape.** Broadcasting grew `[1,3]` into `[2,3]` in section 2. Reductions shrink one. `matMul` makes a third shape out of two. Every time forward changes a shape, backward has to change it back, and getting that direction right is essentially the whole chapter.

It is also a new *category* of bug. In Ch 08 a wrong gradient was a wrong number. Here you can have a perfectly-shaped tensor full of wrong numbers — shapes line up, nothing throws, and the loss even falls for a while. Section 11 is about not letting that happen.

---

## 4. The same rule, running the other way

Before moving on, here is section 2 as a picture — the bias going down in the forward pass, its gradients coming back up in the backward pass. Same numbers, nothing new:

<p align="center">
  <img src="../assets/ch-10/broadcast-sum-duality.svg" alt="Section 2's example drawn as two halves of one operation. The left half, labelled FORWARD, shows the bias b of shape [1,3] holding 10, 20, 30 with an arrow down labelled copied into every row, producing Z of shape [2,3] holding 11, 22, 33 on the first row and 14, 25, 36 on the second, where Z = X + b and X is [[1,2,3],[4,5,6]], giving L = sum(Z) = 141. The right half, labelled BACKWARD, shows Z.grad of shape [2,3] with every one of its six entries equal to 1, and an arrow pointing up labelled 1 + 1 = 2 producing b.grad of shape [1,3] holding 2, 2, 2, annotated sumToShape(Z.grad, [1,3]) = [2,2,2]. A highlighted column sweeps across positions 0, 1 and 2 on both halves at the same time, so each bias entry lines up with the column of gradients that sums into it. A panel below states the rule read both ways: forward broadcasts a shape up so backward sums the gradient down, and forward sums a shape down so backward broadcasts the gradient up, because b was used twice — once per row — so each of its gradients is 1 + 1 = 2." />
</p>

*Figure 2: section 2, drawn. The sweeping column shows which gradients sum into which bias entry.*

That is one direction. There is exactly one more, and it is the mirror image.

**When forward sums, backward broadcasts.** Take a reduction instead of a broadcast — `sum` along axis 1, collapsing three numbers into one:

```
forward :   [ 1  2  3 ]   →  sum axis 1  →   [  6 ]        [2,3] → [2,1]
            [ 4  5  6 ]                      [ 15 ]

backward:   [ 1 ]  →  copied across the row  →  [ 1  1  1 ]    [2,1] → [2,3]
            [ 1 ]                               [ 1  1  1 ]
```

Why: each of the three inputs contributed to its output with a coefficient of exactly 1, so every one of them receives the *same* upstream gradient. Copying one number into many positions is precisely what broadcasting does.

So there are not two rules here to keep straight:

> forward **broadcasts** a shape up  →  backward **sums** the gradient down
> forward **sums** a shape down  →  backward **broadcasts** the gradient up

Both are the single Chapter 08 rule — *a value used in several places collects the sum of their gradients* — read from one end or the other. Section 2 was the first line. This is the second. Nothing else in this chapter needs a third.

---

## 5. `sumToShape` — writing it for any pair of shapes

In section 2 you summed six gradients down into three by hand. `sumToShape` is that same operation, written once so it works for any shapes rather than just `[2,3] → [1,3]`:

```
sumToShape(grad, targetShape)  →  grad, summed back down to targetShape
```

Every broadcasting op's backward calls it, so getting it right once makes `add` and `mul` easy.

You already know *what* it does. The only new thing is *how*, because broadcasting can change a shape in two different ways — and both have to be handled:

1. **Rank was added.** `[3]` broadcast to `[2,3]` gained a leading axis. Sum the leading axes away until the ranks match.
2. **A size-1 axis was stretched.** `[3,1]` broadcast to `[3,4]` kept its rank but grew axis 1. Sum along that axis **with `keepDims = true`**, so the size-1 axis survives and you land on `[3,1]`, not `[3]`.

Do (1) first. Once the ranks agree, one pass comparing axes pairwise handles (2).

> **The `keepDims` trap.** Dropping the axis in case 2 gives `[3]` where the parameter is `[3,1]`. Both hold three numbers, so nothing crashes — it fails later, as a shape mismatch far from the cause, or silently re-broadcasts differently. This is the single most annoying bug in the chapter.

And the no-op case matters: when `grad.shape` already equals `targetShape`, return it unchanged. Every caller depends on that being safe, which is why none of them checks first.

**One note on the numbers, so they don't look like two different stories.** section 2 used two rows and got `[2,2,2]`. The exercise uses the same setup with **four** rows, so `sumToShape(ones([4,3]), [1,3])` gives `[4,4,4]` — one gradient per row, summed. It is the identical rule with a taller tensor, and your implementation should produce both without any special-casing.

---

## 6. Build it (1) — the easy half

Start here, because these four have the fewest moving parts and get you a working file to test against.

**Milestone 1 — `sumToShape`.** The function above. Everything else leans on it.
✅ *Checkpoint:* the three tests in `grad.test.ts` — `[4,3] → [1,3]` gives `[4,4,4]`, an already-matching shape is returned unchanged, and `[3,4] → [1,4]` reduces axis 0.

**Milestone 2 — `topoSortTensor`** in [`engine.ts`](../../src/autograd/engine.ts). Copy the scalar `topoSort` directly above it and change `Value` to `TensorValue`.
✅ *Checkpoint:* it is a character-for-character copy apart from the types. If you found yourself changing anything else, re-read why — the algorithm only ever touches `_inputs`.

**Milestone 3 — `reshape` and `transpose`.** Neither broadcasts, so neither needs `sumToShape`.
- `reshape` backward reshapes the gradient back to the **original** shape, captured from `this.data.shape`.
- `transpose` backward applies the **inverse** permutation. `axes[i] = j` means "output axis i came from input axis j", so the inverse is `inv[axes[i]] = i`.
✅ *Checkpoint:* `axes = [1,2,0]` inverts to `[2,0,1]`. Verify that by hand. Note it is *not* the same array — while for `[1,0]` the inverse **is** `[1,0]`, which is exactly why a 2-D-only test would let a wrong implementation through.

**Milestone 4 — `zeroGrad`.** Set `grad` back to `null`.

---

## 7. `matMul` backward — derive it, don't memorise it

The other genuinely new piece. For `Z = A B`:

$$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial Z}\,B^{\mathsf T}, \qquad \frac{\partial L}{\partial B} = A^{\mathsf T}\frac{\partial L}{\partial Z}$$

Those two lines are the workhorse of every backward pass from here to Chapter 30 — attention is mostly matmuls. So it is worth being able to reconstruct them rather than recall them.

<p align="center">
  <img src="../assets/ch-10/matmul-backward-shapes.svg" alt="A shape derivation for matmul backward. The forward pass shows A of shape [2,3] times B of shape [3,4] giving Z of shape [2,4], so the upstream gradient dZ is also [2,4]. Two panels then derive the backward shapes by constraint. For dA: it must come out [2,3] to match A, and starting from dZ at [2,4] the only factor that produces [2,3] is something shaped [4,3], which is B transposed — giving dA = dZ @ B-transpose. For dB: it must come out [3,4] to match B, and the only factor that produces that from dZ at [2,4] is something shaped [3,2] multiplied on the left, which is A transposed — giving dB = A-transpose @ dZ. A caption notes the transposes are not a trick to memorise, they are the only shapes that fit." />
</p>

*Figure 3: there is only one arrangement that type-checks.*

The reasoning, in full:

```
A : [m, k]      B : [k, n]      Z : [m, n]      dZ : [m, n]

dA must be [m, k].   Starting from dZ [m,n], the only way to reach [m,k]
                     is to multiply by something [n,k].  That is Bᵀ.

dB must be [k, n].   Starting from dZ [m,n], the only way to reach [k,n]
                     is to be multiplied into by something [k,m].  That is Aᵀ.
```

The transposes are not a trick. They are the only shapes that fit. Whenever you are unsure in a later chapter, write down the four shapes and the answer reassembles itself.

> **Which axes does `transpose` swap?** Ch 04's `transpose(t, axes?)` reverses **all** axes by default, which is what you want for a 2-D matrix. For a batched tensor — `[batch, seq, dHead]`, which is what Ch 23 hands you — reversing everything is wrong. You need to swap only the **last two** axes and leave the batch dimensions alone. Pass an explicit `axes` permutation, and consider a small helper for it; attention will call it constantly.
>
> Decide at the same time whether this method uses `matMul` or `matMulBatch`, and whether it should dispatch on `ndim`. The tests here are 2-D. Ch 23 is not.

---

## 8. Build it (2) — `add`, `mul`, `matMul`

**Milestone 5 — `add`.** Forward is Ch 03's broadcasting `add`. Backward is still a router: each parent receives the upstream gradient, wrapped in `sumToShape` for that parent's own shape.
✅ *Checkpoint:* `[2,3] + [2,3]` gives both gradients `[2,3]`. `[2,3] + [3]` gives the second gradient shape `[3]`, with each entry summed over the two rows.

**Milestone 6 — `mul`.** Structurally identical to Ch 08's — the switch still swaps the operands. Every `*` becomes an element-wise tensor `mul`, and each accumulation is wrapped in `sumToShape`.
> **Order matters here.** `mul(other.data, out.grad)` is *itself* a broadcasting operation, so its result has the broadcast shape, not the parent's. `sumToShape` wraps the product; it does not go on `out.grad` first. Reverse those and you will be multiplying mismatched shapes.

**Milestone 7 — `matMul`.** section 7's two formulas.
✅ *Checkpoint:* `[2,3] @ [3,4]` gives `[2,4]`, and after backward the gradients are `[2,3]` and `[3,4]` — each matching its own parameter. That is the shape invariant doing its job, and it is a strong signal you got the transposes the right way round.

---

## 9. Reductions backward — `sum` and `mean`

The mirror side of section 4.

**`sum`.** Every input element contributed with coefficient 1, so every element gets the same upstream gradient — broadcast it back out.

The awkwardness is `keepDims`:

```
input [2,3], sum axis=1, keepDims=true    →  out [2,1]   broadcast works directly
input [2,3], sum axis=1, keepDims=false   →  out [2]     axis is GONE
```

In the second case you must reinsert the axis before broadcasting, or the shapes will not line up — `unsqueeze` from Ch 04 does that. Handle `axis === undefined` too: the output is a scalar and every input element receives that one value.

**`mean`.** `mean` is `sum` divided by the count, so its backward is `sum`'s backward divided by the same count:

$$\frac{\partial}{\partial x_i}\text{mean}(x) = \frac{1}{n}$$

> **Which `n`?** With an axis it is `shape[axis]` — the length of the *reduced axis*, not the total element count. Without an axis it is `data.size`. Getting this wrong scales every gradient by a constant, which looks exactly like a mis-set learning rate and will not fail loudly.

And a callback to Ch 09's deep dive on why `mean` is usually the right choice for a loss: summing makes curvature grow with dataset size, so the safe learning rate shrinks as you add data. Dividing by `n` removes that dependency.

---

## 10. Build it (3) — `sum`, `mean`, `backward`

**Milestone 8 — `sum` and `mean`.** section 9. Do `sum` first and get `mean` by scaling it.

**Milestone 9 — `backward`.** Line for line the Ch 08 version, with two substitutions:

```
topoSort(this)   →   topoSortTensor(this)
this.grad = 1    →   this.grad = ones(this.data.shape)
```

> **Guard the root.** Seeding with ones only means `∂L/∂L` when `L` is a single number. Called on a `[2,3]` node, it quietly computes the gradient of the *sum* of six entries — rarely what anyone intends. PyTorch refuses outright unless you pass an explicit gradient. Either throw when `data.size !== 1` or document the summing behaviour deliberately. This is the deeper reason every loss ends in a `sum` or a `mean`.

✅ *Checkpoint:* the whole of `grad.test.ts` except the gradient checks should now pass.

---

## 11. Verify it — and take this one seriously

**Milestone 10 — `checkTensorGradient`.**

Chapter 07 built `numericalGradientTensor` for exactly this moment. The routine:

1. Run `fn(inputs)` and `backward()` to fill the analytical gradients.
2. For each input, call `numericalGradientTensor` with a scalar-valued wrapper around `fn` — the loss must collapse to one number, so sum the output if it is not already scalar.
3. Compare element by element against a tolerance.

Two traps:

- **Zero the gradients first.** `fn` gets called many times inside the numerical loop, and leftover gradients contaminate the analytical side. This is Ch 08's `−3, −9, −18` problem, now on tensors.
- **Absolute tolerance breaks down.** `1e-5` is fine for gradients around 1 and useless for gradients around `1e6`. For larger matmuls, compare relatively: `|a − n| / max(1, |a|, |n|)`.

Why this milestone matters more than any other in the chapter: **every layer from Ch 11 to Ch 30 sits on top of these six operations.** A wrong scalar gradient in Ch 08 gave you an obviously wrong number. A wrong tensor gradient gives you a correctly-shaped tensor of wrong numbers — shapes line up, nothing throws, the loss even falls for a while — and it surfaces as *"my transformer doesn't learn"* twenty chapters later, with nothing pointing back here.

Run the check on every operation before you move on. This is the last chapter where verification is cheap.

---

## What to Implement

| Symbol | Description |
|---|---|
| `sumToShape(grad, shape)` | Reverse broadcasting: sum over broadcast axes back to `shape` |
| `topoSortTensor(root)` | The Ch 08b sort, for `TensorValue`. In `engine.ts` |
| `TensorValue.add(other)` | Broadcasting forward, `sumToShape` backward |
| `TensorValue.mul(other)` | The switch, wrapped in `sumToShape` |
| `TensorValue.matMul(other)` | `dA = dZ @ Bᵀ`, `dB = Aᵀ @ dZ` |
| `TensorValue.sum(axis?, keepDims?)` | Reduce forward, broadcast backward |
| `TensorValue.mean(axis?, keepDims?)` | As `sum`, scaled by `1/n` |
| `TensorValue.reshape(shape)` | Reshape both ways |
| `TensorValue.transpose(axes?)` | Inverse permutation backward |
| `TensorValue.backward()` | Seed with ones, walk reversed |
| `TensorValue.zeroGrad()` | Back to `null` |
| `checkTensorGradient(fn, inputs, tol?)` | Finite-difference check |

---

## Common Pitfalls

- **Forgetting `sumToShape` after a broadcasting op.** The gradient comes back with the broadcast shape, breaking the invariant.
- **Applying `sumToShape` to `out.grad` before the product in `mul`.** It wraps the product, not the upstream gradient.
- **Dropping `keepDims` in `sumToShape` case 2.** `[3]` where `[3,1]` was expected — same element count, so it fails far from the cause.
- **Transposing the wrong axes in `matMul` backward.** Reversing all axes is right for 2-D and wrong for batched.
- **Implementing `reshape` backward with the forward shape** instead of the captured original.
- **Assuming the inverse of a permutation is itself.** True for `[1,0]`, false in general.
- **Using total size instead of axis length for `mean`'s `n`.** A silent constant factor on every gradient.
- **Calling `backward()` on a non-scalar** and not noticing you asked for the gradient of a sum.
- **Trusting an op without a finite-difference check.** Every later chapter depends on these six.

---

## How to Verify

```bash
bun test src/autograd/grad.test.ts
```
```bash
bun run exercises/ch-10-tensor-autograd.ts
```

There are **15** tests in `grad.test.ts`: three for `sumToShape`, three forward, six backward, three gradient checks. The gradient checks are the real gate — shape tests confirm you did *something* of the right size, and only finite differences confirm you did the right thing.

---

## Self-Check Questions

1. If `x.shape = [3,1]` and `y.shape = [3,4]`, what is the shape of `x.grad` after `z = x + y; loss = sum(z)`? What are its values?
2. Why does reduction backward *broadcast* while broadcasting backward *reduces*? Answer in one sentence that covers both.
3. For `A @ B` with `A: [5,7]` and `B: [7,11]`, derive the shapes of `dA` and `dB` from the constraint that each must match its own parameter.
4. Why is `reshape` backward simpler than `matMul` backward?
5. What exactly breaks if you forget `sumToShape` after a broadcasted `mul` — a crash, or something worse?
6. `topoSort` needed no changes at all for tensors. What does that tell you about which parts of an autograd engine depend on what a node contains?
7. Why is `grad` nullable here when it was `0` in Ch 08? What would it cost to use zero tensors instead?

---

## Further Reading

- [PyTorch internals — autograd](https://pytorch.org/blog/overview-of-pytorch-autograd-engine/) — production autograd; the same ideas with far more bookkeeping.
- [Parr & Howard — The Matrix Calculus You Need For Deep Learning](https://arxiv.org/abs/1802.01528) — derivations for matmul, sum, mean and softmax backward.
- [Justin Domke — Reverse-mode AD](https://people.cs.umass.edu/~domke/courses/sml/08autodiff_nnets.pdf) — clean lecture notes generalising 08b to tensors.

---

## Checkpoint

Part 2 is finished. You have a computation graph (Ch 08a), a backward pass (Ch 08b), an optimizer (Ch 09), and now an engine that operates on tensors rather than single numbers (Ch 10).

That combination is a working deep learning framework. Small, slow, and yours — but nothing in Part 3 through Part 6 adds a new *idea* to it. `Linear`, `LayerNorm`, attention, the whole transformer: they are all expressions built from the operations you just finished, differentiated by the `backward()` you just wrote, and trained by the `SGD` from the chapter before.

Prove it before moving on. Build `Z = (A @ B) + c` where `c` is a bias of shape `[1, n]` broadcast across the rows, take `mean()`, call `backward()`, and confirm that `A.grad`, `B.grad` and `c.grad` all come back with exactly their own shapes — with `c.grad` summed down the broadcast axis. That single expression exercises matmul backward, un-broadcasting, and reduction backward together, and it is a `Linear` layer in everything but name.

---

## Next Chapter

**[Activation Functions](../part-3-neural-net-primitives/ch-11-activation-functions.md)** — with tensor autograd in place, Part 3 begins building the layers. First the nonlinearities, which is where `relu`'s gate and `tanh`'s shrinking derivative stop being scalar curiosities and start deciding whether a deep network trains at all.
