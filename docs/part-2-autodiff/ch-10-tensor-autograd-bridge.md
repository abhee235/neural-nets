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
> | **Read** | 1 → 6 | **Build** `sumToShape`, `reshape`, `transpose`, `zeroGrad` (section 7) |
> | **Read** | 8 | **Build** `add`, `mul`, `matMul` (section 9) |
> | **Read** | 10 | **Build** `sum`, `mean`, `backward` (section 11) |
> | **Read** | — | **Verify** everything (section 12) |
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

## 2. Chapter 08's own graph, with tensors in the nodes

The only honest way to show what this chapter changes is to take something you have **already differentiated by hand** and redo it — changing what is inside the nodes and *nothing else*. Any other example would be changing two things at once, and you would not be able to tell which difference was the point.

So we use Chapter 08b's running example. You have seen this graph three times already:

```
    a ──┐
        ├──► [×] ──► c ──┐
    b ──┘                ├──► [+] ──► L
                         │
    d ────────────────────┘

    L = (a · b) + d          a = 2,  b = -3,  d = 10
```

And you already know every number in it, because you computed them in Ch 08b:

```
forward :  c = -6      L = 4
backward:  L.grad = 1   c.grad = 1   d.grad = 1   a.grad = -3   b.grad = 2
```

### The same graph, one tensor per node

Now the only change: every node holds a **`[2,3]` block** instead of one number. Same names, same values, same operations. `d` stays a **single row** — one bias shared by both rows, which is how a bias is always used:

```
A = [  2   2   2 ]        B = [ -3  -3  -3 ]        d = [ 10  10  10 ]
    [  2   2   2 ]            [ -3  -3  -3 ]

    shape [2,3]                shape [2,3]              shape [1,3]  ← one row
```

**Forward** — every operation is the tensor version of the one Ch 08 used:

```
C = A × B  =  [ -6  -6  -6 ]     ✓ Ch 08's c = -6, six times over
              [ -6  -6  -6 ]

Z = C + d  =  [  4   4   4 ]     ✓ Ch 08's L = 4, six times over
              [  4   4   4 ]        (d's single row is copied into both)

L = sum(Z) =  24                 ← collapse to one number, so backward()
                                    has a single thing to start from
```

**Backward** — now compare each gradient against the one you already know:

| node | Ch 08 gradient | Ch 10 gradient | |
|---|---|---|---|
| `C` | `1` | every entry `1` | **same** |
| `A` | `-3` (that is `b`) | every entry `-3` | **same** |
| `B` | `2` (that is `a`) | every entry `2` | **same** |
| `d` | `1` | **`[ 2  2  2 ]`** | **different** |

Read that table slowly, because it is the entire chapter.

**Three of the four are unchanged.** Not "analogous" — identical, element for element. `A.grad` is `-3` in every position exactly as the scalar `a.grad` was `-3`. The multiply still routes each operand its sibling's value; the add still copies its gradient to both parents. Every rule you wrote in Ch 08 carried over without modification, because every one of those nodes had its own gradient slot for every one of its numbers.

**One is different, and only one.** `d` was a single row, but it was *used in two rows*. So ask the Chapter 08 question: **how many times was `d[0]` — the number 10 — actually used?** Twice. And Ch 08 already told you what happens to a value used in several places:

> *its gradient is the sum of the contributions.*

```
d.grad[0] = Z.grad[0][0] + Z.grad[1][0] = 1 + 1 = 2
d.grad[1] = Z.grad[0][1] + Z.grad[1][1] = 1 + 1 = 2
d.grad[2] = Z.grad[0][2] + Z.grad[1][2] = 1 + 1 = 2
```

That summing-down-the-copies is `sumToShape` — the one genuinely new idea in the first half of the chapter, and you have just done it by hand using a rule you already had. No new mathematics appears anywhere in this section.

> **Why every entry is the same number here.** So the comparison with Ch 08 is exact, line for line. In a real tensor the entries differ from each other and the gradients differ with them — but not one rule changes. If you want to see that, change `A` to `[[1,2,3],[4,5,6]]` and re-run; `A.grad` stays all `-3`, and `d.grad` stays `[2,2,2]`.

> **Check this first.** Once `mul`, `add` and `sum` work, this is the first thing to run. Three gradients matching Ch 08 and `d.grad = [2,2,2]` with shape `[1,3]` means the hard part is done.

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

It is also a new *category* of bug. In Ch 08 a wrong gradient was a wrong number. Here you can have a perfectly-shaped tensor full of wrong numbers — shapes line up, nothing throws, and the loss even falls for a while. Section 12 is about not letting that happen.

---

## 4. Slow down: `d`'s gradient, one number at a time

Section 2 computed `d.grad = [2, 2, 2]` in three lines that looked almost the same. That's worth doing again, much more slowly — one bias number, completely by itself, before moving to the next.

Here is `d` again, and here is `Z.grad`, the tensor of gradients that arrived from upstream. Both are copied straight from section 2, so there is nothing new to set up:

```
d = [ 10  20  30 ]            shape [1,3]

Z.grad = [ 1  1  1 ]          shape [2,3]
         [ 1  1  1 ]
```

**Find `d.grad[0]` — the gradient for the number `10` — and nothing else.**

Forget that `d` has three numbers for a moment; pretend it only has this one. Ask exactly one question: *everywhere `d[0]` was used in the forward pass, which position of `Z.grad` is sitting there now?*

`d[0]` was added into the first column of `X` — that's `Z[0][0]` and `Z[1][0]`. Both of those positions in `Z.grad` hold `1`. So:

```
d.grad[0]  =  Z.grad[0][0]  +  Z.grad[1][0]
           =        1       +        1
           =  2
```

That's it. One number, computed from two cells you can point to.

**Now do `d.grad[1]` — the gradient for `20` — the exact same way, from scratch.**

`d[1]` was added into the *second* column of `X`: `Z[0][1]` and `Z[1][1]`. Both hold `1` in `Z.grad`:

```
d.grad[1]  =  Z.grad[0][1]  +  Z.grad[1][1]
           =        1       +        1
           =  2
```

**And `d.grad[2]`, for `30`, once more:**

`d[2]` lives in the third column: `Z[0][2]` and `Z[1][2]`. Both `1`:

```
d.grad[2]  =  Z.grad[0][2]  +  Z.grad[1][2]
           =        1       +        1
           =  2
```

Three separate questions, asked the same way three times: *which cells did this number feed, and what do their gradients add up to?* Put the three answers side by side and you get section 2's `d.grad = [2, 2, 2]` — the same result, but now you can see where every single one of those `2`s came from, rather than trusting a formula.

That question — *which cells did this number feed, and what is the sum there* — is the whole of what `sumToShape` automates. Nothing else is coming; the rest of this chapter is applying that one question in more places.

---

## 5. The mirror question: one gradient becomes several copies

Section 4 handled *broadcasting* — one small tensor copied into a bigger one. There is exactly one other way a shape changes in this chapter: *reduction*, where a bigger tensor is collapsed into a smaller one. `sum(x, axis)` does this, and it is what every loss function ends with.

Section 2's graph doesn't have a partial reduction to reuse — its only `sum` collapses everything to one number. So here is one small, fresh example, set up the same deliberate way: two rows, three columns, summed along the columns.

```
R = [ 3  1  5 ]          shape [2,3]
    [ 4  2  0 ]
```

**Forward — sum each row into one number:**

```
S = sum(R, axis=1)

S[0]  =  R[0][0] + R[0][1] + R[0][2]  =  3 + 1 + 5  =  9
S[1]  =  R[1][0] + R[1][1] + R[1][2]  =  4 + 2 + 0  =  6

S = [ 9 ]                shape [2,1]
    [ 6 ]
```

Three numbers went in, one came out — for each row. That is a reduction: several inputs feeding one output.

**Backward — say a gradient has already arrived for `S`, from whatever used it next.** For this example, just take it as given, the same way `Z.grad` was simply handed to you in section 4:

```
S.grad = [ 2 ]            shape [2,1]
         [ 5 ]
```

**Find `R.grad[0][0]`, `R.grad[0][1]` and `R.grad[0][2]` — the three numbers that fed `S[0]`.**

Each one contributed to `S[0]` by being added in with a coefficient of exactly 1 — that's what addition inside a sum does to every term. So each of the three gets *the same* gradient back: whatever `S[0]`'s gradient was.

```
R.grad[0][0]  =  S.grad[0]  =  2
R.grad[0][1]  =  S.grad[0]  =  2
R.grad[0][2]  =  S.grad[0]  =  2
```

Notice what this is **not**: it is not a sum of three things landing on one number, the way section 4 was. It is the opposite — *one* number, `S.grad[0] = 2`, handed out as a copy to three different places.

**Now row 1, the same way.** All three of `R[1][0]`, `R[1][1]`, `R[1][2]` fed `S[1]`, whose gradient is `5`:

```
R.grad[1][0]  =  S.grad[1]  =  5
R.grad[1][1]  =  S.grad[1]  =  5
R.grad[1][2]  =  S.grad[1]  =  5
```

Put it together:

```
R.grad = [ 2  2  2 ]      shape [2,3]   ✓ matches R
         [ 5  5  5 ]
```

**Section 4 and section 5, side by side, now that both are concrete rather than abstract:**

<p align="center">
  <img src="../assets/ch-10/two-examples-side-by-side.svg" alt="Two small worked examples shown side by side, matching the numbers already computed in the text. Left, labelled section 4, d gets copied then summed back: d holding 10, 20, 30 is copied into both rows of Z, shown as Z's two rows 11,22,33 and 14,25,36; an upward arrow sweeps across the three columns showing 1 + 1 = 2 being computed for each; d.grad comes out 2, 2, 2, captioned many cells summed to one. Right, labelled section 5, R gets summed then copied back: R holding rows 3,1,5 and 4,2,0 is summed along each row into S holding 9 and 6; a given upstream gradient S.grad of 2 and 5 is shown being copied three times across each row; R.grad comes out as row 0 equal to 2,2,2 and row 1 equal to 5,5,5, captioned one cell copied to many. A bottom bar states the same law applies both times — a value used in several places collects the sum of their gradients — noting that on the left there were three places to sum so the sum shows up, and on the right there was only one place per output so the sum is just that one term." />
</p>

*Figure 2: the two examples above, side by side. Same numbers you just computed by hand.*

| | what forward did | what backward does |
|---|---|---|
| section 4 (`d`) | copied **one** number into **many** cells | **summed** those many cells back into one gradient |
| section 5 (`R`) | **combined many** numbers into one cell | **copied** that one gradient into many gradients |

They are opposite operations, and they get opposite treatments — sum answers broadcasting, broadcasting answers sum. Both of them are still nothing but Chapter 08's rule — *a value used in several places collects the sum of their gradients* — because that rule is what forces section 4's answer to be a sum. Section 5's "copy" isn't a separate rule at all: a single term contributing to a single output has nothing to sum, so the "sum of contributions" is just that one contribution, handed back unchanged. Same law, and section 5 is the case where it only ever had one thing to add up.

---

## 6. `sumToShape` — turning section 4 into code

You have already done section 4's job by hand, for one specific pair of shapes: `[2,3]` down to `[1,3]`. `sumToShape` is that same question — *which cells did this number feed, and what do their gradients add up to* — written once so it works for any two shapes.

```
sumToShape(grad, targetShape)  →  grad, summed back down to targetShape
```

Every broadcasting operation's backward calls it. There are two different ways a forward pass can broadcast a shape up, and `sumToShape` needs to undo both. Take them one at a time — do not try to hold both in your head together.

### Case 1 — the target has fewer axes entirely

Sometimes a bias has no row axis at all — it is a flat list, not a `[1, n]` block. Say the gradient arriving is the same `[2,3]` block from section 4:

```
grad = [ 1  1  1 ]        shape [2,3]
       [ 1  1  1 ]

target shape: [3]         ← rank 1. There is no row axis to keep.
```

The ranks don't match — `grad` has 2 axes, the target has 1 — so the extra leading axis has to disappear completely. Sum it away:

```
result = [ 1+1  1+1  1+1 ]  =  [ 2  2  2 ]        shape [3]
```

Same numbers as section 4's `d.grad`, but notice the shape: `[3]`, not `[1,3]`. There was no size-1 row axis to preserve, because the target never had one.

### Case 2 — the target keeps its rank, but one axis was stretched from size 1

This is `d`'s actual situation. The target is `[1,3]` — rank 2, same as `grad` — and only the row axis shrank, from 2 down to 1. Take a fresh pair of numbers to see it clearly, since this case needs an axis with more than one row in the *target* to show the difference from Case 1:

```
grad = [ 1  1  1  1 ]     shape [3,4]
       [ 2  2  2  2 ]
       [ 3  3  3  3 ]

target shape: [3,1]       ← rank 2, same as grad. Axis 1 must shrink to size 1.
```

Sum along axis 1, and this time keep that axis alive as a size-1 axis rather than deleting it:

```
row 0:  1+1+1+1 = 4
row 1:  2+2+2+2 = 8
row 2:  3+3+3+3 = 12

result = [ 4 ]             shape [3,1]   ✓ matches target
         [ 8 ]
         [ 12 ]
```

**Now watch what happens if `keepDims` is left off.** The three numbers `4, 8, 12` come out exactly the same — but without `keepDims`, the summed axis is *deleted* instead of shrunk to size 1:

```
result = [ 4  8  12 ]      shape [3]   ✗ target was [3,1]
```

Same three numbers. Wrong shape. Nothing here throws an error — `[3]` and `[3,1]` both hold three numbers, so this bug will not announce itself where it happens. It surfaces later, as a shape mismatch somewhere else entirely, or as a silent, wrong re-broadcast. This is the single most annoying bug in the chapter, and now you have seen the exact moment it is born.

### Putting the two cases together

Do case 1 first: sum away whole leading axes until the ranks match. *Then* do case 2: walk the remaining axes one at a time, and wherever the target's size is 1 but `grad`'s isn't, sum that axis with `keepDims = true`.

And one case needs no work at all: if `grad.shape` already equals `targetShape`, return it unchanged. Every caller relies on that being safe, which is why none of them checks first.

**Checking your numbers against a taller version.** The exercise uses the same `d`-shaped situation as section 4, but with **four** rows instead of two: `sumToShape(ones([4,3]), [1,3])`. Walk it exactly like section 4's three separate questions, just with four cells to add per column instead of two — `d.grad[0] = 1+1+1+1 = 4`, and the same for columns 1 and 2. Result: `[4, 4, 4]`. Same question, asked on a taller tensor; no special case needed.

---

## 7. Build it (1) — the easy half

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

## 8. `matMul` backward — derive it, don't memorise it

The other genuinely new piece. For `Z = A B`:

$$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial Z}\,B^{\mathsf T}, \qquad \frac{\partial L}{\partial B} = A^{\mathsf T}\frac{\partial L}{\partial Z}$$

Those two lines are the workhorse of every backward pass from here to Chapter 30 — attention is mostly matmuls. So it is worth being able to reconstruct them rather than recall them.

<p align="center">
  <img src="../assets/ch-10/matmul-backward-shapes.svg" alt="A shape derivation for matmul backward. The forward pass shows A of shape [2,3] times B of shape [3,4] giving Z of shape [2,4], so the upstream gradient dZ is also [2,4]. Two panels then derive the backward shapes by constraint. For dA: it must come out [2,3] to match A, and starting from dZ at [2,4] the only factor that produces [2,3] is something shaped [4,3], which is B transposed — giving dA = dZ @ B-transpose. For dB: it must come out [3,4] to match B, and the only factor that produces that from dZ at [2,4] is something shaped [3,2] multiplied on the left, which is A transposed — giving dB = A-transpose @ dZ. A caption notes the transposes are not a trick to memorise, they are the only shapes that fit." />
</p>

*Figure 3: only one arrangement of transposes makes the shapes fit.*

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

## 9. Build it (2) — `add`, `mul`, `matMul`

**Milestone 5 — `add`.** Forward is Ch 03's broadcasting `add`. Backward is still a router: each parent receives the upstream gradient, wrapped in `sumToShape` for that parent's own shape.
✅ *Checkpoint:* `[2,3] + [2,3]` gives both gradients `[2,3]`. `[2,3] + [3]` gives the second gradient shape `[3]`, with each entry summed over the two rows.

**Milestone 6 — `mul`.** Structurally identical to Ch 08's — the switch still swaps the operands. Every `*` becomes an element-wise tensor `mul`, and each accumulation is wrapped in `sumToShape`.
> **Order matters here.** `mul(other.data, out.grad)` is *itself* a broadcasting operation, so its result has the broadcast shape, not the parent's. `sumToShape` wraps the product; it does not go on `out.grad` first. Reverse those and you will be multiplying mismatched shapes.

**Milestone 7 — `matMul`.** section 8's two formulas.
✅ *Checkpoint:* `[2,3] @ [3,4]` gives `[2,4]`, and after backward the gradients are `[2,3]` and `[3,4]` — each matching its own parameter. That is the shape invariant doing its job, and it is a strong signal you got the transposes the right way round.

---

## 10. Reductions backward — `sum` and `mean`

Section 5 already worked through the mechanism by hand — one gradient, copied out to every input that fed it. This section is the practical, `TensorValue.sum()`-shaped version of that same idea.

**`sum`.** Every input element contributed with coefficient 1, so every element gets the same upstream gradient — broadcast it back out, exactly as in section 5's `R.grad`.

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

## 11. Build it (3) — `sum`, `mean`, `backward`

**Milestone 8 — `sum` and `mean`.** section 10. Do `sum` first and get `mean` by scaling it.

**Milestone 9 — `backward`.** Line for line the Ch 08 version, with two substitutions:

```
topoSort(this)   →   topoSortTensor(this)
this.grad = 1    →   this.grad = ones(this.data.shape)
```

> **Guard the root.** Seeding with ones only means `∂L/∂L` when `L` is a single number. Called on a `[2,3]` node, it quietly computes the gradient of the *sum* of six entries — rarely what anyone intends. PyTorch refuses outright unless you pass an explicit gradient. Either throw when `data.size !== 1` or document the summing behaviour deliberately. This is the deeper reason every loss ends in a `sum` or a `mean`.

✅ *Checkpoint:* the whole of `grad.test.ts` except the gradient checks should now pass.

---

## 12. Verify it — and take this one seriously

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
