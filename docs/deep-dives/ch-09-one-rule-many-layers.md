# Deep Dive — One rule, many layers

> Companion to [Chapter 09: Gradient Descent](../part-2-autodiff/ch-09-gradient-descent.md).
> Read it once `SGD` works on the one-parameter bowl.

Chapter 09 teaches the update rule on a single parameter:

$$w_{\text{new}} = w_{\text{old}} - \eta \cdot \text{gradient}$$

That is honest, and it is genuinely the whole algorithm — but it leaves a fair question unanswered. A real network has parameters buried several layers deep, whose influence on the loss passes through other parameters, a nonlinearity, and possibly forty more layers on the way. **Does one line really cover that too?**

It does, and the reason is worth seeing rather than being told. So let's do the smallest thing that is honestly a *network* — two layers, four parameters, a nonlinearity in the middle — and run one complete step with every number worked out.

Every figure below was computed with the `Value` class you built in Ch 08. None of it is estimated.

---

## The network

One input, one hidden unit, one output:

$$z_1 = w_1 x + b_1, \qquad h = \tanh(z_1), \qquad z_2 = w_2 h + b_2, \qquad L = (z_2 - y)^2$$

Four parameters — `w1, b1, w2, b2` — and this time they are genuinely at different depths. `b2` sits right next to the loss. `w1` is at the far end, with a multiply, a tanh and another multiply between it and `L`.

In code, using nothing beyond Ch 08:

```typescript
const z1   = w1.mul(x).add(b1);
const h    = z1.tanh();
const z2   = w2.mul(h).add(b2);
const loss = z2.add(new Value(-y)).pow(2);
```

Starting values: `w1 = 0.5`, `b1 = 0`, `w2 = 1.0`, `b2 = 0`, with input `x = 1` and target `y = 1`. Learning rate `η = 0.1`.

---

## Step 1 — forward

```
z1 = 0.5 × 1 + 0    =  0.5
h  = tanh(0.5)      =  0.4621
z2 = 1.0 × 0.4621+0 =  0.4621
L  = (0.4621 − 1)²  =  0.2893
```

The model predicts `0.4621`; the target is `1`. It is wrong, and the loss says by how much.

---

## Step 2 — backward, and the part that actually matters

<p align="center">
  <img src="../assets/ch-09/gradient-through-layers.svg" alt="A two-layer network drawn left to right: parameters w1 and b1 feed z1 = w1·x + b1 which equals 0.5; z1 feeds h = tanh(z1) which equals 0.4621; parameters w2 and b2 feed z2 = w2·h + b2 which equals 0.4621; and z2 feeds the loss L = (z2−y)² which equals 0.2893. Below it, a backward row shows a pulse travelling right to left, seeded at 1 on the loss and picking up a factor at each node: ×2(z2−y) gives −1.0758 at z2, ×w2 = 1.0 leaves −1.0758 at h, ×(1−h²) = 0.7864 gives −0.8460 at z1, and ×x = 1.0 leaves −0.8460 at w1. Two panels compare: the gradients reach the parameters through different chain lengths — ∂L/∂b2 = −1.0758 in one factor, ∂L/∂w2 = −0.4971 in two, ∂L/∂w1 = −0.8460 in four — but a single loop, p.data -= 0.1 * p.grad, updates all of them, because step() never learns how deep a parameter was." />
</p>

One call to `loss.backward()` fills in all four gradients:

| Parameter | Gradient | How many factors got multiplied together |
|---|---:|---|
| `b2` | `−1.0758` | 1 |
| `w2` | `−0.4971` | 2 |
| `b1` | `−0.8460` | 3 |
| `w1` | `−0.8460` | 4 |

`b2` is adjacent to the loss, so its gradient is just `∂L/∂z2 = 2(z2 − y) = −1.0758`. Nothing else happens to it.

`w1` is four operations away, so its gradient is a **product of four local derivatives** — exactly the chain rule from Ch 07, applied by the machinery from Ch 08b:

$$\frac{\partial L}{\partial w_1} = \underbrace{\frac{\partial L}{\partial z_2}}_{-1.0758} \times \underbrace{\frac{\partial z_2}{\partial h}}_{w_2 = 1.0} \times \underbrace{\frac{\partial h}{\partial z_1}}_{1 - h^2 = 0.7864} \times \underbrace{\frac{\partial z_1}{\partial w_1}}_{x = 1.0} = -0.8460$$

Follow the pulse in the diagram: it starts at `1` on the loss and picks up one factor per node on the way back. Each node contributes only its own local derivative — the `tanh` node knows `1 − h²` and nothing else — and the accumulated product is the gradient.

**This is the answer to "does the simple rule scale?"** The complexity of being deep is real, but it is entirely contained inside `backward()`. Depth means *more factors in the product*, and nothing else.

---

## Step 3 — the update, which does not care about any of that

Here is the whole of `step()`, applied to all four parameters:

```
w1:  0.5  −  0.1 × (−0.8460)  =  0.5846
b1:  0.0  −  0.1 × (−0.8460)  =  0.0846
w2:  1.0  −  0.1 × (−0.4971)  =  1.0497
b2:  0.0  −  0.1 × (−1.0758)  =  0.1076
```

Read those four lines again and notice what is **not** there:

- No mention of layers.
- No special handling for the parameter that was four operations deep.
- No knowledge of `tanh`, or of the network's shape, or of which parameter feeds which.
- No ordering requirement — you could update them in any order, or in parallel.

By the time `step()` runs, every parameter is exactly two numbers: a value and a gradient. That is the entire interface. The loop in your `SGD.step()` is:

```
for (const p of this.params) {
  p.data -= this.learningRate * p.grad;
}
```

and it is *already* the version that trains a transformer. Not a simplified version of it — the same code. Going from four parameters to four billion changes the length of `this.params` and nothing else.

That separation is the piece worth taking away:

> **`backward()` handles depth. `step()` handles movement. Neither knows what the other is doing.**

It is why you were able to write a working optimizer in Chapter 09 without knowing what a transformer looks like — and why, in Chapter 30, you will not have to revisit it.

---

## Step 4 — repeat, and watch all four descend together

Running the same five stages twenty times:

```
step   0    loss 0.289318    w1=0.5000  b1=0.0000  w2=1.0000  b2=0.0000
step   1    loss 0.077791    w1=0.5846  b1=0.0846  w2=1.0497  b2=0.1076
step   2    loss 0.022981    w1=0.6232  b1=0.1232  w2=1.0823  b2=0.1634
step   3    loss 0.007023    w1=0.6428  b1=0.1428  w2=1.1015  b2=0.1937
step   4    loss 0.002179    w1=0.6533  b1=0.1533  w2=1.1125  b2=0.2104
step   5    loss 0.000681    w1=0.6591  b1=0.1591  w2=1.1187  b2=0.2198
step  10    loss 0.000002    w1=0.6659  b1=0.1659  w2=1.1263  b2=0.2310
step  20    loss 0.000000    w1=0.6663  b1=0.1663  w2=1.1268  b2=0.2317
```

Four parameters, all moving at once, from one `backward()` call per step, with nothing in the code aware that there is more than one of them. The loss falls from `0.289` to essentially zero in about ten steps.

Worth noticing: they move by **different amounts**, because they have different gradients. `b2` moves fastest at first (its gradient is the largest) and `w2` slowest. Nobody assigned those rates — they fall out of the chain rule, which is precisely what makes gradient descent an *algorithm* rather than a set of hand-tuned rules.

---

## What this predicts about deeper networks

Two consequences follow directly from "depth means more factors in the product", and both are things you will meet later in this course.

### Gradients shrink with depth when the factors are small

Compare `b2`'s gradient with `w1`'s: `−1.0758` versus `−0.8460`. The far parameter's is smaller, and you can see exactly where it went — the `tanh` node contributed a factor of `1 − h² = 0.7864`.

That factor is **always at most 1** for tanh, and it approaches 0 as `|z|` grows. One layer costs you 21% here. Ten layers of it multiply out to `0.7864¹⁰ ≈ 0.09`. Forty layers give `≈ 0.00003`. The early layers of a deep network would receive gradients so small that `step()` — correctly applying the rule — moves them by essentially nothing, and they never learn.

This is the **vanishing gradient problem**, and note that nothing is broken: `backward()` is right, `step()` is right, the chain rule is right. The product is simply small. The fixes are architectural — activations whose derivative doesn't shrink (ReLU, Ch 11), normalisation (Ch 20), and residual connections that give the gradient a path around the multiplications entirely (Ch 26). All three exist because of the arithmetic on this page.

### A parameter used twice collects both contributions

In this network each parameter is used once. In a real one — weight sharing, a token embedding referenced at several positions — a parameter appears in more than one path to the loss, and the chain rule says its total gradient is the **sum** over paths.

You already implemented that: it is the `+=` in every `_backward` closure from Ch 08b, and the reason `x.mul(x)` gives `x.grad = 2x` rather than `x`. Nothing in `step()` changes for it either. The parameter still arrives holding one number; that number was just assembled from more places.

---

## Try it yourself

Two experiments that take a minute each and pay for themselves:

1. **Change `w2` from `1.0` to `0.1`** and re-run one backward pass. `w1`'s gradient drops by a factor of ten, because `∂z2/∂h = w2` is one of the four factors. This is the mechanism behind initialisation schemes (Ch 13): the scale of *later* weights directly controls how much gradient reaches *earlier* ones.

2. **Add a second hidden layer** — another `mul`, `add` and `tanh` — and print all six gradients. The parameters in the new first layer will have the smallest gradients of all, and the chain for them will be six factors long. Your `SGD` will need no changes whatsoever, which is the point of this whole document.

---

## Further reading

- [Deep dive: how big a step can you take?](ch-09-how-big-a-step.md) — the other half: given these gradients, how large may `η` be before the loop diverges.
- [Deep dive: why reverse-mode wins](ch-08-why-reverse-mode.md) — why one backward pass fills all four gradients, rather than four passes filling one each.
- [Goodfellow et al. — Deep Learning, §8.2](https://www.deeplearningbook.org/contents/optimization.html) — the formal treatment of what makes deep optimisation hard.
