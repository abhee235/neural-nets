# Deep Dive — One rule, many layers

> Companion to [Chapter 09: Gradient Descent](../part-2-autodiff/ch-09-gradient-descent.md).
> Read it once `SGD` works on the one-parameter bowl.

Chapter 09 teaches the update rule on a single parameter:

$$w_{\text{new}} = w_{\text{old}} - \eta \cdot \text{gradient}$$

and then asserts it scales to a network. That is a lot to take on trust. A real model has parameters buried several operations deep, whose effect on the loss travels through other parameters and through nonlinearities. **Does one subtraction really cover that?**

Rather than argue, let's run it — on a graph you have already differentiated three separate ways.

---

## The graph you already know

Chapter 08 used one running example throughout: forward mode walked it twice, backward mode walked it once, and you hand-derived every gradient on it.

$$f(a, b) = b\sin(a) + b^2$$

Split into one operation per line, exactly as in 08b:

$$u = \sin(a), \qquad v = b\,u, \qquad w = b^2, \qquad f = v + w$$

```
a ──► [sin] ──► u ──┐
                    ├──► [×] ──► v ──┐
b ──────────────────┘                ├──► [+] ──► f
│                                    │
└────────► [b·b] ─────────────► w ───┘
```

Until now this was something to differentiate. Now it becomes something to **minimise** — `a` and `b` stop being inputs and become *parameters*, the two numbers we are allowed to change.

And notice that this graph is a better test than it first appears, because its two parameters reach the output in genuinely different ways:

| Parameter | Gradient (you derived these in 08b) | How it got there |
|---|---|---|
| `a` | `∂f/∂a = b·cos(a)` | a **chain** — the gradient passes *through* the `sin` node and picks up its local derivative |
| `b` | `∂f/∂b = sin(a) + 2b` | a **sum** — `b` reaches `f` by two separate paths, and their contributions add |

That is exactly the pair of complications a deep network presents: gradients that travel through intermediate operations, and parameters that are used more than once. If one update rule handles both here, the question is settled.

---

## Where does this thing even have a minimum?

Worth pinning down before descending, because `b·sin(a) + b²` is not obviously bowl-shaped.

Set both gradients to zero:

$$b\cos(a) = 0 \qquad\text{and}\qquad \sin(a) + 2b = 0$$

The interesting solutions come from `cos(a) = 0`, i.e. `a = π/2`, which forces `b = −sin(a)/2 = −0.5`:

$$f(\pi/2,\; -0.5) = (-0.5)(1) + 0.25 = \boxed{-0.25}$$

There is also a solution at `a = 0, b = 0`, where `f = 0` and **both gradients vanish** — but it is not a minimum. It is a saddle point, and it is about to cause visible trouble.

---

## One step, in full

Start at `a = 1.0`, `b = 3.0`, with `η = 0.1`.

*(Why not the 08b starting point `a = π/2, b = 3`? Because it is a special case worth seeing separately — see the box at the end. Starting a little to the side lets both parameters move.)*

**Forward:**

```
u = sin(1.0)          =  0.8415
v = 3 × 0.8415        =  2.5244
w = 3²                =  9
f = 2.5244 + 9        = 11.5244
```

**Backward** — one call, both gradients:

```
∂f/∂a = b·cos(a)     = 3 × 0.5403        =  1.6209     ← through the chain
∂f/∂b = sin(a) + 2b  = 0.8415 + 6        =  6.8415     ← two paths, summed
```

**Update** — and here is the entire point of this document:

```
a  ←  1.0  −  0.1 × 1.6209  =  0.8379
b  ←  3.0  −  0.1 × 6.8415  =  2.3159
```

New `f = 7.0844`. The loss fell from 11.52 to 7.08 in one step.

Now read those two update lines again and notice what is **not** in them. One of those gradients was assembled by chaining through a `sin` node; the other by summing two independent paths. The update does not know that, and could not act on it if it did. Both parameters arrive at `step()` as exactly two numbers — a value and a gradient — and get the same single line applied.

> **`backward()` handles structure. `step()` handles movement. Neither knows what the other is doing.**

That separation is the whole answer to "does the one-parameter rule scale?". Depth and reuse make the *gradient* harder to compute. They do not make the *update* harder at all.

---

## Twenty steps, then two hundred

<p align="center">
  <img src="../assets/ch-09/bsin-descent.svg" alt="The (a,b) parameter plane with the descent path for f(a,b) = b·sin(a) + b², starting at (1.0, 3.0) with learning rate 0.1. The point sweeps down and to the left through (0.838, 2.316), (0.683, 1.778), (0.545, 1.360), (0.335, 0.787) and (0.099, 0.193), approaching a dashed red circle marking the saddle point at (0,0) where both gradients are zero and f = 0. At step 25 it reaches (0.031, −0.009) and at step 50 it has only crawled to (0.083, −0.034) — twenty-five iterations that barely move, annotated as near-zero gradient. It then escapes, curving right and down through (0.574, −0.234) at step 100 to reach the green minimum marker at (π/2, −0.5) where f = −0.25 by step 200. A side panel recalls the two gradients from Chapter 08b: ∂f/∂a = b·cos(a), a chain through the sin node, and ∂f/∂b = sin(a) + 2b, a sum over b's two paths — the 1 + 6 = 7 computed by hand in that chapter." />
</p>

*Figure: both parameters descending together. Every point computed, not sketched.*

```
step   0    f = 11.524413    a = 1.0000   b =  3.0000
step   1    f =  7.084423    a = 0.8379   b =  2.3159
step   2    f =  4.284883    a = 0.6830   b =  1.7784
step   3    f =  2.553307    a = 0.5450   b =  1.3596
step   5    f =  0.877934    a = 0.3346   b =  0.7871
step  10    f =  0.056520    a = 0.0991   b =  0.1934
step  25    f = -0.000204    a = 0.0313   b = -0.0093     ← ???
step  50    f = -0.001655    a = 0.0827   b = -0.0342     ← 25 steps, almost nothing
step 100    f = -0.072337    a = 0.5744   b = -0.2335
step 200    f = -0.249968    a = 1.5595   b = -0.4999     ← arrived
```

Both parameters descend together, from one `backward()` call per step, with nothing in the code aware that there are two of them or that they reach `f` by different routes. They move by *different amounts* every step, because their gradients differ — and nobody assigned those rates. They fall out of the chain rule.

### The stall in the middle is not a bug

Look at steps 25 to 50. The loss goes from `-0.000204` to `-0.001655`. Twenty-five full iterations produce almost no progress.

That is the **saddle point at `(0, 0)`** — the second critical point we found earlier. The trajectory heads almost straight into it, and near it both gradients are tiny, so the updates are tiny. `step()` is behaving perfectly; it is faithfully multiplying a learning rate by a gradient of roughly `0.01`.

Then it escapes. `a = 0` is unstable — the surface still falls away in the `a` direction — so the small residual gradient slowly pushes the point off the ridge, the gradient grows, and it accelerates down to the true minimum.

This is worth having seen once, because it is exactly the failure mode that Chapter 09 section 12 claims is the realistic one in high dimensions. Here it is, in a two-parameter function, on the course's own running example:

- A loss curve that flattens **does not** mean you have converged.
- It may mean you are near a saddle, and patience will get you out.
- Momentum helps here for precisely the reason section 10 gives: accumulated velocity carries the point across a flat region that instantaneous gradients cannot.

---

## What this predicts about deeper networks

The graph above has one nonlinearity. Stack more and one term in that chain repeats.

`∂f/∂a` picked up a factor of `cos(a)` on its way through the `sin` node. That factor is at most 1, and near `a = π/2` it is close to **zero** — which is why the gradient reaching `a` can be far smaller than the gradient reaching `b`, even though both parameters matter equally to the answer.

Now imagine ten such nodes in a row, as a ten-layer network has. The gradient reaching the first layer is a product of ten local derivatives. If each is around `0.79` — the value `1 − tanh²` takes at a typical activation — then

$$0.79^{10} \approx 0.09, \qquad 0.79^{40} \approx 0.00003$$

The earliest layers receive gradients thousands of times smaller than the last ones, and `step()` — correctly applying the rule — barely moves them. This is the **vanishing gradient problem**, and note that nothing is broken: `backward()` is right, `step()` is right, the chain rule is right. The product is simply small.

Three later chapters exist because of that multiplication:

- **ReLU (Ch 11)** — a derivative of exactly `1` on the active side, so the factor stops shrinking.
- **LayerNorm (Ch 20)** — keeps activations in the range where derivatives are healthy.
- **Residual connections (Ch 26)** — give the gradient a path *around* the multiplications entirely.

You now know what all three are for before meeting any of them.

---

## The 08b starting point is a special case

If you start exactly where Chapter 08b did — `a = π/2`, `b = 3` — something instructive happens. The gradients there are the ones you hand-derived:

```
f      = 12
∂f/∂a  = 3·cos(π/2) = 0
∂f/∂b  = sin(π/2) + 6 = 1 + 6 = 7
```

`a`'s gradient is exactly zero. So `step()` leaves `a` completely alone — and since `∂f/∂a = b·cos(π/2) = 0` for *any* value of `b`, it stays zero forever. **`a` never moves again.** Only `b` descends:

```
step   0    f = 12.000000    a = π/2    b =  3.0000
step   1    f =  7.590000    a = π/2    b =  2.3000
step   2    f =  4.767600    a = π/2    b =  1.7400
step   3    f =  2.961264    a = π/2    b =  1.2920
step  30    f = -0.249981    a = π/2    b = -0.4957
step  60    f = -0.250000    a = π/2    b = -0.5000
```

It still reaches `f = −0.25`, because `a = π/2` happened to be the right value already. But it got there with one parameter frozen throughout.

That is a real phenomenon and not a quirk of this function: **a parameter with zero gradient does not move, and if its gradient stays zero it never learns anything.** You will meet the same thing under a different name in Chapter 11, when a ReLU unit lands on its inactive side and receives exactly zero gradient forever — the "dying ReLU". Same mechanism, same consequence.

---

## Try it yourself

The one obstacle: `Value` has no `sin()` method — Chapter 08a's operation table stops at `add`, `mul`, `pow`, `exp`, `log`, `tanh`, `relu`. Adding it is a five-line exercise following the `exp` pattern exactly, with local derivative `∂/∂x sin(x) = cos(x)`. Do that and the whole of this document runs through your own autograd engine.

Then:

1. **Start at `a = π/2, b = 3`** and confirm `a.grad` is `0` on every single step. Print it — watching a parameter sit still while the loss falls is worth seeing.
2. **Start at `a = 1.0, b = 3`** and print the loss every 10 steps. Find the stall yourself, then try `SGDMomentum` with `β = 0.9` on the same start and see how many steps it saves crossing it.
3. **Start at `a = 3.0, b = -2`** and see which of the two minima it finds. There is another at `(3π/2, 0.5)`, also with `f = −0.25`. Which one you reach depends entirely on where you begin — that is the local-minimum story from section 12, in a function small enough to hold in your head.

---

## Further reading

- [Deep dive: how big a step can you take?](ch-09-how-big-a-step.md) — the other half: given these gradients, how large may `η` be before the loop diverges.
- [Deep dive: why reverse-mode wins](ch-08-why-reverse-mode.md) — why one backward pass fills both gradients rather than one pass each.
- [Goodfellow et al. — Deep Learning, section 8.2](https://www.deeplearningbook.org/contents/optimization.html) — saddle points, plateaus, and why they dominate local minima in high dimensions.
