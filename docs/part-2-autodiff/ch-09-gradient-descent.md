# Chapter 09: Gradient Descent

> **Part 2 of 6 — Autodiff Engine**
> Source: [`src/optim/sgd.ts`](../../src/optim/sgd.ts)
> Tests: [`src/optim/sgd.test.ts`](../../src/optim/sgd.test.ts)
> Exercise: [`exercises/ch-09-gradient-descent.ts`](../../exercises/ch-09-gradient-descent.ts)

---

## Where we left off (and why this chapter exists)

Chapter 08 built an engine that answers one question: *"how does the loss respond to each parameter?"* Call `backward()` and every node in the graph ends up holding its own `∂L/∂θ`.

And then… nothing happens. The gradients sit there. `value.test.ts` checks that they are *correct*, but nothing in the library has ever changed a parameter. The model cannot learn, because knowing which way is downhill is not the same as walking.

This chapter is the walk, and it is startlingly small:

$$\theta \leftarrow \theta - \eta \cdot \frac{\partial L}{\partial \theta}$$

One subtraction per parameter. That is the whole of vanilla gradient descent, and it is the same rule that trains the GPT in Chapter 30 — the only thing that changes between here and there is how many parameters are in the list.

So why give it a chapter? Because this is the first code that **closes the loop**. Forward, loss, backward, update, repeat. Every warning you have been given since Ch 08 — accumulate with `+=`, zero between steps, don't grow the graph — has been theoretical until now. Here they become bugs you can watch happen. The rule takes ten minutes; the loop around it is what takes the chapter.

> **🗺️ Your path through Chapter 9**
> 1. Read to the end of *Concepts*, then implement `SGD` in [`sgd.ts`](../../src/optim/sgd.ts) — three tiny methods.
> 2. Run `bun test src/optim/sgd` until the `SGD` block is green.
> 3. Come back for *Momentum*, implement `SGDMomentum`, get the rest green.
> 4. Run the exercise and **predict each number before you look at it**.
> 5. Victory lap: [how big a step can you take?](../deep-dives/ch-09-how-big-a-step.md) — derives the exact learning rate at which this chapter's bowl starts to diverge.

---

## Learning Goals

By the end of this chapter you can:

- Apply `θ ← θ − η∇L` to one parameter and to a list of parameters, by hand and in code.
- Write the five-stage training loop from memory, and say what breaks if you reorder it.
- Explain why the parameter update must happen *outside* the computation graph.
- Predict — not just observe — whether a given learning rate will converge, oscillate, or diverge.
- Explain what momentum stores between steps, and why that state forces a particular API.
- Diagnose the three classic failures: a loss that climbs, a loss that never moves, and a model that trains but never improves.

---

## Words we'll use in this chapter

| Word | Plain meaning |
|------|---------------|
| **optimizer** | The object that owns the parameters and knows how to move them. |
| **parameter** | A `Value` the model is allowed to change — a weight. Contrast with an input, which is fixed. |
| **learning rate** (`η`, "eta") | Step size. How far to move per unit of gradient. |
| **step** | One application of the update rule to every parameter. |
| **training loop** | forward → loss → backward → step → zero, repeated. |
| **convergence** | The loss settling near a minimum and staying there. |
| **divergence** | The loss growing without bound — almost always too large a learning rate. |
| **velocity** | Momentum's running memory of recent steps; state that survives between steps. |
| **hyperparameter** | A number *you* choose (η, β) rather than one the model learns. |

---

## Intuition First — you already know this

You are standing on a hillside in thick fog. You cannot see the valley, but you can feel the ground under your feet, and that tells you which direction is downhill. So: face downhill, take a step, feel again, repeat.

That is gradient descent, completely. The gradient is the feel of the slope. The learning rate is how long a stride you take. And the fog is real — the model never sees the whole loss surface, only the slope at the one point it happens to be standing on.

The only judgement call is stride length, and it is genuinely a trade-off rather than a thing to get right. Tiny strides are safe but you will still be on the hillside at nightfall. Huge strides cross the valley entirely and land you partway up the opposite slope — higher than where you started.

---

## The Mental Model — three strides, one hill

<p align="center">
  <img src="../assets/ch-09/learning-rate-regimes.svg" alt="Three panels, each showing the same bowl-shaped loss L(w)=(w−5)² with a ball starting at w=0 and the minimum at w=5. Left panel, eta 0.05: the ball inches down the slope, reaching only w=2.05 after six steps. Middle panel, eta 0.2: the ball takes a large first step then progressively smaller ones, settling near the minimum at w=4.61. Right panel, eta 1.1: a single step carries the ball straight past the minimum and up the far wall of the bowl to a point higher than it started, with a dashed arrow showing it continuing to −2.2 then 13.6 and escaping. A caption notes the error shrinks by exactly the factor |1−2eta| each step, so the method converges only when eta is below 1, and eta=0.5 reaches the answer in a single step." />
</p>

*Figure 1: same bowl, same starting point, three learning rates. The left and middle panels differ only in patience. The right one is qualitatively different — it is not "slower", it is broken, and no amount of extra steps will fix it.*

The caption states something worth pausing on: on this particular bowl the outcome is not a matter of taste. The error is multiplied by exactly `|1 − 2η|` every step, so **η < 1 converges and η > 1 diverges**, and you can prove it in three lines. The [deep dive](../deep-dives/ch-09-how-big-a-step.md) does exactly that, and then shows why the threshold shrinks as you add training data.

---

## Concepts

### The update rule, symbol by symbol

$$\theta_{\text{new}} = \theta_{\text{old}} - \eta \cdot \frac{\partial L}{\partial \theta}$$

- $\theta$ — one parameter. In code, `param.data`.
- $\partial L/\partial\theta$ — its gradient. In code, `param.grad`, put there by `backward()`.
- $\eta$ — the learning rate, a number you choose.
- **The minus sign** — the only part with any content. The gradient points in the direction that *increases* the loss, so to decrease it you move the opposite way. Flip that sign and you have written gradient *ascent*: a perfectly good algorithm for maximising the loss, and a bug that is very easy to stare straight through.

Notice the update is **proportional to the gradient**. Nobody has to tell the algorithm to slow down near the bottom — the gradient shrinks as the slope flattens, so the steps shrink automatically. In Figure 1's middle panel the steps go 2.0, 1.2, 0.72, 0.43: each exactly 0.6× the last, with no braking logic anywhere.

### The training loop

```
repeat:
  1. forward     — run the model, building a fresh graph
  2. loss        — collapse the result to ONE number
  3. backward()  — fill in every parameter's .grad
  4. opt.step()  — θ ← θ − η·∇L for every parameter
  5. opt.zeroGrad() — clear the gradients for the next round
```

<p align="center">
  <img src="../assets/ch-09/training-loop.svg" alt="A ring of five stages with a highlight travelling around it: forward (build the graph, Ch 08a), loss (one number: how wrong), backward() (fill every .grad, Ch 08b), opt.step() (theta moves against the gradient, Ch 09), and opt.zeroGrad() (or the next step is wrong). The centre reads 'repeat until the loss stops falling'. A callout warns that stages 3 and 4 must not be separated, because zeroing gradients between them erases the gradients before step() can read them. A footer notes a fresh graph is built every iteration, so backward() is never re-run on the previous step's graph." />
</p>

*Figure 2: the loop. Stages 1–3 are Chapter 08; stages 4–5 are what you are building now.*

Three things about the ordering are load-bearing:

**The loss must be a single number.** `backward()` seeds `∂L/∂L = 1` on the node you call it from. That only means anything if that node is *the* output. A loss that is still a list of per-example errors has no single "the loss" to differentiate — which is why every loss function in Ch 12 ends in a sum or a mean.

**A fresh graph every iteration.** Step 1 rebuilds it from scratch. You never call `backward()` twice on the same graph — Ch 08's `zeroGrad` example showed why: gradients would compound, giving −3, then −9, then −18.

**`zeroGrad` goes after `step`, not between `backward` and `step`.** Both `backward → step → zeroGrad` and `zeroGrad → backward → step` are correct. Slipping it between stages 3 and 4 erases the gradients before `step()` reads them, so the parameters never move — everything runs, no error appears, and the loss simply sits there.

### The update happens *outside* the graph

This is the one genuinely counter-intuitive idea in the chapter.

You have spent Ch 08 learning that every operation on a `Value` should record itself. So it feels natural to write the update using those same operations. **Don't.** The update must reach past the graph and change `param.data` directly.

Two things break otherwise. First, `add`/`mul` return a *new* node — so the parameter your model still points at would be untouched, and nothing would ever train. Second, even if you worked around that, every step would append more nodes to a graph that is never released; after a thousand iterations you are carrying a thousand steps of history and the process runs out of memory.

The distinction is worth naming: the forward pass is **computation**, and belongs in the graph. The update is **bookkeeping about a parameter**, and does not. PyTorch draws the same line with `with torch.no_grad():` around its optimizer step.

### The API: the optimizer owns its parameters

```
const opt = new SGD([w, b], 0.1);   // hand it the parameters ONCE
…
loss.backward();
opt.step();                          // no arguments — it knows
opt.zeroGrad();
```

This mirrors PyTorch's `torch.optim.SGD(model.parameters(), lr=0.1)`, which is what `sgd.ts` means when its header claims equivalence.

It is not just cosmetic. `SGDMomentum` has to remember a velocity **per parameter** between steps, and state like that needs somewhere stable to live. An optimizer handed a fresh array on every call has no way to know that the third element this time is the same parameter as the third element last time. Owning the list solves it once, for every optimizer in the course — including Adam in Ch 14, which keeps *two* running numbers per parameter.

### Momentum — giving the ball mass

Vanilla SGD is memoryless: it goes wherever the current slope points and forgets everything it has ever done. Momentum keeps a running **velocity**:

$$v \leftarrow \beta v - \eta\,\nabla L, \qquad \theta \leftarrow \theta + v$$

with $\beta$ (usually 0.9) deciding how much of the previous velocity survives.

<p align="center">
  <img src="../assets/ch-09/momentum-vs-vanilla.svg" alt="One bowl, L(w)=(w−5)², with two balls run for six steps from w=0 at learning rate 0.1. The blue vanilla SGD ball moves 0, 1.0, 1.8, 2.44, 2.95, 3.36 — each step 0.8 times the last, decaying as it goes. The purple momentum ball with beta 0.9 moves 0, 1.0, 2.7, 4.69, 6.54, 7.90: it reaches the minimum by step three and then sails past it up the far side, marked 'overshoots — inertia, not a bug'. A panel shows the velocity growing 0, 1.0, 1.7, 1.99 even as the gradient shrinks. The caption notes contributions add along a consistent slope, up to about 1/(1−beta) = 10x, and cancel across an oscillation." />
</p>

*Figure 3: the same six steps, with and without inertia.*

Two consequences, and the second is the important one:

- **Along a consistent direction, contributions accumulate.** Watch the velocity panel: the gradient is shrinking, yet `v` keeps growing — 1.0, 1.7, 1.99. Sustained agreement compounds into a step up to about $1/(1-\beta) = 10\times$ larger than vanilla would take.
- **Across an oscillation, contributions cancel.** In a narrow valley where the gradient flips sign every step, consecutive contributions subtract instead of adding, damping the bouncing.

That damping — not the speed — is why momentum survives into every modern optimizer. And note that Figure 3 is honest: this momentum run *overshoots*, badly. β = 0.9 is under-damped on a bowl this simple. That is not a bug in the diagram; it is the trade-off, and it is why Ch 14 spends real effort on tuning.

### The loss landscape, and why local minima matter less than you'd think

For one parameter the loss is a curve; for two, a surface; for GPT's parameters, a surface in a space with more dimensions than you can picture. Gradient descent walks downhill on it and stops where the slope is flat — which need not be the lowest point anywhere.

- **Global minimum** — the lowest loss achievable. Rarely reached, and rarely needed.
- **Local minimum** — a valley that isn't the deepest. Descent can settle here.
- **Saddle point** — downhill in one direction, uphill in another. The gradient is near zero, so progress stalls without you having arrived anywhere.
- **Plateau** — a large flat region. Tiny gradients, tiny steps, training appears frozen.

The textbook worry is getting trapped in a bad local minimum. In practice, for the kind of networks this course builds, that turns out not to be the thing that bites: in very high dimensions, a point is only a local minimum if the surface curves upward in *every one* of thousands of directions, and that is rare. Saddle points and plateaus are far more common, and most local minima that do exist are about as good as each other. Your training runs will fail for other reasons — learning rate, initialisation, a sign error — long before they fail for this one.

---

## What to Implement

| Symbol | Description |
|---|---|
| `SGD` constructor | Store the parameter list and the learning rate |
| `SGD.step()` | Apply `θ ← θ − η·∇L` to every owned parameter |
| `SGD.zeroGrad()` | Reset every owned parameter's `.grad` to 0 |
| `SGDMomentum` constructor | As above, plus `momentum` (default 0.9) and a velocity array allocated **once** |
| `SGDMomentum.step()` | Update each velocity, then move each parameter by it |
| `SGDMomentum.zeroGrad()` | As `SGD.zeroGrad` |

The per-method JSDoc in [`sgd.ts`](../../src/optim/sgd.ts) carries the recipe, the pitfalls, and hand-computed trajectories for both classes. No implementation is given — that is the chapter.

---

## Build Order — milestones with checkpoints

**Milestone 1 — `SGD` constructor.** Store `params` and `learningRate`.
✅ *Checkpoint:* `new SGD([w], 0.1).params.length === 1`, and `params[0]` is the *same object* as `w` — not a copy. If it is a copy, everything below will run and nothing will learn.

**Milestone 2 — `step()`.** One loop, one subtraction, straight onto `.data`.
✅ *Checkpoint:* by hand on `L = (w−5)²` from `w = 0` at `η = 0.1`, where the gradient is `2(w−5)`:

```
step 1:  grad = -10    w ← 0    - 0.1·(-10)  = 1.0
step 2:  grad =  -8    w ← 1.0  - 0.1·(-8)   = 1.8
step 3:  grad = -6.4   w ← 1.8  - 0.1·(-6.4) = 2.44
```

Work those three by hand first. If your code disagrees, print `w.data`, `w.grad`, and `learningRate` — one of the three is wrong, and the trace tells you which.

**Milestone 3 — `zeroGrad()`.** Reset every owned parameter.
✅ *Checkpoint:* run two full loop iterations. Without `zeroGrad`, step 2's update is visibly larger than it should be; with it, the trace above reproduces exactly.

**Milestone 4 — the loop.** Assemble forward → loss → backward → step → zeroGrad and minimise `(w−5)²` for 100 iterations.
✅ *Checkpoint:* `w ≈ 5.0`. Then sweep the learning rate and check the predictions from Figure 1: `0.05` crawls, `0.2` converges cleanly, `0.5` arrives in **one step**, `0.9` oscillates but still converges, `1.1` diverges.

**Milestone 5 — `SGDMomentum`.** Velocity allocated in the constructor, updated before the parameter moves.
✅ *Checkpoint:* same bowl, `η = 0.1`, `β = 0.9`:

```
step 1:  grad = -10     v = 0.9·0    - 0.1·(-10)  = 1.0    w = 1.0
step 2:  grad =  -8     v = 0.9·1.0  - 0.1·(-8)   = 1.7    w = 2.7
step 3:  grad = -4.6    v = 0.9·1.7  - 0.1·(-4.6) = 1.99   w = 4.69
```

Then the algebraic check that catches the classic bug: **`momentum = 0` must behave exactly like vanilla `SGD`**. With a zero coefficient the previous velocity is discarded and the rule collapses to `θ -= η·grad`. If your two classes disagree there, the velocity is being handled wrongly — most likely allocated inside `step()`, where it resets to zero every call.

---

## Common Pitfalls

- **Building the update out of `Value` operations.** Returns a new node, leaves the model pointing at the old one, and grows the graph forever. Assign to `.data`.
- **Forgetting `zeroGrad`.** Gradients compound — Ch 08 measured it: −3, −9, −18. The loss climbs and it looks exactly like too high a learning rate.
- **Calling `zeroGrad` between `backward()` and `step()`.** Everything runs, nothing moves.
- **Copying parameters instead of holding references.** `step()` updates the copies. Silent, and indistinguishable from a learning rate near zero.
- **Allocating momentum's velocity inside `step()`.** Silently degrades to vanilla SGD, which still converges — so only a test that inspects the velocity catches it.
- **Guessing the learning rate.** Sweep on a log scale — `1e-4, 1e-3, 1e-2, 1e-1` — rather than nudging. And read the [deep dive](../deep-dives/ch-09-how-big-a-step.md): on a quadratic you can *derive* the threshold instead.
- **Declaring victory after one step.** Print the loss every N iterations. A number that went down once has told you nothing.
- **Summing the loss instead of averaging it.** Curvature then scales with dataset size, so the safe learning rate shrinks as you add data. This is why E4 needs `0.01` where E1 was fine at `0.1`.

---

## How to Verify

```bash
bun test src/optim/sgd
```
```bash
bun run exercises/ch-09-gradient-descent.ts
```

The exercise is the real gate, and it is worth using properly: **predict every printed number before you run it.** The bowl is `(w−5)²` from `w = 0`, so `wₙ = 5 − 5(1−2η)ⁿ` — which means you can compute the expected output of each call on paper. `minimise(0.01, 200)` should print `4.912`, and both `minimise(0.1, 100)` and `minimise(0.9, 100)` should print `5.000`. Matching a trajectory you predicted is a far stronger signal than watching a number go down.

Part 3's gate arrives next chapter, but the habit starts here: a loss curve that decreases is necessary, not sufficient.

---

## Self-Check Questions

1. `L = w²`, so `∇L = 2w`. With `w = 3` and `η = 0.1`, what is `w` after one step? After ten? (The multiplier is constant — you should not need to iterate by hand.)
2. Why must the parameter update write to `.data` rather than go through `add`? Name both failure modes.
3. What exactly goes wrong if you call `zeroGrad()` between `backward()` and `step()`? What does the loss curve look like?
4. On `L = (w−5)²`, why does `η = 0.5` reach the minimum in a single step? What property of this loss makes that possible, and why doesn't it work for a general loss?
5. Momentum with `β = 0.9` overshot the minimum in Figure 3. Is that a bug? What would you change to stop it, and what would you give up?
6. Why does the optimizer need to own its parameter list, rather than receiving it at each `step()` call? Answer in terms of `SGDMomentum` specifically.
7. E4 sums the squared error over five points and needs `η = 0.01`, while E1 uses `η = 0.1` on one parameter. Where does that factor of 10 come from — and what would you change so the learning rate stops depending on how much data you have?

---

## Further Reading

- [Sebastian Ruder — An overview of gradient descent optimization algorithms](https://ruder.io/optimizing-gradient-descent/) — the tour of SGD variants; several land in Ch 14.
- [3Blue1Brown — Gradient descent, how neural networks learn](https://www.3blue1brown.com/lessons/gradient-descent) — the loss landscape, visually.
- [Distill — Why Momentum Really Works](https://distill.pub/2017/momentum/) — interactive, and the best explanation of the damping effect anywhere.
- [Bottou, Curtis & Nocedal — Optimization Methods for Large-Scale Machine Learning](https://arxiv.org/abs/1606.04838) — the rigorous treatment.

---

## Checkpoint

You now have a complete, working learning system: a computation graph (Ch 08a), an automatic backward pass (Ch 08b), and parameters that move (Ch 09). Everything from here scales that up — it does not replace it.

Prove it to yourself on two parameters. Minimise

$$L = (w_1 - 3)^2 + (w_2 + 1)^2$$

and watch `w₁ → 3` and `w₂ → -1`. Both parameters descend in the same loop, from one `backward()` call, with no code that knows there are two of them. That is the property that makes the next twenty-one chapters possible.

---

## Next Chapter

**[Tensor Autograd Bridge](ch-10-tensor-autograd-bridge.md)** — everything so far has been one number per node, which is unusable at scale: a single transformer weight matrix would be hundreds of thousands of `Value`s. Ch 10 lifts this identical machinery onto tensors, where one node holds an entire matrix and one gradient holds an entire matrix of derivatives.
