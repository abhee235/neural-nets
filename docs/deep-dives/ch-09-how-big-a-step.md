# Deep Dive — How big a step can you take?

> Companion to [Chapter 09: Gradient Descent](../part-2-autodiff/ch-09-gradient-descent.md).
> Optional. Read it after your `SGD` passes its tests.

Every practical guide tells you the learning rate is chosen by trial and error. That is true in general — but on the bowl you are about to optimise in the exercise, you can derive the *exact* threshold in a few lines of algebra. Doing it once turns "try a smaller learning rate" from folklore into something you can predict.

---

## The one-line recurrence

The exercise minimises

$$L(w) = (w - 5)^2$$

Its derivative is $L'(w) = 2(w-5)$, so one SGD step is

$$w \leftarrow w - \eta \cdot 2(w - 5)$$

Now the trick that makes everything fall out: stop tracking $w$ and track the **error** instead,

$$e = w - 5.$$

Subtract 5 from both sides of the update. On the left you get the new error; on the right, $w - 5 - 2\eta(w-5)$, which is just $e - 2\eta e$:

$$\boxed{e_{\text{new}} = e_{\text{old}}\,(1 - 2\eta)}$$

That is the entire dynamics. The error does not decay in some complicated way — it is **multiplied by the same constant every single step**. After $n$ steps,

$$e_n = e_0\,(1 - 2\eta)^n.$$

---

## Reading off the answer

A geometric sequence shrinks exactly when the multiplier is smaller than 1 in absolute value:

$$|1 - 2\eta| < 1 \quad\Longleftrightarrow\quad 0 < \eta < 1$$

So on this bowl **any learning rate below 1 converges, and any learning rate above 1 diverges** — no tuning, no guessing. And the sign of the multiplier tells you *how* it converges:

| η | multiplier `1 − 2η` | behaviour |
|---|---|---|
| 0.05 | +0.90 | same side, crawls in — 10% closer per step |
| 0.2 | +0.60 | same side, comfortable descent |
| **0.5** | **0** | **lands exactly on the minimum in ONE step** |
| 0.9 | −0.80 | overshoots every step, alternating sides, still shrinking |
| 1.0 | −1.00 | bounces between two points forever, never converges |
| 1.1 | −1.20 | overshoots and grows — diverges |

Two of those rows are worth sitting with.

**η = 0.5 is exact.** The multiplier is zero, so the error is annihilated in a single step regardless of where you started. This is not a coincidence: $\eta = 1/L''$ is Newton's method for a quadratic, and Newton's method solves quadratics exactly. When people say second-order optimisers converge fast, this is the cleanest possible example of why.

**η = 0.9 still works.** It overshoots the minimum on every single step, landing on the opposite side each time — yet $|-0.8| < 1$, so the distance still shrinks. A loss curve that oscillates while trending down is not necessarily broken. That is why the exercise file predicts "oscillates but converges" for `lr = 0.9`.

---

## Checking it against the exercise

[`exercises/ch-09-gradient-descent.ts`](../../exercises/ch-09-gradient-descent.ts) runs exactly this bowl from $w_0 = 0$, so $e_0 = -5$ and $w_n = 5 - 5\,(1-2\eta)^n$. Predict before you run:

| call | multiplier | $e_n$ | predicted `w` |
|---|---|---|---|
| `minimise(0.01, 200)` | 0.98 | $-5 \cdot 0.98^{200} = -0.088$ | **4.912** |
| `minimise(0.1, 100)` | 0.80 | $-5 \cdot 0.8^{100} \approx -1\text{e-}10$ | **5.000** |
| `minimise(0.9, 100)` | −0.80 | $-5 \cdot (-0.8)^{100} \approx -1\text{e-}10$ | **5.000** |

Run it. If your `SGD` prints `4.912`, `5.000`, `5.000`, your implementation is not merely "converging" — it is tracing the exact trajectory the algebra predicts. That is a far stronger check than "the number went down".

---

## The general rule, and why your learning rate shrinks as models grow

Nothing above depended on the constants 5 or 2. For any quadratic

$$L(w) = c\,(w - a)^2, \qquad e_{\text{new}} = e_{\text{old}}\,(1 - 2\eta c)$$

so the threshold is

$$\eta < \frac{1}{c}.$$

The number $2c$ is the **second derivative** — the curvature. Steeper bowl, smaller maximum step. That single fact explains most learning-rate advice you will ever read, and it has a consequence that bites immediately in this chapter's own exercise.

E4 fits a line to five points using a **summed** squared error:

```
mse = Σᵢ (slope·xᵢ + intercept − yᵢ)²
```

The curvature with respect to `slope` is $2\sum x_i^2$. With `xs = [0,1,2,3,4]` that is $2(0+1+4+9+16) = 60$, giving a threshold of $\eta < 1/30 \approx 0.033$. The exercise uses `0.01` — comfortably inside it. Now notice what happens if you add more data points: the sum grows, the curvature grows, and the largest safe learning rate *shrinks*. Add twenty more points at the same learning rate and the fit will diverge, for reasons that have nothing to do with the model being harder.

This is exactly why real training code averages the loss instead of summing it:

```
mse = (1/N) Σᵢ (…)²
```

Dividing by `N` divides the curvature by `N` too, which makes the usable learning rate **independent of batch size**. A sum-vs-mean slip is one of the most common causes of "my model trained fine until I increased the batch size."

---

## How much speed can momentum build?

Chapter 09 claims that momentum's steps grow "up to about ten times" vanilla SGD's at `β = 0.9`. That number is not a rule of thumb — it drops out of a geometric series.

Momentum's velocity update is

$$v \leftarrow \beta v - \eta g.$$

Suppose for a moment the gradient is roughly constant at $g_0$ — which is what happens on a long, steady slope. Starting from $v_0 = 0$ and unrolling:

$$v_1 = -\eta g_0$$
$$v_2 = -\eta g_0(1 + \beta)$$
$$v_3 = -\eta g_0(1 + \beta + \beta^2)$$

so after $n$ steps

$$v_n = -\eta g_0 \sum_{k=0}^{n-1}\beta^k = -\eta g_0\,\frac{1-\beta^n}{1-\beta}.$$

As $n$ grows, $\beta^n \to 0$ and the velocity settles at

$$\boxed{v_\infty = -\frac{\eta g_0}{1-\beta}}$$

A vanilla SGD step on the same slope is $-\eta g_0$. So momentum's steady-state step is larger by exactly

$$\frac{1}{1-\beta}$$

| β | speed-up | in words |
|---|---|---|
| 0 | 1× | no memory — vanilla SGD exactly |
| 0.9 | 10× | the usual default |
| 0.99 | 100× | used with correspondingly tiny learning rates |

Two things follow, and the second one bites.

**The build-up is not instant.** The $(1-\beta^n)$ factor means it takes roughly $1/(1-\beta)$ steps to get most of the way to top speed — about 10 steps at `β = 0.9`. Momentum is not fast on step one; it is fast once it has been going the same way for a while. That is exactly the behaviour you want, and it is why Figure 3 in the chapter shows momentum *behind* vanilla after one step and far ahead by step three.

**The effective learning rate is `η/(1−β)`, not `η`.** This is the practical trap. Turning momentum up from `0` to `0.9` multiplies your real step size by ten. If `η` was already near the divergence threshold derived above, adding momentum will push you straight past it — and the failure looks like "momentum is broken" rather than "the learning rate is now effectively 10× larger". The fix is to lower `η` when you raise `β`.

And the mirror image, the damping case: if the gradient *alternates* sign each step (the narrow-valley oscillation), then consecutive terms in that sum subtract rather than add. The series no longer accumulates — it partly cancels — and the oscillation is suppressed. Same formula, opposite outcome, decided entirely by whether the gradient keeps agreeing with itself.

---

## Where this stops being exact

Real loss surfaces are not single quadratics, and the clean threshold generalises only locally:

- **Many parameters.** The curvature becomes a matrix (the Hessian), and the stability condition is governed by its *largest* eigenvalue: $\eta < 2/\lambda_{\max}$. One steep direction caps the learning rate for every direction — including the shallow ones that then crawl. That mismatch is called ill-conditioning, and it is the problem momentum and Adam (Ch 14) exist to soften.
- **Non-quadratic losses.** The curvature changes as you move, so a rate that is safe at initialisation can become unsafe later. Learning-rate schedules are the practical answer.
- **Stochastic gradients.** Once each step uses a different minibatch, the recurrence picks up a noise term and you get convergence to a neighbourhood rather than a point.

None of that invalidates the intuition — it just means the threshold is a moving target. But you now know what it is a moving target *of*, which is more than "try 3e-4 and see".

---

## Further reading

- [Sebastian Ruder — An overview of gradient descent optimization algorithms](https://ruder.io/optimizing-gradient-descent/) — where the variants in Ch 14 come from.
- [Goodfellow et al. — Deep Learning, §4.3 and §8.2](https://www.deeplearningbook.org/contents/numerical.html) — curvature, conditioning, and why the Hessian's eigenvalues set the step size.
