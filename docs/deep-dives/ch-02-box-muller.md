# Deep Dive — Box–Muller: Why It Works, How It's Buffered

> Companion to [Chapter 02: Tensor Creation](../part-1-tensor-library/ch-02-tensor-creation.md).
> **Optional.** Skip this if you only want to build the library; the chapter's tests already
> verify `randn` produces unit-variance, zero-mean samples.

This page collects three things the main chapter intentionally left out:

1. The **intuition** for why the Box–Muller formula produces Normal samples.
2. The **buffering state machine** in pictorial form.
3. A **pen-and-paper exercise** to lock the algorithm into your hands.

---

## 1. Why does Box–Muller work?

A 2-D standard normal $\mathcal{N}(0, I_2)$ is **rotationally symmetric** — its density at
a point depends only on the distance from the origin, not the angle. Picture a perfectly
round, fading-outward fog around the origin.

To sample from a rotationally symmetric 2-D distribution, you can sample the **angle**
and the **radius** independently:

- **Angle** $\theta$ — uniform on $[0, 2\pi)$. That's just $\theta = 2\pi u_2$.
- **Radius** $r$ — must follow the distribution that, after projecting to Cartesian,
  gives unit-variance Normal marginals. The math (change of variables on $r^2$) shows
  the right radius is $r = \sqrt{-2 \ln u_1}$ when $u_1 \sim \text{Uniform}(0, 1]$.

The Cartesian projection $(z_0, z_1) = (r \cos\theta,\, r \sin\theta)$ then gives **two
independent** $\mathcal{N}(0, 1)$ samples for free — independent because of the rotational
symmetry, Normal because of how we chose $r$.

For the formal change-of-variables proof, see the Wikipedia link at the bottom.

---

## 2. The buffering state machine

Box–Muller produces samples in pairs: every call computes both $z_0$ and $z_1$. We only
need one per request, so we return $z_0$ now and stash $z_1$ in a module-level variable
to hand back on the next call. This halves the calls to `Math.random` / `Math.log` /
`cos` / `sin`.

```text
  STATE A (buffer = null)               STATE B (buffer = z₁)
  ┌────────────────────────────┐        ┌────────────────────────────┐
  │ u₁ = Math.random()         │   ──▶  │ z = buffer                 │
  │ u₂ = Math.random()         │        │ buffer = null              │
  │ r  = √(-2 ln u₁)           │        │ return z                   │
  │ θ  = 2π u₂                 │   ◀──  │                            │
  │ z₀ = r·cos θ               │        └────────────────────────────┘
  │ z₁ = r·sin θ               │
  │ buffer = z₁                │
  │ return z₀                  │
  └────────────────────────────┘
```

The transitions:

- **A → B** after returning $z_0$: we cached $z_1$, so the next call should drain it.
- **B → A** after returning the cached $z_1$: the buffer is empty, so the next call must
  compute a fresh pair.

This is exactly what NumPy's `RandomState.standard_normal` and PyTorch's CPU `torch.randn`
do internally.

---

## 3. Pen-and-paper exercise

Reading a formula produces *recognition*, not *recall*. To actually implement Box–Muller
from a blank editor, your hand must have produced the pattern at least once.

1. Copy the boxed Box–Muller pair $(z_0, z_1)$ from the chapter by hand. No peeking.
2. Pick concrete numbers: $u_1 = 0.5$ and $u_2 = 0.25$.
   - Compute $r = \sqrt{-2 \ln 0.5}$ to three decimals. *(Answer: ≈ 1.177.)*
   - Compute $\theta = 2\pi \cdot 0.25 = \pi/2$.
   - Compute $\cos(\pi/2)$ and $\sin(\pi/2)$ exactly. *(0 and 1.)*
   - Therefore $z_0 = 0$ and $z_1 \approx 1.177$.
3. Redraw the three-panel geometry figure from memory: the unit square with $(u_1, u_2)$
   marked, the polar disk with $(r, \theta)$ marked, the Cartesian plane with
   $(z_0, z_1)$ marked.
4. Now open `src/tensor/creation.ts` and type the body of `randn` from memory.

---

## Further reading

- **Wikipedia — Box–Muller transform.** The change-of-variables proof in full.
  <https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform>

- **Glorot & Bengio (2010) — Understanding the difficulty of training deep feedforward
  networks.** Quantifies why initial randomness matters and how it must scale with layer
  size. We return to this in Ch 13.
  <http://proceedings.mlr.press/v9/glorot10a.html>

- **Goodfellow, Bengio & Courville — *Deep Learning*, §8.4 "Parameter Initialization
  Strategies".** The canonical reference on weight-init schemes built on top of `randn`.
  <https://www.deeplearningbook.org/contents/optimization.html>
