# Deep Dives — Optional Selected Readings

This folder holds **optional** material that goes beyond the minimum needed to build
the tiny transformer library. Chapter docs in `docs/part-*/` stay short and
implementation-focused; anything that's "nice to know" but not required to write the
next line of code lives here.

## What goes here

- Mathematical proofs (e.g. why Box–Muller produces Normal samples).
- Historical context (e.g. how Glorot/Kaiming init was discovered).
- Pen-and-paper exercises that aren't strictly required.
- Cross-references to research papers with a short summary.
- Performance optimisations that aren't part of the main implementation.

## What does NOT go here

- Anything required to implement the chapter's code.
- Anything tested in `*.test.ts`.
- Anything a beginner needs to follow the main chapter.

## Naming

`ch-NN-topic.md` — one file per deep-dive topic, prefixed with the chapter number it
extends. A single chapter may have multiple deep-dives (e.g. `ch-22-attention-proof.md`
and `ch-22-flash-attention.md`).

## How chapters reference deep dives

Each chapter's **Further Reading** section may include a single bullet pointing here:

```markdown
- **Deep dive: <topic>.** <one-line description of what's inside.>
  [docs/deep-dives/ch-NN-topic.md](../deep-dives/ch-NN-topic.md)
```

The bullet is *optional* — only add it when there's genuine extra material a curious
reader might want. Never make a deep-dive a prerequisite for the next chapter.

## Index

| Chapter | Deep Dive | Topic |
|---|---|---|
| 02 | [ch-02-box-muller.md](ch-02-box-muller.md) | Why Box–Muller works, buffering details, pen-and-paper exercise |
| 04 | [ch-04-why-matmul.md](ch-04-why-matmul.md) | Proof that row × column is the unique valid definition; history; why AI chose it |
| 05 | [ch-05-why-subtract-the-max.md](ch-05-why-subtract-the-max.md) | Shift-invariance proof for softmax; the overflow it prevents; log-sum-exp connection |
| 05 | [ch-05-the-reduction-family.md](ch-05-the-reduction-family.md) | The reductions not detailed in the chapter — `max`, `min`, `argmin` — with their uses and the one shared algorithm |
| 07 | [ch-07-why-centered-difference.md](ch-07-why-centered-difference.md) | Taylor-series proof that centered differences are O(h²); the round-off trade-off that sets the best `h` |
| 08 | [ch-08-three-ways-to-a-gradient.md](ch-08-three-ways-to-a-gradient.md) | **The proof autograd is just the chain rule**: one function (`b·sin(a)+b²`), one gradient, four routes in teaching order — by hand → tiny per-operation steps (secretly the autograd algorithm, on paper) → autograd replaying it by machine → numerical nudging as independent referee — all landing on the same numbers |
| 08 | [ch-08-why-reverse-mode.md](ch-08-why-reverse-mode.md) | **The cost story**: why one backward pass fills every weight's gradient (~2·M, no N) vs numerical's 2·N·M and forward-mode's N passes; the reuse argument; and why reverse-topological order is the only correct one |
| 08 | [ch-08-symbolic-vs-autodiff.md](ch-08-symbolic-vs-autodiff.md) | **Why not just calculus**: symbolic vs numerical vs automatic — expression swell, formula-vs-value-at-a-point, and why PyTorch never returns a derivative formula |
| 09 | [ch-09-one-rule-many-layers.md](ch-09-one-rule-many-layers.md) | **Does the one-parameter rule really train a network?**: one full step of an actual two-layer net — four parameters, a `tanh` between them, every number computed with the Ch 08 `Value` class. Shows `∂L/∂b2` arriving in one factor and `∂L/∂w1` in a product of four, then the identical update loop moving both; depth lives in `backward()`, never in `step()`. Ends with what that predicts: the vanishing gradient, and why ReLU, LayerNorm and residuals all exist to fight the same multiplication |
| 09 | [ch-09-how-big-a-step.md](ch-09-how-big-a-step.md) | **The learning rate, derived not guessed**: the error on a quadratic is multiplied by exactly `1−2η` each step, so convergence needs `η < 1` and `η = 0.5` is exact in one step; the geometric series behind momentum's `1/(1−β)` speed-up and the effective-learning-rate trap it hides; then why the threshold is `1/curvature`, and why summing instead of averaging a loss makes the safe learning rate shrink as you add data |
