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
