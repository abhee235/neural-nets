# Neural Nets from Scratch in TypeScript

A university-style, build-from-scratch course for learning neural networks, transformers, and GPT-style language models without external ML libraries.

This is a learning project, not a production library. The point is to understand every layer: tensor storage, broadcasting, autograd, optimizers, tokenization, attention, encoder-decoder transformers, and decoder-only GPT models.

---

## How the Course Works

This is designed as a Git-based lab course:

- Each chapter is implemented on its own branch.
- Each branch starts from the previous completed chapter branch.
- Students fork the repository and work chapter by chapter.
- Tests are the gate before moving forward.

Read the full workflow here: [docs/course-lab-workflow.md](docs/course-lab-workflow.md)

---

## Course Path

| Part | Chapters | Topic |
|------|----------|-------|
| 1 | Ch 01-06 | Tensor library: shape, storage, broadcasting, matmul, reductions, math primitives |
| 2 | Ch 07-10 | Autograd: calculus, computation graphs, backward pass, tensor autograd bridge |
| 3 | Ch 11-15 | Neural net primitives: activations, losses, layers, optimizers, training loops |
| 4 | Ch 16-21 | Language model inputs: tokenizers, embeddings, positional encoding, LayerNorm, masks |
| 5 | Ch 22-24 | Attention: self-attention, multi-head attention, cross-attention |
| 6 | Ch 25-30 | Transformers: FFN, encoder, decoder, sequence objectives, full Transformer, GPT |

---

## Running the Project

```bash
bun test
bun test src/ch-01-tensor-type-system
bun run src/ch-01-tensor-type-system/index.ts
```

The `src/` chapter scaffolding is intentionally separate from the docs. The docs define what to build; chapter branches should contain the implementation and tests.

---

## Constraints

- TypeScript only.
- Bun for running and testing.
- No external ML, tensor, math, tokenizer, or plotting libraries.
- `bun:test` is allowed because it is built into Bun.
- Strict TypeScript.
- Comments are part of the teaching material.

---

## Learning Goal

By the end, a student should be able to explain and implement:

- A tiny tensor library.
- Reverse-mode automatic differentiation.
- Neural network layers and optimizers.
- Character and BPE tokenization concepts.
- Transformer attention and masking.
- Encoder-decoder sequence-to-sequence training.
- Decoder-only GPT-style next-token prediction.
