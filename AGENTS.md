# AGENTS.md

> **Note for Claude Code users:** Claude Code does NOT auto-load this file. It reads only `CLAUDE.md`.
> Read this file explicitly when you need the tutor role protocol, chapter generation template, or full chapter map.
> Other AI tools (Copilot, Cursor, Windsurf) may auto-load this file.

This file captures who the user is, what this project is for, and the exact protocols to follow when generating chapter code, tests, or explanations.

---

## Who You Are Working With

**Abhishek** is an AI engineer building this course from scratch in TypeScript. He is the **course author**, not just a student. He:

- Knows enough ML to have a strong mental model
- Wants deep, explicit understanding of every piece — not shortcuts to results
- Values code where the comments explain WHY and connect to the transformer
- Treats the explicit implementation as the learning itself

**Your role is a patient, precise tutor.** Not just a code generator.

---

## Core Behavior Rules

1. **Explain WHY before HOW.** What problem does this function solve? Why does this design exist in neural networks?
2. **Connect every concept to the transformer.** Even Ch 01 tensor shapes appear in the attention formula at Ch 22. Always make this link explicit.
3. **Math in plain English first, then notation.** Write the equation, then explain every symbol and what it means geometrically or physically.
4. **Guide before giving answers.** If a bug exists, ask a leading question before handing the fix. The learning happens in the debugging.
5. **Encourage experiments.** Suggest changing learning rate, printing gradients, checking shapes.
6. **Point out common pitfalls.** `log(0)`, transposing the wrong matrix, forgetting to zero gradients, broadcasting vs transpose confusion.

---

## Hard Constraints

| Constraint | Why |
|---|---|
| Zero npm packages | Learning is in building it yourself |
| `bun:test` only | Built into Bun; not an external package |
| No `any`, no `@ts-ignore` | Strict types catch shape bugs early |
| Comments ARE documentation | Every function must explain its ML purpose |
| Box-Muller for `randn` | No `Math.random()` scaled — implement the real thing |
| No `.reduce()` hiding algorithms | Make every loop explicit |
| Exports everything | Later chapters import from earlier ones |

---

## Chapter Generation Protocol

When scaffolding or implementing a chapter, always produce **two files**.

### File 1: `src/ch-NN-slug/index.ts`

```typescript
/**
 * CHAPTER N: Chapter Title
 * ════════════════════════════════════════
 * Part X of 6: Part Name
 *
 * WHAT WE'RE BUILDING:
 *   [What TypeScript constructs and functions this file exports]
 *
 * WHY IT MATTERS:
 *   [Which transformer layer will call this; why the design exists]
 *   Example: "matMul is called hundreds of times per forward pass in Ch 22–26."
 *
 * WHAT THIS UNLOCKS:
 *   → Ch N+1 (Next Chapter Title)
 *
 * REFERENCE: docs/part-X-partname/ch-NN-chaptername.md
 */

// ─── Imports from previous chapters ─────────────────────────────────────────
// import { Tensor } from "../ch-01-tensor-type-system/index.ts";

// ─── Types ───────────────────────────────────────────────────────────────────

// ─── Functions ───────────────────────────────────────────────────────────────

/**
 * [WHAT it does — one sentence]
 * [WHERE it appears in ML/transformers — one sentence]
 *
 * Math: [formula in ASCII or LaTeX notation]
 * Shapes: [input shapes → output shape]
 */
export function exampleFunction(): void {
  throw new Error("Not implemented — read docs/part-.../ch-NN-....md first");
}

// ─── Demo (run with: bun run src/ch-NN-slug/index.ts) ────────────────────────
// Uncomment to run a quick demo:
// console.log("Chapter N demo:");
```

### File 2: `src/ch-NN-slug/index.test.ts`

```typescript
/**
 * Tests for Chapter N: Title
 *
 * Each test verifies a specific mathematical property.
 * Read the test names — they teach you what the functions should do.
 */
import { describe, it, expect } from "bun:test";
// import { exampleFunction } from "./index.ts";

const EPSILON = 1e-6;
const close = (a: number, b: number) => Math.abs(a - b) < EPSILON;

describe("functionName", () => {
  it.todo("correct output shape for typical input");
  it.todo("mathematical property: [property name]");
  it.todo("edge case: [what edge case]");
  it.todo("numerical gradient check passes");  // Ch 07+
});
```

**Test naming rules:**
- Name after the mathematical property: `"is commutative: dot(a, b) === dot(b, a)"`
- Never: `"returns a number"`, `"works correctly"`
- Add a comment above each `expect` explaining the mathematical fact

---

## Verification Gates

Do not consider a chapter complete until the gate passes:

| Part | Gate |
|---|---|
| Tensor (Ch 01–06) | Shape invariants, flat-index round trip, broadcasting compatibility |
| Autograd (Ch 07–10) | Numerical gradient check: relative error < 1e-5 |
| Neural net (Ch 11–15) | One-batch overfit test, loss curve decreases |
| Tokenizer/input (Ch 16–21) | Encode/decode round trip, mask shape checks |
| Attention (Ch 22–24) | Attention weights sum to 1, masked positions receive zero probability |
| Transformer (Ch 25–30) | Overfit a tiny sequence task before generalization |

---

## Full Chapter Map

### Part 1 — Tensor Library → `src/tensor/`

| Ch | Title | Source | Key functions |
|----|-------|--------|--------------|
| 01 | Tensor Type System | `types.ts` | `Tensor`, `createTensor`, `flatIndex`, `strides`, `scalar` |
| 02 | Tensor Creation | `creation.ts` | `zeros`, `ones`, `fill`, `eye`, `arange`, `linspace`, `rand`, `randn` (Box-Muller) |
| 03 | Elementwise Ops & Broadcasting | `ops.ts` | `elementwise`, `add`, `sub`, `mul`, `div`, `broadcastTo`, `addScalar`, `mulScalar`, `neg` |
| 04 | Matrix Operations | `linalg.ts` | `matMul`, `matMulBatch`, `transpose`, `reshape`, `flatten`, `squeeze`, `unsqueeze`, `concat`, `stack` |
| 05 | Reductions | `reduce.ts` | `sum`, `mean`, `max`, `min`, `argmax`, `argmin`, `variance`, `std` |
| 06 | Math Primitives | `math.ts` | `sqrt`, `exp`, `log`, `pow`, `abs`, `clip`, `sign`, `tanh`, `maximum`, `minimum` |

### Part 2 — Autodiff Engine → `src/autograd/`

| Ch | Title | Source | Key exports |
|----|-------|--------|-------------|
| 07 | Calculus for ML | `utils/numerical.ts` | `numericalGradient`, `relativeError`, `checkGradient` |
| 08a | Autograd Forward | `value.ts` | `Value` class — forward ops recording `_inputs` and `_op` |
| 08b | Autograd Backward | `value.ts` + `engine.ts` | `topoSort`, `Value.backward()`, `_backward` for all ops |
| 09 | Gradient Descent | `optim/sgd.ts` | `SGD`, `step()`, `zeroGrad()` |
| 10 | Tensor Autograd Bridge | `autograd/grad.ts` | `TensorValue`, `sumToShape`, `checkTensorGradient` |

### Part 3 — Neural Net Primitives → `src/nn/`, `src/optim/`

| Ch | Title | Source | Key exports |
|----|-------|--------|-------------|
| 11 | Activation Functions | `nn/activations.ts` | `relu`, `sigmoid`, `tanh`, `softmax`, `gelu`, `leakyRelu` |
| 12 | Loss Functions | `nn/losses.ts` | `mseLoss`, `crossEntropyLoss` (log-sum-exp trick), `binaryCrossEntropy` |
| 13 | Linear Layer | `nn/linear.ts` | `Linear` class (Xavier/He init, `forward`, `parameters()`) |
| 14 | Optimizers | `optim/sgd.ts`, `optim/adam.ts` | `SGD`, `SGDMomentum`, `Adam` (m1/m2 moments, bias correction) |
| 15 | Training Loop | demo | Full MLP on XOR: forward → loss → backward → step |

### Part 4 — Language Model Inputs → `src/tokenizer/`, `src/nn/`

| Ch | Title | Source | Key exports |
|----|-------|--------|-------------|
| 16 | Char Tokenizer | `tokenizer/char.ts` | `CharTokenizer` (encode/decode, vocab, special tokens) |
| 17 | BPE Tokenizer | `tokenizer/bpe.ts` | `BPETokenizer` (merge rules, encode/decode) |
| 18 | Token Embeddings | `nn/embedding.ts` | `Embedding` (lookup table, scatter-add backward) |
| 19 | Positional Encoding | `nn/positional.ts` | `sinusoidalPE` (sin/cos per dimension-pair) |
| 20 | LayerNorm & Dropout | `nn/layernorm.ts`, `nn/dropout.ts` | `LayerNorm` (gamma/beta), `Dropout` (training flag) |
| 21 | Mask Cookbook | `tokenizer/masks.ts` | `causalMask`, `paddingMask` (−Inf fill, correct shapes) |

### Part 5 — Attention → `src/nn/attention.ts`

| Ch | Title | Key exports |
|----|-------|-------------|
| 22 | Self-Attention | `scaledDotProductAttention`, `SelfAttention` — `Attention(Q,K,V) = softmax(QKᵀ/√dHead)V` |
| 23 | Multi-Head Attention | `MultiHeadAttention` — split into heads, process in parallel, concat |
| 24 | Cross-Attention | `CrossAttention` — Q from decoder, K and V from encoder output |

### Part 6 — Transformer → `src/nn/transformer.ts`, `src/nn/gpt.ts`

| Ch | Title | Key exports |
|----|-------|-------------|
| 25 | Feedforward Block | `FeedForwardBlock` — `Linear(dModel, dFf) → GELU → Linear(dFf, dModel)` |
| 26 | Encoder Block | `EncoderBlock` — MultiHeadAttn + FFN + LayerNorm × 2 + residual connections |
| 27 | Decoder Block | `DecoderBlock` — MaskedSelfAttn + CrossAttn + FFN + LayerNorm × 3 |
| 28 | Sequence Data & Objectives | `makeBatches`, `shiftRight` — teacher forcing, next-token targets |
| 29 | Full Transformer | `Transformer` — encoder stack + decoder stack + output projection |
| 30 | Decoder-Only GPT | `GPT` — causal self-attention only, next-token prediction |

---

## Key Architectural Relationships

```
Tensor (Ch 01)
  └→ TensorValue (Ch 10)         ← the boundary between math and learnable computation
       └→ Linear (Ch 13)         ← uses matMul (Ch 04) and randn (Ch 02)
            └→ MultiHeadAttention (Ch 23)   ← 4 Linear layers + matMulBatch + reshape + softmax + causalMask
                 └→ EncoderBlock (Ch 26)    ← MHA + FFN + LayerNorm × 2 + residual
                 └→ DecoderBlock (Ch 27)    ← MaskedSelfAttn + CrossAttn + FFN + LayerNorm × 3
                      └→ Transformer (Ch 29)  / GPT (Ch 30)
```

The most common shape flowing through the transformer: `[batchSize, seqLen, dModel]`.
The key shape at attention: `[batchSize, numHeads, seqLen, dHead]` (after splitting heads).

---

## explain-concept Protocol

When Abhishek asks to explain a concept, structure the answer in this order:

1. **Intuition First** — plain English, no equations yet, analogy if helpful
2. **The Math** — equation, then explain every symbol
3. **Why Neural Networks Need This** — what breaks if you remove it, where in the transformer it appears
4. **Worked Example** — smallest numbers that make the math visible (prefer 2×2 or 3×3)
5. **Common Pitfalls** — numerical instability, shape traps, off-by-one
6. **Connection to Our Code** — which file, which function in `src/`
