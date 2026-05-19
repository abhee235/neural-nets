/**
 * EXERCISES — Ch 28: Sequence Data & Training Objectives
 * ════════════════════════════════════════════════════════
 * Prereq : src/utils/data.ts + tokenizer/char.ts implemented
 * Run    : bun run exercises/ch-28-sequence-data.ts
 *
 * Language models are trained on "predict the next token".
 * Input: [t0, t1, t2, t3]   Target: [t1, t2, t3, t4]
 * This is a shifted copy — the model sees the prefix, predicts the suffix.
 */
import { shiftRight, maskedCrossEntropy, perplexity, makeReversalDataset, splitDataset } from "../src/utils/data.ts";

// ─── E1: Right-shift for language modelling ──────────────────────────────────
// Given input [1,2,3,4,5], target is [2,3,4,5, PAD]
const tokens = [1, 2, 3, 4, 5];
const { input, target } = shiftRight(tokens, 0);   // 0 = PAD_ID
console.log("input :", input,  "  expected: [1,2,3,4,5]");
console.log("target:", target, "  expected: [2,3,4,5,0]");

// ─── E2: Masked cross-entropy (ignore padding) ───────────────────────────────
// Padding positions should not contribute to the loss.
import { TensorValue } from "../src/autograd/grad.ts";
import { createTensor } from "../src/tensor/creation.ts";

const logits  = new TensorValue(createTensor([
  1, 0, 0,    // position 0 — strong prediction for class 0
  0, 1, 0,    // position 1 — strong prediction for class 1
  0, 0, 0,    // position 2 — padding, should be ignored
], [3, 3]));
const targets = [0, 1, -1];   // -1 = ignore (padding)
const loss = maskedCrossEntropy(logits, targets);
console.log("\nmasked CE (2 real, 1 ignored):", loss.data.data[0].toFixed(4));
// Should be close to 0 (both real predictions are correct).

// ─── E3: Perplexity ──────────────────────────────────────────────────────────
// Perplexity = exp(cross-entropy loss)
// Perfect model (loss=0) → perplexity=1
// Random 50k-vocab model (loss≈10.8) → perplexity≈50,000
const perfectLoss = 0.0;
const randomLoss  = Math.log(50000);
console.log("\nperplexity (perfect):", perplexity(perfectLoss).toFixed(2), "  expected: 1.00");
console.log("perplexity (random) :", perplexity(randomLoss).toFixed(0),  "  expected: 50000");

// ─── E4: Reversal dataset ────────────────────────────────────────────────────
// A classic seq2seq toy task: given "abcd", output "dcba".
// Useful for testing encoder-decoder transformer (Ch 29).
const revDataset = makeReversalDataset(100, 5);   // 100 examples, seqLen=5
const { train, val, test } = splitDataset(revDataset, 0.7, 0.15, 0.15);
console.log("\nReversal dataset splits:");
console.log("  train:", train.length, " val:", val.length, " test:", test.length);
console.log("  example src:", train[0]!.src, " →  tgt:", train[0]!.tgt);

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: encode Shakespeare using CharTokenizer, build all (input, target) pairs,
//       and print the first 5 input/target pairs side by side.
