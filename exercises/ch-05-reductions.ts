/**
 * EXERCISES — Ch 05: Reductions
 * ══════════════════════════════
 * Prereq : src/tensor/reduce.ts + creation.ts + linalg.ts implemented
 * Run    : bun run exercises/ch-05-reductions.ts
 *
 * softmax is used in EVERY attention head. sum/mean power LayerNorm (Ch 20).
 */
import { sum, mean, max, min, argmax, softmax } from "../src/tensor/reduce.ts";
import { createTensor } from "../src/tensor/creation.ts";

// ─── E1: sum and mean ────────────────────────────────────────────────────────
const x = createTensor([1, 2, 3, 4, 5, 6], [2, 3]);
console.log("sum (all)    :", sum(x).data[0],   "  expected: 21");
console.log("mean (all)   :", mean(x).data[0],  "  expected: 3.5");

// ─── E2: Axis reductions ──────────────────────────────────────────────────────
// sum along axis 0 (collapse rows) → [3] vector
const s0 = sum(x, 0);
console.log("\nsum axis=0   :", Array.from(s0.data), "  expected: [5,7,9]");
// sum along axis 1 (collapse cols) → [2] vector
const s1 = sum(x, 1);
console.log("sum axis=1   :", Array.from(s1.data), "  expected: [6,15]");

// ─── E3: argmax ─────────────────────────────────────────────────────────────
// argmax picks the predicted class during inference.
const logits = createTensor([0.1, 0.9, 0.3, 0.7], [4]);
console.log("\nargmax:", argmax(logits).data[0], "  expected: 1");

// ─── E4: softmax — the attention distribution ────────────────────────────────
// softmax(x)[i] = exp(x[i]) / Σ exp(x[j])
// Properties to verify:
//   1) all outputs > 0
//   2) outputs sum to 1
const raw = createTensor([2.0, 1.0, 0.1], [3]);
const probs = softmax(raw);
const probSum = Array.from(probs.data).reduce((a, b) => a + b, 0);
console.log("\nsoftmax probs:", Array.from(probs.data).map(v => v.toFixed(4)));
console.log("sum of probs :", probSum.toFixed(6), "  expected: 1.000000");

// ─── E5: Numerical stability — large logits ──────────────────────────────────
// Naive softmax overflows at logit=1000.  Your implementation must subtract max first.
// x - max(x) before exp prevents exp(1000) = Infinity.
const bigLogits = createTensor([1000, 1001, 1002], [3]);
const stableSoftmax = softmax(bigLogits);
const bigSum = Array.from(stableSoftmax.data).reduce((a, b) => a + b, 0);
console.log("\nstable softmax sum:", bigSum.toFixed(6), "  expected: 1.000000");
console.log("no NaN:", !stableSoftmax.data.some(isNaN), "  expected: true");

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: compute softmax along axis=1 for a [4, 3] logit matrix.
//       Each row should sum to 1.
