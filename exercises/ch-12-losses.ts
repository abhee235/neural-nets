/**
 * EXERCISES — Ch 12: Loss Functions
 * ═══════════════════════════════════
 * Prereq : src/nn/losses.ts + activations.ts + autograd/grad.ts implemented
 * Run    : bun run exercises/ch-12-losses.ts
 *
 * The loss function measures how wrong the model is.  Cross-entropy is used
 * for every token prediction in the language model (Ch 28-30).
 */
import { mseLoss, crossEntropyFromLogits, logSumExp } from "../src/nn/losses.ts";
import { TensorValue } from "../src/autograd/grad.ts";
import { createTensor } from "../src/tensor/creation.ts";

// ─── E1: MSE loss ─────────────────────────────────────────────────────────────
// MSE = mean( (y_hat - y)² )
// Perfect prediction → 0, otherwise positive.
const pred   = new TensorValue(createTensor([1.0, 2.0, 3.0], [3]), "pred");
const target = new TensorValue(createTensor([1.0, 2.0, 3.0], [3]), "target");
const mse    = mseLoss(pred, target);
console.log("MSE (perfect):", mse.data.data[0], "  expected: 0");

const pred2 = new TensorValue(createTensor([1.0, 2.0, 3.0], [3]), "pred2");
const tgt2  = new TensorValue(createTensor([0.0, 0.0, 0.0], [3]), "tgt2");
const mse2  = mseLoss(pred2, tgt2);
console.log("MSE ([1,2,3] vs [0,0,0]):", mse2.data.data[0].toFixed(4), "  expected:", ((1+4+9)/3).toFixed(4));

// ─── E2: Cross-entropy from logits ───────────────────────────────────────────
// CE = -log(softmax(logits)[correct_class])
// Logits = [2, 1, 0.5], correct class = 0
// softmax = [0.6652, 0.2447, 0.0900]  →  CE = -log(0.6652) ≈ 0.408
const logits = new TensorValue(createTensor([2, 1, 0.5], [3]), "logits");
const ce = crossEntropyFromLogits(logits, 0);
console.log("\nCE (correct=0):", ce.data.data[0].toFixed(4), "  expected: ≈ 0.4076");

// ─── E3: CE backprop ────────────────────────────────────────────────────────
ce.backward();
console.log("logits.grad:", Array.from(logits.grad!.data).map(v => v.toFixed(4)));
// softmax - one_hot: [0.6652-1, 0.2447-0, 0.0900-0] = [-0.3348, 0.2447, 0.0900]

// ─── E4: logSumExp numerical stability ──────────────────────────────────────
// logSumExp(x) = log(Σ exp(x[i]))  computed stably by subtracting max first.
const bigLogits = new TensorValue(createTensor([1000, 1001, 1002], [3]));
const lse = logSumExp(bigLogits);
console.log("\nlogSumExp([1000,1001,1002]):", lse.data.data[0].toFixed(4));
// expected: 1002 + log(exp(-2)+exp(-1)+exp(0)) ≈ 1002.4076
console.log("no Inf:", !isNaN(lse.data.data[0]) && isFinite(lse.data.data[0]), "  expected: true");

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: compute the CE loss for a batch of 4 tokens from a 5-class vocabulary.
//   logits shape: [4, 5]  targets: [0, 2, 4, 1]
//   Average CE over the batch.
