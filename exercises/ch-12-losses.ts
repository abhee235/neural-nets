/**
 * EXERCISES — Ch 12: Loss Functions
 * ═══════════════════════════════════
 * Prereq : mseLoss for E1–E4; logSumExp and crossEntropyFromLogits for E5–E7
 * Run    : bun run exercises/ch-12-losses.ts
 *
 * Runnable at every stage: exercises for functions you have not implemented
 * yet print "pending" instead of crashing. Finish a milestone, run again.
 *
 * Every number here is one the chapter derives — the one-weight walkthrough,
 * the temperatures, and "the cat ___".
 */
import { mseLoss, crossEntropyFromLogits, logSumExp } from "../src/nn/losses.ts";
import { TensorValue } from "../src/autograd/grad.ts";
import { createTensor } from "../src/tensor/types.ts";

/** Run a block; if the function inside is still a stub, say so and move on. */
function stage(title: string, body: () => void): void {
  console.log(`\n─── ${title} ───`);
  try {
    body();
  } catch (error) {
    if (error instanceof Error && error.message.includes("not implemented")) {
      console.log("  pending —", error.message);
    } else throw error;
  }
}

// ─── E1: One gradient, all the way — the doc's section 2 walkthrough ─────────
// The smallest possible model: p = x·w, one input, one weight.
//     x = 2,  w = 3,  target = 10
// Every number is small enough to check in your head as it prints.
stage("E1: one weight, one step — the whole learning loop", () => {
  const x = new TensorValue(createTensor([2], [1]));
  const w = new TensorValue(createTensor([3], [1]));
  const target = createTensor([10], [1]);

  const prediction = x.mul(w);
  const loss = mseLoss(prediction, target);
  console.log("  forward : p = 2×3 =", prediction.data.data[0], "     L = (6−10)² =", loss.data.data[0]);

  loss.backward();
  // Chain rule, two hops:  dL/dp = 2(p−y) = -8,  then  dL/dw = dL/dp · x = -16.
  console.log("  backward: dL/dp =", prediction.grad!.data[0], "    dL/dw =", w.grad!.data[0]);
  // The engine also filled x.grad = -24 — computed for everything, USED only
  // where there is something to change. x is data, not a parameter.
  console.log("  (unused): dL/dx =", x.grad!.data[0]);

  // The optimizer's job, not backpropagation's — Ch 09's rule:
  const wNew = 3 - 0.1 * w.grad!.data[0]!;
  console.log("  update  : w ← 3 − 0.1·(−16) =", wNew);

  // Close the loop: forward again with the new weight.
  const p2 = new TensorValue(createTensor([2], [1])).mul(new TensorValue(createTensor([wNew], [1])));
  const L2 = mseLoss(p2, target);
  console.log("  again   : p = 2×4.6 =", p2.data.data[0], "   L =", L2.data.data[0], "  (was 16 — the model learned)");
});

// ─── E2: The chapter checkpoint — the temperatures ───────────────────────────
//     predicted   [ 32   28   31 ]
//     actual      [ 35   28   30 ]
//     MSE = 3.333333    gradient = [ -2   0   0.666667 ]
stage("E2: mseLoss on the temperatures", () => {
  const predicted = new TensorValue(createTensor([32, 28, 31], [3]));
  const actual = createTensor([35, 28, 30], [3]);
  const loss = mseLoss(predicted, actual);
  console.log("  loss:", loss.data.data[0]!.toFixed(6), "  expected 3.333333");
  loss.backward();
  console.log("  grad:", Array.from(predicted.grad!.data).map((v) => v.toFixed(6)).join("  "), "  expected -2  0  0.666667");
  // If you see [-6, 0, 2] the mean got lost — that is the sum-not-mean bug.
});

// ─── E3: Why predictions must be a TensorValue — the severed graph ───────────
// Same numbers, history removed: the loss still MEASURES, but can no longer
// TEACH. Job 1 without job 2 (doc, opening section).
stage("E3: the severed graph", () => {
  const w = new TensorValue(createTensor([0.5], [1]));
  const today = new TensorValue(createTensor([30, 27, 31], [3]));
  const actual = createTensor([33.0, 29.7, 34.1], [3]);

  const connected = today.mul(w);
  const detached = new TensorValue(connected.data);       // same numbers, no _inputs
  const loss = mseLoss(detached, actual);
  loss.backward();
  console.log("  loss:", loss.data.data[0]!.toFixed(4), "   identical to the connected version");
  console.log("  w.grad:", w.grad, "        ← the weight got NOTHING. It can never learn.");
});

// ─── E4: Train the one-weight model for real ─────────────────────────────────
// tomorrow ≈ 1.1 × today. Start at w = 0.5 and let the gradient find 1.1.
stage("E4: twelve steps of gradient descent through mseLoss", () => {
  const actual = createTensor([33.0, 29.7, 34.1], [3]);
  let w = 0.5;
  const learningRate = 0.0005;
  for (let step = 0; step <= 12; step++) {
    const wT = new TensorValue(createTensor([w], [1]));
    const prediction = new TensorValue(createTensor([30, 27, 31], [3])).mul(wT);
    const loss = mseLoss(prediction, actual);
    loss.backward();
    if (step % 3 === 0) console.log(`  step ${String(step).padStart(2)}   w = ${w.toFixed(6)}   loss = ${loss.data.data[0]!.toFixed(6)}`);
    w = w - learningRate * wT.grad!.data[0]!;
  }
  console.log("  w found ~1.1 — the rule hidden in the data.");
});

// ─── E5: logSumExp — the row that breaks, handled ────────────────────────────
//     logSumExp([1, 2, 3])          = 3.407606
//     logSumExp([1000, 1001, 1002]) = 1002.407606     same .407606 — the shift
stage("E5: logSumExp stability", () => {
  const small = logSumExp(new TensorValue(createTensor([1, 2, 3], [3])));
  const big = logSumExp(new TensorValue(createTensor([1000, 1001, 1002], [3])));
  console.log("  [1,2,3]          →", small.data.data[0]!.toFixed(6), "  expected 3.407606");
  console.log("  [1000,1001,1002] →", big.data.data[0]!.toFixed(6), "  expected 1002.407606");
  console.log("  finite:", Number.isFinite(big.data.data[0]!), "  — the naive route gives NaN here");
  // Gradient check: d(logSumExp)/dz IS softmax (doc section 13).
  small.backward();
  console.log("  grad :", Array.from((small._inputs[0] as TensorValue).grad!.data).map((v) => v.toFixed(6)).join("  "));
  console.log("          expected 0.090031  0.244728  0.665241 — softmax itself");
});

// ─── E6: Cross-entropy on "the cat ___" ──────────────────────────────────────
//     logits [1, 2, 3] = sat, ran, flew.  Truth "sat" → 2.407606
//     Shift invariance: [1000,1001,1002] must give the identical loss.
stage("E6: crossEntropyFromLogits", () => {
  // NOTE: written for one-hot targets. If you chose class indices, adapt the
  // second argument — the expected numbers do not change.
  const oneHotSat = createTensor([1, 0, 0], [1, 3]);
  const loss = crossEntropyFromLogits(new TensorValue(createTensor([1, 2, 3], [1, 3])), oneHotSat);
  console.log("  truth sat  :", loss.data.data[0]!.toFixed(6), "  expected 2.407606");
  const lossFlew = crossEntropyFromLogits(new TensorValue(createTensor([1, 2, 3], [1, 3])), createTensor([0, 0, 1], [1, 3]));
  console.log("  truth flew :", lossFlew.data.data[0]!.toFixed(6), "  expected 0.407606");
  const shifted = crossEntropyFromLogits(new TensorValue(createTensor([1000, 1001, 1002], [1, 3])), createTensor([0, 0, 1], [1, 3]));
  console.log("  shifted    :", shifted.data.data[0]!.toFixed(6), "  expected 0.407606 — shift invariance");
});

// ─── E7: The gradient is p − y ───────────────────────────────────────────────
//     softmax    [  0.090031   0.244728   0.665241 ]
//     one-hot    [  1          0          0        ]
//     gradient   [ -0.909969   0.244728   0.665241 ]     sums to 0
stage("E7: p − y, read off the graph", () => {
  const logits = new TensorValue(createTensor([1, 2, 3], [1, 3]));
  const loss = crossEntropyFromLogits(logits, createTensor([1, 0, 0], [1, 3]));
  loss.backward();
  const grad = Array.from(logits.grad!.data);
  console.log("  logits.grad:", grad.map((v) => v.toFixed(6)).join("  "));
  console.log("  expected   : -0.909969  0.244728  0.665241");
  // sat under-predicted → negative → its logit RISES. flew took the most it
  // should not have → pushed down hardest. And the row can only redistribute:
  console.log("  sums to    :", grad.reduce((a, b) => a + b, 0).toExponential(2), "  (0 up to float noise)");
});

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: a batch. logits shape [4, 5], four true classes, one CE per row,
//       averaged. Then check the batch gradient still sums to ~0 per row.
