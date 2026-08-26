/**
 * EXERCISES — Ch 13: Linear Layer
 * ═════════════════════════════════
 * Prereq : src/nn/linear.ts + autograd/grad.ts implemented
 * Run    : bun run exercises/ch-13-linear-layer.ts
 *
 * Linear is the Q, K, V projection inside every attention head and both
 * linear transforms inside every FFN block (Ch 25).
 */
import { Linear } from "../src/nn/linear.ts";
import { TensorValue } from "../src/autograd/grad.ts";
// createTensor lives in types.ts; randn in creation.ts.
import { createTensor } from "../src/tensor/types.ts";
import { randn } from "../src/tensor/creation.ts";

// ─── E1: Basic forward pass ───────────────────────────────────────────────────
// input: [batchSize, inFeatures]  →  output: [batchSize, outFeatures]
const layer = new Linear(4, 2, true, "xavier");
const x     = new TensorValue(createTensor([1, 0, -1, 0.5], [1, 4]), "x");
const out   = layer.forward(x);
console.log("output shape:", out.data.shape, "  expected: [1, 2]");

// ─── E2: Batch forward ────────────────────────────────────────────────────────
const x2 = new TensorValue(randn([8, 4]), "x2");
const out2 = layer.forward(x2);
console.log("batch output shape:", out2.data.shape, "  expected: [8, 2]");

// ─── E3: Backward pass + gradient shapes ─────────────────────────────────────
// backward() needs a scalar root (Ch 10's guard), so collapse the [8,2]
// output with .sum() first.
out2.sum().backward();
const [W, b] = layer.parameters();
console.log("\nW.grad shape:", W!.grad?.shape,  "  expected: [2, 4]");
console.log("b.grad shape:", b!.grad?.shape,  "  expected: [2]");

// ─── E3b: the chapter's hand-set layer — where Ch 12's logits came from ──────
// Overwrite the random weights with section 3's chosen values and check the
// layer reproduces [1, 2, 3] from x = [1, 2].
const handSet = new Linear(2, 3);
handSet.weight.data = createTensor([1, 0, 0, 1, 0.5, 0.5], [3, 2]);
handSet.bias!.data = createTensor([0, 0, 1.5], [3]);
const logits = handSet.forward(new TensorValue(createTensor([1, 2], [1, 2])));
console.log("\nhand-set layer output:", Array.from(logits.data.data), "  expected: [1, 2, 3]");

// And the gradients the chapter derives: b.grad = p − y, W.grad row i = (p−y)ᵢ·x
import { crossEntropyFromLogits } from "../src/nn/losses.ts";
crossEntropyFromLogits(logits, createTensor([1, 0, 0], [1, 3])).backward();
console.log("b.grad:", Array.from(handSet.bias!.grad!.data).map((v) => v.toFixed(6)).join("  "));
console.log("        expected -0.909969  0.244728  0.665241   (= p − y)");
console.log("W.grad rows:");
const wg = Array.from(handSet.weight.grad!.data);
for (let i = 0; i < 3; i++) {
  console.log(`  [${wg[2 * i]!.toFixed(6)}, ${wg[2 * i + 1]!.toFixed(6)}]`);
}
console.log("  each row is (p − y)ᵢ × [1, 2] — feature 2 gets twice the gradient");

// ─── E4: Weight initialisation comparison ────────────────────────────────────
// He init → weights scale with sqrt(2/inputDim)  — appropriate before ReLU/GELU.
// Xavier  → scale with sqrt(1/inputDim)          — the doc's forward-only form.
//   (Glorot's original averages forward and backward: sqrt(2/(in+out)). The
//    chapter's section 18 explains the difference; we use the simpler one.)
const heLayer  = new Linear(512, 256, false, "he");
const xvLayer  = new Linear(512, 256, false, "xavier");
const normLayer = new Linear(512, 256, false, "normal");

function weightStd(l: Linear): number {
  const d = l.weight.data.data;
  const mean = Array.from(d).reduce((a, b) => a + b, 0) / d.length;
  const variance = Array.from(d).reduce((a, x) => a + (x - mean) ** 2, 0) / d.length;
  return Math.sqrt(variance);
}
console.log("\nHe    std:", weightStd(heLayer).toFixed(4),   "  expected: ≈", (Math.sqrt(2/512)).toFixed(4));
console.log("Xavier std:", weightStd(xvLayer).toFixed(4),   "  expected: ≈", (Math.sqrt(1/512)).toFixed(4));
console.log("Normal std:", weightStd(normLayer).toFixed(4), "  expected: ≈ 0.0200");

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: stack 3 Linear layers and verify that backprop flows through all.
//   h1 = relu(linear1(x))
//   h2 = relu(linear2(h1))
//   y  = linear3(h2)
//   Compute loss = sum(y), backward, check linear1.weight.grad is not null.
