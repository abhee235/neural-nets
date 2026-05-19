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
import { createTensor, randn } from "../src/tensor/creation.ts";

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
out2.backward();
const [W, b] = layer.parameters();
console.log("\nW.grad shape:", W!.grad?.shape,  "  expected: [2, 4]");
console.log("b.grad shape:", b!.grad?.shape,  "  expected: [2]");

// ─── E4: Weight initialisation comparison ────────────────────────────────────
// He init → weights scale with sqrt(2/fanIn)  — appropriate before ReLU/GELU.
// Xavier  → scale with sqrt(2/(fanIn+fanOut)) — appropriate for output layers.
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
console.log("Xavier std:", weightStd(xvLayer).toFixed(4),   "  expected: ≈", (Math.sqrt(2/(512+256))).toFixed(4));
console.log("Normal std:", weightStd(normLayer).toFixed(4), "  expected: ≈ 0.0200");

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: stack 3 Linear layers and verify that backprop flows through all.
//   h1 = relu(linear1(x))
//   h2 = relu(linear2(h1))
//   y  = linear3(h2)
//   Compute loss = sum(y), backward, check linear1.weight.grad is not null.
