/**
 * EXERCISES — Ch 11: Activation Functions
 * ═════════════════════════════════════════
 * Prereq : src/nn/activations.ts + src/autograd/grad.ts implemented
 * Run    : bun run exercises/ch-11-activations.ts
 *
 * Activations introduce non-linearity.  Without them a stack of linear layers
 * collapses to a single matrix multiply.  GELU is used inside every FFN block.
 */
import { relu, gelu, sigmoid, softmax } from "../src/nn/activations.ts";
import { TensorValue } from "../src/autograd/grad.ts";
import { createTensor } from "../src/tensor/creation.ts";

// ─── E1: ReLU — dead neurons ──────────────────────────────────────────────────
// relu(x) = max(0, x)  →  gradient = 0 for x <= 0  (dead neuron!)
const x = new TensorValue(createTensor([-2, -1, 0, 1, 2], [5]), "x");
const r = relu(x);
r.backward();
console.log("relu output:", Array.from(r.data.data));   // [0, 0, 0, 1, 2]
console.log("relu grad  :", Array.from(x.grad!.data)); // [0, 0, 0, 1, 1]

// ─── E2: GELU — smooth ReLU used in GPT ──────────────────────────────────────
// GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715x³)))
// Unlike ReLU, GELU is smooth at x=0 — gradients never die completely.
const xg = new TensorValue(createTensor([-2, -1, 0, 1, 2], [5]), "xg");
const g  = gelu(xg);
console.log("\ngelu output:", Array.from(g.data.data).map(v => v.toFixed(4)));
// expected approx: [-0.0454, -0.1588,  0.0000,  0.8413,  1.9546]

// ─── E3: Sigmoid saturation ───────────────────────────────────────────────────
// At large |x|, sigmoid saturates → gradient ≈ 0 → vanishing gradient.
const xBig = new TensorValue(createTensor([-10, -1, 0, 1, 10], [5]), "xBig");
const s    = sigmoid(xBig);
s.backward();
console.log("\nsigmoid output:", Array.from(s.data.data).map(v => v.toFixed(6)));
console.log("sigmoid grad  :", Array.from(xBig.grad!.data).map(v => v.toFixed(6)));
// gradients at ±10 should be nearly 0

// ─── E4: Compare activation outputs on same input ────────────────────────────
const vals = [-2, -1, -0.5, 0, 0.5, 1, 2];
console.log("\n   x   |  relu  |   gelu  | sigmoid");
for (const v of vals) {
  const tv = new TensorValue(createTensor([v], [1]));
  const rv = relu(tv).data.data[0]!;
  const gv = gelu(tv).data.data[0]!;
  const sv = sigmoid(tv).data.data[0]!;
  console.log(v.toFixed(1).padStart(5), "|", rv.toFixed(4), "|", gv.toFixed(4), "|", sv.toFixed(4));
}

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: implement swish: swish(x) = x * sigmoid(x)
//       using existing ops in TensorValue.
//       Compare swish vs GELU side-by-side.
