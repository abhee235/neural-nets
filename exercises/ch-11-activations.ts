/**
 * EXERCISES — Ch 11: Activation Functions
 * ═════════════════════════════════════════
 * Prereq : src/nn/activations.ts + src/autograd/grad.ts implemented
 * Run    : bun run exercises/ch-11-activations.ts
 *
 * Activations introduce non-linearity.  Without them a stack of linear layers
 * collapses to a single matrix multiply.  GELU is used inside every FFN block.
 *
 * Note: backward() needs a scalar root (Ch 10's guard), so each example
 * collapses its activation output with .sum() before calling backward.
 * Summing means every output cell gets an upstream gradient of 1, which is
 * exactly the "out.grad = all ones" row the chapter traces by hand.
 */
import { relu, gelu, sigmoid, softmax } from "../src/nn/activations.ts";
import { TensorValue } from "../src/autograd/grad.ts";
import { createTensor } from "../src/tensor/types.ts";

// ─── E1: ReLU — the gate, and dead neurons ───────────────────────────────────
// relu(x) = max(0, x)  →  gradient = 0 for x <= 0  (dead neuron!)
const x = new TensorValue(createTensor([-2, -1, 0, 1, 2], [5]));
const r = relu(x);
r.sum().backward();
console.log("relu output:", Array.from(r.data.data));      // [0, 0, 0, 1, 2]
console.log("relu grad  :", Array.from(x.grad!.data));     // [0, 0, 0, 1, 1]
// The first three cells received exactly 0 — those units cannot learn.

// ─── E2: GELU — the smooth gate used in GPT ──────────────────────────────────
// GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715x³)))
// Unlike ReLU, GELU is smooth at x=0 — gradients never die completely.
const xg = new TensorValue(createTensor([-2, -1, 0, 1, 2], [5]));
const g = gelu(xg);
g.sum().backward();
console.log("\ngelu output:", Array.from(g.data.data).map((v) => v.toFixed(4)));
// expected: [-0.0454, -0.1588, 0.0000, 0.8412, 1.9546]
console.log("gelu grad  :", Array.from(xg.grad!.data).map((v) => v.toFixed(4)));
// expected: [-0.0861, -0.0830, 0.5000, 1.0830, 1.0861]
// Note gelu'(0) = 0.5 — half open — and the two negative entries: gelu is
// not monotonic, it dips below zero before flattening.

// ─── E3: Sigmoid saturation — the vanishing gradient, measured ───────────────
// At large |x|, sigmoid saturates → gradient ≈ 0 → gradients stop flowing.
const xBig = new TensorValue(createTensor([-10, -1, 0, 1, 10], [5]));
const s = sigmoid(xBig);
s.sum().backward();
console.log("\nsigmoid output:", Array.from(s.data.data).map((v) => v.toFixed(6)));
console.log("sigmoid grad  :", Array.from(xBig.grad!.data).map((v) => v.toFixed(6)));
// The ±10 entries are ≈ 0.0000454 — saturated. The peak, at x=0, is only 0.25.
// Ten layers of that best case: 0.25^10 ≈ 9.5e-7.  Ten layers of relu: 1.

// ─── E4: Compare activation outputs on the same input ───────────────────────
const vals = [-2, -1, -0.5, 0, 0.5, 1, 2];
console.log("\n   x   |  relu  |   gelu  | sigmoid");
for (const v of vals) {
  const tv = new TensorValue(createTensor([v], [1]));
  const rv = relu(tv).data.data[0]!;
  const gv = gelu(tv).data.data[0]!;
  const sv = sigmoid(tv).data.data[0]!;
  console.log(v.toFixed(1).padStart(5), "|", rv.toFixed(4), "|", gv.toFixed(4), "|", sv.toFixed(4));
}

// ─── E5: Softmax — a probability distribution, and shift invariance ─────────
const logits = new TensorValue(createTensor([1, 2, 3], [3]));
const p = softmax(logits);
const total = Array.from(p.data.data).reduce((a, b) => a + b, 0);
console.log("\nsoftmax([1,2,3]):", Array.from(p.data.data).map((v) => v.toFixed(6)));
console.log("sums to:", total.toFixed(6), " expected: 1.000000");

// Subtracting a constant changes nothing — this is what makes the max
// subtraction safe, and it is why exp(1000) never appears.
const shifted = softmax(new TensorValue(createTensor([1000, 1001, 1002], [3])));
console.log("softmax([1000,1001,1002]):", Array.from(shifted.data.data).map((v) => v.toFixed(6)));
console.log("  identical to [1,2,3] — shift invariance (Ch 05 deep dive)");

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: implement swish: swish(x) = x * sigmoid(x)
//       This one you CAN compose — sigmoid is now a primitive, and mul is a
//       TensorValue method, so no new _backward is needed. Try it and confirm
//       the gradient still checks out. Then compare swish against gelu on the
//       running row: they are close, which is why both are used in practice.
