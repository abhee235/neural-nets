/**
 * EXERCISES — Ch 06: Math Primitives
 * ════════════════════════════════════
 * Prereq : src/tensor/math.ts + reduce.ts + creation.ts implemented
 * Run    : bun run exercises/ch-06-math-primitives.ts
 *
 * These element-wise math ops underpin every activation function (Ch 11),
 * the log-sum-exp trick (Ch 12), and attention scaling (Ch 22).
 */
import { exp, log, sqrt, pow, abs, clip, tanh, sigmoid } from "../src/tensor/math.ts";
import { createTensor } from "../src/tensor/creation.ts";

// ─── E1: exp / log round-trip ────────────────────────────────────────────────
// log(exp(x)) == x  (within floating-point tolerance)
const x = createTensor([0, 1, 2, -1, 0.5], [5]);
const roundTrip = log(exp(x));
const maxErr = Math.max(...Array.from(x.data).map((v, i) => Math.abs(v - roundTrip.data[i])));
console.log("log(exp(x)) max error:", maxErr.toExponential(2), "  expected: < 1e-10");

// ─── E2: tanh and sigmoid ────────────────────────────────────────────────────
// tanh  output range: (-1, +1)     — gating in LSTMs
// sigmoid output range: (0, +1)    — binary probability
const vals = createTensor([-2, -1, 0, 1, 2], [5]);
const tanhOut    = tanh(vals);
const sigOut     = sigmoid(vals);
console.log("\ntanh  :", Array.from(tanhOut.data).map(v => v.toFixed(4)));
console.log("sigmoid:", Array.from(sigOut.data).map(v => v.toFixed(4)));
// Verify range constraints:
console.log("all tanh in (-1,1)  :", tanhOut.data.every(v => v > -1 && v < 1));
console.log("all sigma in (0,1)  :", sigOut.data.every(v => v > 0 && v < 1));

// ─── E3: clip ────────────────────────────────────────────────────────────────
// Gradient clipping prevents exploding gradients during backprop.
// clip(x, -1, 1) should box every value to [-1, 1].
const raw = createTensor([-5, -1, 0, 1, 5], [5]);
const clipped = clip(raw, -1, 1);
console.log("\nclip([-5,-1,0,1,5], -1, 1):", Array.from(clipped.data));
// expected: [-1, -1, 0, 1, 1]

// ─── E4: pow — weight initialisation scale ───────────────────────────────────
// Xavier init uses sqrt(2 / (fanIn + fanOut)) = (fanIn+fanOut)^-0.5
// pow(t, 0.5) is sqrt; pow(t, -0.5) is reciprocal sqrt.
const fanSum = createTensor([512, 256, 128], [3]);
const xavierScale = pow(fanSum, -0.5);
console.log("\nxavier scale for [512,256,128]:", Array.from(xavierScale.data).map(v => v.toFixed(6)));

// ─── E5: log — cross-entropy loss preview ────────────────────────────────────
// Cross-entropy: L = -log(p_correct)
// Perfect prediction (p=1.0) → loss = 0.
// Random (p=1/vocabSize for vocabSize=50k) → loss ≈ log(50000) ≈ 10.8
const pCorrect = createTensor([1.0, 0.5, 0.1, 1/50000], [4]);
const ce = log(pCorrect); // will be negated
console.log("\n-log(p):", Array.from(ce.data).map(v => (-v).toFixed(4)));
// expected: [0.0000, 0.6931, 2.3026, 10.8198]

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: implement GELU approximation using tanh:
//   gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
// Use only the ops from this chapter.
