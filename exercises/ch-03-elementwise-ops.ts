/**
 * EXERCISES — Ch 03: Element-wise Ops & Broadcasting
 * ════════════════════════════════════════════════════
 * Prereq : src/tensor/ops.ts + creation.ts implemented
 * Run    : bun run exercises/ch-03-elementwise-ops.ts
 *
 * Broadcasting is why you can add a bias vector [dModel] to an
 * activation matrix [batchSize, dModel] without a loop (Ch 13).
 */
import { add, sub, mul, div, addScalar, mulScalar, broadcastShapes, applyFn } from "../src/tensor/ops.ts";
import { createTensor, ones } from "../src/tensor/creation.ts";
import { toString } from "../src/tensor/types.ts";

// ─── E1: Element-wise arithmetic ─────────────────────────────────────────────
const a = createTensor([1, 2, 3, 4], [2, 2]);
const b = createTensor([10, 20, 30, 40], [2, 2]);
console.log("a + b:\n" + toString(add(a, b)));  // [[11,22],[33,44]]
console.log("a * b:\n" + toString(mul(a, b)));  // [[10,40],[90,160]]

// ─── E2: Scalar operations ────────────────────────────────────────────────────
// In the transformer: scale attention scores by 1/sqrt(dModel).
const scores = createTensor([1, 2, 3, 4, 5, 6], [2, 3]);
const dModel = 9;
const scaled = mulScalar(scores, 1 / Math.sqrt(dModel));
console.log("\nscaled (÷√9):", Array.from(scaled.data).map(x => x.toFixed(4)));
// expected: [0.3333, 0.6667, 1.0000, 1.3333, 1.6667, 2.0000]

// ─── E3: broadcastShapes ─────────────────────────────────────────────────────
// Broadcasting rules (numpy-compatible):
//   [3, 1] + [1, 4]  →  [3, 4]
//   [2, 3] + [3]     →  [2, 3]
console.log("\nbroadcast [3,1]+[1,4]:", broadcastShapes([3, 1], [1, 4])); // [3,4]
console.log("broadcast [2,3]+[3] :", broadcastShapes([2, 3], [3]));    // [2,3]
console.log("broadcast [1]+[4,1] :", broadcastShapes([1], [4, 1]));    // [4,1]

// ─── E4: applyFn — custom element-wise transform ─────────────────────────────
// applyFn lets you write any per-element operation.
// Use it to implement clamp-to-zero (a simple ReLU preview):
const x = createTensor([-3, -1, 0, 2, 5], [5]);
const relu = applyFn(x, v => Math.max(0, v));
console.log("\nrelu([-3,-1,0,2,5]):", Array.from(relu.data)); // [0,0,0,2,5]

// ─── E5: Broadcasting add — bias vector ──────────────────────────────────────
// Add bias [1, 3] to activations [4, 3]:
// Each row of activations gets the same bias added — this is nn.Linear's bias.
const activations = createTensor([1,2,3, 4,5,6, 7,8,9, 10,11,12], [4, 3]);
const bias        = createTensor([0.1, 0.2, 0.3], [1, 3]);
const out = add(activations, bias); // must broadcast [4,3] + [1,3]
console.log("\nbias-added row 0:", Array.from(out.data).slice(0, 3)); // [1.1,2.2,3.3]

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: implement z-score normalisation using only add/sub/mul/div + mean/std.
//   z = (x - mean) / std
