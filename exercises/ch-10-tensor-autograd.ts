/**
 * EXERCISES — Ch 10: Tensor Autograd Bridge
 * ══════════════════════════════════════════
 * Prereq : src/autograd/grad.ts + tensor/* implemented
 * Run    : bun run exercises/ch-10-tensor-autograd.ts
 *
 * TensorValue wraps a full Tensor (not a scalar), enabling gradient flow
 * through matrix operations — the foundation of every layer in the network.
 */
import { TensorValue } from "../src/autograd/grad.ts";
import { createTensor } from "../src/tensor/types.ts";

// ─── E1: Simple tensor add ────────────────────────────────────────────────────
const a = new TensorValue(createTensor([1, 2, 3, 4], [2, 2]));
const b = new TensorValue(createTensor([10, 20, 30, 40], [2, 2]));
const c = a.add(b);
// Collapse to one number before backward — the seed needs a scalar root.
const s = c.sum();
s.backward();
console.log("c.data:", Array.from(c.data.data), "  expected: [11,22,33,44]");
// Gradient of add through sum: every cell of a and b receives 1.
console.log("a.grad shape:", a.grad?.shape, "  expected: [2,2], all ones");

// ─── E2: Tensor matMul gradient — GRAPH 2 from the chapter, exactly ──────────
// L = sum(A @ B)
// dL/dA = dL/dC @ Bᵀ        → [10 26 42; 10 26 42]   (row-sums of B)
// dL/dB = Aᵀ @ dL/dC        → [5×4; 7×4; 9×4]        (column-sums of A)
const A = new TensorValue(createTensor([1,2,3,4,5,6], [2,3]));
const B = new TensorValue(createTensor([1,2,3,4,5,6,7,8,9,10,11,12], [3,4]));
const C = A.matMul(B);
const L = C.sum();
L.backward();
console.log("\nC shape:", C.data.shape, "  expected: [2,4]");
console.log("C.data:", Array.from(C.data.data), "  expected: [38,44,50,56,83,98,113,128]");
console.log("dL/dA:", A.grad ? Array.from(A.grad.data) : null, "  expected: [10,26,42,10,26,42]");
console.log("dL/dB shape:", B.grad?.shape, "  expected: [3,4], rows of 5s, 7s, 9s");

// ─── E3: Verify gradients numerically ────────────────────────────────────────
// For a scalar output, the gradient should match finite differences.
import { numericalGradientTensor } from "../src/utils/numerical.ts";
// sum(A @ B) — scalar loss
// TODO: use numericalGradientTensor to verify A.grad element-by-element.
// Expected: max element-wise error < 1e-5

// ─── E4: sumToShape (broadcasting backward) ──────────────────────────────────
// When we broadcast a [1,3] bias across [4,3] activations,
// the gradient of the loss w.r.t. bias must sum across the batch dimension.
import { sumToShape } from "../src/autograd/grad.ts";
const bigGrad = createTensor(Array.from({length:12}, () => 1), [4, 3]);
const reducedGrad = sumToShape(bigGrad, [1, 3]);
console.log("\nsumToShape [4,3]→[1,3]:", Array.from(reducedGrad.data), "  expected: [4,4,4]");

// ─── STRETCH — the chapter's closing checkpoint ──────────────────────────────
// TODO: Build  L = mean( (X @ W) + bias )  with
//   X    [4, 3]   input rows
//   W    [3, 2]   weights
//   bias [1, 2]   broadcast across all 4 rows
// Run backward and confirm every gradient comes back with its OWN shape —
// bias.grad summed down the broadcast axis to [1, 2]. One expression, and it
// exercises matMul backward, sumToShape, and reduction backward together:
// a Linear layer in everything but name (Ch 13 gives it the name).
// (relu on tensors arrives in Ch 11 — nothing here needs it yet.)
