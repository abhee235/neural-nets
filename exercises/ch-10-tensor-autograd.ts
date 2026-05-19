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
import { createTensor, randn } from "../src/tensor/creation.ts";

// ─── E1: Simple tensor add ────────────────────────────────────────────────────
const a = new TensorValue(createTensor([1, 2, 3, 4], [2, 2]), "a");
const b = new TensorValue(createTensor([10, 20, 30, 40], [2, 2]), "b");
const c = a.add(b);
c.backward();
console.log("c.data:", Array.from(c.data.data));  // [11,22,31,42] ... wait [11,22,33,44]
// Gradient of add: dc/da = 1 (ones tensor), dc/db = 1
console.log("a.grad shape:", a.grad?.shape, "  expected: [2,2]");

// ─── E2: Tensor matMul gradient ──────────────────────────────────────────────
// C = A @ B
// dL/dA = dL/dC @ Bᵀ
// dL/dB = Aᵀ @ dL/dC
const A = new TensorValue(createTensor([1,2,3,4,5,6], [2,3]), "A");
const B = new TensorValue(createTensor([1,0,0,1,0,0], [3,2]), "B");
const C = A.matMul(B);
C.backward();
console.log("\nC shape:", C.data.shape, "  expected: [2,2]");
console.log("dL/dA shape:", A.grad?.shape, "  expected: [2,3]");
console.log("dL/dB shape:", B.grad?.shape, "  expected: [3,2]");

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

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: Implement a two-layer computation:
//   h = relu(X @ W1 + b1)
//   y = h @ W2 + b2
// Using TensorValue. Run backward and print all gradients' shapes.
