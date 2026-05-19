/**
 * EXERCISES — Ch 04: Matrix Operations
 * ══════════════════════════════════════
 * Prereq : src/tensor/linalg.ts + creation.ts implemented
 * Run    : bun run exercises/ch-04-matrix-ops.ts
 *
 * matMul is the single most-called function in a transformer forward pass.
 * Every QKV projection, every attention score, every FFN linear step is matMul.
 */
import { matMul, transpose, reshape, flatten, squeeze, unsqueeze } from "../src/tensor/linalg.ts";
import { createTensor, eye, randn } from "../src/tensor/creation.ts";
import { toString } from "../src/tensor/types.ts";

// ─── E1: matMul correctness ──────────────────────────────────────────────────
// A (2×3) @ B (3×2) → C (2×2)
// C[i,j] = Σ_k A[i,k] * B[k,j]
const A = createTensor([1,2,3, 4,5,6], [2, 3]);
const B = createTensor([7,8, 9,10, 11,12], [3, 2]);
const C = matMul(A, B);
console.log("A @ B:\n" + toString(C));
// expected: [[58,64],[139,154]]

// ─── E2: Identity property  A @ I = A ────────────────────────────────────────
const M = createTensor([1,2,3,4,5,6,7,8,9], [3, 3]);
const I = eye(3);
const MI = matMul(M, I);
let identityOk = Array.from(M.data).every((v, i) => Math.abs(v - MI.data[i]) < 1e-10);
console.log("\nM @ I == M:", identityOk, "  expected: true");

// ─── E3: Transpose ───────────────────────────────────────────────────────────
// In attention: scores = Q @ Kᵀ / sqrt(dk)
// We need Kᵀ to go from [seqLen, dk] → [dk, seqLen]
const K = createTensor([1,2,3, 4,5,6], [2, 3]);   // [2, 3]
const Kt = transpose(K);                            // [3, 2]
console.log("\nK shape  :", K.shape);   // [2, 3]
console.log("Kᵀ shape :", Kt.shape);   // [3, 2]
console.log("Kᵀ:\n" + toString(Kt));

// ─── E4: reshape + flatten ───────────────────────────────────────────────────
// After multi-head attention, we concat heads:
// [batch, numHeads, seqLen, dHead] → [batch, seqLen, dModel]
const x = createTensor(Array.from({length: 24}, (_, i) => i), [2, 3, 4]);
const flat = flatten(x);
console.log("\nflatten [2,3,4]:", flat.shape, " expected: [24]");

const r = reshape(x, [6, 4]);
console.log("reshape → [6,4]:", r.shape, " expected: [6,4]");

// ─── E5: squeeze / unsqueeze ─────────────────────────────────────────────────
// Needed constantly to add/remove batch dimensions.
const v = createTensor([1, 2, 3], [1, 3]);
const sq = squeeze(v);
console.log("\nsqueeze [1,3]:", sq.shape, " expected: [3]");
const us = unsqueeze(sq, 0);
console.log("unsqueeze [3] at dim 0:", us.shape, " expected: [1,3]");

// ─── STRETCH: (Aᵀ)ᵀ == A ────────────────────────────────────────────────────
// TODO: prove transpose is its own inverse.
