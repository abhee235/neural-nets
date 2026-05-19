/**
 * EXERCISES — Ch 01: Tensor Type System
 * ══════════════════════════════════════
 * Prereq : src/tensor/types.ts implemented
 * Run    : bun run exercises/ch-01-tensor-type-system.ts
 *
 * A Tensor is just a flat Float64Array + a shape array.
 * Everything else (slicing, broadcasting, autograd) is built on top.
 */
import { createTensor, scalar, isTensor, flatIndex, toString } from "../src/tensor/types.ts";

// ─── E1: Basic construction ───────────────────────────────────────────────────
// TODO: create a 2×3 matrix containing [1..6].
//       Verify shape, ndim, size.
const m = createTensor([1, 2, 3, 4, 5, 6], [2, 3]);
console.log("shape :", m.shape, "   expected: [2,3]");
console.log("ndim  :", m.ndim,  "   expected: 2");
console.log("size  :", m.size,  "   expected: 6");

// ─── E2: Flat-index arithmetic ────────────────────────────────────────────────
// Row-major formula:  offset = Σ_i  index[i] * stride[i]
// For a [2,3] tensor, stride = [3, 1].
// Element at [1, 2]  →  1*3 + 2*1 = 5
const offset = flatIndex(m, [1, 2]);
console.log("\nflatIndex([1,2]):", offset, "  expected: 5");
console.log("data[5]         :", m.data[offset], "  expected: 6");

// ─── E3: Scalar tensor ────────────────────────────────────────────────────────
// A scalar is rank-0: shape=[], ndim=0, size=1.
// This is how loss values are represented in the autograd graph (Ch 07-08).
const loss = scalar(3.14);
console.log("\nscalar shape:", loss.shape, "  expected: []");
console.log("scalar ndim :", loss.ndim,  "  expected: 0");
console.log("scalar data :", loss.data[0], "  expected: 3.14");

// ─── E4: Type guard ───────────────────────────────────────────────────────────
console.log("\nisTensor(m)     :", isTensor(m),      "  expected: true");
console.log("isTensor(42)    :", isTensor(42),     "  expected: false");
console.log("isTensor({})    :", isTensor({}),     "  expected: false");

// ─── E5: Pretty print ────────────────────────────────────────────────────────
// toString should render the matrix row by row, like numpy.
console.log("\n" + toString(m));
// expected:
//   [[1, 2, 3],
//    [4, 5, 6]]

// ─── STRETCH: mismatched shape ────────────────────────────────────────────────
// TODO: try createTensor([1,2,3], [2,3]) — it should throw.
//       Wrap in try/catch and print the error message.
