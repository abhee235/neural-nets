/**
 * EXERCISES — Ch 02: Tensor Creation
 * ════════════════════════════════════
 * Prereq : src/tensor/creation.ts + types.ts implemented
 * Run    : bun run exercises/ch-02-tensor-creation.ts
 *
 * Every neural-net weight matrix starts with one of these factory functions.
 * randn (Box-Muller) seeds every Linear layer (Ch 13).
 */
import { zeros, ones, fill, eye, arange, linspace, randn, fullLike } from "../src/tensor/creation.ts";
import { toString } from "../src/tensor/types.ts";

// ─── E1: zeros / ones / fill ──────────────────────────────────────────────────
const z = zeros([3, 3]);
console.log("zeros(3×3):\n" + toString(z));

const o = ones([2, 4]);
console.log("ones(2×4):\n" + toString(o));

// ─── E2: Identity matrix (eye) ────────────────────────────────────────────────
// Used in residual connections: output = layer(x) + eye * x
const I = eye(4);
console.log("eye(4):\n" + toString(I));
// Every diagonal element must equal 1, off-diagonals 0.
let diagOk = true;
for (let i = 0; i < 4; i++)
  for (let j = 0; j < 4; j++) {
    const expected = i === j ? 1 : 0;
    if (I.data[i * 4 + j] !== expected) diagOk = false;
  }
console.log("diagonal correct:", diagOk, "  expected: true");

// ─── E3: arange + linspace ────────────────────────────────────────────────────
const r = arange(0, 12, 1);   // [0, 1, 2, ..., 11]
console.log("\narange(0,12):", Array.from(r.data).slice(0, 12));

const l = linspace(0, 1, 5);  // [0, 0.25, 0.5, 0.75, 1.0]
console.log("linspace(0,1,5):", Array.from(l.data));

// ─── E4: randn — Box-Muller normal distribution ───────────────────────────────
// TODO: generate 10,000 samples and verify:
//   mean ≈ 0 (within ±0.05)
//   std  ≈ 1 (within ±0.05)
const big = randn([10_000]);
const mean = Array.from(big.data).reduce((s, x) => s + x, 0) / 10_000;
const variance = Array.from(big.data).reduce((s, x) => s + (x - mean) ** 2, 0) / 10_000;
const std = Math.sqrt(variance);
console.log("\nrandn mean:", mean.toFixed(3), "  expected: ≈ 0");
console.log("randn std :", std.toFixed(3),  "  expected: ≈ 1");

// ─── STRETCH: reshape arange into matrix ──────────────────────────────────────
// TODO: use arange(0, 12, 1) and reshape it into [3, 4].
//       Print it — should look like numpy.arange(12).reshape(3,4).
