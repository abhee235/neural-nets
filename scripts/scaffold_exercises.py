#!/usr/bin/env python3
"""
scaffold_exercises.py
Creates exercises/ at the repo root — one .ts file per chapter (Ch 01–30).
Run:  python3 scripts/scaffold_exercises.py
"""

import os, textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX   = os.path.join(ROOT, "exercises")
os.makedirs(EX, exist_ok=True)

# ─── Helper ───────────────────────────────────────────────────────────────────

def write(filename: str, body: str):
    path = os.path.join(EX, filename)
    with open(path, "w") as f:
        f.write(textwrap.dedent(body).lstrip())
    print(f"  {filename}")

# ─── Chapter files ────────────────────────────────────────────────────────────

write("ch-01-tensor-type-system.ts", """
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
console.log("\\nflatIndex([1,2]):", offset, "  expected: 5");
console.log("data[5]         :", m.data[offset], "  expected: 6");

// ─── E3: Scalar tensor ────────────────────────────────────────────────────────
// A scalar is rank-0: shape=[], ndim=0, size=1.
// This is how loss values are represented in the autograd graph (Ch 07-08).
const loss = scalar(3.14);
console.log("\\nscalar shape:", loss.shape, "  expected: []");
console.log("scalar ndim :", loss.ndim,  "  expected: 0");
console.log("scalar data :", loss.data[0], "  expected: 3.14");

// ─── E4: Type guard ───────────────────────────────────────────────────────────
console.log("\\nisTensor(m)     :", isTensor(m),      "  expected: true");
console.log("isTensor(42)    :", isTensor(42),     "  expected: false");
console.log("isTensor({})    :", isTensor({}),     "  expected: false");

// ─── E5: Pretty print ────────────────────────────────────────────────────────
// toString should render the matrix row by row, like numpy.
console.log("\\n" + toString(m));
// expected:
//   [[1, 2, 3],
//    [4, 5, 6]]

// ─── STRETCH: mismatched shape ────────────────────────────────────────────────
// TODO: try createTensor([1,2,3], [2,3]) — it should throw.
//       Wrap in try/catch and print the error message.
""")

write("ch-02-tensor-creation.ts", """
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
console.log("zeros(3×3):\\n" + toString(z));

const o = ones([2, 4]);
console.log("ones(2×4):\\n" + toString(o));

// ─── E2: Identity matrix (eye) ────────────────────────────────────────────────
// Used in residual connections: output = layer(x) + eye * x
const I = eye(4);
console.log("eye(4):\\n" + toString(I));
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
console.log("\\narange(0,12):", Array.from(r.data).slice(0, 12));

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
console.log("\\nrandn mean:", mean.toFixed(3), "  expected: ≈ 0");
console.log("randn std :", std.toFixed(3),  "  expected: ≈ 1");

// ─── STRETCH: reshape arange into matrix ──────────────────────────────────────
// TODO: use arange(0, 12, 1) and reshape it into [3, 4].
//       Print it — should look like numpy.arange(12).reshape(3,4).
""")

write("ch-03-elementwise-ops.ts", """
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
console.log("a + b:\\n" + toString(add(a, b)));  // [[11,22],[33,44]]
console.log("a * b:\\n" + toString(mul(a, b)));  // [[10,40],[90,160]]

// ─── E2: Scalar operations ────────────────────────────────────────────────────
// In the transformer: scale attention scores by 1/sqrt(dModel).
const scores = createTensor([1, 2, 3, 4, 5, 6], [2, 3]);
const dModel = 9;
const scaled = mulScalar(scores, 1 / Math.sqrt(dModel));
console.log("\\nscaled (÷√9):", Array.from(scaled.data).map(x => x.toFixed(4)));
// expected: [0.3333, 0.6667, 1.0000, 1.3333, 1.6667, 2.0000]

// ─── E3: broadcastShapes ─────────────────────────────────────────────────────
// Broadcasting rules (numpy-compatible):
//   [3, 1] + [1, 4]  →  [3, 4]
//   [2, 3] + [3]     →  [2, 3]
console.log("\\nbroadcast [3,1]+[1,4]:", broadcastShapes([3, 1], [1, 4])); // [3,4]
console.log("broadcast [2,3]+[3] :", broadcastShapes([2, 3], [3]));    // [2,3]
console.log("broadcast [1]+[4,1] :", broadcastShapes([1], [4, 1]));    // [4,1]

// ─── E4: applyFn — custom element-wise transform ─────────────────────────────
// applyFn lets you write any per-element operation.
// Use it to implement clamp-to-zero (a simple ReLU preview):
const x = createTensor([-3, -1, 0, 2, 5], [5]);
const relu = applyFn(x, v => Math.max(0, v));
console.log("\\nrelu([-3,-1,0,2,5]):", Array.from(relu.data)); // [0,0,0,2,5]

// ─── E5: Broadcasting add — bias vector ──────────────────────────────────────
// Add bias [1, 3] to activations [4, 3]:
// Each row of activations gets the same bias added — this is nn.Linear's bias.
const activations = createTensor([1,2,3, 4,5,6, 7,8,9, 10,11,12], [4, 3]);
const bias        = createTensor([0.1, 0.2, 0.3], [1, 3]);
const out = add(activations, bias); // must broadcast [4,3] + [1,3]
console.log("\\nbias-added row 0:", Array.from(out.data).slice(0, 3)); // [1.1,2.2,3.3]

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: implement z-score normalisation using only add/sub/mul/div + mean/std.
//   z = (x - mean) / std
""")

write("ch-04-matrix-ops.ts", """
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
console.log("A @ B:\\n" + toString(C));
// expected: [[58,64],[139,154]]

// ─── E2: Identity property  A @ I = A ────────────────────────────────────────
const M = createTensor([1,2,3,4,5,6,7,8,9], [3, 3]);
const I = eye(3);
const MI = matMul(M, I);
let identityOk = Array.from(M.data).every((v, i) => Math.abs(v - MI.data[i]) < 1e-10);
console.log("\\nM @ I == M:", identityOk, "  expected: true");

// ─── E3: Transpose ───────────────────────────────────────────────────────────
// In attention: scores = Q @ Kᵀ / sqrt(dk)
// We need Kᵀ to go from [seqLen, dk] → [dk, seqLen]
const K = createTensor([1,2,3, 4,5,6], [2, 3]);   // [2, 3]
const Kt = transpose(K);                            // [3, 2]
console.log("\\nK shape  :", K.shape);   // [2, 3]
console.log("Kᵀ shape :", Kt.shape);   // [3, 2]
console.log("Kᵀ:\\n" + toString(Kt));

// ─── E4: reshape + flatten ───────────────────────────────────────────────────
// After multi-head attention, we concat heads:
// [batch, numHeads, seqLen, dHead] → [batch, seqLen, dModel]
const x = createTensor(Array.from({length: 24}, (_, i) => i), [2, 3, 4]);
const flat = flatten(x);
console.log("\\nflatten [2,3,4]:", flat.shape, " expected: [24]");

const r = reshape(x, [6, 4]);
console.log("reshape → [6,4]:", r.shape, " expected: [6,4]");

// ─── E5: squeeze / unsqueeze ─────────────────────────────────────────────────
// Needed constantly to add/remove batch dimensions.
const v = createTensor([1, 2, 3], [1, 3]);
const sq = squeeze(v);
console.log("\\nsqueeze [1,3]:", sq.shape, " expected: [3]");
const us = unsqueeze(sq, 0);
console.log("unsqueeze [3] at dim 0:", us.shape, " expected: [1,3]");

// ─── STRETCH: (Aᵀ)ᵀ == A ────────────────────────────────────────────────────
// TODO: prove transpose is its own inverse.
""")

write("ch-05-reductions.ts", """
/**
 * EXERCISES — Ch 05: Reductions
 * ══════════════════════════════
 * Prereq : src/tensor/reduce.ts + creation.ts + linalg.ts implemented
 * Run    : bun run exercises/ch-05-reductions.ts
 *
 * softmax is used in EVERY attention head. sum/mean power LayerNorm (Ch 20).
 */
import { sum, mean, max, min, argmax, softmax } from "../src/tensor/reduce.ts";
import { createTensor } from "../src/tensor/creation.ts";

// ─── E1: sum and mean ────────────────────────────────────────────────────────
const x = createTensor([1, 2, 3, 4, 5, 6], [2, 3]);
console.log("sum (all)    :", sum(x).data[0],   "  expected: 21");
console.log("mean (all)   :", mean(x).data[0],  "  expected: 3.5");

// ─── E2: Axis reductions ──────────────────────────────────────────────────────
// sum along axis 0 (collapse rows) → [3] vector
const s0 = sum(x, 0);
console.log("\\nsum axis=0   :", Array.from(s0.data), "  expected: [5,7,9]");
// sum along axis 1 (collapse cols) → [2] vector
const s1 = sum(x, 1);
console.log("sum axis=1   :", Array.from(s1.data), "  expected: [6,15]");

// ─── E3: argmax ─────────────────────────────────────────────────────────────
// argmax picks the predicted class during inference.
const logits = createTensor([0.1, 0.9, 0.3, 0.7], [4]);
console.log("\\nargmax:", argmax(logits).data[0], "  expected: 1");

// ─── E4: softmax — the attention distribution ────────────────────────────────
// softmax(x)[i] = exp(x[i]) / Σ exp(x[j])
// Properties to verify:
//   1) all outputs > 0
//   2) outputs sum to 1
const raw = createTensor([2.0, 1.0, 0.1], [3]);
const probs = softmax(raw);
const probSum = Array.from(probs.data).reduce((a, b) => a + b, 0);
console.log("\\nsoftmax probs:", Array.from(probs.data).map(v => v.toFixed(4)));
console.log("sum of probs :", probSum.toFixed(6), "  expected: 1.000000");

// ─── E5: Numerical stability — large logits ──────────────────────────────────
// Naive softmax overflows at logit=1000.  Your implementation must subtract max first.
// x - max(x) before exp prevents exp(1000) = Infinity.
const bigLogits = createTensor([1000, 1001, 1002], [3]);
const stableSoftmax = softmax(bigLogits);
const bigSum = Array.from(stableSoftmax.data).reduce((a, b) => a + b, 0);
console.log("\\nstable softmax sum:", bigSum.toFixed(6), "  expected: 1.000000");
console.log("no NaN:", !stableSoftmax.data.some(isNaN), "  expected: true");

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: compute softmax along axis=1 for a [4, 3] logit matrix.
//       Each row should sum to 1.
""")

write("ch-06-math-primitives.ts", """
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
console.log("\\ntanh  :", Array.from(tanhOut.data).map(v => v.toFixed(4)));
console.log("sigmoid:", Array.from(sigOut.data).map(v => v.toFixed(4)));
// Verify range constraints:
console.log("all tanh in (-1,1)  :", tanhOut.data.every(v => v > -1 && v < 1));
console.log("all sigma in (0,1)  :", sigOut.data.every(v => v > 0 && v < 1));

// ─── E3: clip ────────────────────────────────────────────────────────────────
// Gradient clipping prevents exploding gradients during backprop.
// clip(x, -1, 1) should box every value to [-1, 1].
const raw = createTensor([-5, -1, 0, 1, 5], [5]);
const clipped = clip(raw, -1, 1);
console.log("\\nclip([-5,-1,0,1,5], -1, 1):", Array.from(clipped.data));
// expected: [-1, -1, 0, 1, 1]

// ─── E4: pow — weight initialisation scale ───────────────────────────────────
// Xavier init uses sqrt(2 / (fanIn + fanOut)) = (fanIn+fanOut)^-0.5
// pow(t, 0.5) is sqrt; pow(t, -0.5) is reciprocal sqrt.
const fanSum = createTensor([512, 256, 128], [3]);
const xavierScale = pow(fanSum, -0.5);
console.log("\\nxavier scale for [512,256,128]:", Array.from(xavierScale.data).map(v => v.toFixed(6)));

// ─── E5: log — cross-entropy loss preview ────────────────────────────────────
// Cross-entropy: L = -log(p_correct)
// Perfect prediction (p=1.0) → loss = 0.
// Random (p=1/vocabSize for vocabSize=50k) → loss ≈ log(50000) ≈ 10.8
const pCorrect = createTensor([1.0, 0.5, 0.1, 1/50000], [4]);
const ce = log(pCorrect); // will be negated
console.log("\\n-log(p):", Array.from(ce.data).map(v => (-v).toFixed(4)));
// expected: [0.0000, 0.6931, 2.3026, 10.8198]

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: implement GELU approximation using tanh:
//   gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
// Use only the ops from this chapter.
""")

write("ch-07-calculus-for-ml.ts", """
/**
 * EXERCISES — Ch 07: Calculus for ML
 * ════════════════════════════════════
 * Prereq : src/tensor/math.ts + ops.ts implemented
 * Run    : bun run exercises/ch-07-calculus-for-ml.ts
 *
 * Before writing autograd, feel these derivatives by hand:
 * finite-difference approximation: df/dx ≈ (f(x+h) - f(x-h)) / 2h
 */

// ─── E1: Derivative of x² at x=3 ────────────────────────────────────────────
// Analytical: d/dx x² = 2x  →  at x=3 → 6
function f1(x: number): number { return x * x; }
function numericalGrad(f: (x: number) => number, x: number, h = 1e-5): number {
  return (f(x + h) - f(x - h)) / (2 * h);
}
const grad_x2 = numericalGrad(f1, 3);
console.log("d/dx x² at x=3:", grad_x2.toFixed(6), "  expected: 6.000000");

// ─── E2: Derivative of sigmoid ───────────────────────────────────────────────
// sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
function sigmoidFn(x: number): number { return 1 / (1 + Math.exp(-x)); }
const x = 0.5;
const numerical = numericalGrad(sigmoidFn, x);
const sig        = sigmoidFn(x);
const analytical = sig * (1 - sig);
console.log("\\nsigmoid'(0.5) numerical :", numerical.toFixed(8));
console.log("sigmoid'(0.5) analytical:", analytical.toFixed(8));
console.log("match:", Math.abs(numerical - analytical) < 1e-5, "  expected: true");

// ─── E3: Chain rule — d/dx sin(x²) at x=1 ───────────────────────────────────
// Chain rule:  d/dx f(g(x)) = f'(g(x)) * g'(x)
// g(x) = x², g'(x) = 2x
// f(u) = sin(u), f'(u) = cos(u)
// → d/dx sin(x²) = cos(x²) * 2x  at x=1 = cos(1) * 2 ≈ 1.0806
function chainFn(x: number): number { return Math.sin(x * x); }
const grad_chain  = numericalGrad(chainFn, 1);
const analytical2 = Math.cos(1 * 1) * (2 * 1);
console.log("\\nd/dx sin(x²) at x=1 numerical :", grad_chain.toFixed(8));
console.log("d/dx sin(x²) at x=1 analytical:", analytical2.toFixed(8));

// ─── E4: Gradient descent step ───────────────────────────────────────────────
// Minimise f(x) = (x - 3)² starting from x=0.
// Gradient: f'(x) = 2(x - 3)
// Update:   x ← x - lr * f'(x)
// After enough steps x should be near 3.
let xParam = 0.0;
const lr = 0.1;
for (let step = 0; step < 50; step++) {
  const grad = 2 * (xParam - 3);
  xParam -= lr * grad;
}
console.log("\\nGD minimising (x-3)²: x =", xParam.toFixed(6), "  expected: ≈ 3.0");

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: numerically verify the gradient of tanh at x=0, x=1, x=2.
//   Analytical:  tanh'(x) = 1 - tanh(x)²
""")

write("ch-08-autograd.ts", """
/**
 * EXERCISES — Ch 08: Autograd (Forward + Backward)
 * ══════════════════════════════════════════════════
 * Prereq : src/autograd/value.ts + engine.ts implemented
 * Run    : bun run exercises/ch-08-autograd.ts
 *
 * Build a tiny computation graph by hand, then let .backward() fill in
 * all gradients automatically via reverse-mode autodiff.
 */
import { Value } from "../src/autograd/value.ts";

// ─── E1: Forward pass — simple expression ────────────────────────────────────
// f = (a + b) * c
// forward: a=2, b=3, c=4  →  f = (2+3)*4 = 20
const a = new Value(2, "a");
const b = new Value(3, "b");
const c = new Value(4, "c");
const sum = a.add(b);           // node: a+b = 5
const f   = sum.mul(c);         // node: (a+b)*c = 20
console.log("f.data:", f.data, "  expected: 20");

// ─── E2: Backward pass ───────────────────────────────────────────────────────
// df/da = c = 4
// df/db = c = 4
// df/dc = a+b = 5
f.backward();
console.log("\\ndf/da:", a.grad, "  expected: 4");
console.log("df/db:", b.grad, "  expected: 4");
console.log("df/dc:", c.grad, "  expected: 5");

// ─── E3: Non-linear — tanh activation ────────────────────────────────────────
// y = tanh(w * x + b)
// Verify gradient numerically using finite differences.
import { numericalGradient } from "../src/utils/numerical.ts";

const w = new Value(0.5, "w");
const x = new Value(1.0, "x");
const bv = new Value(0.2, "b");
const y  = w.mul(x).add(bv).tanh();
y.backward();

function yFn(wVal: number): number {
  return Math.tanh(wVal * 1.0 + 0.2);
}
const numGrad = numericalGradient(yFn, 0.5);
console.log("\\ndy/dw autograd  :", w.grad.toFixed(8));
console.log("dy/dw numerical :", numGrad.toFixed(8));
console.log("match:", Math.abs(w.grad - numGrad) < 1e-5, "  expected: true");

// ─── E4: Gradient accumulation ───────────────────────────────────────────────
// When a value is used twice, gradients accumulate (sum).
// z = a * a   →   dz/da = 2a
const av = new Value(3, "a");
const z = av.mul(av);
z.backward();
console.log("\\ndz/d(a) where z=a*a, a=3:", av.grad, "  expected: 6");

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: build a small 2-input neuron:
//   n = relu(w1*x1 + w2*x2 + b)
// Set w1=0.4, w2=-0.6, b=0.1, x1=1.0, x2=0.5.
// Verify all gradients numerically.
""")

write("ch-09-gradient-descent.ts", """
/**
 * EXERCISES — Ch 09: Gradient Descent & SGD
 * ══════════════════════════════════════════
 * Prereq : src/autograd/value.ts + src/optim/sgd.ts implemented
 * Run    : bun run exercises/ch-09-gradient-descent.ts
 *
 * Gradient descent is the engine that trains every model.
 * Here you watch it minimise a 2D bowl, then overfit a tiny dataset.
 */
import { Value } from "../src/autograd/value.ts";
import { SGD }   from "../src/optim/sgd.ts";

// ─── E1: Minimise a quadratic ─────────────────────────────────────────────────
// f(w) = (w - 5)²  →  minimum at w=5, f=0
let w = new Value(0.0, "w");
const sgd = new SGD([w], 0.1);

for (let step = 0; step < 100; step++) {
  const loss = w.add(new Value(-5)).pow(2);
  loss.backward();
  sgd.step();
  sgd.zeroGrad();
}
console.log("w after 100 steps:", w.data.toFixed(4), "  expected: ≈ 5.0");

// ─── E2: Learning-rate sensitivity ────────────────────────────────────────────
// Too large lr → diverges. Too small → converges slowly.
// TODO: try lr = 0.5 (should still converge), lr = 1.1 (should diverge).
function minimise(lr: number, steps: number): number {
  let ww = new Value(0.0);
  const opt = new SGD([ww], lr);
  for (let i = 0; i < steps; i++) {
    const loss = ww.add(new Value(-5)).pow(2);
    loss.backward();
    opt.step();
    opt.zeroGrad();
  }
  return ww.data;
}
console.log("\\nlr=0.01 →", minimise(0.01, 200).toFixed(3), "  (slower, still converges)");
console.log("lr=0.1  →", minimise(0.1,  100).toFixed(3), "  (fast convergence)");
console.log("lr=0.9  →", minimise(0.9,  100).toFixed(3), "  (oscillates but converges)");

// ─── E3: SGD with momentum ────────────────────────────────────────────────────
import { SGDMomentum } from "../src/optim/sgd.ts";

let wm = new Value(0.0, "wm");
const optM = new SGDMomentum([wm], 0.1, 0.9);
for (let step = 0; step < 50; step++) {
  const loss = wm.add(new Value(-5)).pow(2);
  loss.backward();
  optM.step();
  optM.zeroGrad();
}
console.log("\\nSGD+momentum after 50 steps:", wm.data.toFixed(4), "  expected: ≈ 5.0");

// ─── E4: Fit y = 2x + 1 on 5 points ─────────────────────────────────────────
// Dataset: xs=[0,1,2,3,4], ys=[1,3,5,7,9]
// Model: y_hat = slope*x + intercept   (linear regression via autograd)
const xs = [0, 1, 2, 3, 4];
const ys = [1, 3, 5, 7, 9];
let slope     = new Value(0.0, "slope");
let intercept = new Value(0.0, "intercept");
const linReg = new SGD([slope, intercept], 0.01);

for (let step = 0; step < 500; step++) {
  let mse = new Value(0);
  for (let i = 0; i < xs.length; i++) {
    const yHat = slope.mul(new Value(xs[i]!)).add(intercept);
    const err  = yHat.add(new Value(-ys[i]!));
    mse = mse.add(err.pow(2));
  }
  mse.backward();
  linReg.step();
  linReg.zeroGrad();
}
console.log("\\nLinear fit: slope =", slope.data.toFixed(3), " expected ≈ 2");
console.log("Linear fit: intercept =", intercept.data.toFixed(3), " expected ≈ 1");

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: plot the loss curve (print loss every 50 steps).
//       Does it decrease smoothly? What shape is it?
""")

write("ch-10-tensor-autograd.ts", """
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
console.log("\\nC shape:", C.data.shape, "  expected: [2,2]");
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
console.log("\\nsumToShape [4,3]→[1,3]:", Array.from(reducedGrad.data), "  expected: [4,4,4]");

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: Implement a two-layer computation:
//   h = relu(X @ W1 + b1)
//   y = h @ W2 + b2
// Using TensorValue. Run backward and print all gradients' shapes.
""")

write("ch-11-activations.ts", """
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
console.log("\\ngelu output:", Array.from(g.data.data).map(v => v.toFixed(4)));
// expected approx: [-0.0454, -0.1588,  0.0000,  0.8413,  1.9546]

// ─── E3: Sigmoid saturation ───────────────────────────────────────────────────
// At large |x|, sigmoid saturates → gradient ≈ 0 → vanishing gradient.
const xBig = new TensorValue(createTensor([-10, -1, 0, 1, 10], [5]), "xBig");
const s    = sigmoid(xBig);
s.backward();
console.log("\\nsigmoid output:", Array.from(s.data.data).map(v => v.toFixed(6)));
console.log("sigmoid grad  :", Array.from(xBig.grad!.data).map(v => v.toFixed(6)));
// gradients at ±10 should be nearly 0

// ─── E4: Compare activation outputs on same input ────────────────────────────
const vals = [-2, -1, -0.5, 0, 0.5, 1, 2];
console.log("\\n   x   |  relu  |   gelu  | sigmoid");
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
""")

write("ch-12-losses.ts", """
/**
 * EXERCISES — Ch 12: Loss Functions
 * ═══════════════════════════════════
 * Prereq : src/nn/losses.ts + activations.ts + autograd/grad.ts implemented
 * Run    : bun run exercises/ch-12-losses.ts
 *
 * The loss function measures how wrong the model is.  Cross-entropy is used
 * for every token prediction in the language model (Ch 28-30).
 */
import { mseLoss, crossEntropyFromLogits, logSumExp } from "../src/nn/losses.ts";
import { TensorValue } from "../src/autograd/grad.ts";
import { createTensor } from "../src/tensor/creation.ts";

// ─── E1: MSE loss ─────────────────────────────────────────────────────────────
// MSE = mean( (y_hat - y)² )
// Perfect prediction → 0, otherwise positive.
const pred   = new TensorValue(createTensor([1.0, 2.0, 3.0], [3]), "pred");
const target = new TensorValue(createTensor([1.0, 2.0, 3.0], [3]), "target");
const mse    = mseLoss(pred, target);
console.log("MSE (perfect):", mse.data.data[0], "  expected: 0");

const pred2 = new TensorValue(createTensor([1.0, 2.0, 3.0], [3]), "pred2");
const tgt2  = new TensorValue(createTensor([0.0, 0.0, 0.0], [3]), "tgt2");
const mse2  = mseLoss(pred2, tgt2);
console.log("MSE ([1,2,3] vs [0,0,0]):", mse2.data.data[0].toFixed(4), "  expected:", ((1+4+9)/3).toFixed(4));

// ─── E2: Cross-entropy from logits ───────────────────────────────────────────
// CE = -log(softmax(logits)[correct_class])
// Logits = [2, 1, 0.5], correct class = 0
// softmax = [0.6652, 0.2447, 0.0900]  →  CE = -log(0.6652) ≈ 0.408
const logits = new TensorValue(createTensor([2, 1, 0.5], [3]), "logits");
const ce = crossEntropyFromLogits(logits, 0);
console.log("\\nCE (correct=0):", ce.data.data[0].toFixed(4), "  expected: ≈ 0.4076");

// ─── E3: CE backprop ────────────────────────────────────────────────────────
ce.backward();
console.log("logits.grad:", Array.from(logits.grad!.data).map(v => v.toFixed(4)));
// softmax - one_hot: [0.6652-1, 0.2447-0, 0.0900-0] = [-0.3348, 0.2447, 0.0900]

// ─── E4: logSumExp numerical stability ──────────────────────────────────────
// logSumExp(x) = log(Σ exp(x[i]))  computed stably by subtracting max first.
const bigLogits = new TensorValue(createTensor([1000, 1001, 1002], [3]));
const lse = logSumExp(bigLogits);
console.log("\\nlogSumExp([1000,1001,1002]):", lse.data.data[0].toFixed(4));
// expected: 1002 + log(exp(-2)+exp(-1)+exp(0)) ≈ 1002.4076
console.log("no Inf:", !isNaN(lse.data.data[0]) && isFinite(lse.data.data[0]), "  expected: true");

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: compute the CE loss for a batch of 4 tokens from a 5-class vocabulary.
//   logits shape: [4, 5]  targets: [0, 2, 4, 1]
//   Average CE over the batch.
""")

write("ch-13-linear-layer.ts", """
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
console.log("\\nW.grad shape:", W!.grad?.shape,  "  expected: [2, 4]");
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
console.log("\\nHe    std:", weightStd(heLayer).toFixed(4),   "  expected: ≈", (Math.sqrt(2/512)).toFixed(4));
console.log("Xavier std:", weightStd(xvLayer).toFixed(4),   "  expected: ≈", (Math.sqrt(2/(512+256))).toFixed(4));
console.log("Normal std:", weightStd(normLayer).toFixed(4), "  expected: ≈ 0.0200");

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: stack 3 Linear layers and verify that backprop flows through all.
//   h1 = relu(linear1(x))
//   h2 = relu(linear2(h1))
//   y  = linear3(h2)
//   Compute loss = sum(y), backward, check linear1.weight.grad is not null.
""")

write("ch-14-optimizers.ts", """
/**
 * EXERCISES — Ch 14: Optimizers (SGD + Adam)
 * ════════════════════════════════════════════
 * Prereq : src/optim/sgd.ts + src/optim/adam.ts + src/nn/linear.ts implemented
 * Run    : bun run exercises/ch-14-optimizers.ts
 *
 * Adam's adaptive moments let it handle sparse gradients and noisy objectives —
 * it's the default optimizer for training transformers (Ch 29-30).
 */
import { Adam }          from "../src/optim/adam.ts";
import { SGD }           from "../src/optim/sgd.ts";
import { Linear }        from "../src/nn/linear.ts";
import { TensorValue }   from "../src/autograd/grad.ts";
import { createTensor }  from "../src/tensor/creation.ts";
import { mseLoss }       from "../src/nn/losses.ts";

// ─── E1: Adam minimises a quadratic ──────────────────────────────────────────
// f(w) = (w - 5)²   with Adam. Should converge faster than plain SGD.
import { Value } from "../src/autograd/value.ts";
let wAdam = new Value(0.0, "w");
const adam = new Adam([wAdam], 0.1);
for (let i = 0; i < 100; i++) {
  const loss = wAdam.add(new Value(-5)).pow(2);
  loss.backward();
  adam.step();
  adam.zeroGrad();
}
console.log("Adam w (should ≈ 5):", wAdam.data.toFixed(4));

// ─── E2: Adam vs SGD convergence speed ───────────────────────────────────────
function trainSteps(optType: "sgd" | "adam", lr: number, steps: number): number[] {
  const layer = new Linear(4, 1, false, "normal");
  const xs    = createTensor([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1], [4, 4]);
  const ys    = createTensor([1, 2, 3, 4], [4, 1]);
  const params = layer.parameters();
  const opt    = optType === "adam" ? new Adam(params, lr) : new SGD(params, lr);
  const losses: number[] = [];
  for (let i = 0; i < steps; i++) {
    const xv   = new TensorValue(xs, "x");
    const yv   = new TensorValue(ys, "y");
    const pred = layer.forward(xv);
    const loss = mseLoss(pred, yv);
    loss.backward();
    opt.step();
    opt.zeroGrad();
    if (i % 20 === 0) losses.push(loss.data.data[0]!);
  }
  return losses;
}
const sgdLosses  = trainSteps("sgd",  0.01, 200);
const adamLosses = trainSteps("adam", 0.01, 200);
console.log("\\nSGD  losses (every 20 steps):", sgdLosses.map(v => v.toFixed(3)));
console.log("Adam losses (every 20 steps):", adamLosses.map(v => v.toFixed(3)));

// ─── E3: Adam hyperparameters ─────────────────────────────────────────────────
// Default: lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8
// TODO: try lr=0.01 vs lr=0.001 — print final loss for each.
// What happens with very high lr (1.0)?

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: implement a simple learning-rate scheduler:
//   lr(step) = lr_base * 0.1^(step / decay_steps)
//   Apply it by mutating adam.lr each step.
""")

write("ch-15-training-loop.ts", """
/**
 * EXERCISES — Ch 15: Training Loop
 * ══════════════════════════════════
 * Prereq : src/nn/linear.ts + activations + losses + optim/adam implemented
 * Run    : bun run exercises/ch-15-training-loop.ts
 *
 * A complete training loop: forward → loss → backward → optimizer step.
 * We train a 2-layer MLP to classify 2D spiral data.
 */
import { Linear }       from "../src/nn/linear.ts";
import { relu }         from "../src/nn/activations.ts";
import { crossEntropyFromLogits } from "../src/nn/losses.ts";
import { Adam }         from "../src/optim/adam.ts";
import { TensorValue }  from "../src/autograd/grad.ts";
import { createTensor } from "../src/tensor/creation.ts";

// ─── E1: Spiral dataset generator ────────────────────────────────────────────
function generateSpiral(n: number, classes: number): { x: number[][], y: number[] } {
  const xs: number[][] = [];
  const ys: number[] = [];
  for (let c = 0; c < classes; c++) {
    for (let i = 0; i < n; i++) {
      const r    = i / n;
      const t    = (i / n) * 4 + c * (4 / classes) + (Math.random() - 0.5) * 0.2;
      xs.push([r * Math.sin(t), r * Math.cos(t)]);
      ys.push(c);
    }
  }
  return { x: xs, y: ys };
}

const dataset = generateSpiral(50, 3);   // 150 samples, 3 classes
console.log("Dataset: 150 samples, 3 classes");

// ─── E2: 2-layer MLP ─────────────────────────────────────────────────────────
const W1 = new Linear(2, 16, true, "he");
const W2 = new Linear(16, 3, true, "xavier");
const params = [...W1.parameters(), ...W2.parameters()];
const opt    = new Adam(params, 0.01);

// ─── E3: Training loop ────────────────────────────────────────────────────────
for (let epoch = 0; epoch < 200; epoch++) {
  const xData = dataset.x.flat();
  const xv    = new TensorValue(createTensor(xData, [150, 2]));

  const h     = relu(W1.forward(xv));
  const logit = W2.forward(h);

  // Compute cross-entropy for each sample
  let totalLoss = new TensorValue(createTensor([0], [1]));
  for (let i = 0; i < 150; i++) {
    const rowLogits = new TensorValue(
      createTensor(Array.from(logit.data.data.slice(i * 3, i * 3 + 3)), [3])
    );
    const ce = crossEntropyFromLogits(rowLogits, dataset.y[i]!);
    totalLoss = totalLoss.add(ce);
  }
  // Average loss
  const avgLoss = totalLoss.mul(new TensorValue(createTensor([1 / 150], [1])));
  avgLoss.backward();
  opt.step();
  opt.zeroGrad();

  if (epoch % 40 === 0) {
    console.log(\`epoch \${epoch}: loss = \${avgLoss.data.data[0]!.toFixed(4)}\`);
  }
}

// ─── E4: Accuracy ────────────────────────────────────────────────────────────
let correct = 0;
for (let i = 0; i < 150; i++) {
  const xv     = new TensorValue(createTensor(dataset.x[i]!, [1, 2]));
  const h      = relu(W1.forward(xv));
  const logits = W2.forward(h);
  const pred   = Array.from(logits.data.data).indexOf(Math.max(...Array.from(logits.data.data)));
  if (pred === dataset.y[i]) correct++;
}
console.log(\`\\nFinal accuracy: \${(correct / 150 * 100).toFixed(1)}%  (random=33%, good=>85%)\`);

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: add a 3rd hidden layer (16→16→16→3).
//       Does accuracy improve? Does training slow down?
""")

write("ch-16-char-tokenizer.ts", """
/**
 * EXERCISES — Ch 16: Character Tokenizer
 * ════════════════════════════════════════
 * Prereq : src/tokenizer/char.ts implemented
 * Run    : bun run exercises/ch-16-char-tokenizer.ts
 *
 * Every language model starts here: turning raw text into integer IDs.
 * The char tokenizer is the simplest possible vocabulary — one token per char.
 */
import { buildVocab, CharTokenizer, PAD_ID, BOS_ID, EOS_ID } from "../src/tokenizer/char.ts";

// ─── E1: Build vocabulary ─────────────────────────────────────────────────────
const text = "hello world! neural nets are fun.";
const vocab = buildVocab(text);
console.log("vocab size:", vocab.size, "  (special tokens + unique chars)");
console.log("contains 'h':", vocab.has("h"), "  expected: true");
console.log("contains 'z':", vocab.has("z"), "  expected: false");

// ─── E2: Encode & decode round-trip ──────────────────────────────────────────
const tokenizer = new CharTokenizer(vocab);
const encoded   = tokenizer.encode("hello");
const decoded   = tokenizer.decode(encoded);
console.log("\\nencode('hello'):", encoded);
console.log("decode back    :", decoded, "  expected: 'hello'");
console.log("round-trip ok  :", decoded === "hello", "  expected: true");

// ─── E3: Special tokens ───────────────────────────────────────────────────────
// BOS (beginning of sequence) and EOS (end of sequence) frame every sequence.
const withSpecial = [BOS_ID, ...tokenizer.encode("hi"), EOS_ID];
console.log("\\n[BOS] + 'hi' + [EOS]:", withSpecial);
// BOS=2, EOS=3 → [2, id_h, id_i, 3]

// ─── E4: Unknown characters ───────────────────────────────────────────────────
// Chars not in vocab should map to UNK_ID.
const unkEncoded = tokenizer.encode("héllo");   // é not in vocab
console.log("\\nencode('héllo') with unk:", unkEncoded);
// 'é' should be UNK_ID=1

// ─── E5: Batch encode ────────────────────────────────────────────────────────
// Batch encode pads all sequences to the same length.
const sentences = ["hi", "hello", "hey!"];
const batch     = tokenizer.encodeBatch(sentences);
console.log("\\nbatch shapes:");
for (const [i, seq] of batch.entries()) {
  console.log(\`  [\${i}]: length=\${seq.length}\`);
}
// All sequences should have the same length (padded with PAD_ID=0).

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: encode Shakespeare's first line, print token count vs character count.
//   "To be, or not to be, that is the question"
""")

write("ch-17-bpe-tokenizer.ts", """
/**
 * EXERCISES — Ch 17: BPE Tokenizer
 * ══════════════════════════════════
 * Prereq : src/tokenizer/bpe.ts + char.ts implemented
 * Run    : bun run exercises/ch-17-bpe-tokenizer.ts
 *
 * BPE (Byte Pair Encoding) builds a subword vocabulary by merging the most
 * frequent adjacent byte-pairs. GPT-2 uses 50,257 BPE tokens.
 */
import { BPETokenizer, countPairs, mergePair } from "../src/tokenizer/bpe.ts";

// ─── E1: Count pairs ─────────────────────────────────────────────────────────
// Before any merges, each word is split into chars.
const corpus = ["low", "low", "low", "lower", "newer", "wider"];
const tokens = corpus.map(w => w.split(""));
const pairs  = countPairs(tokens);
console.log("Most common pair:", [...pairs.entries()].sort((a,b) => b[1]-a[1])[0]);
// expected: ["l,o", 4]  or similar

// ─── E2: Merge a pair ────────────────────────────────────────────────────────
const merged = mergePair(tokens, ["l", "o"]);
console.log("\\nAfter merge l+o:");
console.log(merged.slice(0, 3)); // ["lo", "w"], ["lo","w"], ["lo","w"]

// ─── E3: Train BPE on tiny corpus ────────────────────────────────────────────
const bpe = new BPETokenizer();
const trainText = "the cat sat on the mat. the cat is fat.";
bpe.train(trainText, 20);   // learn 20 merges
console.log("\\nBPE vocab size after 20 merges:", bpe.vocabSize);

// ─── E4: Encode + decode ─────────────────────────────────────────────────────
const enc = bpe.encode("the cat");
const dec = bpe.decode(enc);
console.log("encode('the cat'):", enc);
console.log("decode back      :", dec);
console.log("round-trip ok    :", dec === "the cat");

// ─── E5: Token efficiency ────────────────────────────────────────────────────
// Compare token count for char tokenizer vs BPE on the same sentence.
import { CharTokenizer, buildVocab } from "../src/tokenizer/char.ts";
const charTok = new CharTokenizer(buildVocab(trainText));
const sentence = "the cat sat on the mat";
const charTokens = charTok.encode(sentence);
const bpeTokens  = bpe.encode(sentence);
console.log("\\nChar tokens:", charTokens.length);
console.log("BPE tokens :", bpeTokens.length, "  (should be fewer)");

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: train BPE on a larger text (50+ merges) and print the 10 most useful merges.
""")

write("ch-18-embeddings.ts", """
/**
 * EXERCISES — Ch 18: Token Embeddings
 * ═════════════════════════════════════
 * Prereq : src/nn/embedding.ts + tokenizer/char.ts implemented
 * Run    : bun run exercises/ch-18-embeddings.ts
 *
 * Embeddings map discrete token IDs to continuous vectors.
 * The embedding table is a [vocabSize, dModel] learned matrix —
 * lookup = matMul(one-hot, W_embed).
 */
import { Embedding }    from "../src/nn/embedding.ts";
import { TensorValue }  from "../src/autograd/grad.ts";
import { createTensor } from "../src/tensor/creation.ts";

// ─── E1: Basic lookup ────────────────────────────────────────────────────────
const embed = new Embedding(10, 8);    // vocabSize=10, dModel=8
const ids   = createTensor([0, 2, 5, 9], [4]);
const vecs  = embed.forward(ids);
console.log("embedding output shape:", vecs.data.shape, "  expected: [4, 8]");

// ─── E2: Same ID → same vector ───────────────────────────────────────────────
// Token 3 should always map to the same embedding vector.
const ids2  = createTensor([3, 3, 3], [3]);
const vecs2 = embed.forward(ids2);
const row0  = Array.from(vecs2.data.data.slice(0, 8));
const row1  = Array.from(vecs2.data.data.slice(8, 16));
const same  = row0.every((v, i) => v === row1[i]);
console.log("same ID → same vector:", same, "  expected: true");

// ─── E3: Gradient flows through embedding ────────────────────────────────────
// Backprop should update only the rows that were looked up.
const ids3  = createTensor([1, 3], [2]);
const vecs3 = embed.forward(ids3);
vecs3.backward();
console.log("\\nembedding weight grad shape:", embed.parameters()[0]!.grad?.shape);
// expected: [vocabSize, dModel]

// ─── E4: Cosine similarity of embeddings ─────────────────────────────────────
// Randomly initialised embeddings have no structure.
// After training, semantically similar tokens should be closer.
// Here we verify the math of cosine similarity.
function cosineSim(a: Float64Array, b: Float64Array): number {
  const dot  = Array.from(a).reduce((s, v, i) => s + v * b[i]!, 0);
  const normA = Math.sqrt(Array.from(a).reduce((s, v) => s + v * v, 0));
  const normB = Math.sqrt(Array.from(b).reduce((s, v) => s + v * v, 0));
  return dot / (normA * normB);
}
const v0 = embed.forward(createTensor([0], [1])).data.data;
const v1 = embed.forward(createTensor([1], [1])).data.data;
console.log("\\ncos(embed[0], embed[1]):", cosineSim(v0, v1).toFixed(4), "  (random ≈ near 0)");

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: encode "hello" with CharTokenizer, pass through Embedding, print shape.
//   Verify output shape: [len("hello"), dModel].
""")

write("ch-19-positional-encoding.ts", """
/**
 * EXERCISES — Ch 19: Positional Encoding
 * ════════════════════════════════════════
 * Prereq : src/nn/positional.ts + embedding.ts implemented
 * Run    : bun run exercises/ch-19-positional-encoding.ts
 *
 * Self-attention is permutation-invariant — it sees a SET of tokens, not a SEQUENCE.
 * Positional encodings inject position information so the model knows word order.
 *
 * PE(pos, 2i)   = sin(pos / 10000^(2i/dModel))
 * PE(pos, 2i+1) = cos(pos / 10000^(2i/dModel))
 */
import { PositionalEncoding } from "../src/nn/positional.ts";
import { TensorValue }        from "../src/autograd/grad.ts";
import { createTensor, zeros } from "../src/tensor/creation.ts";

// ─── E1: Shape check ─────────────────────────────────────────────────────────
const pe    = new PositionalEncoding(512, 100);  // dModel=512, maxLen=100
const input = new TensorValue(createTensor(
  Array.from({length: 10 * 512}, () => 0), [10, 512]
));
const out = pe.forward(input);
console.log("PE output shape:", out.data.shape, "  expected: [10, 512]");

// ─── E2: Sinusoidal pattern at position 0 ────────────────────────────────────
// At position 0, all sin terms = 0 and all cos terms = 1.
// So PE[0, 0] = sin(0) = 0, PE[0, 1] = cos(0) = 1.
const pe2  = new PositionalEncoding(8, 20);
const zero = new TensorValue(zeros([1, 8]));
const p0   = pe2.forward(zero);
console.log("\\nPE at pos 0, dim 0:", p0.data.data[0]!.toFixed(6), "  expected: 0.000000 (sin(0))");
console.log("PE at pos 0, dim 1:", p0.data.data[1]!.toFixed(6), "  expected: 1.000000 (cos(0))");

// ─── E3: Different positions → different encodings ───────────────────────────
// PE at pos 5 should differ from PE at pos 6.
const seq = new TensorValue(zeros([10, 8]));
const enc = pe2.forward(seq);
const pos5 = Array.from(enc.data.data.slice(5 * 8, 6 * 8));
const pos6 = Array.from(enc.data.data.slice(6 * 8, 7 * 8));
const distinct = pos5.some((v, i) => Math.abs(v - pos6[i]!) > 1e-6);
console.log("\\npos 5 ≠ pos 6:", distinct, "  expected: true");

// ─── E4: High-frequency vs low-frequency dimensions ─────────────────────────
// Dimension 0 uses frequency 1/10000^0 = 1 (fast oscillation).
// Dimension dModel-2 uses frequency 1/10000^1 ≈ very slow oscillation.
// Print dim 0 and dim dModel-2 for positions 0..7.
const pe3 = new PositionalEncoding(16, 50);
const longSeq = new TensorValue(zeros([8, 16]));
const longEnc = pe3.forward(longSeq);
console.log("\\nDim 0 values (fast freq):", Array.from({length:8}, (_, p) => longEnc.data.data[p*16]!.toFixed(3)));
console.log("Dim 14 values (slow freq):", Array.from({length:8}, (_, p) => longEnc.data.data[p*16+14]!.toFixed(3)));

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: verify that PE values are bounded in [-1, 1] for any sequence length.
//       Hint: sin and cos are always in [-1,1], so the output should be too.
""")

write("ch-20-layernorm-dropout.ts", """
/**
 * EXERCISES — Ch 20: LayerNorm & Dropout
 * ════════════════════════════════════════
 * Prereq : src/nn/layernorm.ts + dropout.ts implemented
 * Run    : bun run exercises/ch-20-layernorm-dropout.ts
 *
 * LayerNorm stabilises training — without it, deep transformers diverge.
 * Dropout prevents co-adaptation (overfitting) during training.
 */
import { LayerNorm }    from "../src/nn/layernorm.ts";
import { Dropout }      from "../src/nn/dropout.ts";
import { TensorValue }  from "../src/autograd/grad.ts";
import { createTensor } from "../src/tensor/creation.ts";

// ─── E1: LayerNorm normalises each row ────────────────────────────────────────
// After LayerNorm, each row should have mean ≈ 0 and std ≈ 1.
const ln = new LayerNorm(4);
const x  = new TensorValue(createTensor([10, 20, 30, 40,  1, 2, 3, 4], [2, 4]));
const n  = ln.forward(x);

function rowStats(data: Float64Array, start: number, len: number) {
  const vals = Array.from(data).slice(start, start + len);
  const mean = vals.reduce((a, b) => a + b, 0) / len;
  const std  = Math.sqrt(vals.reduce((a, v) => a + (v - mean) ** 2, 0) / len);
  return { mean, std };
}
const row0 = rowStats(n.data.data, 0, 4);
const row1 = rowStats(n.data.data, 4, 4);
console.log("row 0 mean:", row0.mean.toFixed(6), "  expected: ≈ 0");
console.log("row 0 std :", row0.std.toFixed(6),  "  expected: ≈ 1");
console.log("row 1 mean:", row1.mean.toFixed(6), "  expected: ≈ 0");
console.log("row 1 std :", row1.std.toFixed(6),  "  expected: ≈ 1");

// ─── E2: Learnable gamma and beta ────────────────────────────────────────────
// gamma starts at 1 (no scaling), beta at 0 (no shift).
const [gamma, beta] = ln.parameters();
console.log("\\ngamma initial:", Array.from(gamma!.data.data), "  expected: [1,1,1,1]");
console.log("beta  initial:", Array.from(beta!.data.data),  "  expected: [0,0,0,0]");

// ─── E3: LayerNorm gradient ───────────────────────────────────────────────────
n.backward();
console.log("\\ngamma.grad:", Array.from(gamma!.grad!.data));
console.log("beta.grad :", Array.from(beta!.grad!.data));

// ─── E4: Dropout — train vs eval mode ────────────────────────────────────────
const drop = new Dropout(0.5);
drop.train();
const xDrop  = new TensorValue(createTensor(Array.from({length: 1000}, () => 1.0), [1000]));
const outTrain = drop.forward(xDrop);
const zeros1k  = Array.from(outTrain.data.data).filter(v => v === 0).length;
console.log("\\nDropout (train, p=0.5) zeros:", zeros1k, "  expected: ≈ 500 (±50)");
// Inverted dropout scales surviving activations by 1/(1-p) to preserve expected value.
const meanTrain = Array.from(outTrain.data.data).reduce((a, b) => a + b, 0) / 1000;
console.log("Mean after dropout:", meanTrain.toFixed(3), "  expected: ≈ 1.0 (inverted scaling)");

drop.eval();
const outEval = drop.forward(xDrop);
const zerosEval = Array.from(outEval.data.data).filter(v => v === 0).length;
console.log("Dropout (eval) zeros:", zerosEval, "  expected: 0");

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: verify LayerNorm is invariant to input scale.
//   ln.forward(x) and ln.forward(10*x) should give identical outputs.
""")

write("ch-21-masks.ts", """
/**
 * EXERCISES — Ch 21: Mask Cookbook
 * ══════════════════════════════════
 * Prereq : src/tokenizer/masks.ts + char.ts implemented
 * Run    : bun run exercises/ch-21-masks.ts
 *
 * Masks are boolean tensors that control which tokens can attend to which.
 * Two types: causal masks (decoder self-attention) + padding masks (variable-length inputs).
 */
import {
  causalMask, expandCausalMask,
  binaryMaskFromIds, toAdditiveMask, expandPaddingMask,
  combineMasks, makeDecoderSelfAttentionMask,
  MASK_VALUE
} from "../src/tokenizer/masks.ts";
import { PAD_ID } from "../src/tokenizer/char.ts";

// ─── E1: Causal mask ─────────────────────────────────────────────────────────
// A 4×4 causal mask should be lower-triangular: position i can only see j ≤ i.
const cm = causalMask(4);
console.log("causal mask (4×4):");
for (let i = 0; i < 4; i++) {
  console.log(Array.from(cm.data.slice(i * 4, i * 4 + 4)));
}
// row 0: [true, false, false, false]
// row 1: [true, true,  false, false]
// etc.

// ─── E2: Additive mask ────────────────────────────────────────────────────────
// Before softmax, masked positions get -1e9 added → exp(-1e9) ≈ 0.
const additive = toAdditiveMask(cm);
console.log("\\nadditive mask row 0:", Array.from(additive.data.slice(0, 4)));
// [0, -1e9, -1e9, -1e9]

// ─── E3: Padding mask ─────────────────────────────────────────────────────────
// Token IDs with PAD_ID should be masked out.
const ids = [5, 3, PAD_ID, PAD_ID];   // 2 real tokens, 2 padding
const bm  = binaryMaskFromIds(ids);
console.log("\\nbinary padding mask:", Array.from(bm.data));
// [true, true, false, false]

// ─── E4: Combine masks ────────────────────────────────────────────────────────
// Decoder self-attention uses causal + padding mask combined.
const paddingIds  = [5, 3, PAD_ID, PAD_ID];
const decoderMask = makeDecoderSelfAttentionMask(paddingIds);
console.log("\\ndecoder self-attn mask (4×4):");
for (let i = 0; i < 4; i++) {
  console.log(Array.from(decoderMask.data.slice(i * 4, i * 4 + 4)));
}
// Position 2 and 3 are padding — no token should attend to them.

// ─── E5: Expand for batch + heads ────────────────────────────────────────────
// Attention expects mask shape [batch, numHeads, seqLen, seqLen].
const expanded = expandCausalMask(cm, 2, 4);   // batch=2, numHeads=4
console.log("\\nexpanded mask shape:", expanded.data.shape, "  expected: [2,4,4,4]");

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: verify that after applying the causal mask and softmax,
//   position 0 can only attend to position 0 (attention weight ≈ 1.0 for pos 0→0).
""")

write("ch-22-self-attention.ts", """
/**
 * EXERCISES — Ch 22: Self-Attention
 * ═══════════════════════════════════
 * Prereq : src/nn/attention.ts + linear.ts + masks.ts implemented
 * Run    : bun run exercises/ch-22-self-attention.ts
 *
 * Self-attention lets every token look at every other token in the sequence.
 * Scores = softmax(Q @ Kᵀ / √dk) · V
 */
import { scaledDotProductAttention, SelfAttention } from "../src/nn/attention.ts";
import { TensorValue }  from "../src/autograd/grad.ts";
import { createTensor, randn } from "../src/tensor/creation.ts";
import { causalMask, toAdditiveMask } from "../src/tokenizer/masks.ts";

// ─── E1: Scaled dot-product attention — shapes ───────────────────────────────
// seqLen=5, dk=8
const seqLen = 5, dk = 8;
const Q = new TensorValue(randn([seqLen, dk]), "Q");
const K = new TensorValue(randn([seqLen, dk]), "K");
const V = new TensorValue(randn([seqLen, dk]), "V");

const attnOut = scaledDotProductAttention(Q, K, V);
console.log("attn output shape:", attnOut.data.shape, "  expected: [5, 8]");

// ─── E2: Attention with causal mask ──────────────────────────────────────────
// With a causal mask, position 0 only sees itself, position 2 sees [0,1,2], etc.
const mask    = toAdditiveMask(causalMask(seqLen));
const maskTV  = new TensorValue(mask, "mask");
const maskedOut = scaledDotProductAttention(Q, K, V, maskTV);
console.log("masked attn output shape:", maskedOut.data.shape, "  expected: [5, 8]");

// ─── E3: Attention weights sum to 1 ──────────────────────────────────────────
// After softmax, each row of the attention weight matrix sums to 1.
// We can't read the internal weights unless SelfAttention exposes them.
// Instead: a * Q + b * (1-a) * Q should equal Q when a=1.
// Simpler check: feed Q=K=V=identity and verify output ≈ input.
const I = createTensor([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1], [4, 4]);
const Qid = new TensorValue(I, "Qid");
const Kid = new TensorValue(I, "Kid");
const Vid = new TensorValue(I, "Vid");
const outId = scaledDotProductAttention(Qid, Kid, Vid);
console.log("\\nWith Q=K=V=I, row 0:", Array.from(outId.data.data.slice(0, 4)).map(v => v.toFixed(4)));
// Should be near [0.25, 0.25, 0.25, 0.25] (uniform distribution over all rows)

// ─── E4: SelfAttention layer ─────────────────────────────────────────────────
const sa = new SelfAttention(16);   // dModel=16
const x  = new TensorValue(randn([6, 16]), "x");
const saOut = sa.forward(x);
console.log("\\nSelfAttention output shape:", saOut.data.shape, "  expected: [6, 16]");

// ─── E5: Gradient through attention ──────────────────────────────────────────
saOut.backward();
console.log("Input grad shape:", x.grad?.shape, "  expected: [6, 16]");

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: verify that attention is scale-invariant in the following sense:
//   scaledDotProductAttention(Q, K, V) ≈ scaledDotProductAttention(2Q, 2K, V)
//   because the scaling factor cancels.
""")

write("ch-23-multi-head-attention.ts", """
/**
 * EXERCISES — Ch 23: Multi-Head Attention
 * ═════════════════════════════════════════
 * Prereq : src/nn/attention.ts (MultiHeadAttention) implemented
 * Run    : bun run exercises/ch-23-multi-head-attention.ts
 *
 * Multiple heads let the model attend to different representation subspaces
 * simultaneously. Each head sees a [seqLen, dHead] slice of the full dModel.
 */
import { MultiHeadAttention } from "../src/nn/attention.ts";
import { TensorValue }        from "../src/autograd/grad.ts";
import { randn }              from "../src/tensor/creation.ts";

// ─── E1: Shape contract ───────────────────────────────────────────────────────
// dModel=32, numHeads=4 → dHead = 32/4 = 8
const mha     = new MultiHeadAttention(32, 4);
const batchSz = 2, seqLen = 6;
const x       = new TensorValue(randn([batchSz, seqLen, 32]), "x");
const out     = mha.forward(x);
console.log("MHA output shape:", out.data.shape, "  expected: [2, 6, 32]");

// ─── E2: With causal mask ─────────────────────────────────────────────────────
import { expandCausalMask, causalMask, toAdditiveMask } from "../src/tokenizer/masks.ts";
const cm   = toAdditiveMask(causalMask(seqLen));
const mask = new TensorValue(expandCausalMask(cm, batchSz, 4), "mask");
const maskedOut = mha.forward(x, mask);
console.log("Masked MHA output shape:", maskedOut.data.shape, "  expected: [2, 6, 32]");

// ─── E3: Backward pass ────────────────────────────────────────────────────────
maskedOut.backward();
console.log("\\nInput grad shape:", x.grad?.shape, "  expected: [2, 6, 32]");

// ─── E4: Parameter count ─────────────────────────────────────────────────────
// MHA has 4 Linear projections: Wq, Wk, Wv, Wo
// Each [dModel, dModel] + bias [dModel] → 4*(dModel² + dModel) params
const params = mha.parameters();
const totalParams = params.reduce((s, p) => s + p.data.size, 0);
const expected    = 4 * (32 * 32 + 32);   // Wq+Wk+Wv+Wo weights + biases
console.log("\\nTotal params:", totalParams, "  expected:", expected);

// ─── E5: splitHeads / mergeHeads ─────────────────────────────────────────────
// These reshape operations convert between:
//   [batch, seq, dModel]  ↔  [batch, numHeads, seq, dHead]
// Verify the shapes are correct by running a forward pass with known input sizes.
const mha2 = new MultiHeadAttention(64, 8);
const x2   = new TensorValue(randn([3, 10, 64]), "x2");
const out2 = mha2.forward(x2);
console.log("\\nMHA(dModel=64, heads=8) output shape:", out2.data.shape, "  expected: [3, 10, 64]");

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: verify that numHeads must divide dModel evenly.
//       Try MultiHeadAttention(32, 5) — should throw at construction.
""")

write("ch-24-cross-attention.ts", """
/**
 * EXERCISES — Ch 24: Cross-Attention
 * ════════════════════════════════════
 * Prereq : src/nn/attention.ts (CrossAttention) implemented
 * Run    : bun run exercises/ch-24-cross-attention.ts
 *
 * Cross-attention lets the decoder query the encoder's output.
 * Q comes from the decoder; K, V come from the encoder.
 * This is the bridge between encoder and decoder (Ch 27).
 */
import { CrossAttention }   from "../src/nn/attention.ts";
import { TensorValue }      from "../src/autograd/grad.ts";
import { randn }            from "../src/tensor/creation.ts";

// ─── E1: Shape contract ───────────────────────────────────────────────────────
// Encoder output: [batch, srcLen, dModel]
// Decoder state : [batch, tgtLen, dModel]
// Cross-attention output: [batch, tgtLen, dModel]  ← query length preserved
const dModel  = 32, numHeads = 4;
const srcLen  = 10, tgtLen = 6;
const batch   = 2;

const crossAttn  = new CrossAttention(dModel, numHeads);
const encOut     = new TensorValue(randn([batch, srcLen, dModel]), "encOut");
const decState   = new TensorValue(randn([batch, tgtLen, dModel]), "decState");
const out        = crossAttn.forward(decState, encOut);
console.log("cross-attn output shape:", out.data.shape, "  expected: [2, 6, 32]");

// ─── E2: Query length drives output length ────────────────────────────────────
// Changing tgtLen (decoder sequence length) changes output rows.
// Changing srcLen (encoder sequence length) does NOT change output rows.
const decState2 = new TensorValue(randn([batch, 4, dModel]));   // tgtLen=4
const out2      = crossAttn.forward(decState2, encOut);          // srcLen still 10
console.log("\\noutput shape with tgtLen=4:", out2.data.shape, "  expected: [2, 4, 32]");

// ─── E3: Backward pass ────────────────────────────────────────────────────────
out.backward();
console.log("decState grad shape:", decState.grad?.shape, "  expected: [2, 6, 32]");
console.log("encOut   grad shape:", encOut.grad?.shape,   "  expected: [2, 10, 32]");

// ─── E4: Cross-attention vs self-attention ───────────────────────────────────
// In self-attention, Q=K=V (all come from the same sequence).
// In cross-attention, Q≠K=V (Q from decoder, K/V from encoder).
// TODO: demonstrate this by passing encOut as both query and memory —
//       it should behave identically to SelfAttention.
import { MultiHeadAttention } from "../src/nn/attention.ts";
const selfAttn = new MultiHeadAttention(dModel, numHeads);
const crossSelf = new CrossAttention(dModel, numHeads);
// Use same weights ... this is conceptual; just verify shapes match.
const selfOut  = selfAttn.forward(encOut);
const crossOut = crossAttn.forward(encOut, encOut);
console.log("\\nself-attn  output shape:", selfOut.data.shape);
console.log("cross-attn output shape:", crossOut.data.shape);
// Both should be [2, 10, 32].

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: in translation, encoder sees English (srcLen tokens),
//       decoder generates French (tgtLen tokens).
//       Model: srcLen=15, tgtLen=12, dModel=64, numHeads=8.
//       Verify all shapes.
""")

write("ch-25-feedforward-block.ts", """
/**
 * EXERCISES — Ch 25: Feed-Forward Block
 * ═══════════════════════════════════════
 * Prereq : src/nn/feedforward.ts + layernorm + attention implemented
 * Run    : bun run exercises/ch-25-feedforward-block.ts
 *
 * Every transformer block applies attention THEN a 2-layer FFN:
 *   FFN(x) = GELU(x @ W1 + b1) @ W2 + b2
 * The inner dimension (dFf) is typically 4× dModel.
 */
import { FFN }           from "../src/nn/feedforward.ts";
import { TensorValue }   from "../src/autograd/grad.ts";
import { randn }         from "../src/tensor/creation.ts";

// ─── E1: Basic forward pass ───────────────────────────────────────────────────
const ffn = new FFN(32, 128);   // dModel=32, dFf=128
const x   = new TensorValue(randn([4, 10, 32]), "x");  // [batch, seqLen, dModel]
const out = ffn.forward(x);
console.log("FFN output shape:", out.data.shape, "  expected: [4, 10, 32]");

// ─── E2: Default dFf = 4 × dModel ───────────────────────────────────────────
// GPT-2 uses dFf = 4 * dModel.  The default constructor should enforce this.
const ffnDefault = new FFN(64);  // dFf defaults to 256
const xDef = new TensorValue(randn([2, 8, 64]));
const outDef = ffnDefault.forward(xDef);
console.log("FFN(64) default output shape:", outDef.data.shape, "  expected: [2, 8, 64]");

// ─── E3: Backward pass ────────────────────────────────────────────────────────
out.backward();
console.log("\\nInput grad shape:", x.grad?.shape, "  expected: [4, 10, 32]");
const paramShapes = ffn.parameters().map(p => p.data.shape);
console.log("Param shapes:", paramShapes);
// expected: [[128,32], [128], [32,128], [32]]  (W1, b1, W2, b2)

// ─── E4: Parameter count ─────────────────────────────────────────────────────
// W1: [dFf, dModel] + b1: [dFf] + W2: [dModel, dFf] + b2: [dModel]
// = 128*32 + 128 + 32*128 + 32 = 8320
const totalFFN = ffn.parameters().reduce((s, p) => s + p.data.size, 0);
const expected = 128 * 32 + 128 + 32 * 128 + 32;
console.log("\\nTotal params:", totalFFN, "  expected:", expected);

// ─── E5: Residual connection ─────────────────────────────────────────────────
// In the transformer, FFN is used with a residual: output = x + FFN(LN(x))
import { LayerNorm } from "../src/nn/layernorm.ts";
const ln  = new LayerNorm(32);
const res = x.add(ffn.forward(ln.forward(x)));
console.log("\\nResidual output shape:", res.data.shape, "  expected: [4, 10, 32]");

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: try dFf = dModel (no expansion). Compare final loss on Ch 15 spiral task.
""")

write("ch-26-encoder-block.ts", """
/**
 * EXERCISES — Ch 26: Encoder Block
 * ══════════════════════════════════
 * Prereq : src/nn/transformer.ts (EncoderBlock) implemented
 * Run    : bun run exercises/ch-26-encoder-block.ts
 *
 * Encoder block = LayerNorm → MultiHeadAttention → residual
 *               + LayerNorm → FFN → residual
 * (Pre-LN variant used by most modern transformers.)
 */
import { EncoderBlock }  from "../src/nn/transformer.ts";
import { TensorValue }   from "../src/autograd/grad.ts";
import { randn }         from "../src/tensor/creation.ts";

// ─── E1: Forward pass shape ───────────────────────────────────────────────────
const enc    = new EncoderBlock(32, 4, 128);   // dModel, numHeads, dFf
const x      = new TensorValue(randn([2, 8, 32]), "x");
const out    = enc.forward(x);
console.log("encoder block output shape:", out.data.shape, "  expected: [2, 8, 32]");

// ─── E2: With padding mask ────────────────────────────────────────────────────
import { binaryMaskFromIds, toAdditiveMask, expandPaddingMask } from "../src/tokenizer/masks.ts";
const paddedIds   = [1, 2, 3, 0, 0, 0, 0, 0];   // 3 real tokens, 5 padding
const binaryMask  = binaryMaskFromIds(paddedIds);
const additiveMask = toAdditiveMask(binaryMask);
const expandedMask = expandPaddingMask(additiveMask, 2, 4);   // batch=2, heads=4
const maskTV = new TensorValue(expandedMask, "mask");
const maskedOut = enc.forward(x, maskTV);
console.log("\\nmasked encoder output shape:", maskedOut.data.shape, "  expected: [2, 8, 32]");

// ─── E3: Backward through the entire block ───────────────────────────────────
maskedOut.backward();
console.log("Input grad shape:", x.grad?.shape, "  expected: [2, 8, 32]");

// ─── E4: Stack two encoder blocks ─────────────────────────────────────────────
const enc2 = new EncoderBlock(32, 4, 128);
const out2 = enc2.forward(enc.forward(x));
console.log("\\nStacked encoder output shape:", out2.data.shape, "  expected: [2, 8, 32]");

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: count total parameters in one EncoderBlock with dModel=512, numHeads=8, dFf=2048.
//   Expected: 2 * LayerNorm + MultiHeadAttention + FFN
//           = 2*(512+512) + 4*(512²+512) + 2*(512*2048+2048+2048*512+512)
""")

write("ch-27-decoder-block.ts", """
/**
 * EXERCISES — Ch 27: Decoder Block
 * ══════════════════════════════════
 * Prereq : src/nn/transformer.ts (DecoderBlock) implemented
 * Run    : bun run exercises/ch-27-decoder-block.ts
 *
 * Decoder block = masked self-attention + cross-attention + FFN (each with LN + residual).
 * The causal mask in self-attention prevents the decoder from seeing future tokens.
 */
import { DecoderBlock }  from "../src/nn/transformer.ts";
import { TensorValue }   from "../src/autograd/grad.ts";
import { randn }         from "../src/tensor/creation.ts";
import { causalMask, toAdditiveMask, expandCausalMask } from "../src/tokenizer/masks.ts";

// ─── E1: Forward pass shape ───────────────────────────────────────────────────
const dec     = new DecoderBlock(32, 4, 128);
const srcLen  = 10, tgtLen = 6, batch = 2;
const encOut  = new TensorValue(randn([batch, srcLen, 32]), "encOut");
const decIn   = new TensorValue(randn([batch, tgtLen, 32]), "decIn");
const out     = dec.forward(decIn, encOut);
console.log("decoder block output shape:", out.data.shape, "  expected: [2, 6, 32]");

// ─── E2: With causal mask on decoder self-attention ──────────────────────────
const cm        = toAdditiveMask(causalMask(tgtLen));
const causalTV  = new TensorValue(expandCausalMask(cm, batch, 4), "causalMask");
const maskedOut = dec.forward(decIn, encOut, causalTV);
console.log("masked decoder output:", maskedOut.data.shape, "  expected: [2, 6, 32]");

// ─── E3: Encoder output gradient ─────────────────────────────────────────────
maskedOut.backward();
console.log("encOut grad shape:", encOut.grad?.shape, "  expected: [2, 10, 32]");
console.log("decIn  grad shape:", decIn.grad?.shape,  "  expected: [2, 6, 32]");

// ─── E4: Auto-regressive generation preview ──────────────────────────────────
// In inference we run the decoder one token at a time:
// step 1: tgtLen=1, step 2: tgtLen=2, …
for (let t = 1; t <= 4; t++) {
  const tgt   = new TensorValue(randn([1, t, 32]));
  const enc_o = new TensorValue(randn([1, srcLen, 32]));
  const o     = dec.forward(tgt, enc_o);
  console.log(\`  tgtLen=\${t} → output shape:\`, o.data.shape);
}

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: verify that the decoder block has 3 sub-layers:
//   masked self-attention, cross-attention, FFN.
//   Count parameters and compare to the formula.
""")

write("ch-28-sequence-data.ts", """
/**
 * EXERCISES — Ch 28: Sequence Data & Training Objectives
 * ════════════════════════════════════════════════════════
 * Prereq : src/utils/data.ts + tokenizer/char.ts implemented
 * Run    : bun run exercises/ch-28-sequence-data.ts
 *
 * Language models are trained on "predict the next token".
 * Input: [t0, t1, t2, t3]   Target: [t1, t2, t3, t4]
 * This is a shifted copy — the model sees the prefix, predicts the suffix.
 */
import { shiftRight, maskedCrossEntropy, perplexity, makeReversalDataset, splitDataset } from "../src/utils/data.ts";

// ─── E1: Right-shift for language modelling ──────────────────────────────────
// Given input [1,2,3,4,5], target is [2,3,4,5, PAD]
const tokens = [1, 2, 3, 4, 5];
const { input, target } = shiftRight(tokens, 0);   // 0 = PAD_ID
console.log("input :", input,  "  expected: [1,2,3,4,5]");
console.log("target:", target, "  expected: [2,3,4,5,0]");

// ─── E2: Masked cross-entropy (ignore padding) ───────────────────────────────
// Padding positions should not contribute to the loss.
import { TensorValue } from "../src/autograd/grad.ts";
import { createTensor } from "../src/tensor/creation.ts";

const logits  = new TensorValue(createTensor([
  1, 0, 0,    // position 0 — strong prediction for class 0
  0, 1, 0,    // position 1 — strong prediction for class 1
  0, 0, 0,    // position 2 — padding, should be ignored
], [3, 3]));
const targets = [0, 1, -1];   // -1 = ignore (padding)
const loss = maskedCrossEntropy(logits, targets);
console.log("\\nmasked CE (2 real, 1 ignored):", loss.data.data[0].toFixed(4));
// Should be close to 0 (both real predictions are correct).

// ─── E3: Perplexity ──────────────────────────────────────────────────────────
// Perplexity = exp(cross-entropy loss)
// Perfect model (loss=0) → perplexity=1
// Random 50k-vocab model (loss≈10.8) → perplexity≈50,000
const perfectLoss = 0.0;
const randomLoss  = Math.log(50000);
console.log("\\nperplexity (perfect):", perplexity(perfectLoss).toFixed(2), "  expected: 1.00");
console.log("perplexity (random) :", perplexity(randomLoss).toFixed(0),  "  expected: 50000");

// ─── E4: Reversal dataset ────────────────────────────────────────────────────
// A classic seq2seq toy task: given "abcd", output "dcba".
// Useful for testing encoder-decoder transformer (Ch 29).
const revDataset = makeReversalDataset(100, 5);   // 100 examples, seqLen=5
const { train, val, test } = splitDataset(revDataset, 0.7, 0.15, 0.15);
console.log("\\nReversal dataset splits:");
console.log("  train:", train.length, " val:", val.length, " test:", test.length);
console.log("  example src:", train[0]!.src, " →  tgt:", train[0]!.tgt);

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: encode Shakespeare using CharTokenizer, build all (input, target) pairs,
//       and print the first 5 input/target pairs side by side.
""")

write("ch-29-full-transformer.ts", """
/**
 * EXERCISES — Ch 29: Full Transformer
 * ═════════════════════════════════════
 * Prereq : src/nn/transformer.ts (Transformer) + utils/data.ts implemented
 * Run    : bun run exercises/ch-29-full-transformer.ts
 *
 * The complete encoder-decoder transformer: encode source → decode to target.
 * We train it on the sequence reversal task from Ch 28.
 */
import { Transformer, TransformerConfig } from "../src/nn/transformer.ts";
import { TensorValue }   from "../src/autograd/grad.ts";
import { createTensor }  from "../src/tensor/creation.ts";
import { Adam }          from "../src/optim/adam.ts";
import { makeReversalDataset, splitDataset, maskedCrossEntropy } from "../src/utils/data.ts";
import { CharTokenizer, buildVocab } from "../src/tokenizer/char.ts";

// ─── E1: Build and inspect the model ─────────────────────────────────────────
const config: TransformerConfig = {
  vocabSize: 20, dModel: 32, numHeads: 4,
  numEncoderLayers: 2, numDecoderLayers: 2,
  dFf: 64, maxSeqLen: 20, dropout: 0.0,
};
const model  = new Transformer(config);
const params = model.parameters();
const nParams = params.reduce((s, p) => s + p.data.size, 0);
console.log("Total parameters:", nParams);

// ─── E2: Forward pass ─────────────────────────────────────────────────────────
const srcIds = createTensor([3, 4, 5, 6, 7], [1, 5]);
const tgtIds = createTensor([3, 7, 6, 5, 4], [1, 5]);   // reversed
const logits = model.forward(srcIds, tgtIds);
console.log("\\nlogits shape:", logits.data.shape, "  expected: [1, 5, 20]");

// ─── E3: Loss on one example ──────────────────────────────────────────────────
const targets = [7, 6, 5, 4, 0];   // shifted right, last = PAD
const loss = maskedCrossEntropy(logits, targets);
console.log("initial loss:", loss.data.data[0].toFixed(4), "  (random ≈ log(20) ≈ 3.0)");

// ─── E4: Train on reversal task ───────────────────────────────────────────────
const dataset = makeReversalDataset(200, 5);
const { train } = splitDataset(dataset, 0.8, 0.1, 0.1);
const opt = new Adam(model.parameters(), 3e-4);

for (let epoch = 0; epoch < 5; epoch++) {
  let totalLoss = 0;
  for (const ex of train.slice(0, 20)) {   // mini-batch of 20 for speed
    const src   = createTensor(ex.src, [1, ex.src.length]);
    const tgt   = createTensor(ex.tgt, [1, ex.tgt.length]);
    const logit = model.forward(src, tgt);
    const tgtShift = [...ex.tgt.slice(1), 0];
    const l = maskedCrossEntropy(logit, tgtShift);
    l.backward();
    opt.step();
    opt.zeroGrad();
    totalLoss += l.data.data[0]!;
  }
  console.log(\`epoch \${epoch+1}: avg loss = \${(totalLoss/20).toFixed(4)}\`);
}

// ─── E5: generate() ──────────────────────────────────────────────────────────
import { generate } from "../src/nn/transformer.ts";
const srcTest = [3, 4, 5, 6, 7];
const prediction = generate(model, srcTest, 5, 0);   // maxLen=5, BOS=0
console.log("\\nSource  :", srcTest);
console.log("Predicted:", prediction, "  (target: [7,6,5,4,3])");

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: train for 50 epochs and compute accuracy on the test set.
//       Does the model learn to reverse sequences perfectly?
""")

write("ch-30-decoder-only-gpt.ts", """
/**
 * EXERCISES — Ch 30: Decoder-Only GPT
 * ═════════════════════════════════════
 * Prereq : src/nn/gpt.ts + all prior modules implemented
 * Run    : bun run exercises/ch-30-decoder-only-gpt.ts
 *
 * GPT removes the encoder entirely. It's trained on a single objective:
 * predict the next token.  The causal mask ensures it cannot cheat by
 * looking at future tokens during training.
 */
import { GPT, GPTConfig, makeLanguageModelingBatch, generateText } from "../src/nn/gpt.ts";
import { CharTokenizer, buildVocab } from "../src/tokenizer/char.ts";
import { Adam }         from "../src/optim/adam.ts";
import { maskedCrossEntropy } from "../src/utils/data.ts";

// ─── E1: Build a tiny GPT ─────────────────────────────────────────────────────
const config: GPTConfig = {
  vocabSize: 50, dModel: 32, numHeads: 4,
  numLayers: 2, dFf: 64, maxSeqLen: 64, dropout: 0.0,
};
const gpt    = new GPT(config);
const nParam = gpt.parameters().reduce((s, p) => s + p.data.size, 0);
console.log("GPT parameter count:", nParam);

// ─── E2: Forward pass ─────────────────────────────────────────────────────────
import { createTensor } from "../src/tensor/creation.ts";
const ids    = createTensor([1, 5, 12, 7, 3], [1, 5]);   // [batch=1, seqLen=5]
const logits = gpt.forward(ids);
console.log("logits shape:", logits.data.shape, "  expected: [1, 5, 50]");

// ─── E3: Language modeling batch ─────────────────────────────────────────────
// The language modeling objective uses right-shifted targets.
const tokenSequence = [1, 2, 3, 4, 5, 6, 7, 8];
const { input: lmInput, target: lmTarget } = makeLanguageModelingBatch(tokenSequence, 0);
console.log("\\nLM input :", lmInput, "  expected: [1,2,3,4,5,6,7,8]");
console.log("LM target:", lmTarget, "  expected: [2,3,4,5,6,7,8,0]");

// ─── E4: Train on "hello world" repeated ─────────────────────────────────────
const trainText = "hello world! hello world! hello world!";
const vocab     = buildVocab(trainText);
const tokenizer = new CharTokenizer(vocab);
const gptSmall: GPTConfig = {
  vocabSize: vocab.size, dModel: 32, numHeads: 4,
  numLayers: 2, dFf: 64, maxSeqLen: 40, dropout: 0.0,
};
const model = new GPT(gptSmall);
const opt   = new Adam(model.parameters(), 1e-3);

const trainIds = tokenizer.encode(trainText);
for (let epoch = 0; epoch < 100; epoch++) {
  const { input: inp, target: tgt } = makeLanguageModelingBatch(trainIds, 0);
  const inpT   = createTensor(inp, [1, inp.length]);
  const logit  = model.forward(inpT);
  const l      = maskedCrossEntropy(logit, tgt);
  l.backward();
  opt.step();
  opt.zeroGrad();
  if (epoch % 20 === 0)
    console.log(\`epoch \${epoch}: loss = \${l.data.data[0]!.toFixed(4)}\`);
}

// ─── E5: Text generation (autoregressive) ────────────────────────────────────
const prompt    = tokenizer.encode("hello");
const generated = generateText(model, prompt, 20, { temperature: 0.8 });
console.log("\\nGenerated:", tokenizer.decode(generated));

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: after training, measure perplexity on a held-out sentence.
//       Try greedy decoding (temperature=0) vs temperature=1.2.
//       What is the difference in output diversity?
""")

# ─── Summary ──────────────────────────────────────────────────────────────────
print(f"\nCreated {len(os.listdir(EX))} exercise files in exercises/")
print("Run any exercise with:  bun run exercises/ch-XX-name.ts")
