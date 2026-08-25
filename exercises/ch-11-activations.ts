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

// ═════════════════════════════════════════════════════════════════════════════
//  E6-E10: TRAIN THEM.  Everything above tested one activation on one row.
//  These train whole networks on XOR — the problem from section 1 — and show
//  what the choice of activation actually buys you.
//
//  XOR:  (0,0) -> 0    (0,1) -> 1    (1,0) -> 1    (1,1) -> 0
//  The two classes sit on opposite diagonals, so no straight line separates
//  them.  That is the whole point: it is the smallest problem a linear model
//  provably cannot solve.
// ═════════════════════════════════════════════════════════════════════════════
import { randn } from "../src/tensor/creation.ts";
import { sub, mulScalar } from "../src/tensor/ops.ts";

/** The four XOR inputs as one [4,2] batch, and the four targets as [4,1]. */
const XOR_X = new TensorValue(createTensor([0, 0, 0, 1, 1, 0, 1, 1], [4, 2]));
const XOR_Y = createTensor([0, 1, 1, 0], [4, 1]);
// Pre-negated targets, so the loss can use add() instead of a subtract we
// never wrote: (out - y) is spelled out.add(negY).
const XOR_NEG_Y = new TensorValue(mulScalar(XOR_Y, -1));

type Activation = "none" | "relu" | "sigmoid";

/**
 * Build the weights and biases for an MLP.  `sizes` lists layer widths, so
 * [2, 8, 1] means two inputs, eight hidden units, one output.
 *
 * He initialisation: scale by sqrt(2 / fanIn).  The scale matters — E10(d)
 * shows what happens when it is wrong, and E10(a) shows why it cannot be zero.
 */
function makeNet(sizes: number[], scale = 1): TensorValue[] {
  const params: TensorValue[] = [];
  for (let l = 0; l < sizes.length - 1; l++) {
    const fanIn = sizes[l]!, fanOut = sizes[l + 1]!;
    params.push(new TensorValue(mulScalar(randn([fanIn, fanOut]), scale * Math.sqrt(2 / fanIn))));
    params.push(new TensorValue(createTensor(new Array(fanOut).fill(0), [1, fanOut])));
  }
  return params;
}

/**
 * Forward pass.  Every layer is matMul then add — the linear layer of Ch 13,
 * built here by hand — with the activation applied between layers only.  The
 * output layer stays linear so the net can produce any real number.
 */
function forward(params: TensorValue[], x: TensorValue, act: Activation): TensorValue {
  let h = x;
  const numLayers = params.length / 2;
  for (let l = 0; l < numLayers; l++) {
    h = h.matMul(params[2 * l]!).add(params[2 * l + 1]!);
    if (l < numLayers - 1 && act !== "none") h = act === "relu" ? relu(h) : sigmoid(h);
  }
  return h;
}

/** Mean squared error, built from the ops TensorValue already has. */
function mse(out: TensorValue, negTarget: TensorValue): TensorValue {
  const diff = out.add(negTarget);
  return diff.mul(diff).mean();
}

/** One SGD step: forward, backward, subtract lr * grad from every parameter. */
function trainStep(params: TensorValue[], act: Activation, lr: number): number {
  for (const p of params) p.zeroGrad();
  const loss = mse(forward(params, XOR_X, act), XOR_NEG_Y);
  loss.backward();
  if (lr !== 0) {
    for (const p of params) if (p.grad) p.data = sub(p.data, mulScalar(p.grad, lr));
  }
  return loss.data.data[0]!;
}

function train(params: TensorValue[], act: Activation, lr: number, steps: number): number {
  let loss = NaN;
  for (let i = 0; i < steps; i++) loss = trainStep(params, act, lr);
  return loss;
}

const fmt = (v: number) => (Number.isFinite(v) ? v.toFixed(4) : String(v)).padStart(8);
const preds = (params: TensorValue[], act: Activation) =>
  Array.from(forward(params, XOR_X, act).data.data).map(fmt).join("");

// ─── E6: The exact solution, by hand — two relu units are enough ─────────────
// No training at all. These weights are set deliberately, and they are the
// smallest network that solves XOR exactly:
//
//     h1 = relu(x1 + x2)         counts how many inputs are on
//     h2 = relu(x1 + x2 - 1)     stays ASLEEP unless BOTH are on
//     y  = h1 - 2*h2
//
// Read it as XOR = OR - 2*AND. h2 is a detector that fires only at (1,1), and
// subtracting it twice cancels the one case a straight line could not handle.
// This is the "one bend" of section 6, doing exactly one job.
const W1 = new TensorValue(createTensor([1, 1, 1, 1], [2, 2]));   // both units see x1+x2
const B1 = new TensorValue(createTensor([0, -1], [1, 2]));        // the -1 puts h2 to sleep
const W2 = new TensorValue(createTensor([1, -2], [2, 1]));        // h1 - 2*h2
const B2 = new TensorValue(createTensor([0], [1, 1]));
console.log("\n\n=== E6: hand-built XOR — 2 relu units, zero training ===");
console.log("   h1 = relu(x1+x2)   h2 = relu(x1+x2-1)   y = h1 - 2*h2");
console.log("   predictions:", preds([W1, B1, W2, B2], "relu"));
console.log("   want       :", Array.from(XOR_Y.data).map(fmt).join(""));
// Exact, not approximate — every prediction is 0 or 1 on the nose.

// ─── E7: Linear only — the 0.25 floor that no amount of training moves ──────
// Ch 09 proved linear composed with linear is linear. So this net, whatever
// its depth, is exactly equivalent to ONE matrix. It cannot bend, so the best
// it can do on XOR is predict the average of the targets, 0.5, everywhere:
//     MSE = ((0.5)^2 + (0.5)^2 + (0.5)^2 + (0.5)^2) / 4 = 0.25
console.log("\n=== E7: linear only — no activation ===");
for (const sizes of [[2, 1], [2, 8, 1], [2, 8, 8, 1]]) {
  const net = makeNet(sizes);
  const loss = train(net, "none", 0.05, 5000);
  console.log(`   ${JSON.stringify(sizes).padEnd(12)} loss ${fmt(loss)}   ${preds(net, "none")}`);
}
console.log("   0.25 is the FLOOR. Depth does not help — it is the same one matrix.");
// TODO: multiply the trained weight matrices of the [2,8,8,1] net together
//       with matMul from linalg.ts. You get a single [2,1] matrix — proof the
//       three layers collapsed. Then check its predictions match the net's.

// ─── E8: One relu layer — the same problem, solved ──────────────────────────
console.log("\n=== E8: one relu hidden layer ===");
for (const h of [2, 8]) {
  const net = makeNet([2, h, 1]);
  const loss = train(net, "relu", 0.1, 8000);
  console.log(`   2 -> ${String(h).padStart(2)} -> 1   loss ${fmt(loss)}   ${preds(net, "relu")}`);
}
console.log("   Width 8 reaches ~0. Width 2 is the theoretical minimum (E6 proves");
console.log("   a solution EXISTS) but training finds it only ~30% of the time —");
console.log("   one dead unit and the remaining one cannot do the job alone.");
// Results vary run to run: randn() is unseeded, so re-run this a few times.

// ─── E9: The vanishing gradient, measured ───────────────────────────────────
// Section 10 says sigmoid'(z) <= 0.25, so n layers multiply to at most 0.25^n,
// while relu' is exactly 1 on the active side and 1^n = 1 forever.
// Measure it: one backward pass on a FRESH net (lr = 0, nothing moves) and
// print how much gradient actually reaches each layer.
console.log("\n=== E9: gradient reaching each layer of a 12-layer net (at init) ===");
const gradNorm = (p: TensorValue) =>
  Math.sqrt(Array.from(p.grad!.data).reduce((s, v) => s + v * v, 0));  // Euclidean norm of one weight matrix's gradient
for (const act of ["sigmoid", "relu"] as const) {
  const deep = makeNet([2, ...new Array(12).fill(16), 1]);
  trainStep(deep, act, 0);
  const norms = deep.filter((_, i) => i % 2 === 0).map(gradNorm);   // weight matrices only, skip biases
  console.log(`   ${act.padEnd(8)}`, norms.map((v) => v.toExponential(1)).join(" "));
  console.log(`   ${" ".repeat(8)} last / first = ${(norms[norms.length - 1]! / norms[0]!).toExponential(1)}x`);
}
console.log("   sigmoid starves its early layers by many orders of magnitude.");
console.log("   Compare the arithmetic:  0.25^12 =", (0.25 ** 12).toExponential(2), "  vs  1^12 =", 1 ** 12);

// ─── E10: What goes wrong ───────────────────────────────────────────────────
console.log("\n=== E10: the five failure modes ===");

// (a) Zero init — every hidden unit receives the SAME gradient, so the units
//     stay identical copies of each other forever. Eight units, one unit of
//     actual capacity. This is why initialisation is random.
const zeroNet = makeNet([2, 8, 1], 0);
const zeroLoss = train(zeroNet, "relu", 0.5, 3000);
console.log(`   (a) zero init      loss ${fmt(zeroLoss)}   ${preds(zeroNet, "relu")}`);
console.log(`       layer-1 weights all still: ${Array.from(zeroNet[0]!.data.data).every((v) => v === 0)}`);

// (b) Learning rate. Too small and it crawls; too large and it diverges to
//     NaN. The usable band is narrower than beginners expect.
console.log("   (b) learning rate");
for (const lr of [0.01, 0.1, 1, 10]) {
  const net = makeNet([2, 8, 1]);
  console.log(`       lr=${String(lr).padStart(5)}   loss ${fmt(train(net, "relu", lr, 3000))}`);
}

// (c) Dying relu. A unit whose input is negative for EVERY example gets
//     gradient exactly 0 (E1 showed this), so it never updates, so its input
//     stays negative. Permanent. Count how often a narrow net fails.
let failures = 0;
const TRIALS = 20;
for (let t = 0; t < TRIALS; t++) if (train(makeNet([2, 4, 1]), "relu", 0.5, 3000) > 0.01) failures++;
console.log(`   (c) dying relu     ${failures}/${TRIALS} runs of 2->4->1 (lr=0.5) never reached loss < 0.01`);
console.log("       This is what gelu fixes: a small negative-side gradient, so units revive.");

// (d) Sigmoid saturation. Oversized initial weights push z far from 0, where
//     the curve is flat. |z| > 4 means sigmoid'(z) < 0.018 — pinned before
//     training even starts.
// (d) Sigmoid saturation. Oversized initial weights push the pre-activation z
//     far from 0, where the curve is flat. Measured at init, before any
//     training has happened:
console.log("   (d) sigmoid saturation at init");
for (const scale of [1, 25]) {
  const net = makeNet([2, 8, 8, 1], scale);
  // Layer 1's pre-activations: z = X @ W1 + b1, one per unit per example.
  const z1 = Array.from(XOR_X.matMul(net[0]!).add(net[1]!).data.data);
  const saturated = z1.filter((z) => Math.abs(z) > 4).length;
  console.log(`       init scale x${String(scale).padStart(2)}   |z|>4 in ${String(saturated).padStart(2)}/${z1.length} units`);
}
console.log("       |z| > 4 means sigmoid'(z) < 0.018 — the unit is pinned near 0 or 1");
console.log("       and its gradient is nearly gone before step 1.");
// Honest caveat: on a problem this small the FINAL loss is erratic — a
// saturated net sometimes still recovers. Saturation is reliably visible in
// the gradient, not in the loss. The reliable training failure is (e).

// (e) Depth is what actually kills sigmoid. Same init, same lr, just deeper —
//     this is E9's 0.25^n arriving as a training outcome.
console.log("   (e) depth vs sigmoid (normal init, lr=0.05)");
for (const depth of [2, 6]) {
  const losses = [0, 1, 2].map(() => train(makeNet([2, ...new Array(depth).fill(8), 1], 1), "sigmoid", 0.05, 8000));
  console.log(`       ${depth} hidden layers   losses ${losses.map(fmt).join("")}`);
}
console.log("       2 layers learn. 6 layers sit at 0.2500 — the 0.25 floor of E7,");
console.log("       reached not because the net CANNOT represent XOR but because the");
console.log("       gradient never reaches the early layers. Compare E9's numbers.");

// TODO: swap relu for gelu in forward() and re-run E10(c). Does the failure
//       count drop? That is the entire argument for gelu, measured.

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: implement swish: swish(x) = x * sigmoid(x)
//       This one you CAN compose — sigmoid is now a primitive, and mul is a
//       TensorValue method, so no new _backward is needed. Try it and confirm
//       the gradient still checks out. Then compare swish against gelu on the
//       running row: they are close, which is why both are used in practice.
