/**
 * EXERCISES — Ch 14: Optimizers
 * ════════════════════════════════════════════
 * Prereq : src/optim/sgd.ts, src/optim/adam.ts, src/nn/linear.ts
 * Run    : bun run exercises/ch-14-optimizers.ts
 *
 * Every number here is one the chapter derives. Run it after implementing
 * each optimizer and check the printout against the doc.
 */
import { SGD, SGDMomentum } from "../src/optim/sgd.ts";
import { Adam, AdamW } from "../src/optim/adam.ts";
import { Linear } from "../src/nn/linear.ts";
import { TensorValue } from "../src/autograd/grad.ts";
import { createTensor } from "../src/tensor/types.ts";
import { mseLoss } from "../src/nn/losses.ts";

/** Run a block; if the optimizer inside is still a stub, say so and move on. */
function stage(title: string, body: () => void): void {
  console.log(`\n─── ${title} ───`);
  try {
    body();
  } catch (error) {
    if (error instanceof Error && error.message.includes("not implemented")) {
      console.log("  pending —", error.message);
    } else throw error;
  }
}

const param = (v: number) => new TensorValue(createTensor([v], [1]));
const setGrad = (p: TensorValue, v: number) => {
  p.grad = createTensor([v], p.data.shape);
};

/** Move per step with lr = 1, which IS the optimizer's internal ratio. */
function ratios(make: (p: TensorValue) => { step(): void }, grads: number[]): number[] {
  const p = param(0);
  const opt = make(p);
  const out: number[] = [];
  let previous = 0;
  for (const g of grads) {
    setGrad(p, g);
    opt.step();
    out.push(previous - p.data.data[0]!);
    previous = p.data.data[0]!;
  }
  return out;
}

// ─── E1: the bowl — SGD's trace from the chapter ─────────────────────────────
// L = (θ − 5)², θ₀ = 0, lr = 0.1  →  1, 1.8, 2.44, 2.952, each step 0.8× the last
stage("E1: SGD on the bowl", () => {
  const theta = param(0);
  const opt = new SGD([theta], 0.1);
  const target = createTensor([5], [1]);
  const seen: number[] = [];
  for (let i = 0; i < 4; i++) {
    opt.zeroGrad();
    mseLoss(theta, target).backward();
    opt.step();
    seen.push(theta.data.data[0]!);
  }
  console.log("  θ:", seen.map((v) => v.toFixed(4)).join("  "));
  console.log("  expected: 1.0000  1.8000  2.4400  2.9520");
});

// ─── E2: momentum — agreement accumulates, disagreement cancels ──────────────
stage("E2: momentum, the same rule run twice", () => {
  const same = ratios((p) => new SGDMomentum([p], 1, 0.9), Array(8).fill(1));
  const alt = ratios((p) => new SGDMomentum([p], 1, 0.9),
    Array.from({ length: 8 }, (_, i) => (i % 2 ? -1 : 1)));
  console.log("  gradients all +1 :", same.map((v) => v.toFixed(3)).join("  "));
  console.log("  expected         : 1.000  1.900  2.710  3.439  4.095  4.686  5.217  5.695");
  console.log("  alternating ±1   :", alt.map((v) => v.toFixed(3)).join("  "));
  console.log("  expected         : 1.000 -0.100  0.910 -0.181  0.837 -0.247  0.778 -0.300");
  console.log("  same rule, same |g| — only the signs differ, and the outcomes are 5.695 vs -0.300");
});

// ─── E3: Adam's first step — what bias correction buys ───────────────────────
stage("E3: Adam's first step", () => {
  const p = param(0);
  setGrad(p, 1);
  new Adam([p], 1e-3).step();
  console.log("  moved by:", (-p.data.data[0]!).toExponential(6), "  expected ≈ lr = 1.000000e-3");
  console.log("  without correction it would be 0.1/√0.001 · lr =",
    ((0.1 / Math.sqrt(0.001)) * 1e-3).toExponential(3), "— three times too far");
});

// ─── E4: the ratio ignores gradient SIZE, but not agreement ──────────────────
stage("E4: what m̂/√v̂ comes out to", () => {
  const show = (name: string, gs: number[]) =>
    console.log(`  ${name.padEnd(22)}`,
      ratios((p) => new Adam([p], 1), gs).slice(-4).map((v) => v.toFixed(3).padStart(7)).join(" "));
  show("all +1", Array(20).fill(1));
  show("all +1000", Array(20).fill(1000));
  show("alternating +3, −3", Array.from({ length: 20 }, (_, i) => (i % 2 ? -3 : 3)));
  show("noisy, mixed signs", [2, -1, 5, -3, 1, 4, -2, 3, -1, 2, 6, -4, 1, 3, -2, 5, -1, 2, 4, -3]);
  console.log("  +1 and +1000 are identical — the gradient scale has cancelled.");
  console.log("  the alternating row is near zero — Adam steps small when unsure.");
});

// ─── E5: the two-parameter problem the chapter opens with ────────────────────
// A receives gradients ~100, B receives ~0.001. Should one lr suit both?
stage("E5: parameter A vs parameter B", () => {
  const a = param(0), b = param(0);
  const sgd = new SGD([a, b], 1e-3);
  setGrad(a, 100);
  setGrad(b, 0.001);
  sgd.step();
  console.log("  SGD    A moved", (-a.data.data[0]!).toExponential(2),
    "  B moved", (-b.data.data[0]!).toExponential(2), "  ratio 100,000 : 1");
  const c = param(0), d = param(0);
  const adam = new Adam([c, d], 1e-3);
  setGrad(c, 100);
  setGrad(d, 0.001);
  adam.step();
  console.log("  Adam   A moved", (-c.data.data[0]!).toExponential(2),
    "  B moved", (-d.data.data[0]!).toExponential(2), "  identical");
});

// ─── E6: AdamW's decoupled decay ─────────────────────────────────────────────
stage("E6: AdamW", () => {
  const p = param(10);
  setGrad(p, 0);
  new AdamW([p], 0.1, 0.01).step();
  console.log("  θ=10, lr=0.1, λ=0.01, zero gradient →", p.data.data[0]!.toFixed(6),
    "  expected 9.990000");
  // Decoupled: Adam's share must not depend on how large θ is.
  const share = (theta: number) => {
    const q = param(theta);
    const opt = new AdamW([q], 0.1, 0.01);
    setGrad(q, 1);
    opt.step();
    return theta - q.data.data[0]! - 0.1 * 0.01 * theta;
  };
  console.log("  Adam's share at θ=1:", share(1).toFixed(8), "  at θ=100:", share(100).toFixed(8));
  console.log("  identical → the decay never entered m or v. That is the 'W'.");
});

// ─── E7: all four on a real Linear layer ─────────────────────────────────────
// An honest comparison: on a clean, well-conditioned problem plain SGD is
// hard to beat. Adam's advantage is robustness, not raw speed here.
stage("E7: training a Linear(2,3) to output [1,2,3]", () => {
  const x = new TensorValue(createTensor([1, 2], [1, 2]));
  const want = createTensor([1, 2, 3], [1, 3]);
  const run = (name: string, make: (ps: TensorValue[]) => { step(): void; zeroGrad(): void }, steps: number) => {
    const layer = new Linear(2, 3);
    const opt = make(layer.parameters());
    let loss = 0;
    for (let i = 0; i < steps; i++) {
      opt.zeroGrad();
      const l = mseLoss(layer.forward(x), want);
      l.backward();
      opt.step();
      loss = l.data.data[0]!;
    }
    console.log(`  ${name.padEnd(14)} after ${steps} steps, loss ${loss.toExponential(2)}`);
  };
  run("SGD", (ps) => new SGD(ps, 0.05), 200);
  run("SGDMomentum", (ps) => new SGDMomentum(ps, 0.01, 0.9), 200);
  run("Adam", (ps) => new Adam(ps, 0.05), 200);
  run("AdamW", (ps) => new AdamW(ps, 0.05, 0.001), 200);
});

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO 1: a learning-rate schedule. `learningRate` is readonly, so build a new
//         optimizer each phase, or make the field mutable and decay it:
//             lr(step) = lrBase · 0.1 ^ (step / decaySteps)
//         Does it help SGD more than Adam? Why would you expect that?
//
// TODO 2: reproduce the chapter's ravine figure. L = 0.5x² + 10y² with two
//         parameters. Count how many times each optimizer crosses y = 0 in
//         28 steps — SGD 28, momentum 8 at matched effective step size.
