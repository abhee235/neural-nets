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
console.log("\nlr=0.01 →", minimise(0.01, 200).toFixed(3), "  (slower, still converges)");
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
console.log("\nSGD+momentum after 50 steps:", wm.data.toFixed(4), "  expected: ≈ 5.0");

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
console.log("\nLinear fit: slope =", slope.data.toFixed(3), " expected ≈ 2");
console.log("Linear fit: intercept =", intercept.data.toFixed(3), " expected ≈ 1");

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: plot the loss curve (print loss every 50 steps).
//       Does it decrease smoothly? What shape is it?
