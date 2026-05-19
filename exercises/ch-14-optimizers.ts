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
console.log("\nSGD  losses (every 20 steps):", sgdLosses.map(v => v.toFixed(3)));
console.log("Adam losses (every 20 steps):", adamLosses.map(v => v.toFixed(3)));

// ─── E3: Adam hyperparameters ─────────────────────────────────────────────────
// Default: lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8
// TODO: try lr=0.01 vs lr=0.001 — print final loss for each.
// What happens with very high lr (1.0)?

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: implement a simple learning-rate scheduler:
//   lr(step) = lr_base * 0.1^(step / decay_steps)
//   Apply it by mutating adam.lr each step.
