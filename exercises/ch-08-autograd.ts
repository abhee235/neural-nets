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
console.log("\ndf/da:", a.grad, "  expected: 4");
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
console.log("\ndy/dw autograd  :", w.grad.toFixed(8));
console.log("dy/dw numerical :", numGrad.toFixed(8));
console.log("match:", Math.abs(w.grad - numGrad) < 1e-5, "  expected: true");

// ─── E4: Gradient accumulation ───────────────────────────────────────────────
// When a value is used twice, gradients accumulate (sum).
// z = a * a   →   dz/da = 2a
const av = new Value(3, "a");
const z = av.mul(av);
z.backward();
console.log("\ndz/d(a) where z=a*a, a=3:", av.grad, "  expected: 6");

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: build a small 2-input neuron:
//   n = relu(w1*x1 + w2*x2 + b)
// Set w1=0.4, w2=-0.6, b=0.1, x1=1.0, x2=0.5.
// Verify all gradients numerically.
