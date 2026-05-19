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
console.log("\nsigmoid'(0.5) numerical :", numerical.toFixed(8));
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
console.log("\nd/dx sin(x²) at x=1 numerical :", grad_chain.toFixed(8));
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
console.log("\nGD minimising (x-3)²: x =", xParam.toFixed(6), "  expected: ≈ 3.0");

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: numerically verify the gradient of tanh at x=0, x=1, x=2.
//   Analytical:  tanh'(x) = 1 - tanh(x)²
