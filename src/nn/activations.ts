/**
 * nn/activations.ts
 * ══════════════════════════════════════════════════════════
 * Differentiable activation functions operating on TensorValues.
 * Equivalent to torch.nn.functional.relu, gelu, sigmoid, softmax.
 *
 * Chapter: 11 — Activation Functions
 * Doc:     docs/part-3-neural-net-primitives/ch-11-activation-functions.md
 *
 * ──────────────────────────────────────────────────────────────────────────
 * WHY THIS FILE EXISTS
 * ──────────────────────────────────────────────────────────────────────────
 * `TensorValue` knows seven operations: add, mul, matMul, sum, mean, reshape,
 * transpose. There is NO combination of them — however long — that produces
 * relu, sigmoid, tanh or exp. Adds and multiplies build polynomials, and none
 * of these is a polynomial.
 *
 * So a nonlinearity cannot be composed. It has to be ADDED to the engine as a
 * new primitive: a node that knows its own forward value and its own
 * derivative. You did exactly this in Ch 08, when Value gained exp, log, tanh
 * and relu on top of add and mul. This file is that same act, with tensors.
 *
 * Without it, depth is worthless — Ch 09's deep dive proved two linear layers
 * collapse into one. These four functions are what make stacking mean
 * something.
 *
 * ──────────────────────────────────────────────────────────────────────────
 * THE PATTERN — one recipe, used four times
 * ──────────────────────────────────────────────────────────────────────────
 *   1. forward:  apply the function to every cell   → a Tensor
 *   2. wrap:     const out = new TensorValue(thatTensor)
 *   3. wire:     out._inputs = [x]                  ← by hand, as in Ch 10
 *   4. backward: out._backward = () => accumulate into x
 *   5. return out
 *
 * ──────────────────────────────────────────────────────────────────────────
 * WHY THIS CHAPTER IS EASIER THAN CH 10
 * ──────────────────────────────────────────────────────────────────────────
 * Three of these four are ELEMENTWISE: output cell i depends on input cell i
 * and nothing else. So the shape never changes, and backward has nothing to
 * repair — no sumToShape, no unsqueeze, no broadcasting:
 *
 *     x.grad  +=  f'(x)  ⊙  out.grad          ⊙ = element-wise mul (Ch 03)
 *
 * That one line is the backward of relu, sigmoid and gelu. Only softmax is
 * different, because its outputs are tied together by summing to 1.
 *
 * ──────────────────────────────────────────────────────────────────────────
 * THE RUNNING EXAMPLE — the doc's row, used in every trace below
 * ──────────────────────────────────────────────────────────────────────────
 *     x         = [ -2       -1       0       1       2      ]   shape [5]
 *
 *     relu(x)   = [  0.0000   0.0000  0.0000  1.0000  2.0000 ]
 *     gelu(x)   = [ -0.0454  -0.1588  0.0000  0.8412  1.9546 ]
 *     sigmoid(x)= [  0.1192   0.2689  0.5000  0.7311  0.8808 ]
 *
 * and for softmax, a three-element row:
 *
 *     [ 1  2  3 ]  →  [ 0.090031  0.244728  0.665241 ]   (sums to 1)
 *
 * ──────────────────────────────────────────────────────────────────────────
 * WHAT YOU WILL NEED TO IMPORT
 * ──────────────────────────────────────────────────────────────────────────
 * The FORWARD passes already exist — you are not writing exp from scratch:
 *
 *   tensor/math.ts     sigmoid, tanh, exp, pow      ← Ch 06
 *   tensor/reduce.ts   softmax, sum                 ← Ch 05 (softmax is stable)
 *   tensor/ops.ts      mul, sub, applyFn            ← Ch 03
 *
 * `applyFn(t, v => ...)` maps a plain function over every cell — the simplest
 * way to build both a forward and a local-derivative tensor.
 *
 * THE WORK IN THIS FILE IS THE BACKWARD. That is the part the engine cannot
 * infer and you must supply.
 *
 * ──────────────────────────────────────────────────────────────────────────
 * ONE DECISION BEFORE YOU START
 * ──────────────────────────────────────────────────────────────────────────
 * Every backward below has to accumulate into `x.grad` with the null-aware
 * pattern from Ch 10 — first contribution assigns, later ones add:
 *
 *     if (x.grad === null)  x.grad = contribution;
 *     else                  x.grad = add(x.grad, contribution);
 *
 * You already wrote exactly this as `accumulate` in grad.ts — but it is a
 * private function there, so this file cannot import it. Either add `export`
 * to it (recommended: one keyword, and all four activations reuse it) or
 * write those two lines here. Decide once, now, rather than four times.
 */
import { TensorValue } from "../autograd/grad.ts";

/**
 * ReLU: max(0, x).
 * Gradient: 1 for x > 0, 0 for x ≤ 0.
 *
 * ── SCALAR (Ch 08) — your own value.ts, unchanged ───────────────────────────
 *     relu(): Value {
 *       const out = new Value(Math.max(0, this.data), [this], "relu");
 *       out._backward = () => {
 *         this.grad += (this.data > 0 ? 1 : 0) * out.grad;
 *       };
 *       return out;
 *     }
 *
 * ── TENSOR (Ch 11) ──────────────────────────────────────────────────────────
 * The same three jobs, with "every cell" in place of "the number":
 *
 *     forward :  applyFn(x.data, v => Math.max(0, v))
 *     local   :  applyFn(x.data, v => (v > 0 ? 1 : 0))     ← the 0/1 gate
 *     backward:  x.grad += mul(local, out.grad)
 *
 * ── IT IS A GATE, NOT A SCALE ───────────────────────────────────────────────
 * Where the input was positive the gradient passes through untouched (×1);
 * where it was negative nothing passes at all (×0). A unit whose input stays
 * negative receives exactly zero forever and never learns again — the dying
 * ReLU, the same mechanism as the frozen parameter in Ch 09's deep dive.
 *
 * ── WORKED TRACE — the running row ──────────────────────────────────────────
 *
 *     x         = [ -2  -1   0   1   2 ]
 *     relu(x)   = [  0   0   0   1   2 ]     ← negatives flattened
 *
 *     relu'(x)  = [  0   0   0   1   1 ]     ← the gate
 *     out.grad  = [  1   1   1   1   1 ]     ← say the upstream is all ones
 *                  ─────────────────────  ⊙
 *     x.grad    = [  0   0   0   1   1 ]     shape [5] ✓ unchanged
 *
 * ── PITFALL: x = 0 exactly ──────────────────────────────────────────────────
 * The derivative genuinely does not exist at the corner. Pick a convention and
 * document it — we use 0, matching Ch 08. This is why the test says "(x ≠ 0)":
 * a centered finite difference across the corner averages the two one-sided
 * slopes to 0.5 and disagrees with every convention.
 */
export function relu(x: TensorValue): TensorValue {
  throw new Error("relu not implemented");
}

/**
 * GELU — Gaussian Error Linear Unit.
 * Approximation: 0.5 x (1 + tanh(√(2/π)(x + 0.044715 x³)))
 *
 * Used inside every FFN block in GPT-2 and most modern transformers.
 * Smoother than ReLU near x = 0.
 *
 * ── SCALAR (Ch 08) ──────────────────────────────────────────────────────────
 * None — Ch 08 never had gelu. But its `tanh` is the closest relative you
 * have written, and the same shape of backward applies.
 *
 * ── TENSOR (Ch 11) ──────────────────────────────────────────────────────────
 * Forward: the approximation above, cell by cell with `applyFn`. Name the
 * constant — √(2/π) ≈ 0.7978845608.
 *
 * Backward: differentiate the approximation (product rule on `0.5·x·(1+tanh u)`
 * with `u = k(x + 0.044715x³)`, chain rule through the tanh) and evaluate it
 * cell by cell with `applyFn`, exactly as relu does. Then the same one line:
 *
 *     x.grad += mul(local, out.grad)
 *
 * ── WHY IT IS WORTH THE EXTRA ALGEBRA ───────────────────────────────────────
 * relu makes a hard decision at zero: pass fully, or block fully. gelu makes a
 * soft one — compare the same inputs:
 *
 *     x        = [ -2       -1       0       1       2      ]
 *     relu(x)  = [  0.0000   0.0000  0.0000  1.0000  2.0000 ]
 *     gelu(x)  = [ -0.0454  -0.1588  0.0000  0.8412  1.9546 ]
 *                              ↑
 *              relu returns a hard 0 here and its gate shuts;
 *              gelu returns -0.1588, so a gradient still flows
 *              and the unit can recover
 *
 * ── WORKED TRACE — the derivative row ───────────────────────────────────────
 *
 *     gelu'(x) = [ -0.0861  -0.0830  0.5000  1.0830  1.0861 ]
 *
 * Two surprises worth checking against, because they catch sign errors:
 * at x = 0 the derivative is 0.5 (half open, not 0 or 1), and at x = -2 and
 * x = -1 it is NEGATIVE — gelu is not monotonic; it dips below zero before
 * flattening.
 *
 * ── PITFALL: verify this one numerically ────────────────────────────────────
 * Its derivative has the most terms and the least intuition in this file, so a
 * dropped constant produces plausible-looking numbers. `checkTensorGradient`
 * from Ch 10 catches that instantly; re-reading the formula does not.
 */
export function gelu(x: TensorValue): TensorValue {
  throw new Error("gelu not implemented");
}

/**
 * Sigmoid: σ(x) = 1 / (1 + e^{−x}).
 * Output in (0, 1).  Gradient: σ(x)(1 − σ(x)).
 *
 * ── SCALAR (Ch 08) ──────────────────────────────────────────────────────────
 * Ch 08's `tanh` is the pattern to copy — note especially what its backward
 * reads from:
 *
 *     tanh(): Value {
 *       const out = new Value(Math.tanh(this.data), [this], "tanh");
 *       out._backward = () => {
 *         this.grad += (1 - out.data ** 2) * out.grad;   // ← out.data, not this.data
 *       };
 *       return out;
 *     }
 *
 * ── TENSOR (Ch 11) ──────────────────────────────────────────────────────────
 * Forward: `sigmoid(x.data)` from Ch 06 — already written, already careful.
 *
 * Backward: σ' = σ(1 − σ), and σ is the value you JUST computed. Build the
 * local derivative from `out.data`:
 *
 *     local = applyFn(out.data, s => s * (1 - s))
 *     x.grad += mul(local, out.grad)
 *
 * Reusing the output is the same trick as Ch 08's exp and tanh, for the same
 * two reasons: it avoids recomputing an exponential, and it cannot drift from
 * the value the forward pass actually used.
 *
 * ── WORKED TRACE — the running row ──────────────────────────────────────────
 *
 *     x           = [ -2       -1       0       1       2      ]
 *     sigmoid(x)  = [  0.1192   0.2689  0.5000  0.7311  0.8808 ]   ← out.data
 *     sigmoid'(x) = [  0.1050   0.1966  0.2500  0.1966  0.1050 ]
 *                                         ↑
 *                              the maximum — 0.25, at x = 0
 *
 * ── PITFALL: out.data, NOT x.data ───────────────────────────────────────────
 * `x.data * (1 - x.data)` is a real and popular bug: it happens to be right at
 * x = 0 and is wrong everywhere else — the same trap as Ch 08's tanh backward,
 * which is why that test is checked at x = 1 rather than the origin.
 *
 * ── WHAT THAT 0.25 CEILING COSTS ────────────────────────────────────────────
 * sigmoid's derivative never exceeds 0.25 anywhere, and Ch 09's deep dive
 * showed a gradient reaching an early layer is the PRODUCT of every local
 * derivative on the way. Ten sigmoid layers, best case:
 *
 *     0.25¹⁰ ≈ 9.5e-7          vs.   relu:  1¹⁰ = 1
 *
 * That is the whole reason relu replaced sigmoid in hidden layers.
 */
export function sigmoid(x: TensorValue): TensorValue {
  throw new Error("sigmoid not implemented");
}

/**
 * Numerically stable softmax along axis (default: last axis).
 * Output sums to 1.0 — a probability distribution.
 *
 * ── THE ONE THAT IS NOT ELEMENTWISE ─────────────────────────────────────────
 * Every function above had output cell i depending on input cell i alone.
 * softmax breaks that: its denominator sums over EVERY element along the axis,
 * so changing one input moves all the outputs. They are tied together by
 * having to sum to 1.
 *
 * ── TENSOR (Ch 11) ──────────────────────────────────────────────────────────
 * Forward: `softmax(x.data, axis)` from Ch 05 — one call. It already subtracts
 * the max, which is what keeps it safe (see below). Default the axis to the
 * last one: `axis ?? x.data.ndim - 1`.
 *
 * Backward: because outputs are coupled, the local derivatives form a matrix
 * (a Jacobian) rather than a single number per cell:
 *
 *     ∂sᵢ/∂xⱼ = sᵢ(δᵢⱼ − sⱼ)
 *
 * which collapses to something much friendlier than it looks. With s = out.data:
 *
 *     x.grad += s ⊙ ( out.grad − Σₖ(out.grad ⊙ s) )
 *                                  └── summed along `axis`, keepDims = true ──┘
 *
 * In words: take the upstream gradient, subtract its s-weighted average, then
 * scale by s. One weighted sum plus two elementwise ops — no matrix is ever
 * built. `keepDims = true` makes the sum broadcast back cleanly against the
 * full tensor: the Ch 10 machinery doing exactly what it was built for.
 *
 * ── WORKED TRACE — the three-element row ────────────────────────────────────
 *
 *     x          = [ 1         2         3        ]
 *     softmax(x) = [ 0.090031  0.244728  0.665241 ]     sums to 1.0
 *
 * Shift invariance — subtracting any constant changes nothing:
 *
 *     softmax([0, 1, 2])       = [ 0.090031  0.244728  0.665241 ]   identical
 *     softmax([1000,1001,1002])= [ 0.090031  0.244728  0.665241 ]   identical
 *
 * The last one is why the max subtraction matters: exp(1000) is Infinity, and
 * Infinity/Infinity is NaN. Ch 05's deep dive proves the invariance:
 * docs/deep-dives/ch-05-why-subtract-the-max.md
 *
 * ── YOU WILL RARELY CALL THIS BACKWARD ──────────────────────────────────────
 * softmax is almost always followed by cross-entropy, and the two together
 * have a famously simple combined gradient — s − y_true, the output minus the
 * one-hot label. Ch 12 implements that fused form, which is both faster and
 * more stable. You implement the standalone version here because attention
 * (Ch 22) uses softmax with no loss attached.
 */
export function softmax(x: TensorValue, axis?: number): TensorValue {
  throw new Error("softmax not implemented");
}
