/**
 * CHAPTER 14: Optimizers — Adam
 * ════════════════════════════════════════
 * Part 3 of 6: Neural Net Primitives
 *
 * WHAT WE'RE BUILDING:  class Adam — per-parameter adaptive step sizes, built
 *                       from two running averages of the gradient.
 * WHY IT MATTERS:       Ch 15's training loop drives it, and it is the
 *                       optimizer every transformer in this course trains with.
 * WHAT THIS UNLOCKS:    → Ch 15 (The Training Loop) — the last piece before a
 *                       model can actually be trained end to end.
 *
 * REFERENCE: docs/part-3-neural-net-primitives/ch-14-optimizers.md
 *
 * Update rule (per parameter, per step t):
 *   m  = β₁ m  + (1−β₁) g          ← first moment  ("which way?")
 *   v  = β₂ v  + (1−β₂) g²         ← second moment ("how big?")
 *   m̂  = m / (1−β₁ᵗ)               ← bias correction
 *   v̂  = v / (1−β₂ᵗ)
 *   θ -= lr * m̂ / (√v̂ + ε)
 *
 * ──────────────────────────────────────────────────────────────────────────
 * WHY TensorValue[] AND NOT Value[]
 * ──────────────────────────────────────────────────────────────────────────
 * Ch 09's SGD in sgd.ts takes scalar `Value` objects, because Ch 09 came
 * before TensorValue existed. Adam is written after Ch 13, and Ch 13's
 * Linear.parameters() returns TensorValue[] — so that is what Adam consumes.
 * This is the whole point of the parameters() contract: a layer hands over a
 * flat list, and the optimizer walks it knowing nothing else about the model.
 *
 * The API shape matches Ch 09's deliberately: parameters go in the
 * CONSTRUCTOR, then step() and zeroGrad() take no arguments. That is what
 * lets the optimizer own per-parameter state, which is exactly what Adam is.
 *
 * ──────────────────────────────────────────────────────────────────────────
 * THE STATE — one m and one v PER PARAMETER, persisting across steps
 * ──────────────────────────────────────────────────────────────────────────
 * Not one shared pair. If W is [3,2] and b is [3], then Adam holds:
 *
 *     m[W], v[W]   shape [3,2]        m[b], v[b]   shape [3]
 *
 * all starting at zeros. Recreating them inside step() is the classic bug:
 * everything runs, no error appears, and Adam silently degrades into
 * plain SGD with a strange scale factor — because m and v never accumulate
 * any history. The `t` counter must persist too; it drives bias correction.
 *
 * ──────────────────────────────────────────────────────────────────────────
 * WORKED TRACE — the doc's "Let's watch the first Adam step", g = 1
 * ──────────────────────────────────────────────────────────────────────────
 *     β₁ = 0.9   β₂ = 0.999   m₀ = 0   v₀ = 0   t = 1
 *
 *     m₁ = 0.9·0     + 0.1·1     = 0.1
 *     v₁ = 0.999·0   + 0.001·1²  = 0.001
 *
 *     m̂₁ = 0.1   / (1 − 0.9¹)   = 0.1   / 0.1   = 1
 *     v̂₁ = 0.001 / (1 − 0.999¹) = 0.001 / 0.001 = 1
 *
 *     update = 1 / (√1 + ε) ≈ 1        ← a sensible first step, not 0.1
 *
 * Without bias correction that first step would be 0.1/√0.001 ≈ 3.16 — wrong
 * in the other direction. Both corrections are needed, and they fade fast:
 * the m̂ factor is ×10 at t=1, ×1.54 by t=10, ×1.005 by t=50.
 *
 * ──────────────────────────────────────────────────────────────────────────
 * PITFALL: the update must NOT touch the graph
 * ──────────────────────────────────────────────────────────────────────────
 * step() mutates .data directly with plain tensor functions (sub, mulScalar,
 * add, applyFn from tensor/*). Using TensorValue METHODS here would build
 * graph nodes on every training step — the graph would grow without bound
 * across an entire run, and the parameters would stop being leaves. Ch 11's
 * rule, one level up: forward passes build the graph, everything else does
 * not.
 */
import {
  add,
  addScalar,
  div,
  mul,
  mulScalar,
  sqrt,
  sub,
  zeros,
  type Tensor,
} from "../tensor/index.ts";
import { TensorValue } from "../autograd/grad.ts";

export class Adam {
  readonly params: TensorValue[];
  readonly learningRate: number;
  readonly beta1: number;
  readonly beta2: number;
  readonly epsilon: number;

  /** First moments — "which way", one per parameter, same order as params. */
  private readonly ms: Tensor[];
  /** Second moments — "how big", one per parameter. */
  private readonly vs: Tensor[];
  /** Steps taken so far. Drives bias correction, so it must persist. */
  private t: number;

  /**
   * ── STEPS ─────────────────────────────────────────────────────────────────
   *   1. Store params and the four hyperparameters. Defaults are the ones
   *      everyone uses: lr = 1e-3, β₁ = 0.9, β₂ = 0.999, ε = 1e-8.
   *   2. Allocate m and v as zeros matching EACH parameter's shape, and keep
   *      them alongside this.params (parallel arrays are fine — the i-th
   *      entry belongs to the i-th parameter).
   *   3. Initialise the step counter t = 0. It increments once per step(),
   *      not once per parameter.
   *
   * ── PITFALL: β₂ = 0.999 means v has a ~1000-step memory ───────────────────
   * That is deliberate — the size estimate should be stable. β₁ = 0.9 gives
   * m a ~10-step memory, so direction reacts faster than scale. Swapping the
   * two betas produces a working-looking optimizer that trains badly.
   */
  constructor(
    params: TensorValue[],
    learningRate?: number,
    beta1?: number,
    beta2?: number,
    epsilon?: number
  ) {
    this.params = params;
    this.learningRate = learningRate !== undefined ? learningRate : 1e-3;
    this.beta1 = beta1 !== undefined ? beta1 : 0.9;
    this.beta2 = beta2 !== undefined ? beta2 : 0.999;
    this.epsilon = epsilon !== undefined ? epsilon : 1e-8;

    // The state, allocated ONCE. One m and one v per parameter, each matching
    // that parameter's own shape. Building these inside step() would restart
    // both averages from zero every iteration — no error, and Adam silently
    // stops being Adam.
    this.ms = params.map((param) => zeros(param.data.shape));
    this.vs = params.map((param) => zeros(param.data.shape));
    this.t = 0;
  }

  /**
   * Apply one Adam step to all parameters.
   *
   * ── STEPS (for each parameter i, all on RAW tensors) ──────────────────────
   *   0. t += 1, once, before the loop.
   *   1. g  = params[i].grad          (skip the parameter if it is null)
   *   2. m[i] = β₁·m[i] + (1−β₁)·g
   *   3. v[i] = β₂·v[i] + (1−β₂)·g⊙g
   *   4. m̂ = m[i] / (1 − β₁ᵗ)         v̂ = v[i] / (1 − β₂ᵗ)
   *   5. params[i].data -= lr · m̂ / (√v̂ + ε)      elementwise
   *
   * Steps 2–5 are all elementwise, so applyFn / mul / mulScalar / add / sub
   * from tensor/ops.ts cover everything. No new backward: this is not part
   * of the graph at all.
   *
   * ── WHY THE DIVISION IS THE POINT ─────────────────────────────────────────
   * m̂ tracks the gradient and √v̂ tracks its size, so their ratio is roughly
   * ±1 regardless of how large the gradients are. Multiply every gradient by
   * 10 and m̂ scales by 10, √v̂ scales by 10, and the update is unchanged —
   * measured identical to 2.6e-9 over 30 steps. That is why Adam needs so
   * little learning-rate tuning, and why its steps are bounded near lr.
   */
  step(): void {
    // Once per step, before the loop — every parameter shares this clock.
    this.t += 1;

    // Bias-correction denominators depend only on t, so compute them once.
    const correctM = 1 - Math.pow(this.beta1, this.t);
    const correctV = 1 - Math.pow(this.beta2, this.t);

    this.params.forEach((param, i) => {
      const g = param.grad;
      if (g === null) return;

      // m = β₁·m + (1−β₁)·g          which way the gradient usually points
      this.ms[i] = add(
        mulScalar(this.ms[i]!, this.beta1),
        mulScalar(g, 1 - this.beta1),
      );

      // v = β₂·v + (1−β₂)·g⊙g        how large the gradients usually are.
      // mul(g, g), not mulScalar — squaring is elementwise, tensor by tensor.
      this.vs[i] = add(
        mulScalar(this.vs[i]!, this.beta2),
        mulScalar(mul(g, g), 1 - this.beta2),
      );

      // Dividing a tensor by a NUMBER is mulScalar by its reciprocal;
      // div() expects two tensors.
      const mHat = mulScalar(this.ms[i]!, 1 / correctM);
      const vHat = mulScalar(this.vs[i]!, 1 / correctV);

      // θ -= lr · m̂ / (√v̂ + ε). The ε is added AFTER the root, and addScalar
      // is what adds a number to a tensor.
      const update = div(mHat, addScalar(sqrt(vHat), this.epsilon));
      param.data = sub(param.data, mulScalar(update, this.learningRate));
    });
  }

  /**
   * Zero gradients on all parameters.
   *
   * Same reason as Ch 09's SGD.zeroGrad: backward() accumulates with +=, so
   * without this the previous step's gradients ride into the next one and the
   * effective learning rate grows every iteration. It looks exactly like a
   * learning rate set too high.
   *
   * Setting each .grad back to null is enough — Ch 10's accumulate() treats
   * null as "first contribution" and assigns rather than adds.
   */
  zeroGrad(): void {
    for (const param of this.params) {
      param.grad = null;
    }
  }
}
