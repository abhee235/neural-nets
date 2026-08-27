/**
 * CHAPTER 14: Optimizers — SGD and SGD with momentum
 * ════════════════════════════════════════
 * Part 3 of 6: Neural Net Primitives
 *
 * WHAT WE'RE BUILDING:  class SGD          — θ ← θ − lr·g
 *                       class SGDMomentum  — v ← β·v + g,  then θ ← θ − lr·v
 * WHY IT MATTERS:       the two rungs the chapter climbs before Adam. Adam's
 *                       `m` IS momentum; `v` is the thing momentum lacked.
 * WHAT THIS UNLOCKS:    → Adam (adam.ts), then Ch 15's training loop.
 *
 * REFERENCE: docs/part-3-neural-net-primitives/ch-14-optimizers.md
 *
 * ──────────────────────────────────────────────────────────────────────────
 * THIS IS A REBUILD, NOT A NEW IDEA — read optim/sgd-scalar.ts alongside
 * ──────────────────────────────────────────────────────────────────────────
 * You already wrote both of these in Ch 09, on scalar `Value`. That file is
 * kept as optim/sgd-scalar.ts, untouched and still passing its 16 tests, so
 * you can put the two versions side by side.
 *
 * The RULE does not change at all. What changes is that `.data` and `.grad`
 * are Tensors instead of numbers, so arithmetic that was infix becomes a
 * function call:
 *
 *     Ch 09 (scalar)                    Ch 14 (tensor)
 *     ─────────────────────────────     ──────────────────────────────────
 *     param.data -= lr * param.grad     param.data = sub(param.data,
 *                                           mulScalar(param.grad, lr))
 *     param.grad = 0                    param.grad = null
 *     v = beta * v + g                  v = add(mulScalar(v, beta), g)
 *
 * That is the whole migration. If you find yourself writing anything
 * cleverer than that, you have probably wandered off.
 *
 * ──────────────────────────────────────────────────────────────────────────
 * WHY TENSORS NOW
 * ──────────────────────────────────────────────────────────────────────────
 * Ch 13's `Linear.parameters()` returns `TensorValue[]`. An optimizer taking
 * `Value[]` cannot consume a single real layer, so nothing built after Ch 13
 * could be trained with it. This is the `parameters()` contract being paid
 * off: a layer hands over a flat list, the optimizer walks it, and neither
 * side knows anything else about the other.
 *
 * ──────────────────────────────────────────────────────────────────────────
 * PITFALL FOR BOTH CLASSES: stay off the graph
 * ──────────────────────────────────────────────────────────────────────────
 * `step()` mutates `.data` using PLAIN TENSOR FUNCTIONS from tensor/ops.ts —
 * sub, add, mulScalar. Never the TensorValue methods. A method call builds a
 * graph node, and a node built inside step() is never freed: the graph would
 * grow by one node per parameter per iteration for the entire run, and the
 * parameters would stop being leaves. Forward passes build the graph.
 * Nothing else does.
 *
 * ──────────────────────────────────────────────────────────────────────────
 * THE RUNNING EXAMPLE — the doc's bowl, checkable by hand
 * ──────────────────────────────────────────────────────────────────────────
 *     L = (θ − 5)²      dL/dθ = 2(θ − 5)      θ₀ = 0      lr = 0.1
 *
 *     step 1:  g = 2(0 − 5)   = −10     θ = 0   − 0.1·(−10)  = 1
 *     step 2:  g = 2(1 − 5)   = −8      θ = 1   − 0.1·(−8)   = 1.8
 *     step 3:  g = 2(1.8 − 5) = −6.4    θ = 1.8 − 0.1·(−6.4) = 2.44
 *     then     2.952 → 3.3616 → 3.68928 → … → 5
 *
 * Each step is 0.8× the previous one, because the slope flattens as θ nears
 * 5. Work the first two by hand before running anything.
 */
import type { Tensor } from "../tensor/index.ts";
import { TensorValue } from "../autograd/grad.ts";
import { sub, mulScalar, add, zeros } from "../tensor/index.ts";

/**
 * Plain stochastic gradient descent:  θ ← θ − lr · g
 *
 * The whole optimizer. It keeps no state beyond its two constructor
 * arguments, which is exactly what makes it the baseline the next two
 * classes are measured against.
 *
 * ── CONSTRUCTOR ───────────────────────────────────────────────────────────
 * Store `params` and `learningRate`. Parameters go HERE, not into step() —
 * the shape you chose in Ch 09, and PyTorch's. An optimizer is bound to one
 * list of parameters once, and from then on step() takes no arguments.
 *
 * ── step() ────────────────────────────────────────────────────────────────
 * For each parameter, skipping any whose `.grad` is still null:
 *
 *     param.data = sub(param.data, mulScalar(param.grad, this.learningRate))
 *
 * One line. No state, no graph.
 *
 * ── zeroGrad() ────────────────────────────────────────────────────────────
 * Set every `.grad` back to `null`. Ch 10's `accumulate()` treats null as
 * "first contribution" and assigns rather than adds, so null is a true reset.
 *
 * ── WHY zeroGrad EXISTS — Ch 08 measured this ─────────────────────────────
 * backward() accumulates with `+=`. Skip the reset and the previous
 * iteration's gradients ride into this one: Ch 08 measured −3, then −9, then
 * −18 on a three-node graph, compounding with depth because interior nodes
 * carry the contamination forward. In a training loop that inflates the
 * effective learning rate every iteration and the loss climbs — it presents
 * exactly like a learning rate set far too high.
 *
 * Call it any time between one step() and the next backward(). Calling it
 * BETWEEN backward() and step() erases the gradients before they are used,
 * and the parameters never move at all — everything runs, the loss just sits
 * there.
 */
export class SGD {
  readonly params: TensorValue[];
  readonly learningRate: number;

  constructor(params: TensorValue[], learningRate: number) {
    this.params = params;
    this.learningRate = learningRate;
  }

  step(): void {
    for (const param of this.params) {
      if (param.grad !== null) {
        param.data = sub(param.data, mulScalar(param.grad, this.learningRate));
      }
    }
  }

  zeroGrad(): void {
    for (const param of this.params) {
      param.grad = null;
    }
  }
}

/**
 * SGD with momentum:  v ← β·v + g,  then  θ ← θ − lr · v
 *
 * The first optimizer in this course that REMEMBERS anything. That single
 * difference is what the rest of the chapter is built on — Adam's first
 * moment is this same running average, and its second moment is the thing
 * this class still cannot do.
 *
 * ── CONSTRUCTOR ───────────────────────────────────────────────────────────
 * Store params, learningRate, and momentum (default 0.9). Then allocate one
 * velocity tensor PER PARAMETER, zeros, matching that parameter's own shape.
 * Parallel arrays are fine: the i-th velocity belongs to the i-th parameter.
 *
 * ── PITFALL: the velocity must persist, and must not be shared ────────────
 * Allocate the velocities in the CONSTRUCTOR. Creating them inside step()
 * means every step starts from v = 0, which collapses the update back to
 * `θ − lr·g` — plain SGD wearing a momentum costume. Nothing errors, nothing
 * looks wrong, and the entire point of the class disappears silently.
 *
 * And one velocity per parameter, never one shared tensor: W and b have
 * different shapes and completely different gradient histories.
 *
 * ── step() ────────────────────────────────────────────────────────────────
 * For each parameter i, on raw tensors:
 *
 *     v[i]       = add(mulScalar(v[i], this.momentum), param.grad)
 *     param.data = sub(param.data, mulScalar(v[i], this.learningRate))
 *
 * Order matters: update the velocity first, then step with the NEW velocity.
 *
 * ── WORKED TRACE — the doc's "watch momentum on a simple sequence" ────────
 * With β = 0.9, v₀ = 0, and a gradient of +1 arriving every step:
 *
 *     v₁ = 0.9·0    + 1 = 1
 *     v₂ = 0.9·1    + 1 = 1.9
 *     v₃ = 0.9·1.9  + 1 = 2.71
 *     v₄ = 0.9·2.71 + 1 = 3.439        … and after 8 steps: 5.695
 *
 * Now the same rule with the gradient ALTERNATING +1, −1, +1, −1 — identical
 * magnitudes, only the signs differ:
 *
 *     1, −0.1, 0.91, −0.181, …         … and after 8 steps: −0.300
 *
 * Agreement accumulates; disagreement cancels. If your consistent run does
 * not reach 5.695, or your alternating one drifts away from zero, the
 * velocity is being reset or reused somewhere it should not be.
 *
 * ── zeroGrad() ────────────────────────────────────────────────────────────
 * Identical to SGD's — and note what it must NOT touch: the velocities.
 * Gradients are per-iteration scratch and get cleared; velocity is the
 * memory that survives across iterations, which is the whole class.
 */
export class SGDMomentum {
  readonly params: TensorValue[];
  readonly learningRate: number;
  readonly momentum: number;

  /** One velocity per parameter, same shape, in the same order as `params`. */
  private readonly velocities: Tensor[];

  constructor(params: TensorValue[], learningRate: number, momentum?: number) {
    this.params = params;
    this.learningRate = learningRate;
    this.momentum = momentum !== undefined ? momentum : 0.9;

    // Allocated ONCE, here — this is the memory that makes it momentum.
    // Building these inside step() would restart from v = 0 every iteration
    // and silently collapse the update back to plain SGD.
    this.velocities = params.map((param) => zeros(param.data.shape));
  }

  step(): void {
    this.params.forEach((param, i) => {
      if (param.grad === null) return;

      // Velocity first, then step with the NEW velocity.
      this.velocities[i] = add(
        mulScalar(this.velocities[i]!, this.momentum),
        param.grad,
      );
      param.data = sub(
        param.data,
        mulScalar(this.velocities[i]!, this.learningRate),
      );
    });
  }

  zeroGrad(): void {
    // Clears gradients only. The velocities are deliberately untouched —
    // they are the history this class exists to keep.
    for (const param of this.params) {
      param.grad = null;
    }
  }
}
