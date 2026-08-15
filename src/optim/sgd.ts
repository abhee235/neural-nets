/**
 * optim/sgd.ts
 * ══════════════════════════════════════════════════════════
 * Stochastic Gradient Descent (plain and with momentum).
 * Equivalent to torch.optim.SGD.
 *
 * Chapters: 09 (gradient descent), 14 (optimizers)
 * Doc:      docs/part-2-autodiff/ch-09-gradient-descent.md
 *
 * ──────────────────────────────────────────────────────────────────────────
 * WHERE THIS SITS IN THE COURSE
 * ──────────────────────────────────────────────────────────────────────────
 * Ch 08 gave you gradients. This chapter is the other half of learning: a
 * gradient tells you which way is uphill, and an optimizer is what actually
 * MOVES the parameter. Nothing here is clever — the whole of vanilla SGD is
 * one subtraction per parameter. What makes the chapter worth doing carefully
 * is that this is the first code that closes the loop, so every bug you have
 * been warned about since Ch 08 (accumulating gradients, stale state, updates
 * leaking into the graph) becomes visible here for the first time.
 *
 * ──────────────────────────────────────────────────────────────────────────
 * THE API: the optimizer OWNS its parameters
 * ──────────────────────────────────────────────────────────────────────────
 * You construct it once with the list of parameters it is responsible for,
 * then call `step()` and `zeroGrad()` with no arguments:
 *
 *     const opt = new SGD([w, b], 0.1);
 *     loss.backward();
 *     opt.step();
 *     opt.zeroGrad();
 *
 * This mirrors PyTorch (`torch.optim.SGD(model.parameters(), lr=0.1)`), which
 * is why the header above can honestly claim equivalence. It also matters for
 * `SGDMomentum` below: momentum keeps one velocity number PER PARAMETER
 * between steps, and state like that needs a stable parameter list to live
 * alongside. An optimizer handed a fresh array on every call has nowhere to
 * keep it.
 *
 * ──────────────────────────────────────────────────────────────────────────
 * THE ONE RULE
 * ──────────────────────────────────────────────────────────────────────────
 *     θ  ←  θ  −  η · ∂L/∂θ
 *     ─      ─     ─    ────
 *     new    old   the  the gradient backward() just computed
 *                  learning rate (step size)
 *
 * The minus sign is the entire idea: the gradient points UPHILL, and we want
 * to go down. Every optimizer in this course — SGD, momentum, Adam (Ch 14) —
 * is a variation on how to choose the size and direction of that step.
 */
import type { Value } from "../autograd/value.ts";

/**
 * Vanilla SGD.
 * Update rule:  param.data -= learningRate * param.grad
 *
 * ── FIELDS TO ASSIGN ──────────────────────────────────────────────────────
 *   params        ← the list handed in; keep the reference, these are the
 *                   very same nodes the loss graph was built from
 *   learningRate  ← η, the step size
 *
 * ── PITFALL: keep the reference, do not copy the numbers ──────────────────
 * `params` must stay an array of the ACTUAL `Value` nodes. If you copy their
 * `.data` into a plain number array, `step()` will faithfully update your
 * copies and the model will never change — a silent no-op that looks exactly
 * like "the learning rate is too small".
 */
export class SGD {
  readonly params: Value[];
  readonly learningRate: number;

  constructor(params: Value[], learningRate: number) {
    throw new Error("SGD constructor not implemented");
  }

  /**
   * Apply one gradient step to every parameter.
   *
   * ── WHAT TO DO ────────────────────────────────────────────────────────────
   * Walk `this.params` and move each one against its gradient:
   *
   *     θ.data  ←  θ.data  −  η · θ.grad
   *
   * That is the whole method — one loop, one subtraction. No graph, no nodes.
   *
   * ── PITFALL: update `.data` DIRECTLY, never through Value ops ─────────────
   * It is tempting to write the update using the operations you just built —
   * something that produces a new `Value`. Do not. Two things break:
   *   1. The parameter would become a NEW node, so the next forward pass would
   *      still reference the old one and nothing would ever train.
   *   2. Every step would append to the computation graph, which would grow
   *      without bound across training and eventually exhaust memory.
   * The update is bookkeeping ON a parameter, not a computation WITHIN the
   * model. This is what PyTorch's `with torch.no_grad():` is for.
   *
   * ── PITFALL: do not zero gradients in here ────────────────────────────────
   * Keep the two responsibilities separate. `step()` reads gradients;
   * `zeroGrad()` clears them. Fusing them feels tidy but makes it impossible
   * to inspect gradients after a step, which is the first thing you will want
   * to do when training misbehaves.
   *
   * ── WORKED EXAMPLE — exercise E1: minimise f(w) = (w − 5)² ────────────────
   * Start w = 0, η = 0.1. The gradient of (w−5)² is 2(w−5):
   *
   *     step 1:  w.grad = 2(0 − 5) = -10   →  w ← 0    − 0.1·(-10) = 1.0
   *     step 2:  w.grad = 2(1 − 5) =  -8   →  w ← 1.0  − 0.1·(-8)  = 1.8
   *     step 3:  w.grad = 2(1.8 − 5) = -6.4 →  w ← 1.8 − 0.1·(-6.4) = 2.44
   *
   * The steps shrink as the slope flattens — 1.0, 0.8, 0.64 — each exactly 0.8×
   * the last. After 100 steps w is 5 to ten decimal places. Work the first two
   * by hand before you run it; if your code disagrees, you know which of the
   * three numbers to print.
   */
  step(): void {
    throw new Error("SGD.step not implemented");
  }

  /**
   * Zero all parameter gradients before the next backward pass.
   *
   * ── WHAT TO DO ────────────────────────────────────────────────────────────
   * Walk `this.params` and reset each `.grad` to 0. `Value.zeroGrad()` from
   * Ch 08 already does one node — this is that, across the list.
   *
   * ── WHY THIS METHOD EXISTS ────────────────────────────────────────────────
   * Because `backward()` accumulates with `+=`. Ch 08 showed what skipping it
   * costs on a three-node graph: gradients went -3, then -9, then -18 — not
   * merely doubling, because interior nodes compound the contamination with
   * depth. In a training loop that error rides straight into `step()`, so the
   * effective learning rate grows every iteration and the loss climbs. It
   * presents exactly like a learning rate set too high.
   *
   * ── WHERE IT GOES IN THE LOOP ─────────────────────────────────────────────
   * Any time between one `step()` and the next `backward()`. Both of these are
   * correct and you will see both in the wild:
   *
   *     backward → step → zeroGrad → (next iteration)
   *     zeroGrad → backward → step → (next iteration)
   *
   * What is NOT correct is calling it between `backward()` and `step()` — that
   * erases the gradients before they have been used, and the parameters never
   * move at all. Everything runs, the loss just sits there.
   */
  zeroGrad(): void {
    throw new Error("SGD.zeroGrad not implemented");
  }
}

/**
 * SGD with momentum.
 * Update rule:  v = momentum*v - lr*grad
 *               param.data += v
 *
 * Momentum dampens oscillations and accelerates convergence.
 *
 * ── THE INTUITION ─────────────────────────────────────────────────────────
 * Vanilla SGD is a ball with no mass: it goes wherever the current slope
 * points and forgets everything. Momentum gives it inertia. `v` is a running
 * velocity that keeps a fraction (`momentum`, usually 0.9) of the previous
 * step and adds the new gradient contribution to it.
 *
 * Two consequences, and they are why every real optimizer has something like
 * this:
 *   - Along a consistent downhill direction, contributions ACCUMULATE and the
 *     effective step grows — up to about 1/(1−momentum) = 10× at 0.9.
 *   - Across a narrow valley where the gradient flips sign each step, the
 *     contributions CANCEL, damping the oscillation instead of amplifying it.
 *
 * ── FIELDS TO ASSIGN ──────────────────────────────────────────────────────
 *   params, learningRate  ← as above
 *   momentum              ← default 0.9 when the caller omits it
 *   velocities            ← one number PER PARAMETER, all starting at 0,
 *                           and the array must be the same length as params
 *
 * ── PITFALL: velocity is state that must SURVIVE between steps ────────────
 * This is the whole difficulty of the class, and the reason the optimizer owns
 * its parameter list. Allocate `velocities` ONCE in the constructor. If you
 * create it inside `step()`, it resets to zero every call, the momentum term
 * is always 0, and you have silently re-implemented vanilla SGD — which still
 * converges, so the tests you would think to write still pass.
 *
 * The test that catches it is in sgd.test.ts: "velocity accumulates across
 * steps". Also note "momentum=0 is equivalent to vanilla SGD" — with a zero
 * coefficient the previous velocity is discarded entirely and the rule
 * collapses back to θ -= lr·grad, which is a good algebraic check on your
 * implementation before you trust the 0.9 case.
 *
 * ── WORKED EXAMPLE — same bowl as E1, f(w) = (w−5)², η = 0.1, β = 0.9 ─────
 *     start: w = 0, v = 0
 *     step 1:  grad = -10  →  v = 0.9·0    − 0.1·(-10) = 1.0   →  w = 1.0
 *     step 2:  grad = -8   →  v = 0.9·1.0  − 0.1·(-8)  = 1.7   →  w = 2.7
 *     step 3:  grad = -4.6 →  v = 0.9·1.7  − 0.1·(-4.6) = 1.99 →  w = 4.69
 *
 * Compare against vanilla's 1.0, 1.8, 2.44 at the same learning rate. By step
 * three momentum has nearly arrived while plain SGD is not yet halfway. It
 * will also overshoot past 5 and come back — that overshoot is the inertia
 * doing its job, not a bug.
 */
export class SGDMomentum {
  readonly params: Value[];
  readonly learningRate: number;
  readonly momentum: number;
  /** Running velocity, one entry per parameter. Allocated once, in the constructor. */
  readonly velocities: number[];

  constructor(params: Value[], learningRate: number, momentum?: number) {
    throw new Error("SGDMomentum constructor not implemented — default momentum=0.9");
  }

  /**
   * One momentum step across every parameter.
   *
   * For each parameter i, in this order:
   *   1. update the stored velocity from the OLD velocity and the current grad
   *   2. move the parameter by the NEW velocity
   *
   * Getting those two backwards — moving by the old velocity, then updating —
   * lags the optimizer one step behind and is nearly invisible on a smooth
   * bowl. Check against the three hand-computed steps above.
   */
  step(): void {
    throw new Error("SGDMomentum.step not implemented");
  }

  /** Zero gradients on all parameters. Identical in spirit to SGD.zeroGrad. */
  zeroGrad(): void {
    throw new Error("SGDMomentum.zeroGrad not implemented");
  }
}
