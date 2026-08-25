/**
 * CHAPTER 12: Loss Functions
 * ════════════════════════════════════════
 * Part 3 of 6: Neural Net Primitives
 *
 * WHAT WE'RE BUILDING:  mseLoss, logSumExp, crossEntropyFromLogits — the
 *                       functions that turn "the model said X, the truth was
 *                       Y" into ONE number with useful gradients.
 * WHY IT MATTERS:       Ch 15's training loop minimises crossEntropyFromLogits;
 *                       it is the training objective of every language model
 *                       in this course, including Ch 30's GPT.
 * WHAT THIS UNLOCKS:    → Ch 13 (Linear Layer) — something to optimise against.
 *
 * REFERENCE: docs/part-3-neural-net-primitives/ch-12-loss-functions.md
 *
 * ──────────────────────────────────────────────────────────────────────────
 * WHY THIS FILE EXISTS
 * ──────────────────────────────────────────────────────────────────────────
 * Nothing in Ch 01–11 ever compares an output against an answer. The engine
 * computes, differentiates, and transforms — but it does not know what
 * "wrong" means. These three functions add that, and they are the top of
 * every training graph: the single number backward() starts from.
 *
 * A loss has two jobs (doc, opening section):
 *   Job 1 — measure:  say how bad the prediction is.
 *   Job 2 — teach:    produce gradients that say how to improve.
 * Every design choice below is in service of job 2. A loss is judged by its
 * gradients, not by whether its value looks sensible.
 *
 * ──────────────────────────────────────────────────────────────────────────
 * HOW THIS DIFFERS FROM CH 11
 * ──────────────────────────────────────────────────────────────────────────
 * Ch 11 added PRIMITIVES: new nodes with hand-written _backward closures.
 * This chapter mostly COMPOSES: mseLoss is add + mul + mean, all of which
 * already know their own backward, so autograd derives the gradient for you.
 * You write forward code and get backward for free — the payoff for Ch 10.
 *
 * The one exception is logSumExp — see the decision below.
 *
 * ──────────────────────────────────────────────────────────────────────────
 * THE RUNNING EXAMPLES — the doc's numbers, used in every trace below
 * ──────────────────────────────────────────────────────────────────────────
 * Regression (doc section 1) — three days of temperatures:
 *
 *     predicted   [ 32   28   31 ]     shape [3]
 *     actual      [ 35   28   30 ]     shape [3]
 *     MSE = 3.333333    gradient = [ -2   0   0.666667 ]
 *
 * Classification (doc sections 3–13) — "the cat ___", truth = "sat":
 *
 *     logits      [ 1          2          3        ]     shape [3]
 *                   sat        ran        flew
 *     softmax     [ 0.090031   0.244728   0.665241 ]
 *     loss (truth = sat)  = 2.407606
 *     loss (truth = flew) = 0.407606
 *
 * ──────────────────────────────────────────────────────────────────────────
 * ONE DECISION BEFORE YOU START — TensorValue has no exp, log, or max
 * ──────────────────────────────────────────────────────────────────────────
 * Check grad.ts: the seven ops are add, mul, matMul, sum, mean, reshape,
 * transpose. The doc's recipe for logSumExp says "exp, then sum, then log" —
 * but exp and log DO NOT EXIST as differentiable ops yet. Two honest routes:
 *
 *   (a) Add exp and log as primitives first, with Ch 11's five-step recipe.
 *       Both are elementwise one-liners, and Ch 08 already taught their
 *       backwards:  exp'(x) = exp(x) = out.data (reuse the output, like
 *       sigmoid);  log'(x) = 1/x. Then compose logSumExp from them.
 *
 *   (b) Make logSumExp itself ONE new primitive: forward with the plain
 *       tensor functions from tensor/math.ts and tensor/reduce.ts, and a
 *       hand-written _backward. Its derivative is a fact the doc proves in
 *       section 13:  ∂logSumExp/∂z_i = softmax(z)_i. One line of backward.
 *
 *   Route (a) is more reusable (exp/log return in later chapters); route (b)
 *   is less code today. Either passes the tests. Decide, and write which one
 *   you chose in logSumExp's JSDoc.
 *
 * The max inside logSumExp is a CONSTANT either way — read it from .data
 * with tensor/reduce.ts's max(). It cancels exactly in the mathematics
 * (doc section 9), so it carries no gradient.
 */
import type { Tensor } from "../tensor/index.ts";
import { TensorValue } from "../autograd/grad.ts";

/**
 * Mean Squared Error: mean((predictions − targets)²).
 *
 * The regression loss — Job 1: average squared distance. Job 2: a gradient
 * (2/n)(p − y) that points every prediction at its target.
 *
 * ── THE ML IDEA vs OUR ENGINE (doc section 2) ──────────────────────────────
 * The idea is just: subtract, square, average. The asymmetric signature is
 * an ENGINE detail: `predictions` is a TensorValue so the graph records the
 * ops; `targets` is a plain Tensor because the truth is a constant — nothing
 * produced it, nothing in it gets updated. Do NOT wrap targets in a
 * TensorValue: backward() would compute a meaningless gradient for the labels.
 *
 * ── COMPOSE IT (Ch 10's ops — no new primitive needed) ─────────────────────
 * TensorValue has no sub and no pow. But:
 *     subtract  =  add the negation      (mulScalar(targets, -1), then wrap)
 *     square    =  mul a thing by itself (mul records BOTH parents — that is
 *                                         why d(x²) = 2x comes out for free)
 *     collapse  =  .mean()               (no axis → one number, the scalar
 *                                         root backward() demands)
 *
 * ── WORKED TRACE — the temperatures (doc section 1) ────────────────────────
 *
 *     predicted     [ 32   28   31 ]     shape [3]
 *     actual        [ 35   28   30 ]     shape [3]
 *                   ────────────────  −
 *     difference    [ -3    0    1 ]     shape [3]
 *     squared       [  9    0    1 ]     shape [3]
 *     mean          → (9 + 0 + 1) / 3 = 3.333333        ← ONE number
 *
 *     backward, gradient (2/3)·difference:
 *     grad          [ -2    0    0.666667 ]     shape [3] ✓ matches predictions
 *
 * Signs: Monday was 3° too LOW → negative gradient → step() pushes it UP.
 * Wednesday too high → pushed down. Tuesday exact → untouched.
 *
 * ── PITFALL: mean, not sum ─────────────────────────────────────────────────
 * Sum and the loss grows with batch size, so the learning rate tuned on 32
 * examples is wrong for 64. The tests check the 1/n is there via the
 * gradient: with sum it would be [-6, 0, 2], not [-2, 0, 0.666667].
 */
export function mseLoss(predictions: TensorValue, targets: Tensor): TensorValue {
  throw new Error("mseLoss not implemented");
}

/**
 * Numerically stable log-sum-exp along an axis (default: the last).
 *
 *     logSumExp(z) = max(z) + log( Σ exp(z_i − max(z)) )
 *
 * The helper that makes cross-entropy computable. Exact — not an
 * approximation: e^z = e^m · e^(z−m), factor e^m out of the sum, and
 * log(ab) = log a + log b puts m back outside (doc section 9).
 *
 * ── WHY THE SUBTRACTION CLOSES BOTH FAILURES ───────────────────────────────
 * Every z_i − max is ≤ 0, so:
 *     exp never sees anything bigger than 0  →  e⁰ = 1, nothing overflows
 *     the sum includes the max's own term, exactly 1  →  total ≥ 1,
 *                                               log(total) ≥ 0, log(0) impossible
 *
 * ── BUILD IT (route (a) or (b) — see the file-header decision) ─────────────
 *     1. max along the axis, keepDims: true   ← a CONSTANT, from .data
 *     2. subtract it (everything now ≤ 0)
 *     3. exp, sum along the axis, log
 *     4. add the max back on
 * Default the axis like Ch 11's softmax:  axis ?? x.data.ndim - 1.
 *
 * ── WORKED TRACE — the numbers that broke, and the check (doc section 9) ───
 *
 *     the naive route on [1000, 1001, 1002]:
 *     exp        [ Infinity   Infinity   Infinity ]
 *     probs      [ NaN        NaN        NaN      ]      ← destroyed
 *
 *     this function on the same row:
 *     z          [ 1000   1001   1002 ]     shape [3]
 *     max        1002
 *     z − max    [ -2     -1      0   ]     all ≤ 0 ✓
 *     exp        [ 0.135335   0.367879   1.000000 ]      biggest is e⁰ = 1 ✓
 *     sum        1.503215
 *     log        0.407606
 *     + max      1002.407606                              ← no NaN anywhere
 *
 *     logSumExp([1, 2, 3]) = 3.407606 — same fractional digits .407606,
 *     because shifting every logit by 999 shifts the answer by exactly 999.
 *     That is softmax's shift invariance (Ch 05 deep dive) and your best
 *     self-check.
 *
 * ── PITFALL: keepDims on the max ───────────────────────────────────────────
 * Without keepDims: true the max drops the axis and the subtraction cannot
 * broadcast back against the full tensor. Ch 10 section 4 is the machinery.
 *
 * ── GRADIENT CHECK (route (b) especially) ──────────────────────────────────
 * ∂logSumExp/∂z_i = softmax(z)_i — proven in doc section 13. On [1,2,3] the
 * numerical gradient must come out [0.090031, 0.244728, 0.665241]. If you
 * see all-ones or a uniform 1/3, the max leaked into the graph or the axis
 * sum is wrong.
 */
export function logSumExp(x: TensorValue, axis?: number): TensorValue {
  throw new Error("logSumExp not implemented");
}

/**
 * Cross-entropy directly from raw logits — no softmax is ever built:
 *
 *     L = logSumExp(logits) − logit[true class]        (per row, then mean)
 *
 * The training objective of every language model in this course. The name
 * says FromLogits and it means it: hand this the output of softmax and it
 * returns a confidently wrong answer with no error. Raw scores only.
 *
 * ── WHERE THE FORMULA COMES FROM (doc section 11) ──────────────────────────
 * −log(softmax(z)_y) with the log pushed through the division:
 *     log(e^{z_y} / Σe^{z_j}) = z_y − logSumExp(z)
 * The exp and the log cancel — no division, no exp of a raw logit, no log of
 * a tiny probability. That cancellation is the whole reason this function
 * exists instead of "softmax, then log".
 *
 * ── DECIDE: what is `targets`? ─────────────────────────────────────────────
 * Class indices ([0] for "sat") or one-hot rows ([1,0,0])? Both work:
 *     indices → gather one logit per row (plain code — it picks, nothing
 *               differentiable happens in the picking)
 *     one-hot → mul by the mask and sum along the axis; every other term is
 *               ×0 (the doc's section-6 picture). The mask is a CONSTANT.
 * Pick one, write it in this JSDoc, stay consistent — mixing the two
 * conventions is the most common bug in this file.
 *
 * ── WORKED TRACE — "the cat ___", truth = "sat" (doc sections 3–6) ─────────
 *
 *     logits       [ 1          2          3        ]     shape [3]
 *                    sat        ran        flew
 *     logSumExp    3.407606
 *     z_sat        1
 *     loss         3.407606 − 1 = 2.407606
 *
 *     check against the long way (doc section 6):
 *     softmax      [ 0.090031   0.244728   0.665241 ]
 *     −log(0.090031) = 2.407606                          ✓ identical
 *
 *     and with truth = "flew":  3.407606 − 3 = 0.407606
 *     — the losses for truths sat/ran/flew are 2.407606 / 1.407606 / 0.407606,
 *     differing by exactly the logit gaps, because logSumExp does not care
 *     which class is true.
 *
 * ── THE GRADIENT YOU GET FOR FREE (doc section 13) ─────────────────────────
 *
 *     softmax      [  0.090031   0.244728   0.665241 ]     p
 *     one-hot      [  1          0          0        ]     y
 *                  ───────────────────────────────────  −
 *     gradient     [ -0.909969   0.244728   0.665241 ]     p − y
 *
 * "What you predicted, minus what was true." sat is under-predicted →
 * negative → its logit rises; flew took the most it should not have →
 * pushed down hardest. The row sums to 0: this loss can only REDISTRIBUTE
 * confidence, never move all logits together. If your numerical gradient
 * check does not sum to ~0, the mean over the batch or the gather is wrong.
 *
 * ── PITFALL: batch mean ────────────────────────────────────────────────────
 * One loss per row, then average — same reason as mseLoss. Sum instead and
 * the learning rate silently depends on batch size.
 */
export function crossEntropyFromLogits(
  logits: TensorValue,
  targets: Tensor
): TensorValue {
  throw new Error("crossEntropyFromLogits not implemented");
}
