/**
 * autograd/value.ts
 * ══════════════════════════════════════════════════════════
 * Scalar Value node for automatic differentiation.
 *
 * During the forward pass each operation creates a new Value that records
 * its _inputs and a _backward closure.  Calling .backward() on the output
 * propagates gradients all the way back to the leaf parameters.
 *
 * Chapters: 08a (forward), 08b (backward)
 * Doc:      docs/part-2-autodiff/ch-08a-autograd-forward.md
 *           docs/part-2-autodiff/ch-08b-autograd-backward.md
 *
 * ──────────────────────────────────────────────────────────────────────────
 * HOW TO USE THIS FILE
 * ──────────────────────────────────────────────────────────────────────────
 * Every method below carries three comment blocks:
 *
 *   WHAT IT IS      — the math, and where it shows up later in the transformer
 *   FORWARD (08a)   — the recipe for the recording half: compute + build a node
 *   BACKWARD (08b)  — the local derivative, i.e. this op's row of the table
 *
 * Work through 08a for EVERY method first (forward only, `_backward` left as
 * the no-op), then come back and do a second pass for 08b. Do not try to write
 * both halves of one method at a time — you will conflate "what happened" with
 * "what its derivative is", which is exactly the confusion the two-chapter
 * split exists to prevent.
 *
 * ──────────────────────────────────────────────────────────────────────────
 * THE ONE PATTERN — every operation does exactly two jobs
 * ──────────────────────────────────────────────────────────────────────────
 * Job 1 (08a): compute the forward number.
 * Job 2 (08a): return a NEW Value that remembers which nodes produced it
 *              and which operation was used.
 * Job 3 (08b): before returning, attach an `out._backward` closure that pushes
 *              gradient from `out` back into the parent(s).
 *
 * ch-08a-autograd-forward.md writes this pattern out in full for `mul` — that
 * one worked example is the only implementation you have been given. Every
 * other method on this class is the same three jobs with a different formula.
 *
 * ──────────────────────────────────────────────────────────────────────────
 * THE ONE RULE — all of backpropagation, in a sentence
 * ──────────────────────────────────────────────────────────────────────────
 *   parent.grad  +=  out.grad  ×  (∂out/∂parent)
 *                    ─────────    ──────────────
 *                    incoming     local derivative — the ONLY thing that
 *                    (already     changes between methods; everything else
 *                    computed)    on this page is identical boilerplate.
 *
 * Two things about that line are easy to get wrong and cost hours:
 *
 *   `+=` NOT `=`  — a node feeding two children (e.g. `x.mul(x)`) receives one
 *                   contribution per child, and the true derivative is their
 *                   SUM. `=` silently keeps only the last one. This is the #1
 *                   backprop bug in existence.
 *
 *   out.grad READ AT CALL TIME — the closure must read `out.grad` when it RUNS
 *                   (during backward), not capture a snapshot of it when it was
 *                   created (during forward), because at creation time it is
 *                   still 0. Reading the property inside the closure body gives
 *                   you this for free; just don't destructure it out early.
 *
 * ──────────────────────────────────────────────────────────────────────────
 * A TYPESCRIPT TRAP WAITING FOR YOU
 * ──────────────────────────────────────────────────────────────────────────
 * The five fields below are declared with no initializer. That type-checks
 * TODAY only because the constructor's body is a single `throw` — the end of
 * the constructor is unreachable, so TS's definite-assignment analysis passes
 * vacuously. The moment you replace that throw with real code, strict mode will
 * demand that EVERY field is assigned on EVERY path through the constructor.
 * That is not a bug; it is the type system telling you a node with no `_op` or
 * no `_backward` must never exist.
 */

import { topoSort } from "./engine";

/**
 * A single node in the scalar computation graph.
 *
 * Every arithmetic op creates a new Value whose _backward function knows
 * how to push gradients back to its _inputs.
 *
 * MENTAL MODEL (08a): a Value is a number carrying a sticky note that reads
 * "I came from these parents, via this operation." A plain `number` throws that
 * note away; keeping it is the entire difference between a calculator and an
 * autodiff engine.
 *
 * WHY NOT COMPUTE THE DERIVATIVE NOW? Because that is forward mode — the `Dual`
 * warm-up in 08a — and it costs one full pass PER INPUT. A network has millions
 * of inputs (the weights) and one output (the loss), so we record now and
 * differentiate later, answering for all weights in a single backward sweep.
 * Deferring the derivative work IS the design; it is not laziness.
 */
export class Value {
  /** The scalar number this node holds. */
  data: number;
  /** Accumulated gradient ∂L/∂this. Starts at 0; the root is set to 1. */
  grad: number;
  /** Input nodes that produced this value. */
  _inputs: Value[];
  /** Operation label for debugging, e.g. "+", "*", "tanh". */
  _op: string;
  /** Runs the local backward step and accumulates into _inputs' grads. */
  _backward: () => void;

  /**
   * Build a node.
   *
   * Leaves (weights, inputs — anything you type by hand) are created with no
   * parents: `new Value(2)`. Operations create interior nodes by passing the
   * parents they consumed and a label: `new Value(sum, [this, other], "+")`.
   *
   * ── FORWARD (08a) — what to assign ────────────────────────────────────────
   *   data       ← the number passed in
   *   grad       ← 0        (no gradient has reached this node yet; 08b fills it)
   *   _inputs    ← the parents, defaulting to an EMPTY ARRAY for a leaf
   *   _op        ← the label, defaulting to an EMPTY STRING for a leaf
   *   _backward  ← a NO-OP closure, `() => {}`
   *
   * ── WHY _backward STARTS AS A NO-OP ───────────────────────────────────────
   * `backward()` will call `_backward()` on every node in the graph, leaves
   * included. A leaf has no parents to push gradient into, so its correct
   * behaviour is "do nothing" — and a no-op default means `backward()` needs no
   * special case for leaves. Leaving this field undefined instead would crash
   * the sweep the first time it reaches a weight.
   *
   * ── PITFALL ───────────────────────────────────────────────────────────────
   * Use an ARRAY for `_inputs`, never a Set. The backward pass depends on
   * stable parent ordering, and `pow`/`div` care which operand is which.
   *
   * ── WORKED EXAMPLE — the three leaves of Figure 3 (08a) ───────────────────
   * `new Value(2)`, `new Value(-3)`, `new Value(10)` — the a, b, d that the
   * running example is built from. Each must come out as:
   *
   *     data      =  2         (or -3, or 10)
   *     grad      =  0         ← no gradient has reached it yet
   *     _inputs   =  []        ← a leaf: nothing produced it
   *     _op       =  ""        ← a leaf: no operation produced it
   *     _backward =  () => {}  ← nothing to push to; the no-op IS the point
   *
   * Contrast with the INTERIOR node `mul` will build two methods down:
   *
   *     data      =  -6
   *     grad      =  0
   *     _inputs   =  [a, b]    ← non-empty is what makes it interior
   *     _op       =  "*"
   *     _backward =  the closure that knows how to reach a and b
   *
   * Same constructor, same five fields, both times. "Leaf" is not a flag or a
   * subclass — it is just a node whose `_inputs` happens to be empty. That is
   * exactly what the two defaults are for: they are what make `new Value(2)`
   * come out as a valid leaf without the caller saying anything.
   *
   * ✅ CHECKPOINT (doc milestone 1):
   *      new Value(2).data === 2
   *      new Value(2).grad === 0
   *      new Value(2)._inputs  →  []
   *
   * ⚠️ NOTE — exercises/ch-08-autograd.ts calls `new Value(2, "a")`, passing a
   * LABEL as the second argument. That contradicts this signature (and the doc,
   * which specifies `(data, inputs = [], op = "")`). One of the two has to
   * change; see the note at the bottom of this file before you start.
   */
  constructor(data: number, _inputs?: Value[], _op?: string) {
    this.data = data;
    this.grad = 0;
    this._inputs = _inputs || [];
    this._op = _op || "";
    this._backward = () => {};
  }

  /**
   * Forward: this + other
   *
   * ── WHERE THIS SHOWS UP ───────────────────────────────────────────────────
   * Every bias add, and every residual connection in the transformer
   * (`x + attention(x)`, Ch 26/27) is this operation.
   *
   * ── FORWARD (08a) ─────────────────────────────────────────────────────────
   * 1. Compute the sum of the two `.data` numbers.
   * 2. Return a NEW Value built from that sum, recording BOTH parents in order
   *    and the label "+".
   * 3. Return the node — never a bare `number`, or the graph dead-ends here and
   *    gradients stop flowing at this point with no error message.
   *
   * ── BACKWARD (08b) — "+ is a ROUTER" ──────────────────────────────────────
   *   ∂out/∂this  = 1
   *   ∂out/∂other = 1
   * Both local derivatives are 1, so addition COPIES its incoming gradient to
   * both parents unchanged — it routes, it never scales. Apply the one rule
   * with each of those, remembering `+=`.
   *
   * ── WORKED EXAMPLE — Figure 3, second step: L = c.add(d) ──────────────────
   * With c.data = -6 (built by `mul` below) and d.data = 10, the node you
   * return must be:
   *
   *     data      =  -6 + 10  =  4
   *     grad      =  0
   *     _inputs   =  [c, d]    ← the receiver first, then the argument
   *     _op       =  "+"
   *
   * Then in 08b, once the sweep has seeded L.grad = 1:
   *
   *     c.grad  +=  L.grad × 1  =  1 × 1  =  1
   *     d.grad  +=  L.grad × 1  =  1 × 1  =  1
   *
   * Both parents end up with exactly what arrived — unchanged. That is the
   * router, and it matches Figure 2's ∂L/∂c = ∂L/∂d = 1.
   *
   * ── SELF-CHECK ────────────────────────────────────────────────────────────
   * Why does `x.add(x)` give `x.grad === 2` and not `1`? If your answer isn't
   * immediately obvious, you have just found the reason `+=` exists.
   */
  add(other: Value): Value {
    const out = new Value(this.data + other.data, [this, other], "+");
    out._backward = () => {
      this.grad += out.grad;
      other.grad += out.grad;
    };
    return out;
  }

  /**
   * Forward: this * other
   *
   * ── WHERE THIS SHOWS UP ───────────────────────────────────────────────────
   * Every weight-times-activation product. A matMul (Ch 04) is nothing but a
   * large pile of these plus adds — which is why getting this one node right
   * is what eventually makes attention differentiable.
   *
   * ── FORWARD (08a) ─────────────────────────────────────────────────────────
   * Written out in full in ch-08a-autograd-forward.md, under "Each operation
   * does two jobs". Read it there, then reproduce it here from memory rather
   * than copying — the pattern has to be in your fingers, because the next six
   * methods are the same shape.
   *
   * ── BACKWARD (08b) — "× is a SWITCH" ──────────────────────────────────────
   *   ∂out/∂this  = other.data      ← scaled by the OTHER operand
   *   ∂out/∂other = this.data
   * Each parent's gradient is scaled by its sibling's forward value. This is
   * why the forward `.data` must be STORED on the node: the backward pass is a
   * separate sweep that happens later and has no other way to recover it.
   *
   * ── PITFALL ───────────────────────────────────────────────────────────────
   * Read `.data` off the parent nodes, not off local variables you computed
   * before building `out` — same number today, but the habit breaks in Ch 10
   * where tensors are involved.
   *
   * ── WORKED EXAMPLE — Figure 3, first step: c = a.mul(b) ───────────────────
   * With a.data = 2 and b.data = -3, the node you return must be:
   *
   *     data      =  2 × (-3)  =  -6
   *     grad      =  0
   *     _inputs   =  [a, b]
   *     _op       =  "*"
   *
   * Then in 08b, once `add` has handed this node c.grad = 1:
   *
   *     a.grad  +=  c.grad × b.data  =  1 × (-3)  =  -3
   *     b.grad  +=  c.grad × a.data  =  1 ×   2   =   2
   *
   * Read those two lines slowly: `a` receives `b`'s value and `b` receives
   * `a`'s. Crossed over — that is the switch. And -3 and 2 are exactly what
   * calculus says ∂L/∂a and ∂L/∂b are for L = a·b + d, which is Figure 2's
   * answer. Your autograd just did Ch 07's chain rule on its own.
   *
   * ── WORKED EXAMPLE 2 — why `+=` is not optional: z = a.mul(a), a = 3 ──────
   * This is exercise E4. The SAME node arrives as both operands, so:
   *
   *     _inputs  =  [a, a]     ← one node, listed twice
   *
   * The closure still runs exactly once, but both of its accumulations land on
   * that one node:
   *
   *     a.grad  +=  1 × 3   →  a.grad = 3
   *     a.grad  +=  1 × 3   →  a.grad = 6      ← correct: d(a²)/da = 2a = 6
   *
   * Write `=` instead of `+=` and the second assignment overwrites the first:
   * you get 3, exactly half the right answer, and nothing anywhere errors.
   */
  mul(other: Value): Value {
    const out = new Value(this.data * other.data, [this, other], "*");
    out._backward = () => {
      this.grad += other.data * out.grad;
      other.grad += this.data * out.grad;
    };
    return out;
  }

  /**
   * Forward: this ^ exponent
   *
   * Note the asymmetry: `exponent` is a plain `number`, NOT a Value. It is a
   * constant of the graph, so it has no gradient and never becomes a parent —
   * `_inputs` holds one node here, not two.
   *
   * ── WHERE THIS SHOWS UP ───────────────────────────────────────────────────
   * Squaring in MSE loss (Ch 12), and `pow(-1)` is how `div` gets built without
   * needing its own derivative rule.
   *
   * ── FORWARD (08a) ─────────────────────────────────────────────────────────
   * Same three jobs; the label conventionally records the exponent, e.g.
   * `"**2"`, so `printGraph` output stays readable.
   *
   * ── BACKWARD (08b) ────────────────────────────────────────────────────────
   *   ∂out/∂this = n · this.data^(n−1)        (the power rule from Ch 07)
   * Only ONE parent receives gradient. The exponent is a constant — if you
   * catch yourself wanting to give it a `.grad`, re-read why it isn't a Value.
   *
   * ── PITFALL ───────────────────────────────────────────────────────────────
   * The exponent here must be `n − 1`, not `n`. Off-by-one in the exponent is
   * the classic slip, and it stays invisible at `this.data === 1` — where n and
   * n−1 give the same answer — so test at some other point.
   *
   * ── WORKED EXAMPLE — z = a.pow(2), a = 3 ──────────────────────────────────
   *     data      =  3²  =  9
   *     grad      =  0
   *     _inputs   =  [a]       ← ONE parent. The 2 is not a node.
   *     _op       =  "**2"
   *
   * Backward, with z.grad = 1:
   *
   *     a.grad  +=  z.grad × (2 × 3¹)  =  1 × 6  =  6
   *
   * Compare with `mul`'s second worked example: `a.mul(a)` at a = 3 also came
   * out as 6, by a completely different route — two accumulations of 3, versus
   * one of 6. Same value, same gradient, different mechanism. When two of your
   * rules agree like that, both are probably right.
   */
  pow(exponent: number): Value {
    const out = new Value(this.data ** exponent, [this], `**${exponent}`);
    out._backward = () => {
      this.grad += exponent * this.data ** (exponent - 1) * out.grad;
    };
    return out;
  }

  /**
   * Forward: e^this
   *
   * ── WHERE THIS SHOWS UP ───────────────────────────────────────────────────
   * softmax (Ch 05/11) and therefore the last step of every attention head.
   *
   * ── FORWARD (08a) ─────────────────────────────────────────────────────────
   * One parent, label "exp".
   *
   * ── BACKWARD (08b) ────────────────────────────────────────────────────────
   *   ∂out/∂this = e^(this.data) = out.data        ← reuse the OUTPUT
   * `exp` is its own derivative, which means the number you need has already
   * been computed and is sitting in `out.data`. Use it. Calling `Math.exp`
   * a second time costs a transcendental call per node per backward pass and
   * can drift by an ulp from the value the forward pass actually used.
   *
   * ── WORKED EXAMPLE — p = x.exp(), x = 1 ───────────────────────────────────
   *     data      =  e¹  =  2.7182818
   *     grad      =  0
   *     _inputs   =  [x]
   *     _op       =  "exp"
   *
   * Backward, with p.grad = 1:
   *
   *     x.grad  +=  p.grad × out.data  =  1 × 2.7182818  =  2.7182818
   *
   * The gradient is equal to the forward output. Nothing else on this page has
   * that property, and it is why this closure needs no stored input at all.
   *
   * ✅ CHECKPOINT: `new Value(0).exp().data === 1` — a forward test already
   *    waiting for you in value.test.ts.
   *
   * The doc's TypeScript Hints section shows this method wired end-to-end
   * (forward + backward) as the worked example for 08b.
   */
  exp(): Value {
    const out = new Value(Math.exp(this.data), [this], "exp");
    out._backward = () => {
      this.grad += out.data * out.grad;
    };
    return out;
  }

  /**
   * Forward: ln(this)
   *
   * ── WHERE THIS SHOWS UP ───────────────────────────────────────────────────
   * Cross-entropy loss (Ch 12) — `−log(p_correct)`, the objective every
   * language model in this course is trained against.
   *
   * ── FORWARD (08a) ─────────────────────────────────────────────────────────
   * One parent, label "log".
   *
   * ── BACKWARD (08b) ────────────────────────────────────────────────────────
   *   ∂out/∂this = 1 / this.data                   ← the INPUT, not the output
   * Unlike exp/tanh, log's derivative needs the input value.
   *
   * ── PITFALL ───────────────────────────────────────────────────────────────
   * `log(0)` is −Infinity and its derivative is 1/0 = Infinity, which turns
   * into NaN one multiply later and then silently poisons every gradient
   * upstream of it. When you meet this again in Ch 12, the fix is the
   * log-sum-exp trick, not an epsilon patch. For now just know that a training
   * run full of NaNs usually starts at a log.
   *
   * ── WORKED EXAMPLE — q = x.log(), x = 4 ───────────────────────────────────
   *     data      =  ln 4  =  1.3862944
   *     grad      =  0
   *     _inputs   =  [x]
   *     _op       =  "log"
   *
   * Backward, with q.grad = 1:
   *
   *     x.grad  +=  q.grad × (1 / x.data)  =  1 × (1/4)  =  0.25
   *
   * Notice which number the rule needed: 4, the INPUT — not 1.3862944, the
   * output. exp and tanh reach for `out.data`; log reaches back for
   * `this.data`. Being able to say which, without looking it up, is the sign
   * you have actually absorbed the local-gradient table.
   */
  log(): Value {
    const out = new Value(Math.log(this.data), [this], "log");
    out._backward = () => {
      this.grad += (1 / this.data) * out.grad;
    };
    return out;
  }

  /**
   * Forward: tanh(this)
   *
   * ── WHERE THIS SHOWS UP ───────────────────────────────────────────────────
   * The classic squashing nonlinearity. Ch 11 replaces it with ReLU/GELU for
   * real networks, but tanh is the one worth differentiating by hand first
   * because its derivative has the prettiest closed form.
   *
   * ── FORWARD (08a) ─────────────────────────────────────────────────────────
   * One parent, label "tanh".
   *
   * ✅ CHECKPOINT: `new Value(0).tanh().data === 0`, and `new Value(0).exp()
   *    .data === 1` — two of the forward-pass tests already waiting in
   *    value.test.ts.
   *
   * ── BACKWARD (08b) ────────────────────────────────────────────────────────
   *   ∂out/∂this = 1 − tanh²(this.data) = 1 − out.data²    ← use the OUTPUT
   *
   * ── PITFALL (explicitly called out in the doc) ────────────────────────────
   * `1 − out.data²`, NOT `1 − this.data²`. Squaring the input instead of the
   * output is a real and popular bug: both expressions are 1 at x = 0, so the
   * cheapest test you might write passes while every other point is wrong. Test
   * at x = 1.
   *
   * ── WHY THIS MATTERS LATER ────────────────────────────────────────────────
   * That derivative is at most 1 and approaches 0 as |x| grows. Stack forty
   * layers of it and the product of forty such factors underflows — the
   * vanishing gradient problem, and half the reason the transformer is built
   * out of residual connections.
   *
   * ── WORKED EXAMPLE — the neuron from exercise E3: y = tanh(w·x + b) ───────
   * With w = 0.5, x = 1.0, b = 0.2, the forward pass builds three nodes:
   *
   *     u = w.mul(x)  →  data = 0.5        _inputs = [w, x]  _op = "*"
   *     v = u.add(b)  →  data = 0.7        _inputs = [u, b]  _op = "+"
   *     y = v.tanh()  →  data = 0.6043678  _inputs = [v]     _op = "tanh"
   *
   * Backward, seeded with y.grad = 1, one node at a time:
   *
   *     v.grad  +=  1 × (1 − 0.6043678²)   =  0.6347396   ← tanh scales
   *     u.grad  +=  0.6347396 × 1          =  0.6347396   ← the add routes:
   *     b.grad  +=  0.6347396 × 1          =  0.6347396     both unchanged
   *     w.grad  +=  0.6347396 × x.data(1.0)=  0.6347396   ← the mul switches:
   *     x.grad  +=  0.6347396 × w.data(0.5)=  0.3173698     each gets the other
   *
   * That 0.6347396 on `w` is precisely the number exercise E3 compares against
   * Ch 07's `numericalGradient` — three of your operators verified at once, by
   * a tool you already built two chapters ago.
   *
   * Now run the WRONG version on these same numbers: `1 − this.data²` gives
   * 1 − 0.7² = 0.51 instead of 0.6347396. Not a crash, not obviously wrong —
   * just a network that quietly learns the wrong thing. This is exactly the
   * class of bug the numerical gradient check exists to catch.
   */
  tanh(): Value {
    const out = new Value(Math.tanh(this.data), [this], "tanh");
    out._backward = () => {
      this.grad += (1 - out.data ** 2) * out.grad;
    };
    return out;
  }

  /**
   * Forward: max(0, this)
   *
   * ── WHERE THIS SHOWS UP ───────────────────────────────────────────────────
   * The activation inside every feedforward block (Ch 25).
   *
   * ── FORWARD (08a) ─────────────────────────────────────────────────────────
   * One parent, label "relu".
   *
   * ── BACKWARD (08b) — a GATE, not a scale ──────────────────────────────────
   *   ∂out/∂this = 1  if this.data > 0
   *                0  otherwise
   * ReLU doesn't scale the gradient, it gates it: gradient passes untouched
   * through active units and is blocked entirely at inactive ones. A unit that
   * lands on the wrong side receives exactly nothing and cannot recover — the
   * "dying ReLU" problem, and the reason leakyRelu exists in Ch 11.
   *
   * You can express the gate either as a branch or by multiplying by a 0/1
   * indicator; both are fine, and both must still use `+=`.
   *
   * ── PITFALL ───────────────────────────────────────────────────────────────
   * x = 0 is a genuine kink — the derivative does not exist there. Every
   * library picks a convention (0 is the usual choice) and moves on. Note that
   * the gradient-check test in value.test.ts is titled "relu gradient matches
   * finite differences (x ≠ 0)": the parenthetical is there because a centered
   * finite difference straddling the kink averages the two one-sided slopes to
   * 0.5 and would disagree with ANY convention you pick. Choose your test point
   * away from the origin.
   *
   * ── WORKED EXAMPLE — the STRETCH neuron at the end of the exercise file ───
   * n = relu(w1·x1 + w2·x2 + b), with w1 = 0.4, w2 = -0.6, b = 0.1,
   * x1 = 1.0, x2 = 0.5:
   *
   *     w1·x1  =   0.4        w2·x2  =  -0.3
   *     sum    =   0.1        + b    =   0.2     ← the pre-activation
   *     n      =   relu(0.2)  =  0.2             ← positive: the gate is OPEN
   *
   * Backward, with n.grad = 1:
   *
   *     pre-activation.grad  +=  1 × 1  =  1     ← gate open: passes untouched
   *     b.grad   +=  1                           ← the adds route it onward
   *     w1.grad  +=  1 × x1.data( 1.0)  =   1.0
   *     x1.grad  +=  1 × w1.data( 0.4)  =   0.4
   *     w2.grad  +=  1 × x2.data( 0.5)  =   0.5
   *     x2.grad  +=  1 × w2.data(-0.6)  =  -0.6
   *
   * Now change b from 0.1 to -0.5 and re-run. The pre-activation becomes -0.4,
   * relu outputs 0, the gate SHUTS — and every gradient in that list becomes
   * exactly 0. Not small: zero. Nothing upstream of a dead ReLU learns
   * anything, ever. That is the whole "dying ReLU" story, in one edit you can
   * make yourself.
   */
  relu(): Value {
    const out = new Value(Math.max(0, this.data), [this], "relu");
    out._backward = () => {
      this.grad += (this.data > 0 ? 1 : 0) * out.grad;
    };
    return out;
  }

  /**
   * Run reverse-mode autodiff from this node.
   * Sets this.grad = 1, then visits all nodes in reverse topological order
   * calling _backward to accumulate gradients into leaf parameters.
   *
   * Chapter 08b
   *
   * ── THE WHOLE ALGORITHM, IN THREE LINES ───────────────────────────────────
   *   1. order = topoSort(this)         ← from ./engine.ts; inputs-before-outputs
   *   2. this.grad = 1                  ← the seed: ∂L/∂L = 1, because L = L
   *   3. walk `order` BACKWARD, calling node._backward() on each
   *
   * That is the complete method. Every interesting decision was already made
   * inside the individual `_backward` closures; this function only chooses the
   * ORDER in which they fire.
   *
   * ── WHY REVERSE TOPOLOGICAL ORDER, SPECIFICALLY ───────────────────────────
   * A node must not push gradient to its parents until it has received the FULL
   * gradient from all of its own children — otherwise it forwards a
   * half-finished number and everything upstream is quietly wrong. Topological
   * order lists every node after its inputs; walking it in reverse therefore
   * guarantees every node is visited only after all of its children have
   * already contributed to it. It is not one valid order among several — it is
   * the correctness condition.
   *
   * ── WHY SEED WITH 1 ───────────────────────────────────────────────────────
   * `grad` on a node means ∂(this output)/∂(that node). For the output itself
   * that is ∂L/∂L, and the derivative of anything with respect to itself is 1.
   * Seeding 0 would make the entire sweep multiply zeros; forgetting to seed at
   * all does the same thing and is a silent, all-zero failure.
   *
   * ── DO NOT ZERO GRADIENTS HERE ────────────────────────────────────────────
   * Tempting, but wrong: `backward()` accumulates BY DESIGN, and zeroing at the
   * start of the sweep would erase contributions the caller may be deliberately
   * summing. Resetting is the caller's job — which is exactly why PyTorch makes
   * you write `optimizer.zero_grad()` by hand, and why `zeroGrad` below is a
   * separate method.
   *
   * ── PITFALL ───────────────────────────────────────────────────────────────
   * Do not call `backward()` twice on the same graph and expect the same
   * numbers. The second call accumulates on top of the first. Build a fresh
   * forward graph for each training step.
   *
   * ── WORKED EXAMPLE — the entire sweep, on Figure 3's graph ────────────────
   * Build `L = (a·b) + d` with a = 2, b = -3, d = 10, then call L.backward().
   * Here is every step this method performs, in order:
   *
   *   1. topoSort(L)      →  [a, b, c, d, L]     (inputs before outputs)
   *   2. L.grad = 1                               (the seed)
   *   3. walk that REVERSED  →  L, d, c, b, a
   *
   *        L._backward()  →  c.grad += 1×1 = 1,  d.grad += 1×1 = 1
   *        d._backward()  →  leaf: the no-op runs, nothing happens
   *        c._backward()  →  a.grad += 1×(-3) = -3,  b.grad += 1×2 = 2
   *        b._backward()  →  leaf: no-op
   *        a._backward()  →  leaf: no-op
   *
   *   Final state:  a.grad = -3,  b.grad = 2,  c.grad = 1,  d.grad = 1
   *
   * That is Figure 2, exactly. And look at the one thing that made it work:
   * `c` had its full gradient (handed over by L) BEFORE its own closure ran.
   * That is reverse topological order earning its keep.
   *
   * Walk the SAME list forward instead — a, b, c, d, L — and `c._backward()`
   * fires while c.grad is still 0, so a and b each receive 1×0×… = 0. The
   * method returns normally, every gradient is 0 or wrong, and nothing throws.
   * This is why the order is a correctness condition and not a preference.
   *
   * ✅ CHECKPOINT — the two-path case, adapted from the doc's hand-derivation.
   *    08b derives it on `f = b·sin(a) + b²`; there is no `sin` on this class,
   *    so build the same SHAPE from the ops you have:
   *
   *        f = a.mul(b).add(b.pow(2))        at  a = 1, b = 3
   *
   *    Forward: f.data = 3 + 9 = 12. Backward must give ∂f/∂a = 3 and
   *    ∂f/∂b = 7 — and that 7 is 1 + 6, `b`'s two paths through the graph
   *    meeting and ADDING, the same junction you watched in the warm-up's
   *    Run 2. If you get 1 or 6 instead of 7, exactly one of your `+=`
   *    accumulations is an `=`.
   */
  backward(): void {
    const order = topoSort(this);
    this.grad = 1;
    // Walk the topological order in REVERSE: every node fires only after all of
    // its children have already accumulated into it.
    for (let i = order.length - 1; i >= 0; i--) {
      // `!` because noUncheckedIndexedAccess types order[i] as Value | undefined;
      // i is bounded by order.length, so the element is always present.
      order[i]!._backward();
    }
  }

  /**
   * Reset grad to 0.  Call before each new forward/backward pass.
   *
   * ── SCOPE QUESTION TO SETTLE BEFORE YOU WRITE IT ──────────────────────────
   * Does this reset only THIS node, or every node reachable from it? The doc
   * (08b, "Zeroing gradients between steps") shows the one-line, single-node
   * form and says "call on every node before re-running backward" — so the
   * per-node reading is the intended one, and the test is titled simply
   * "zeroGrad resets grad to 0".
   *
   * The graph-wide version is a reasonable thing to want, and you already have
   * the tool for it: `topoSort` hands you every reachable node. But make that a
   * deliberate choice rather than an accident, and if you go that way, say so
   * in the JSDoc — a caller who expects a one-node reset and gets a graph-wide
   * one will lose an afternoon.
   *
   * ── WHY THIS EXISTS AT ALL ────────────────────────────────────────────────
   * Because `backward()` uses `+=`. Skip the reset between training steps and
   * step 2's gradients land on top of step 1's, the effective step size grows
   * without bound, and the loss diverges. It looks like a bad learning rate,
   * which is why it wastes so much time. You will meet this again for real in
   * Ch 14 when the optimizer starts stepping.
   *
   * ── WORKED EXAMPLE — what "forgetting to reset" actually looks like ───────
   * Take Figure 3's graph once more (a = 2, b = -3, d = 10) and call
   * backward() repeatedly WITHOUT zeroing in between:
   *
   *     after the 1st backward():   a.grad =  -3     ← correct
   *     after the 2nd backward():   a.grad =  -9
   *     after the 3rd backward():   a.grad = -18
   *
   * Note it does NOT merely double each time. `backward()` re-SEEDS the root
   * (`this.grad = 1`, an assignment), but every INTERIOR node still accumulates
   * — so on run 2, `c.grad` is already 2 before it multiplies into `a`, giving
   * -3 + (-3×2) = -9; on run 3, `c.grad` is 3, giving -9 + (-3×3) = -18. The
   * contamination compounds with the DEPTH of the graph, which is why it gets
   * dramatically worse on a real network than this three-node example suggests.
   * (The sequence is -3·n(n+1)/2 if you want to check yourself.)
   *
   * Nothing throws, and every individual number looks plausible. The gradient
   * is simply wrong by a factor that grows every single step, so the optimizer
   * takes ever-larger strides and the loss climbs. It presents exactly like a
   * learning rate set too high — which is why this one is so expensive to find
   * by staring at the loss curve.
   */
  zeroGrad(): void {
    this.grad = 0;
  }

  toString(): string {
    return `Value(data=${this.data.toFixed(4)}, grad=${this.grad.toFixed(4)})`;
  }
}

/**
 * ══════════════════════════════════════════════════════════════════════════
 * NOT YET DECLARED — three gaps between this file and the chapter docs
 * ══════════════════════════════════════════════════════════════════════════
 *
 * Work through the build order in 08a and you will reach for these; none of
 * them is stubbed above. Flagging rather than adding, since it changes this
 * module's public surface:
 *
 *   1. `printGraph(v)` — MILESTONE 2, and the doc tells you to write it BEFORE
 *      the interesting ops because it is your eyes for everything after. Walks
 *      `_inputs` recursively, indenting one level per depth. Belongs here or in
 *      engine.ts next to topoSort.
 *
 *   2. `neg()`, `sub(other)`, `div(other)` — MILESTONE 4. The doc builds the
 *      last two by COMPOSITION rather than as primitives:
 *          sub  =  add(other.neg())
 *          div  =  mul(other.pow(-1))
 *      Composition means no new `_backward` rules to write in 08b — the
 *      existing add/mul/pow closures already handle them, and the chain rule
 *      composes automatically. That is the payoff, and it is worth doing this
 *      way even though a direct implementation looks shorter. The checkpoint:
 *      `a.sub(b)` must build TWO nodes (a neg, then an add) — confirm with
 *      printGraph.
 *
 *   3. The constructor signature vs. exercises/ch-08-autograd.ts (see the
 *      constructor's note above): the exercise passes a debug LABEL second,
 *      `new Value(2, "a")`, while this signature and the doc both expect
 *      `(data, inputs, op)`. Either the exercise is stale, or the class is
 *      meant to carry an optional label. Decide which, because the exercise
 *      will not type-check against the current signature.
 *
 * Also note `relu` exists here but is absent from 08a's "Operations to
 * support" table, while `neg`/`sub`/`div` are in the table but absent here.
 * The tests are the tiebreaker: value.test.ts gradient-checks relu, so the
 * union of both lists is what you actually want.
 */

const printGraph = (v: Value, indent = 0): void => {
  console.log(`${" ".repeat(indent)}${v.toString()} (op=${v._op})`);
  for (const input of v._inputs) {
    printGraph(input, indent + 2);
  }
}
