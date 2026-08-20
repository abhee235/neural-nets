/**
 * autograd/grad.ts
 * ══════════════════════════════════════════════════════════
 * Tensor-aware automatic differentiation.
 *
 * Extends the scalar Value concept to full Tensors.
 * Weight matrices and activation maps now carry gradients of the same shape
 * as their data — this is what makes it possible to train neural networks.
 *
 * Chapter: 10 — Tensor Autograd Bridge
 * Doc:     docs/part-2-autodiff/ch-10-tensor-autograd-bridge.md
 *
 * ──────────────────────────────────────────────────────────────────────────
 * HOW TO READ THIS FILE
 * ──────────────────────────────────────────────────────────────────────────
 * Every method below is presented as a PAIR:
 *
 *   ── SCALAR (Ch 08) ──   the code you already wrote in value.ts, quoted
 *   ── TENSOR (Ch 10) ──   what changes, and what does not
 *
 * Read the scalar block first. You wrote it, you debugged it, you trust it.
 * Then read the tensor block and look only for the delta. For several methods
 * the delta is smaller than you expect; for two of them it is the whole point
 * of the chapter.
 *
 * Nothing about the GRAPH changes here. Not the topological order, not the
 * reverse walk, not `+=` accumulation, not the seed, not `zeroGrad`, and not
 * your `SGD`. Only the CONTENTS of a node change: one number becomes one
 * tensor.
 *
 * ──────────────────────────────────────────────────────────────────────────
 * THE ONE NEW IDEA: shape has to be conserved
 * ──────────────────────────────────────────────────────────────────────────
 * A gradient answers "if I nudge this, how does the loss respond?" — so there
 * must be exactly one gradient number per data number:
 *
 *     node.grad.shape  ===  node.data.shape        ALWAYS. No exceptions.
 *
 * Forward operations are allowed to CHANGE shape — broadcasting grows one,
 * reductions shrink one — so backward has to put the shape back. Which
 * direction it goes is the whole trick:
 *
 *     forward BROADCASTS a shape up   →   backward SUMS the gradient down
 *     forward SUMS a shape down       →   backward BROADCASTS the gradient up
 *
 * They are mirror images. The chapter doc walks both by hand (sections 4-5).
 *
 * ──────────────────────────────────────────────────────────────────────────
 * WHAT YOU WILL NEED TO IMPORT
 * ──────────────────────────────────────────────────────────────────────────
 * Everything on the right-hand side of the tensor blocks below already exists.
 * Wire these up as you go — knowing where each piece lives is part of the work:
 *
 *   tensor/ops.ts      add, mul, broadcast, broadcastShapes, mulScalar
 *   tensor/linalg.ts   matMul, matMulBatch, transpose, reshape, unsqueeze
 *   tensor/reduce.ts   sum, mean
 *   tensor/creation.ts zeros, ones, fullLike
 *   autograd/engine.ts topoSortTensor            ← the Ch 10 sort in engine.ts
 *   utils/numerical.ts numericalGradientTensor   ← the referee for the checks
 */
import type { Tensor } from "../tensor/index.ts";

/**
 * A tensor-valued node in the autograd graph.
 * .data and .grad are both Tensors with the same shape.
 *
 * ── SCALAR (Ch 08) ────────────────────────────────────────────────────────
 *     class Value {
 *       data: number;
 *       grad: number;          // starts at 0
 *       _inputs: Value[];
 *       _op: string;
 *       _backward: () => void;
 *     }
 *
 * ── TENSOR (Ch 10) ────────────────────────────────────────────────────────
 * `data` becomes a Tensor. `grad` becomes `Tensor | null`.
 *
 * Why nullable, when the scalar version just used 0? Because "a gradient of
 * zero" and "no gradient yet" stop being the same statement once shape is
 * involved. A zero gradient has to be a whole tensor of the right shape, and
 * allocating one per parameter before training even starts is real memory
 * spent to represent "nothing has happened". PyTorch draws the same line —
 * `.grad` is `None` until the first backward pass.
 *
 * The price is that every accumulation site has to answer "first contribution,
 * or another one?". That is two lines, and it is the same two lines everywhere:
 *
 *     if (node.grad === null)  node.grad = <the contribution>;
 *     else                     node.grad = add(node.grad, <the contribution>);
 *
 * That `add` does exactly what `+=` did in Ch 08 — element-wise, across the
 * whole tensor. Consider writing it once as a small private helper rather than
 * eight times; you will be glad of it by the time you reach `matMul`.
 *
 * Note there is no `_op` field here. It was debug-only in Ch 08 — add it back
 * if you want readable graph dumps, but nothing in this file depends on it.
 */
export class TensorValue {
  data: Tensor;
  /** Accumulated gradient — same shape as data. null until backward runs. */
  grad: Tensor | null;
  _inputs: TensorValue[];
  _backward: () => void;

  /**
   * Build a node.
   *
   * ── SCALAR (Ch 08) ──────────────────────────────────────────────────────
   *     constructor(data, _inputs = [], _op = "") {
   *       this.data = data;
   *       this.grad = 0;
   *       this._inputs = _inputs || [];
   *       this._op = _op || "";
   *       this._backward = () => {};
   *     }
   *
   * ── TENSOR (Ch 10) ──────────────────────────────────────────────────────
   * Same idea, with `grad` starting at `null` rather than 0 — done below.
   *
   * ── NOTE ON THIS SIGNATURE ──────────────────────────────────────────────
   * Unlike Ch 08, this constructor takes ONLY `data`. So interior nodes have
   * to set `_inputs` and `_backward` after construction, inside each
   * operation — you will be writing `out._inputs = [this, other]` by hand in
   * every method below.
   */
  constructor(data: Tensor) {
     this.data = data;
     this.grad = null;
     this._inputs = [];
     this._backward = () => {};
  }

  /**
   * Elementwise add with broadcast-aware backward.
   *
   * ── SCALAR (Ch 08) ──────────────────────────────────────────────────────
   *     add(other: Value): Value {
   *       const out = new Value(this.data + other.data, [this, other], "+");
   *       out._backward = () => {
   *         this.grad += out.grad;
   *         other.grad += out.grad;
   *       };
   *       return out;
   *     }
   *
   * ── TENSOR (Ch 10) ──────────────────────────────────────────────────────
   * Forward: `this.data + other.data` becomes `add(this.data, other.data)`
   * from Ch 03 — which BROADCASTS. That one word is the entire difference,
   * and it creates the entire problem.
   *
   * Backward: addition is still a router, the local derivative is still 1,
   * and each parent still receives the upstream gradient unchanged... except
   * that "unchanged" now has to mean "unchanged, AND of the parent's shape".
   *
   * If a `[1,3]` was broadcast against a `[2,3]`, then `out.grad` is `[2,3]`,
   * and handing that straight to the `[1,3]` parent violates the invariant.
   * Each original element was COPIED into two positions, so the chain rule
   * says it collects the SUM of both gradients. That is `sumToShape`:
   *
   *     this.grad  += sumToShape(out.grad, this.data.shape)
   *     other.grad += sumToShape(out.grad, other.data.shape)
   *
   * ── PITFALL ─────────────────────────────────────────────────────────────
   * When the shapes already match, `sumToShape` must be a no-op — so it is
   * always safe to call. Do not try to be clever and skip it conditionally.
   *
   * ✅ CHECKPOINT: the doc's section 2 example — `Z = (A×B) + d` with `d` of
   *    shape `[1,3]` — must give `d.grad = [2,2,2]` with shape `[1,3]`.
   */
  add(other: TensorValue): TensorValue {
    throw new Error("TensorValue.add not implemented");
  }

  /**
   * Elementwise multiply with broadcast-aware backward.
   *
   * ── SCALAR (Ch 08) ──────────────────────────────────────────────────────
   *     mul(other: Value): Value {
   *       const out = new Value(this.data * other.data, [this, other], "*");
   *       out._backward = () => {
   *         this.grad  += other.data * out.grad;
   *         other.grad += this.data  * out.grad;
   *       };
   *       return out;
   *     }
   *
   * ── TENSOR (Ch 10) ──────────────────────────────────────────────────────
   * Structurally identical. The switch still swaps the operands; each `*`
   * becomes an element-wise tensor `mul`; each accumulation gets wrapped in
   * `sumToShape` for the same reason as `add`.
   *
   *     this.grad  += sumToShape( mul(other.data, out.grad), this.data.shape  )
   *     other.grad += sumToShape( mul(this.data,  out.grad), other.data.shape )
   *
   * ── PITFALL: the inner multiplications broadcast too ────────────────────
   * `mul(other.data, out.grad)` is itself a broadcasting operation, and its
   * result has the BROADCAST shape, not the parent's. That is exactly why
   * `sumToShape` wraps the product rather than being applied to `out.grad`
   * first — reverse the order and you will be multiplying mismatched shapes.
   */
  mul(other: TensorValue): TensorValue {
    throw new Error("TensorValue.mul not implemented");
  }

  /**
   * Matrix multiply.
   * Backward: dA = dZ @ Bᵀ,  dB = Aᵀ @ dZ
   *
   * ── SCALAR (Ch 08) ──────────────────────────────────────────────────────
   * There is none. For single numbers, matMul *is* mul. This operation is
   * genuinely new, and it is the one you will lean on for the rest of the
   * course: attention is mostly matmuls.
   *
   * ── TENSOR (Ch 10) ──────────────────────────────────────────────────────
   * Forward: `matMul(this.data, other.data)` from Ch 04.
   *
   * Backward, for `Z = A @ B`:
   *
   *     ∂L/∂A  =  ∂L/∂Z  @  Bᵀ
   *     ∂L/∂B  =  Aᵀ     @  ∂L/∂Z
   *
   * ── HOW TO REMEMBER IT: let the shapes derive it for you ────────────────
   * Do not memorise those two lines. There is only one arrangement of the
   * pieces that fits, so you can always rebuild them:
   *
   *     A : [m, k]      B : [k, n]      Z : [m, n]      dZ : [m, n]
   *
   *     dA must be [m,k].  From dZ [m,n], the only way to reach [m,k] is to
   *                        multiply by something [n,k] — that is Bᵀ.
   *     dB must be [k,n].  From dZ [m,n], the only way to reach [k,n] is to
   *                        be multiplied INTO by something [k,m] — that is Aᵀ.
   *
   * And with non-square test shapes, getting a transpose wrong does not give
   * a wrong number — it gives a shape mismatch that THROWS. Use that.
   *
   * ── PITFALL: which axes does `transpose` swap? ──────────────────────────
   * Ch 04's `transpose(t, axes?)` reverses ALL axes by default. For a 2-D
   * matrix that is what you want. For a BATCHED tensor — `[batch, seq, dHead]`,
   * which is what Ch 23 will hand you — reversing everything is wrong: swap
   * only the LAST TWO axes and leave the batch dimensions alone. Pass an
   * explicit `axes` permutation, and consider a small local helper for it.
   *
   * ── PITFALL: `matMul` vs `matMulBatch` ──────────────────────────────────
   * Ch 04 gave you both. Decide now which this method uses, and whether it
   * should dispatch on `ndim`. The tests here are 2-D; Ch 23 is not.
   *
   * ✅ CHECKPOINT: `[2,3] @ [3,4]` gives `[2,4]`; after backward the gradients
   *    are `[2,3]` and `[3,4]` — each matching its own parameter.
   */
  matMul(other: TensorValue): TensorValue {
    throw new Error("TensorValue.matMul not implemented");
  }

  /**
   * Sum reduction. Backward broadcasts upstream gradient back to input shape.
   *
   * ── SCALAR (Ch 08) ──────────────────────────────────────────────────────
   * None — there is nothing to reduce in a single number.
   *
   * ── TENSOR (Ch 10) ──────────────────────────────────────────────────────
   * Forward: `sum(this.data, axis, keepDims)` from Ch 05.
   *
   * Backward is the MIRROR of broadcasting (doc, section 5): every input
   * element contributed with a coefficient of exactly 1, so every element
   * receives the SAME upstream gradient. Copying one number into many
   * positions is precisely what `broadcast` does.
   *
   * ── PITFALL: keepDims decides whether this is easy or fiddly ────────────
   * With `keepDims = true` the reduced axis survives as size 1, so `out.grad`
   * is `[2,1]` for an input `[2,3]` and `broadcast` handles it directly.
   * With `keepDims = false` the axis is GONE — `out.grad` is `[2]` — and you
   * must reinsert it before broadcasting, or the shapes will not line up.
   * `unsqueeze` from Ch 04 does that.
   *
   * Handle `axis === undefined` too: the output is a scalar, and every input
   * element receives that one gradient value.
   */
  sum(axis?: number, keepDims?: boolean): TensorValue {
    throw new Error("TensorValue.sum not implemented");
  }

  /**
   * Mean reduction. Backward distributes grad / n to each input element.
   *
   * ── TENSOR (Ch 10) ──────────────────────────────────────────────────────
   * `mean` is `sum` divided by the count, so its backward is `sum`'s backward
   * divided by the same count:
   *
   *     ∂mean/∂inputᵢ  =  1/n        where n is the number of elements reduced
   *
   * Broadcast the upstream gradient back out exactly as `sum` does, then
   * scale by `1/n` (`mulScalar` from Ch 03).
   *
   * ── PITFALL: which n? ───────────────────────────────────────────────────
   * With an axis, `n` is `this.data.shape[axis]` — the LENGTH OF THE REDUCED
   * AXIS, not the total element count. With no axis it is `this.data.size`.
   * Getting this wrong scales every gradient by a constant factor, which
   * looks exactly like a mis-set learning rate and will not fail loudly.
   */
  mean(axis?: number, keepDims?: boolean): TensorValue {
    throw new Error("TensorValue.mean not implemented");
  }

  /**
   * Reshape forward; backward reshapes grad to original shape.
   *
   * ── TENSOR (Ch 10) ──────────────────────────────────────────────────────
   * The easiest operation in the file, and worth doing early for the
   * momentum. Reshape moves no numbers — it reinterprets the same flat buffer
   * under a new shape. So no gradient VALUE changes; only the shape label
   * does, and backward just puts the original label back:
   *
   *     forward :  out.data   = reshape(this.data, newShape)
   *     backward:  this.grad += reshape(out.grad, <the ORIGINAL shape>)
   *
   * ── PITFALL ─────────────────────────────────────────────────────────────
   * Reshape backward to the ORIGINAL shape, captured from `this.data.shape`
   * — not to `newShape`, and not with the forward call's argument.
   */
  reshape(newShape: number[]): TensorValue {
    throw new Error("TensorValue.reshape not implemented");
  }

  /**
   * Transpose forward; backward applies the inverse permutation.
   *
   * ── TENSOR (Ch 10) ──────────────────────────────────────────────────────
   * Forward: `transpose(this.data, axes)` from Ch 04.
   *
   * Backward: apply the INVERSE permutation. For a plain 2-D transpose the
   * inverse is the same swap — which is exactly why this is easy to get
   * wrong: it will pass every 2-D test while being wrong for 3-D and above.
   *
   * ── HOW TO INVERT A PERMUTATION ─────────────────────────────────────────
   * `axes[i] = j` means "output axis i comes from input axis j". To undo that
   * you need the array `inv` with `inv[axes[i]] = i`. One loop.
   *
   *     axes = [1, 2, 0]   →   inv = [2, 0, 1]     (≠ axes!)
   *     axes = [1, 0]      →   inv = [1, 0]        (happens to equal axes)
   *
   * ── PITFALL: the default ────────────────────────────────────────────────
   * When `axes` is undefined, Ch 04's `transpose` reverses all axes. Reversal
   * is its own inverse, so backward can simply reverse again — but only if
   * you handle the undefined case explicitly instead of feeding `undefined`
   * into your inversion loop.
   */
  transpose(axes?: number[]): TensorValue {
    throw new Error("TensorValue.transpose not implemented");
  }

  /**
   * Run reverse-mode autodiff from this tensor node.
   *
   * ── SCALAR (Ch 08) ──────────────────────────────────────────────────────
   *     backward(): void {
   *       const order = topoSort(this);
   *       this.grad = 1;
   *       for (let i = order.length - 1; i >= 0; i--) {
   *         order[i]!._backward();
   *       }
   *     }
   *
   * ── TENSOR (Ch 10) ──────────────────────────────────────────────────────
   * Line for line the same, with exactly two substitutions:
   *
   *     topoSort(this)   →   topoSortTensor(this)
   *     this.grad = 1    →   this.grad = ones(this.data.shape)
   *
   * The seed is still "the derivative of the output with respect to itself",
   * which is still 1 — there is just one of them per element now.
   *
   * ── GUARD THE ROOT ──────────────────────────────────────────────────────
   * Seeding with ones only means ∂L/∂L when L is a single number. Call
   * `backward()` on a `[2,3]` node and you are quietly asking for the
   * gradient of the SUM of its six entries — rarely what anyone intends.
   * PyTorch refuses outright unless you pass an explicit gradient. Either
   * throw when `this.data.size !== 1`, or document the summing behaviour as
   * a deliberate choice.
   *
   * ── UNCHANGED ───────────────────────────────────────────────────────────
   * Reverse topological order, for the same correctness reason as Ch 08b.
   * No zeroing here — still the caller's job. Still one sweep, all gradients.
   */
  backward(): void {
    throw new Error("TensorValue.backward not implemented");
  }

  /**
   * Reset grad to null.
   *
   * ── SCALAR (Ch 08) ──────────────────────────────────────────────────────
   *     zeroGrad(): void { this.grad = 0; }
   *
   * ── TENSOR (Ch 10) ──────────────────────────────────────────────────────
   * Set `grad` back to `null` rather than to a zero tensor. Same meaning as
   * Ch 08's 0 — "nothing has accumulated yet" — and it releases the gradient
   * memory instead of holding a full tensor of zeros per parameter.
   */
  zeroGrad(): void {
    throw new Error("TensorValue.zeroGrad not implemented");
  }
}

/**
 * Reverse broadcasting: sum gradient over axes that were broadcast.
 *
 * When forward computed a→b (a smaller → b larger via broadcast),
 * backward must sum b's gradient back down to a's original shape.
 *
 * ── THE IDEA IN ONE SENTENCE ──────────────────────────────────────────────
 * Broadcasting COPIED one input element into many output positions, and the
 * chain rule says an element used in many places collects the SUM of the
 * gradients from all of them. So un-broadcasting is summing — Ch 08's `+=`
 * for a node used twice, applied across whole axes at once.
 *
 * The chapter doc walks this by hand in section 4 (one bias entry at a time)
 * before generalising it in section 6. Read those first.
 *
 * ── THE ALGORITHM (doc, section 6) ────────────────────────────────────────
 * Two distinct cases, and both must be handled:
 *
 *   1. RANK was added. `[3]` broadcast to `[2,3]` gained a leading axis. Sum
 *      the leading axes away until the ranks match.
 *   2. A size-1 axis was STRETCHED. `[3,1]` broadcast to `[3,4]` kept its
 *      rank but grew axis 1. Sum along that axis with `keepDims = true`, so
 *      the size-1 axis survives and the result is `[3,1]`, not `[3]`.
 *
 * Do (1) first. Once the ranks agree, one pass comparing axes pairwise
 * handles (2).
 *
 * ── PITFALL: keepDims is not optional in case 2 ───────────────────────────
 * Dropping the axis gives `[3]` where the parameter is `[3,1]`. Both hold
 * three numbers, so nothing crashes here — it fails later, far from the
 * cause. This is the single most annoying bug in the chapter.
 *
 * ── PITFALL: the no-op case ───────────────────────────────────────────────
 * When `grad.shape` already equals `targetShape`, return it unchanged. Every
 * caller relies on this being safe, which is why none of them checks first.
 *
 * ✅ CHECKPOINTS (the three tests waiting in grad.test.ts):
 *      sumToShape( ones([4,3]), [1,3] )  →  [4,4,4] shaped [1,3]
 *      sumToShape( ones([2,3]), [2,3] )  →  unchanged
 *      sumToShape( ones([3,4]), [1,4] )  →  reduces axis 0
 */
export function sumToShape(grad: Tensor, targetShape: number[]): Tensor {
  throw new Error("sumToShape not implemented");
}

/**
 * Numerical gradient check for tensor-valued operations.
 * Uses finite differences to verify the analytical backward pass.
 *
 * ── WHY THIS MATTERS MORE HERE THAN ANYWHERE ELSE ─────────────────────────
 * In Ch 08 a wrong gradient was a wrong number. Here it is a wrong number in
 * a correctly-shaped tensor, which is far harder to spot: shapes line up,
 * nothing throws, and the loss even goes down for a while. Every layer from
 * Ch 11 to Ch 30 sits on top of these six operations, so a quiet error here
 * surfaces as "my transformer doesn't learn" twenty chapters later.
 *
 * Run this on every op before moving on.
 *
 * ── HOW IT WORKS ──────────────────────────────────────────────────────────
 * The tensor version of Ch 07's check, using `numericalGradientTensor`:
 *
 *   1. Run `fn(inputs)` and `backward()` to fill the analytical gradients.
 *   2. For each input, call `numericalGradientTensor` with a scalar-valued
 *      wrapper around `fn` — the loss must collapse to ONE number, so sum
 *      the output if it is not already scalar.
 *   3. Compare element by element against `tolerance` (default 1e-5).
 *
 * ── PITFALL: zero the gradients before you start ──────────────────────────
 * `fn` gets called many times inside the numerical loop. If gradients from
 * an earlier call are still sitting on the inputs, the analytical side is
 * contaminated — Ch 08's −3, −9, −18 problem, now on tensors.
 *
 * ── PITFALL: relative vs absolute error ───────────────────────────────────
 * An absolute tolerance of 1e-5 is fine for gradients around 1 and useless
 * for gradients around 1e6. For larger matmuls, prefer a relative
 * comparison: |a − n| / max(1, |a|, |n|).
 */
export function checkTensorGradient(
  fn: (inputs: TensorValue[]) => TensorValue,
  inputs: TensorValue[],
  tolerance?: number
): boolean {
  throw new Error("checkTensorGradient not implemented");
}
