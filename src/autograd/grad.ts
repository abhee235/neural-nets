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
 * THE RUNNING EXAMPLES — the doc's two graphs, nothing new
 * ──────────────────────────────────────────────────────────────────────────
 * Every worked trace below uses one of the two graphs you already know, so
 * every number here is one you have already derived by hand in the doc.
 *
 * GRAPH 1 (doc sections 2-6) — Ch 08's  L = (a·b) + d,  with tensors:
 *
 *     A = all 2   [2,3]     B = all -3  [2,3]     d = [10,10,10]  [1,3]
 *     C = A×B = all -6      Z = C+d = all 4       L = sum(Z) = 24
 *
 *     backward:  Z.grad = C.grad = all 1
 *                A.grad = all -3    B.grad = all 2    d.grad = [2,2,2]
 *
 * GRAPH 2 (doc section 8) —  L = sum(A @ B):
 *
 *     A = [1 2 3]      B = [1  2  3  4]      Z = [38  44  50  56]
 *         [4 5 6]          [5  6  7  8]          [83  98 113 128]
 *       [2,3]              [9 10 11 12]        [2,4]     L = 610
 *                        [3,4]
 *
 *     backward:  A.grad = [10 26 42]      B.grad = [5 5 5 5]
 *                         [10 26 42]               [7 7 7 7]
 *                                                  [9 9 9 9]
 *
 * If any number in a trace surprises you, the doc derives it cell by cell in
 * the section named beside the trace.
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
import {
  add, broadcast, matMul, mean, mul, mulScalar, ones, reshape, sum,
  transpose, unsqueeze, type Tensor,
} from "../tensor/index.ts";
import { topoSortTensor } from "./engine.ts";
import { numericalGradientTensor } from "../utils/numerical.ts";

/**
 * The tensor version of Ch 08's `+=`: fold one gradient contribution into a
 * node, handling the "no gradient yet" case. Written once because every
 * `_backward` closure in this file needs the identical two lines.
 *
 * `contribution` must already have the node's own shape — that is the
 * caller's job (usually via `sumToShape`).
 */
function accumulate(node: TensorValue, contribution: Tensor): void {
  // First contribution replaces the null; later ones add element-wise.
  node.grad = node.grad === null ? contribution : add(node.grad, contribution);
}

/**
 * Invert a permutation. `axes[i] = j` means "output axis i came from input
 * axis j", so the inverse satisfies `inv[axes[i]] = i` — one loop.
 *
 *     axes = [1, 2, 0]  →  inv = [2, 0, 1]      (not the same array!)
 *     axes = [1, 0]     →  inv = [1, 0]         (happens to equal axes)
 */
function invertPermutation(axes: number[]): number[] {
  const inv = new Array<number>(axes.length);
  for (let i = 0; i < axes.length; i++) {
    inv[axes[i]!] = i;
  }
  return inv;
}

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
   * ── THE RECIPE (every binary op below follows these same four steps) ─────
   * 1. out = a new TensorValue holding the forward result
   * 2. out._inputs = [this, other]      ← by hand; this constructor won't
   * 3. out._backward = a closure that accumulates each parent's contribution
   *    (null-aware, as in the class comment) — for add, the two `sumToShape`
   *    lines above
   * 4. return out
   *
   * ── WORKED TRACE — GRAPH 1's  Z = C.add(d)   (doc section 2) ────────────
   *
   * forward — d's single row is copied into both rows of C (broadcast):
   *
   *     C.data = [ -6  -6  -6 ]          d.data = [ 10  10  10 ]  [1,3]
   *              [ -6  -6  -6 ]  [2,3]
   *
   *     Z.data = [  4   4   4 ]
   *              [  4   4   4 ]  [2,3]
   *
   * backward — Z.grad arrives as all ones (L = sum seeded it):
   *
   *     Z.grad = [ 1  1  1 ]
   *              [ 1  1  1 ]  [2,3]
   *
   *     C.grad += sumToShape(Z.grad, [2,3])  =  [ 1  1  1 ]   ← same shape:
   *                                             [ 1  1  1 ]     the no-op case
   *
   *     d.grad += sumToShape(Z.grad, [1,3])  =  [ 2  2  2 ]   ← each column
   *                                               [1,3]          summed: 1+1
   *
   * — the section 2 table's rows for C and d, exactly.
   */
  add(other: TensorValue): TensorValue {
    // Forward: the Ch 03 free function — Tensor has no methods.
    const out = new TensorValue(add(this.data, other.data));
    out._inputs = [this, other];
    out._backward = () => {
      // `!` — safe for the same reason as order[i]! in Ch 08's backward():
      // reverse topological order guarantees out.grad was seeded (root) or
      // accumulated by out's consumers before this closure fires.
      accumulate(this, sumToShape(out.grad!, this.data.shape));
      accumulate(other, sumToShape(out.grad!, other.data.shape));
    };
    return out;
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
   *
   * ── WORKED TRACE — GRAPH 1's  C = A.mul(B)   (doc section 2) ────────────
   *
   * forward — same shape, element by element:
   *
   *     A.data = [ 2  2  2 ]          B.data = [ -3  -3  -3 ]
   *              [ 2  2  2 ]  [2,3]            [ -3  -3  -3 ]  [2,3]
   *
   *     C.data = [ -6  -6  -6 ]
   *              [ -6  -6  -6 ]  [2,3]
   *
   * backward — C.grad arrives as all ones; the switch crosses the operands:
   *
   *     A.grad += sumToShape( mul(B.data, C.grad), [2,3] )
   *             = [ -3  -3  -3 ]        ← A receives B's values
   *               [ -3  -3  -3 ]
   *
   *     B.grad += sumToShape( mul(A.data, C.grad), [2,3] )
   *             = [ 2  2  2 ]           ← B receives A's values
   *               [ 2  2  2 ]
   *
   * Ch 08's  a.grad = -3, b.grad = 2 — once per cell.
   * (Recipe: add's four steps, with these wrapped products as step 3.)
   */
  mul(other: TensorValue): TensorValue {
    const out = new TensorValue(mul(this.data, other.data));
    out._inputs = [this, other];
    out._backward = () => {
      // The switch: each parent's gradient is scaled by the SIBLING's data —
      // and the product is what gets summed to the parent's shape, because
      // the product itself has the broadcast shape (see the pitfall above).
      accumulate(this, sumToShape(mul(other.data, out.grad!), this.data.shape));
      accumulate(other, sumToShape(mul(this.data, out.grad!), other.data.shape));
    };
    return out;
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
   * It is NOT new calculus, though. One cell of A is used once per column of
   * Z's row i, multiplied by B[k][j] each time — Ch 08's switch rule, summed
   * over the uses:
   *
   *     A.grad[i][k] = Σⱼ dZ[i][j] · B[k][j]  =  (dZ @ Bᵀ)[i][k]
   *
   * The transpose exists because that sum walks ALONG a row of B while a
   * matmul walks DOWN a column. Doc section 8 derives it cell by cell with
   * concrete numbers — read that before implementing this.
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
   * ── THE RECIPE ──────────────────────────────────────────────────────────
   * add's four steps; step 3's two contributions are the formulas above,
   * built from `matMul` and `transpose`. In 2-D each contribution already
   * has its parent's exact shape (no broadcasting happens in a plain matmul),
   * so no `sumToShape` is needed here.
   *
   * ── WORKED TRACE — GRAPH 2's  L = sum(A @ B)   (doc section 8) ──────────
   *
   * forward:
   *
   *     A.data = [ 1  2  3 ]          B.data = [ 1   2   3   4 ]
   *              [ 4  5  6 ]  [2,3]            [ 5   6   7   8 ]
   *                                            [ 9  10  11  12 ]  [3,4]
   *
   *     Z.data = [ 38  44   50   56 ]
   *              [ 83  98  113  128 ]  [2,4]          L = sum(Z) = 610
   *
   * backward — Z.grad = all ones [2,4]:
   *
   *     A.grad += matMul(Z.grad, Bᵀ)              [2,4] @ [4,3] → [2,3]
   *             = [ 10  26  42 ]       each entry = the sum of the matching
   *               [ 10  26  42 ]       ROW of B       (10 = 1+2+3+4)
   *
   *     B.grad += matMul(Aᵀ, Z.grad)              [3,2] @ [2,4] → [3,4]
   *             = [ 5  5  5  5 ]       each row = the sum of the matching
   *               [ 7  7  7  7 ]       COLUMN of A     (5 = 1+4)
   *               [ 9  9  9  9 ]
   *
   * Section 8 derives the 10 and the 5 cell by cell. If either number
   * surprises you, re-read it before writing this method.
   */
  matMul(other: TensorValue): TensorValue {
    const out = new TensorValue(matMul(this.data, other.data));
    out._inputs = [this, other];
    out._backward = () => {
      // dA = dZ @ Bᵀ,  dB = Aᵀ @ dZ.  In 2-D both contributions already have
      // their parent's exact shape, so no sumToShape is needed here.
      accumulate(this, matMul(out.grad!, transpose(other.data)));
      accumulate(other, matMul(transpose(this.data), out.grad!));
    };
    return out;
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
   *
   * ── THE RECIPE ──────────────────────────────────────────────────────────
   * 1. capture this.data.shape — backward must broadcast back to the ORIGINAL
   * 2. out = new TensorValue(sum(this.data, axis, keepDims)); wire _inputs
   * 3. backward: take out.grad; if the axis was dropped (keepDims false and
   *    an axis was given), unsqueeze it back in; broadcast to the captured
   *    shape; accumulate into this.grad
   * 4. return out
   *
   * ── WORKED TRACE 1 — GRAPH 1's  L = Z.sum()   (doc section 2) ───────────
   *
   * forward — all six cells collapse into one number:
   *
   *     Z.data = [ 4  4  4 ]          →       L.data = [ 24 ]   shape []
   *              [ 4  4  4 ]  [2,3]
   *
   * backward — L.grad is the seed itself, a lone 1 of shape []:
   *
   *     Z.grad += broadcast(L.grad, [2,3])  =  [ 1  1  1 ]
   *                                            [ 1  1  1 ]
   *
   * (a scalar broadcasts to any shape directly — no unsqueeze needed when
   * axis is undefined)
   *
   * ── WORKED TRACE 2 — the axis case   (doc section 5's R) ────────────────
   *
   * forward — sum(axis=1, keepDims=true): each row folds into one number:
   *
   *     R.data = [ 3  1  5 ]          →       S.data = [ 9 ]
   *              [ 4  2  0 ]  [2,3]                    [ 6 ]  [2,1]
   *
   * backward — say the upstream gradient arriving for S is:
   *
   *     S.grad = [ 2 ]           R.grad += broadcast(S.grad, [2,3])
   *              [ 5 ]  [2,1]            = [ 2  2  2 ]  ← row 0's 2, copied
   *                                        [ 5  5  5 ]  ← row 1's 5, copied
   *
   * With keepDims=false the same upstream arrives as  [ 2  5 ]  shape [2] —
   * unsqueeze(grad, 1) restores [2,1], then the identical broadcast.
   */
  sum(axis?: number, keepDims?: boolean): TensorValue {
    const originalShape = this.data.shape;
    const out = new TensorValue(sum(this.data, axis, keepDims));
    out._inputs = [this];
    out._backward = () => {
      let grad = out.grad!;
      if (axis !== undefined && !keepDims) {
        grad = unsqueeze(grad, axis);
      }
      accumulate(this, broadcast(grad, originalShape));
    };
    return out;
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
   *
   * ── WORKED TRACE — GRAPH 1's Z again ────────────────────────────────────
   *
   * forward:
   *
   *     Z.data = [ 4  4  4 ]          →       Z.mean() = [ 4 ]   shape []
   *              [ 4  4  4 ]  [2,3]                      (24 / 6)
   *
   * backward — sum's broadcast, then scaled by 1/n, n = size = 6 (no axis):
   *
   *     Z.grad += [ 1/6  1/6  1/6 ]
   *               [ 1/6  1/6  1/6 ]
   *
   * (Recipe: sum's recipe plus one `mulScalar(…, 1/n)` on the way back.)
   */
  mean(axis?: number, keepDims?: boolean): TensorValue {
    const originalShape = this.data.shape;
    // Forward must BE mean — not sum. (With sum here, Z.mean() returns 24
    // instead of 4 while the backward scales by 1/n: forward and backward
    // would disagree about what the op is.)
    const out = new TensorValue(mean(this.data, axis, keepDims));
    out._inputs = [this];
    out._backward = () => {
      let grad = out.grad!;
      if (axis !== undefined && !keepDims) {
        grad = unsqueeze(grad, axis);
      }
      // `!` — axis was valid in the forward pass or mean() would have thrown.
      const n = axis === undefined ? this.data.size : this.data.shape[axis]!;
      accumulate(this, mulScalar(broadcast(grad, originalShape), 1 / n));
    };
    return out;
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
   *
   * ── WORKED TRACE — GRAPH 2's A ──────────────────────────────────────────
   *
   *     A.data = [ 1  2  3 ]    → reshape([6]) →   [ 1  2  3  4  5  6 ]  [6]
   *              [ 4  5  6 ]  [2,3]
   *
   * backward — a [6] gradient arrives; reshape it to the captured [2,3]:
   *
   *     [ g0  g1  g2  g3  g4  g5 ]   →   [ g0  g1  g2 ]
   *                           [6]        [ g3  g4  g5 ]  [2,3]
   *
   * Same six numbers both directions; only the shape label changes.
   */
  reshape(newShape: number[]): TensorValue {
    const originalShape = this.data.shape;
    // Free functions again — Tensor has no methods (same fix as add's forward).
    const out = new TensorValue(reshape(this.data, newShape));
    out._inputs = [this];
    out._backward = () => {
      accumulate(this, reshape(out.grad!, originalShape));
    };
    return out;
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
   *
   * ── WHERE YOU USE IT FIRST ──────────────────────────────────────────────
   * matMul's backward, just above: Bᵀ is `transpose(B.data)` — GRAPH 2's
   * B [3,4] becomes [4,3], turning B's rows into columns so the backward
   * matmul can dot rows of Z.grad with rows of B (doc section 8).
   */
  transpose(axes?: number[]): TensorValue {
    const out = new TensorValue(transpose(this.data, axes));
    out._inputs = [this];
    out._backward = () => {
      // No axes → forward reversed all axes; reversal is its own inverse, so
      // undefined again undoes it. With axes, invert the permutation.
      const invAxes = axes ? invertPermutation(axes) : undefined;
      accumulate(this, transpose(out.grad!, invAxes));
    };
    return out;
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
   *
   * ── WORKED TRACE — the whole of GRAPH 1, one sweep (doc section 2) ──────
   * Build  C = A.mul(B);  Z = C.add(d);  L = Z.sum();  then L.backward():
   *
   *   1. order = topoSortTensor(L) = [A, B, C, d, Z, L]
   *   2. seed:  L.grad = ones([]) — a lone 1
   *   3. walk reversed — L, Z, d, C, B, A — each closure fills a grid:
   *
   *      L fires:   Z.grad = [ 1  1  1 ]
   *                          [ 1  1  1 ]
   *
   *      Z fires:   C.grad = [ 1  1  1 ]      d.grad = [ 2  2  2 ]  [1,3]
   *                          [ 1  1  1 ]
   *
   *      d fires:   leaf — the constructor's no-op runs
   *
   *      C fires:   A.grad = [ -3  -3  -3 ]   B.grad = [ 2  2  2 ]
   *                          [ -3  -3  -3 ]            [ 2  2  2 ]
   *
   *      B, A:      leaves — no-ops
   *
   *   Final: the section 2 table, exactly — three gradients element-for-
   *   element equal to Ch 08's scalars, and d summed down its rows. Run this
   *   graph as your first end-to-end check the moment backward compiles.
   */
  backward(): void {
    // Guard the root (see above): seeding with ones only means ∂L/∂L when L
    // is a single number. Anything else is almost always a missing .sum().
    if (this.data.size !== 1) {
      throw new Error(
        `backward() needs a scalar root, got shape [${this.data.shape}] — collapse with .sum() or .mean() first`,
      );
    }
    const order = topoSortTensor(this);
    this.grad = ones(this.data.shape);
    for (let i = order.length - 1; i >= 0; i--) {
      order[i]!._backward();
    }
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
    this.grad = null;
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
 * ── WORKED TRACE — GRAPH 1's d   (doc sections 2 and 4) ───────────────────
 *
 *     grad in  = [ 1  1  1 ]          target shape: [1,3]   (d's shape)
 *                [ 1  1  1 ]  [2,3]
 *
 *     ranks match; axis 0 has size 2 where the target has 1
 *       →  sum axis 0, keepDims:  [ 1+1  1+1  1+1 ]  =  [ 2  2  2 ]  [1,3]
 *
 *     — d.grad, the number you derived one entry at a time in section 4.
 *
 * Section 6's other two situations:
 *
 *     case 1 — rank dropped:     [ 1  1  1 ]   →  [3]   =  [ 2  2  2 ]  [3]
 *                                [ 1  1  1 ]
 *
 *     case 2 — the keepDims      [ 1  1  1  1 ]              [  4 ]
 *              trap:             [ 2  2  2  2 ]  → [3,1]  =  [  8 ]
 *                                [ 3  3  3  3 ]              [ 12 ]  [3,1]
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
  let result = grad;

  // Case 1: RANK was added. Sum the extra leading axes away one at a time —
  // NO keepDims here, the axis must disappear. `sum` takes a single axis, and
  // after each sum the remaining extra axis is axis 0 again, so always sum 0.
  while (result.shape.length > targetShape.length) {
    result = sum(result, 0, false);
  }

  // Case 2: a size-1 axis was STRETCHED. Sum along it WITH keepDims so the
  // axis survives as size 1. keepDims never shifts the axis numbering, so
  // summing one axis at a time in order is safe.
  for (let axis = 0; axis < targetShape.length; axis++) {
    if (result.shape[axis] !== targetShape[axis]) {
      if (targetShape[axis] === 1) {
        result = sum(result, axis, true);
      } else {
        throw new Error(
          `Cannot sum to shape: incompatible shapes ${grad.shape} and ${targetShape}`,
        );
      }
    }
  }

  // No-op case: if neither loop fired, result is still the original grad.
  return result;
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
 * ── WORKED TRACE — verifying GRAPH 1's d.grad ─────────────────────────────
 *     fn = ([c, d]) => c.add(d).sum()
 *
 *     analytical, from your backward:        d.grad   = [ 2  2  2 ]
 *     numerical, one nudge at a time:        measured = [ 2  2  2 ]   ✓
 *       (each d[j] sits in TWO cells of Z, so a nudge of h moves L by 2h)
 *
 * When the two rows agree to tolerance on every operation, the chapter's
 * verification gate is met.
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
  const tol = tolerance ?? 1e-5;

  // The loss BOTH sides agree on: sum(fn(...)) — one number, always.
  // (backward() insists on a scalar root, and the numerical side needs a
  // plain number to measure a slope against.)
  const scalarLoss = (nodes: TensorValue[]): TensorValue => {
    const out = fn(nodes);
    return out.data.size === 1 ? out : out.sum();
  };

  // Analytical side. Zero first — fn may have run before, and stale gradients
  // would contaminate the comparison (Ch 08's −3, −9, −18, on tensors).
  for (const input of inputs) input.zeroGrad();
  scalarLoss(inputs).backward();

  // Numerical side, one input at a time. The perturbed copy is substituted at
  // POSITION i — not always position 0 — and every other input is rebuilt as
  // a fresh leaf, so no graph state leaks between evaluations.
  for (let i = 0; i < inputs.length; i++) {
    const input = inputs[i]!;

    const numerical = numericalGradientTensor((t: Tensor) => {
      const fresh = inputs.map((node, j) =>
        j === i ? new TensorValue(t) : new TensorValue(node.data),
      );
      // Plain-number readout of the scalar loss — no backward involved here.
      return scalarLoss(fresh).data.data[0]!;
    }, input.data);

    // Compare element by element. A null analytical gradient means this input
    // never influenced the output — which agrees with a numerical zero.
    for (let k = 0; k < numerical.size; k++) {
      const analytical = input.grad === null ? 0 : input.grad.data[k]!;
      if (Math.abs(analytical - numerical.data[k]!) > tol) {
        return false;
      }
    }
  }

  return true;
}
