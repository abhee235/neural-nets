/**
 * CHAPTER 13: The Linear Layer
 * ════════════════════════════════════════
 * Part 3 of 6: Neural Net Primitives
 *
 * WHAT WE'RE BUILDING:  class Linear — y = x @ Wᵀ + b, with owned parameters
 *                       and the parameters() contract. The course's first class.
 * WHY IT MATTERS:       every Q/K/V projection (Ch 22), both FFN layers (Ch 25)
 *                       and the vocabulary head (Ch 30) ARE this class — 73 of
 *                       them in a 12-block GPT-2, holding most of its weights.
 * WHAT THIS UNLOCKS:    → Ch 14 (Optimizers) — Adam updates whatever
 *                       parameters() hands it, without knowing what a layer is.
 *
 * REFERENCE: docs/part-3-neural-net-primitives/ch-13-linear-layer.md
 *
 * ──────────────────────────────────────────────────────────────────────────
 * WHY THIS FILE EXISTS
 * ──────────────────────────────────────────────────────────────────────────
 * Ch 12 trained on logits [1, 2, 3] that we typed by hand. Something has to
 * PRODUCE scores — and it has been written inline since Ch 11:
 *
 *     x.matMul(W).add(b)
 *
 * This file gives that line a home. What is new is not the math — it is
 * OWNERSHIP: W and b must live somewhere across forward passes and training
 * steps, so this is the course's first class, and parameters() is the
 * contract every optimizer from Ch 14 onward is built against.
 *
 * There is NO new backward anywhere in this file. forward composes matMul,
 * transpose and add — all Ch 10 graph methods — so autograd already knows
 * every gradient. The work is bookkeeping done exactly right.
 *
 * ──────────────────────────────────────────────────────────────────────────
 * THE RUNNING EXAMPLE — the doc's hand-set layer, used in every trace below
 * ──────────────────────────────────────────────────────────────────────────
 * Two features of "the cat" in, three word scores out (doc section 2):
 *
 *     x  =  [ 1   2 ]                        shape [1, 2]
 *
 *     W  =  [ 1     0   ]     b = [ 0    0    1.5 ]
 *           [ 0     1   ]
 *           [ 0.5   0.5 ]                    W shape [3, 2] = [outputDim, inputDim]
 *
 *     y  =  [ 1   2   3 ]                    ← Ch 12's logits, exactly
 *
 * With Ch 12's loss on top (truth "sat") the gradients are:
 *
 *     b.grad  =  [ -0.909969   0.244728   0.665241 ]        = p − y
 *     W.grad  =  [ -0.909969  -1.819938 ]                   row i = (p−y)ᵢ · x
 *                [  0.244728   0.489456 ]
 *                [  0.665241   1.330482 ]
 *
 * Every number machine-verified. The tests check these exact values.
 *
 * ──────────────────────────────────────────────────────────────────────────
 * THE ONE CONVENTION TO GET RIGHT — W is [outputDim, inputDim]
 * ──────────────────────────────────────────────────────────────────────────
 * Row i of W is everything output unit i knows: its weight per input. This
 * is PyTorch's convention, and every published checkpoint stores W this way.
 * The price is one transpose in forward (x is [batch, inputDim], so the
 * multiply needs Wᵀ [inputDim, outputDim]). The price is worth paying once,
 * here, so Ch 22 can load real attention weights without re-indexing.
 *
 * Beware the silent version of getting it backwards: with inputDim ==
 * outputDim, a flipped W still runs and still trains — just worse. The
 * tests use 2→3 precisely so a flip is a loud shape error.
 */
import { TensorValue } from "../autograd/grad.ts";

/**
 * A single fully-connected linear transformation.
 *
 * Weight W has shape [outputDim, inputDim] (PyTorch convention).
 * Forward: y = x @ Wᵀ + b
 *   x shape: [*, inputDim]  →  y shape: [*, outputDim]
 *
 * This is the Q, K, V projection layer and every FFN sub-layer
 * in the transformer.
 */
export class Linear {
  readonly weight: TensorValue;
  readonly bias: TensorValue | null;
  readonly inputDim: number;
  readonly outputDim: number;

  /**
   * ── WHAT THE CONSTRUCTOR DECIDES (doc section 6) ──────────────────────────
   * The starting weights, and the famous wrong answer is zeros: every unit
   * then computes 0, receives the identical gradient, takes the identical
   * step, and stays a copy of its siblings forever — Ch 11's exercise E10(a)
   * measured it (loss parked at 0.2500, weights still exactly zero after
   * 3000 steps). The randomness is load-bearing.
   *
   * Random, then — but scaled. Each output sums inputDim products, and a sum
   * of n independent terms grows by √n, so divide by √n to cancel it (the
   * doc's section 6 measurement: raw randn explodes ×10/layer through
   * width-100 layers, ×0.01 vanishes ÷10/layer, √(1/n) holds steady at 1).
   *
   *     "xavier"   randn([outputDim, inputDim]) × √(1/inputDim)
   *     "he"       randn([outputDim, inputDim]) × √(2/inputDim)   ← DEFAULT
   *                (the ×2 compensates relu zeroing half the signal — Ch 11)
   *     "normal"   randn([outputDim, inputDim]) × 0.02            (GPT-2 style)
   *
   * ── STEPS ─────────────────────────────────────────────────────────────────
   *   1. randn from tensor/creation.ts (Box-Muller — already yours, Ch 02),
   *      shape [outputDim, inputDim]. Scale with mulScalar on the RAW tensor,
   *      then wrap ONCE in a TensorValue — same move as mseLoss's constant,
   *      except this wrap matters: it is the leaf the optimizer will update.
   *   2. bias: zeros([outputDim]) wrapped as a TensorValue — or null when
   *      bias === false. Zero is SAFE for the bias: the weights being random
   *      already breaks the symmetry, and zero just means "no initial
   *      opinion". Shape [outputDim], not [1, outputDim] — Ch 10's
   *      broadcasting stretches it across any batch.
   *   3. Store inputDim / outputDim; default bias to true, init to "he".
   *
   * ── PITFALL: √(2/inputDim), not √(2/outputDim) ────────────────────────────
   * The sum being tamed runs over the INPUTS. With a square layer the mixup
   * is invisible; the variance test uses uneven dims to catch it.
   *
   * @param init  "he"     — Kaiming init (before ReLU/GELU layers)
   *              "xavier" — Xavier init   (output projections)
   *              "normal" — small N(0, 0.02) (GPT-2 style)
   */
  constructor(
    inputDim: number,
    outputDim: number,
    bias?: boolean,
    init?: "he" | "xavier" | "normal"
  ) {
    throw new Error("Linear constructor not implemented");
  }

  /**
   * y = x @ Wᵀ + b
   *
   * ── COMPOSE IT (graph methods only — this is a forward pass) ──────────────
   * One line: matMul against the TRANSPOSED weight, then add the bias.
   * Both transpose and add must be the TensorValue METHODS:
   *
   *   - transpose inside the graph is what delivers W.grad in W's own
   *     [outputDim, inputDim] shape — the backward of transpose un-flips it.
   *     Transposing .data by hand and wrapping fresh would sever W from the
   *     graph: the exact bug crossEntropyFromLogits' first draft had, and
   *     the layer would never train.
   *   - add broadcasts bias [outputDim] across [batch, outputDim], and its
   *     backward (Ch 10's sumToShape) collects each unit's bias gradient by
   *     summing over the batch. You wrote that machinery; here it earns out.
   *
   * ── WORKED TRACE — the running example ────────────────────────────────────
   *
   *     x [1,2] @ Wᵀ [2,3]  →  [1,3]:
   *
   *     y_sat  = 1·1   + 2·0     + 0    = 1
   *     y_ran  = 1·0   + 2·1     + 0    = 2
   *     y_flew = 1·0.5 + 2·0.5   + 1.5  = 3
   *
   *     y = [ 1   2   3 ]     shape [1, 3] ✓
   *
   * Backward (with Ch 12's loss, truth "sat") — nothing new to write, but
   * know what to expect:
   *
   *     b.grad = p − y (Ch 12's gradient, arriving at the biases untouched)
   *     W.grad row i = (p − y)ᵢ · x   (blame = unit's error × the input it
   *                                    was multiplying)
   *     x.grad also exists — unused here, but it is how blame reaches the
   *     layer BELOW when layers stack (Ch 25).
   *
   * ── PITFALL: null bias ────────────────────────────────────────────────────
   * When bias is null, skip the add entirely. Adding a zeros tensor instead
   * would be numerically identical but builds a dead graph node per forward
   * — and its gradient accumulates into a tensor no optimizer will ever see.
   */
  forward(x: TensorValue): TensorValue {
    throw new Error("Linear.forward not implemented");
  }

  /**
   * Return [weight, bias] (or [weight] when bias=false).
   *
   * ── THE CONTRACT (doc section 5) ──────────────────────────────────────────
   * Every layer hands over its trainable tensors as a flat list; the
   * optimizer walks the list and updates each one. Neither side knows
   * anything else about the other. This one method is how Ch 09's step()
   * scales to GPT without changing.
   *
   * Two rules the tests enforce:
   *   - the SAME objects, not copies: parameters()[0] === this.weight must
   *     hold, or the optimizer faithfully updates tensors nothing reads.
   *   - no hole when bias is off: [weight], length 1 — never [weight, null].
   */
  parameters(): TensorValue[] {
    throw new Error("Linear.parameters not implemented");
  }
}
