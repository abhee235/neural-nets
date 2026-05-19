/**
 * nn/linear.ts
 * ══════════════════════════════════════════════════════════
 * Fully-connected linear layer: y = x Wᵀ + b.
 * Equivalent to torch.nn.Linear.
 *
 * Chapter: 13 — Linear Layer
 * Doc:     docs/part-3-neural-net-primitives/ch-13-linear-layer.md
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

  /** y = x @ Wᵀ + b */
  forward(x: TensorValue): TensorValue {
    throw new Error("Linear.forward not implemented");
  }

  /** Return [weight, bias] (or [weight] when bias=false). */
  parameters(): TensorValue[] {
    throw new Error("Linear.parameters not implemented");
  }
}
