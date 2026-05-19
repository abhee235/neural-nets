/**
 * nn/layernorm.ts
 * ══════════════════════════════════════════════════════════
 * Layer normalisation (pre-norm variant used in GPT-2 and most modern
 * transformer implementations).
 * Equivalent to torch.nn.LayerNorm.
 *
 * Chapter: 20 — LayerNorm & Dropout
 * Doc:     docs/part-4-tokenizer-and-inputs/ch-20-layernorm-dropout.md
 */
import { TensorValue } from "../autograd/grad.ts";

/**
 * Normalise over the last (dModel) dimension for each token.
 *
 *   xHat = (x − mean(x)) / sqrt(var(x) + eps)
 *   y    = gamma * xHat + beta
 *
 * gamma: learnable per-feature scale (initialised to 1)
 * beta:  learnable per-feature shift (initialised to 0)
 *
 * x shape: [batch, seq, dModel] → y shape: same
 */
export class LayerNorm {
  readonly gamma: TensorValue;
  readonly beta: TensorValue;
  readonly dModel: number;
  readonly eps: number;

  constructor(dModel: number, eps?: number) {
    throw new Error("LayerNorm constructor not implemented — default eps=1e-5");
  }

  forward(x: TensorValue): TensorValue {
    throw new Error("LayerNorm.forward not implemented");
  }

  parameters(): TensorValue[] {
    throw new Error("LayerNorm.parameters not implemented");
  }
}
