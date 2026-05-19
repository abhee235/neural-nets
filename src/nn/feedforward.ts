/**
 * nn/feedforward.ts
 * ══════════════════════════════════════════════════════════
 * Position-wise Feed-Forward Network — the second sublayer in every
 * transformer block.
 *
 * Chapter: 25 — Feed-Forward Block
 * Doc:     docs/part-6-transformer/ch-25-feedforward-block.md
 */
import { TensorValue } from "../autograd/grad.ts";
import { Linear } from "./linear.ts";

/**
 * Two-layer MLP with GELU activation and a 4× hidden expansion.
 *
 *   FFN(x) = linear2( GELU( linear1(x) ) )
 *
 *   linear1: dModel → dFf       (default dFf = 4 × dModel)
 *   linear2: dFf    → dModel
 *
 * Accounts for ~⅔ of all parameters in a transformer.
 * x shape: [batch, seq, dModel] → same.
 */
export class FFN {
  readonly linear1: Linear;
  readonly linear2: Linear;
  readonly dModel: number;
  readonly dFf: number;

  constructor(dModel: number, dFf?: number) {
    throw new Error("FFN constructor not implemented");
  }

  forward(x: TensorValue): TensorValue {
    throw new Error("FFN.forward not implemented");
  }

  parameters(): TensorValue[] {
    throw new Error("FFN.parameters not implemented");
  }
}
