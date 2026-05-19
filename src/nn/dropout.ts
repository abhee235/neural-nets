/**
 * nn/dropout.ts
 * ══════════════════════════════════════════════════════════
 * Inverted dropout regularisation.
 * Equivalent to torch.nn.Dropout.
 *
 * Chapter: 20 — LayerNorm & Dropout
 * Doc:     docs/part-4-tokenizer-and-inputs/ch-20-layernorm-dropout.md
 */
import { TensorValue } from "../autograd/grad.ts";

/**
 * Training:  zero random elements with probability rate,
 *            scale survivors by 1/(1−rate) to preserve expected value.
 * Eval:      identity — no dropout applied.
 *
 * Inverted scaling means inference code needs no special handling.
 */
export class Dropout {
  readonly rate: number;
  private _training: boolean;

  constructor(rate: number) {
    throw new Error("Dropout constructor not implemented");
  }

  forward(x: TensorValue): TensorValue {
    throw new Error("Dropout.forward not implemented");
  }

  /** Switch to training mode — dropout is active. */
  train(): void {
    throw new Error("Dropout.train not implemented");
  }

  /** Switch to eval mode — dropout is bypassed. */
  eval(): void {
    throw new Error("Dropout.eval not implemented");
  }
}
