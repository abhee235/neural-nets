/**
 * optim/adam.ts
 * ══════════════════════════════════════════════════════════
 * Adam — the standard optimizer for training transformers.
 * Equivalent to torch.optim.Adam.
 *
 * Chapter: 14 — Optimizers
 * Doc:     docs/part-3-neural-net-primitives/ch-14-optimizers.md
 *
 * Update rule (per parameter, per step t):
 *   m  = β₁ m  + (1−β₁) g          ← first moment  (mean)
 *   v  = β₂ v  + (1−β₂) g²         ← second moment (variance)
 *   m̂  = m / (1−β₁ᵗ)               ← bias correction
 *   v̂  = v / (1−β₂ᵗ)
 *   θ -= α * m̂ / (√v̂ + ε)
 */
import type { Value } from "../autograd/value.ts";

export class Adam {
  readonly learningRate: number;
  readonly beta1: number;
  readonly beta2: number;
  readonly epsilon: number;

  constructor(
    learningRate?: number,
    beta1?: number,
    beta2?: number,
    epsilon?: number
  ) {
    throw new Error(
      "Adam constructor not implemented — defaults: lr=1e-3, β1=0.9, β2=0.999, ε=1e-8"
    );
  }

  /** Apply one Adam step to all parameters. */
  step(params: Value[]): void {
    throw new Error("Adam.step not implemented");
  }

  /** Zero gradients on all parameters. */
  zeroGrad(params: Value[]): void {
    throw new Error("Adam.zeroGrad not implemented");
  }
}
