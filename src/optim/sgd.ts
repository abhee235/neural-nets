/**
 * optim/sgd.ts
 * ══════════════════════════════════════════════════════════
 * Stochastic Gradient Descent (plain and with momentum).
 * Equivalent to torch.optim.SGD.
 *
 * Chapters: 09 (gradient descent), 14 (optimizers)
 * Doc:      docs/part-2-autodiff/ch-09-gradient-descent.md
 */
import type { Value } from "../autograd/value.ts";

/**
 * Vanilla SGD.
 * Update rule:  param.data -= learningRate * param.grad
 */
export class SGD {
  readonly learningRate: number;

  constructor(learningRate: number) {
    throw new Error("SGD constructor not implemented");
  }

  /** Apply one gradient step to all parameters. */
  step(params: Value[]): void {
    throw new Error("SGD.step not implemented");
  }

  /** Zero all parameter gradients before the next backward pass. */
  zeroGrad(params: Value[]): void {
    throw new Error("SGD.zeroGrad not implemented");
  }
}

/**
 * SGD with momentum.
 * Update rule:  v = momentum*v - lr*grad
 *               param.data += v
 *
 * Momentum dampens oscillations and accelerates convergence.
 */
export class SGDMomentum {
  readonly learningRate: number;
  readonly momentum: number;

  constructor(learningRate: number, momentum?: number) {
    throw new Error("SGDMomentum constructor not implemented — default momentum=0.9");
  }

  step(params: Value[]): void {
    throw new Error("SGDMomentum.step not implemented");
  }

  zeroGrad(params: Value[]): void {
    throw new Error("SGDMomentum.zeroGrad not implemented");
  }
}
