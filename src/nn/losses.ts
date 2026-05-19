/**
 * nn/losses.ts
 * ══════════════════════════════════════════════════════════
 * Loss functions.
 * Equivalent to torch.nn.MSELoss and torch.nn.CrossEntropyLoss.
 *
 * Chapter: 12 — Loss Functions
 * Doc:     docs/part-3-neural-net-primitives/ch-12-loss-functions.md
 */
import type { Tensor } from "../tensor/index.ts";
import { TensorValue } from "../autograd/grad.ts";

/**
 * Mean Squared Error: mean((predictions − targets)²).
 */
export function mseLoss(predictions: TensorValue, targets: Tensor): TensorValue {
  throw new Error("mseLoss not implemented");
}

/**
 * Numerically stable log-sum-exp:
 *   LSE(x) = max(x) + log( Σ exp(x_i − max(x)) )
 *
 * Used inside crossEntropyFromLogits.
 */
export function logSumExp(x: TensorValue, axis?: number): TensorValue {
  throw new Error("logSumExp not implemented");
}

/**
 * Cross-entropy from raw logits (no separate softmax step needed):
 *   L = −logit[correct_class] + logSumExp(logits)
 *
 * Gradient shortcut:  ∂L/∂logits = softmax(logits) − one_hot(targets)
 *
 * This is the training objective for every language model in this course.
 */
export function crossEntropyFromLogits(
  logits: TensorValue,
  targets: Tensor
): TensorValue {
  throw new Error("crossEntropyFromLogits not implemented");
}
