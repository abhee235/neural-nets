/**
 * nn/activations.ts
 * ══════════════════════════════════════════════════════════
 * Differentiable activation functions operating on TensorValues.
 * Equivalent to torch.nn.functional.relu, gelu, sigmoid, softmax.
 *
 * Chapter: 11 — Activation Functions
 * Doc:     docs/part-3-neural-net-primitives/ch-11-activation-functions.md
 */
import { TensorValue } from "../autograd/grad.ts";

/**
 * ReLU: max(0, x).
 * Gradient: 1 for x > 0, 0 for x ≤ 0.
 */
export function relu(x: TensorValue): TensorValue {
  throw new Error("relu not implemented");
}

/**
 * GELU — Gaussian Error Linear Unit.
 * Approximation: 0.5 x (1 + tanh(√(2/π)(x + 0.044715 x³)))
 *
 * Used inside every FFN block in GPT-2 and most modern transformers.
 * Smoother than ReLU near x = 0.
 */
export function gelu(x: TensorValue): TensorValue {
  throw new Error("gelu not implemented");
}

/**
 * Sigmoid: σ(x) = 1 / (1 + e^{−x}).
 * Output in (0, 1).  Gradient: σ(x)(1 − σ(x)).
 */
export function sigmoid(x: TensorValue): TensorValue {
  throw new Error("sigmoid not implemented");
}

/**
 * Numerically stable softmax along axis (default: last axis).
 * Output sums to 1.0 — a probability distribution.
 */
export function softmax(x: TensorValue, axis?: number): TensorValue {
  throw new Error("softmax not implemented");
}
