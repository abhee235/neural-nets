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
 */
import type { Tensor } from "../tensor/index.ts";

/**
 * A tensor-valued node in the autograd graph.
 * .data and .grad are both Tensors with the same shape.
 */
export class TensorValue {
  data: Tensor;
  /** Accumulated gradient — same shape as data. null until backward runs. */
  grad: Tensor | null;
  _inputs: TensorValue[];
  _backward: () => void;

  constructor(data: Tensor) {
    throw new Error("TensorValue constructor not implemented");
  }

  /** Elementwise add with broadcast-aware backward. */
  add(other: TensorValue): TensorValue {
    throw new Error("TensorValue.add not implemented");
  }

  /** Elementwise multiply with broadcast-aware backward. */
  mul(other: TensorValue): TensorValue {
    throw new Error("TensorValue.mul not implemented");
  }

  /**
   * Matrix multiply.
   * Backward: dA = dZ @ Bᵀ,  dB = Aᵀ @ dZ
   */
  matMul(other: TensorValue): TensorValue {
    throw new Error("TensorValue.matMul not implemented");
  }

  /**
   * Sum reduction. Backward broadcasts upstream gradient back to input shape.
   */
  sum(axis?: number, keepDims?: boolean): TensorValue {
    throw new Error("TensorValue.sum not implemented");
  }

  /**
   * Mean reduction. Backward distributes grad / n to each input element.
   */
  mean(axis?: number, keepDims?: boolean): TensorValue {
    throw new Error("TensorValue.mean not implemented");
  }

  /** Reshape forward; backward reshapes grad to original shape. */
  reshape(newShape: number[]): TensorValue {
    throw new Error("TensorValue.reshape not implemented");
  }

  /** Transpose forward; backward applies the inverse permutation. */
  transpose(axes?: number[]): TensorValue {
    throw new Error("TensorValue.transpose not implemented");
  }

  /** Run reverse-mode autodiff from this tensor node. */
  backward(): void {
    throw new Error("TensorValue.backward not implemented");
  }

  /** Reset grad to null. */
  zeroGrad(): void {
    throw new Error("TensorValue.zeroGrad not implemented");
  }
}

/**
 * Reverse broadcasting: sum gradient over axes that were broadcast.
 *
 * When forward computed a→b (a smaller → b larger via broadcast),
 * backward must sum b's gradient back down to a's original shape.
 */
export function sumToShape(grad: Tensor, targetShape: number[]): Tensor {
  throw new Error("sumToShape not implemented");
}

/**
 * Numerical gradient check for tensor-valued operations.
 * Uses finite differences to verify the analytical backward pass.
 */
export function checkTensorGradient(
  fn: (inputs: TensorValue[]) => TensorValue,
  inputs: TensorValue[],
  tolerance?: number
): boolean {
  throw new Error("checkTensorGradient not implemented");
}
