/**
 * autograd/value.ts
 * ══════════════════════════════════════════════════════════
 * Scalar Value node for automatic differentiation.
 *
 * During the forward pass each operation creates a new Value that records
 * its _inputs and a _backward closure.  Calling .backward() on the output
 * propagates gradients all the way back to the leaf parameters.
 *
 * Chapters: 08a (forward), 08b (backward)
 * Doc:      docs/part-2-autodiff/ch-08a-autograd-forward.md
 *           docs/part-2-autodiff/ch-08b-autograd-backward.md
 */

/**
 * A single node in the scalar computation graph.
 *
 * Every arithmetic op creates a new Value whose _backward function knows
 * how to push gradients back to its _inputs.
 */
export class Value {
  /** The scalar number this node holds. */
  data: number;
  /** Accumulated gradient ∂L/∂this. Starts at 0; the root is set to 1. */
  grad: number;
  /** Input nodes that produced this value. */
  _inputs: Value[];
  /** Operation label for debugging, e.g. "+", "*", "tanh". */
  _op: string;
  /** Runs the local backward step and accumulates into _inputs' grads. */
  _backward: () => void;

  constructor(data: number, _inputs?: Value[], _op?: string) {
    throw new Error("Value constructor not implemented");
  }

  /** Forward: this + other */
  add(other: Value): Value {
    throw new Error("Value.add not implemented");
  }

  /** Forward: this * other */
  mul(other: Value): Value {
    throw new Error("Value.mul not implemented");
  }

  /** Forward: this ^ exponent */
  pow(exponent: number): Value {
    throw new Error("Value.pow not implemented");
  }

  /** Forward: e^this */
  exp(): Value {
    throw new Error("Value.exp not implemented");
  }

  /** Forward: ln(this) */
  log(): Value {
    throw new Error("Value.log not implemented");
  }

  /** Forward: tanh(this) */
  tanh(): Value {
    throw new Error("Value.tanh not implemented");
  }

  /** Forward: max(0, this) */
  relu(): Value {
    throw new Error("Value.relu not implemented");
  }

  /**
   * Run reverse-mode autodiff from this node.
   * Sets this.grad = 1, then visits all nodes in reverse topological order
   * calling _backward to accumulate gradients into leaf parameters.
   *
   * Chapter 08b
   */
  backward(): void {
    throw new Error("Value.backward not implemented");
  }

  /** Reset grad to 0.  Call before each new forward/backward pass. */
  zeroGrad(): void {
    throw new Error("Value.zeroGrad not implemented");
  }

  toString(): string {
    return `Value(data=${this.data.toFixed(4)}, grad=${this.grad.toFixed(4)})`;
  }
}
