#!/usr/bin/env python3
"""
scaffold_library.py
===================
Replaces the old chapter-per-folder src/ structure with a proper
PyTorch-style library layout:

  src/
    tensor/       - Tensor type, creation, ops, linalg, reduce, math
    autograd/     - Value node, backward engine, tensor gradients
    nn/           - activations, losses, linear, embeddings, attention,
                    transformer, GPT
    optim/        - SGD, Adam
    tokenizer/    - CharTokenizer, BPETokenizer, masks
    utils/        - numerical gradient check, dataset helpers
    index.ts      - main entry point (re-exports everything)

Usage: python3 scripts/scaffold_library.py
"""

import os, shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.join(ROOT, "src")

# ── 1. Remove old chapter folders ────────────────────────────────────────────
removed = 0
if os.path.exists(SRC):
    for name in os.listdir(SRC):
        if name.startswith("ch-"):
            shutil.rmtree(os.path.join(SRC, name))
            removed += 1
print(f"Removed {removed} old ch-* folders")

# ── 2. File definitions ───────────────────────────────────────────────────────
# Each entry: (path_relative_to_src, file_content)
FILES: list[tuple[str, str]] = []

# =============================================================================
# src/tensor/
# =============================================================================

FILES.append(("tensor/types.ts", """\
/**
 * tensor/types.ts
 * ══════════════════════════════════════════════════════════
 * The core Tensor type: a flat Float64Array + shape descriptor.
 * Equivalent to numpy.ndarray's raw layout.
 *
 * Chapter: 01 — Tensor Type System
 * Doc:     docs/part-1-tensor-library/ch-01-tensor-type-system.md
 */

// ─── Core type ────────────────────────────────────────────────────────────────

/**
 * A multi-dimensional array stored in row-major (C-contiguous) flat layout.
 *
 * Example — 2×3 matrix:
 *   shape = [2, 3]
 *   data  = Float64Array [1,2,3, 4,5,6]
 *   element[i,j] lives at data[i * 3 + j]
 */
export interface Tensor {
  /** Flat storage, row-major order. */
  readonly data: Float64Array;
  /** Size of each dimension, e.g. [batch, seq, dModel]. */
  readonly shape: number[];
  /** Number of dimensions — shape.length. */
  readonly ndim: number;
  /** Total element count — product of all shape values. */
  readonly size: number;
}

// ─── Construction ────────────────────────────────────────────────────────────

/**
 * Create a Tensor from a flat number array and a shape.
 * Validates data.length === product(shape).
 */
export function createTensor(data: number[], shape: number[]): Tensor {
  throw new Error("createTensor not implemented — docs/part-1-tensor-library/ch-01-tensor-type-system.md");
}

/**
 * Create a rank-0 scalar Tensor (shape = [], size = 1).
 */
export function scalar(value: number): Tensor {
  throw new Error("scalar not implemented");
}

// ─── Utilities ────────────────────────────────────────────────────────────────

/**
 * Runtime type guard — narrows unknown to Tensor.
 */
export function isTensor(value: unknown): value is Tensor {
  throw new Error("isTensor not implemented");
}

/**
 * Convert a multi-dimensional index array to the flat offset into data[].
 * Row-major formula:  offset = Σ_i  indices[i] * stride[i]
 */
export function flatIndex(shape: number[], indices: number[]): number {
  throw new Error("flatIndex not implemented");
}

/**
 * Human-readable representation showing shape and sampled values.
 * Use this instead of console.log(tensor) for debugging.
 */
export function toString(t: Tensor): string {
  throw new Error("toString not implemented");
}
"""))

FILES.append(("tensor/types.test.ts", """\
/**
 * Tests for tensor/types.ts
 * Chapter 01 — Tensor Type System
 *
 * Run: bun test src/tensor/types.test.ts
 */
import { describe, it, expect } from "bun:test";
// import { createTensor, scalar, isTensor, flatIndex } from "./types.ts";

describe("createTensor", () => {
  it.todo("stores data flat in the same order as the input array");
  it.todo("records shape exactly as given");
  it.todo("size equals the product of all shape dimensions");
  it.todo("ndim equals shape.length");
  it.todo("throws when data.length does not equal the shape product");
});

describe("scalar", () => {
  it.todo("creates a tensor with shape [] (rank 0)");
  it.todo("stores exactly one element");
});

describe("flatIndex", () => {
  it.todo("[0,0] maps to offset 0 for any shape");
  it.todo("row-major: [i,j] maps to i*cols + j for a 2-D tensor");
  it.todo("throws when an index is out of bounds");
});

describe("isTensor", () => {
  it.todo("returns true for a valid Tensor object");
  it.todo("returns false for a plain number");
  it.todo("returns false for null and undefined");
});
"""))

FILES.append(("tensor/creation.ts", """\
/**
 * tensor/creation.ts
 * ══════════════════════════════════════════════════════════
 * Factory functions that produce Tensors filled with specific values.
 * Equivalent to numpy.zeros, numpy.ones, numpy.random.randn, numpy.eye.
 *
 * Chapter: 02 — Tensor Creation
 * Doc:     docs/part-1-tensor-library/ch-02-tensor-creation.md
 */
import type { Tensor } from "./types.ts";

/** Tensor filled with 0.0. Used to initialise bias vectors and masks. */
export function zeros(shape: number[]): Tensor {
  throw new Error("zeros not implemented");
}

/** Tensor filled with 1.0. Used for LayerNorm gamma initialisation. */
export function ones(shape: number[]): Tensor {
  throw new Error("ones not implemented");
}

/** Tensor filled with a constant value. */
export function fill(shape: number[], value: number): Tensor {
  throw new Error("fill not implemented");
}

/**
 * Tensor of N(0,1) samples using Box-Muller transform.
 * Do NOT use Math.random() scaled — implement Box-Muller explicitly:
 *   Z = sqrt(-2 ln U1) * cos(2π U2),  where U1, U2 ~ Uniform(0,1)
 *
 * Used to initialise weight matrices in Linear and Embedding layers.
 */
export function randn(shape: number[]): Tensor {
  throw new Error("randn not implemented — use Box-Muller, not Math.random() scaling");
}

/** n×n identity matrix. 1 on diagonal, 0 elsewhere. */
export function eye(n: number): Tensor {
  throw new Error("eye not implemented");
}

/**
 * 1-D tensor [start, start+step, ...] up to (not including) stop.
 * Like numpy.arange.
 */
export function arange(start: number, stop: number, step?: number): Tensor {
  throw new Error("arange not implemented");
}

/**
 * 1-D tensor of n evenly-spaced values from start to stop (inclusive).
 * Like numpy.linspace.
 */
export function linspace(start: number, stop: number, n: number): Tensor {
  throw new Error("linspace not implemented");
}

/** New tensor with the same shape as t, filled with value. */
export function fullLike(t: Tensor, value: number): Tensor {
  throw new Error("fullLike not implemented");
}
"""))

FILES.append(("tensor/creation.test.ts", """\
/**
 * Tests for tensor/creation.ts
 * Chapter 02 — Tensor Creation
 *
 * Run: bun test src/tensor/creation.test.ts
 */
import { describe, it, expect } from "bun:test";
// import { zeros, ones, fill, randn, eye, arange, linspace } from "./creation.ts";

const EPSILON = 1e-6;
const close = (a: number, b: number) => Math.abs(a - b) < EPSILON;

describe("zeros", () => {
  it.todo("every element is 0.0");
  it.todo("shape matches the argument");
});

describe("ones", () => {
  it.todo("every element is 1.0");
});

describe("randn", () => {
  it.todo("output shape matches the requested shape");
  it.todo("large sample mean ≈ 0 (Box-Muller property)");
  it.todo("large sample standard deviation ≈ 1 (Box-Muller property)");
  it.todo("values are not all identical — genuine randomness");
});

describe("eye", () => {
  it.todo("diagonal elements are 1.0");
  it.todo("off-diagonal elements are 0.0");
  it.todo("shape is [n, n]");
});

describe("arange", () => {
  it.todo("arange(0, 5, 1) produces [0, 1, 2, 3, 4]");
  it.todo("step defaults to 1");
});

describe("linspace", () => {
  it.todo("linspace(0, 1, 5) produces [0, 0.25, 0.5, 0.75, 1.0]");
  it.todo("first element === start, last element === stop");
});
"""))

FILES.append(("tensor/ops.ts", """\
/**
 * tensor/ops.ts
 * ══════════════════════════════════════════════════════════
 * Elementwise arithmetic and NumPy-style broadcasting.
 * Equivalent to torch's +, -, *, / tensor operators.
 *
 * Chapter: 03 — Elementwise Ops & Broadcasting
 * Doc:     docs/part-1-tensor-library/ch-03-elementwise-ops-broadcasting.md
 */
import type { Tensor } from "./types.ts";

/**
 * Compute the output shape when broadcasting a and b (NumPy rules).
 * Align right; each dim must be equal or one of them is 1.
 *   broadcastShapes([3,1], [1,4]) → [3,4]
 */
export function broadcastShapes(a: number[], b: number[]): number[] {
  throw new Error("broadcastShapes not implemented");
}

/**
 * Expand t to the target shape by repeating along size-1 dimensions.
 */
export function broadcast(t: Tensor, shape: number[]): Tensor {
  throw new Error("broadcast not implemented");
}

/** Elementwise a + b with broadcasting. */
export function add(a: Tensor, b: Tensor): Tensor {
  throw new Error("add not implemented");
}

/** Elementwise a - b with broadcasting. */
export function sub(a: Tensor, b: Tensor): Tensor {
  throw new Error("sub not implemented");
}

/** Elementwise a * b with broadcasting. */
export function mul(a: Tensor, b: Tensor): Tensor {
  throw new Error("mul not implemented");
}

/** Elementwise a / b with broadcasting. */
export function div(a: Tensor, b: Tensor): Tensor {
  throw new Error("div not implemented");
}

/** Add scalar s to every element. */
export function addScalar(t: Tensor, s: number): Tensor {
  throw new Error("addScalar not implemented");
}

/** Multiply every element by scalar s. */
export function mulScalar(t: Tensor, s: number): Tensor {
  throw new Error("mulScalar not implemented");
}

/**
 * Apply function fn to every element.
 * The building block for all activation functions and math primitives.
 */
export function applyFn(t: Tensor, fn: (x: number) => number): Tensor {
  throw new Error("applyFn not implemented");
}
"""))

FILES.append(("tensor/ops.test.ts", """\
/**
 * Tests for tensor/ops.ts
 * Chapter 03 — Elementwise Ops & Broadcasting
 *
 * Run: bun test src/tensor/ops.test.ts
 */
import { describe, it, expect } from "bun:test";
// import { add, sub, mul, div, mulScalar, broadcastShapes, broadcast } from "./ops.ts";
// import { createTensor } from "./types.ts";

describe("broadcastShapes", () => {
  it.todo("[3,1] and [1,4] broadcast to [3,4]");
  it.todo("[5] and [3,5] broadcast to [3,5]");
  it.todo("equal shapes broadcast to themselves");
  it.todo("throws for incompatible shapes, e.g. [3] and [4]");
});

describe("add", () => {
  it.todo("add([1,2],[3,4]) produces [4,6]");
  it.todo("add broadcasts [3,1] + [1,4] to shape [3,4]");
});

describe("mul / div", () => {
  it.todo("elementwise multiplication is correct");
  it.todo("div(a, a) produces all-ones for nonzero a");
});

describe("mulScalar", () => {
  it.todo("doubles every element when s=2");
  it.todo("produces all zeros when s=0");
});

describe("applyFn", () => {
  it.todo("applies the function independently to every element");
  it.todo("output shape matches input shape");
});
"""))

FILES.append(("tensor/linalg.ts", """\
/**
 * tensor/linalg.ts
 * ══════════════════════════════════════════════════════════
 * Linear algebra: matmul, transpose, reshape, flatten, squeeze/unsqueeze.
 * These are the structural ops used in every attention and linear layer.
 *
 * Chapter: 04 — Matrix Operations
 * Doc:     docs/part-1-tensor-library/ch-04-matrix-ops.md
 */
import type { Tensor } from "./types.ts";

/**
 * Matrix multiply.
 *   2-D:  (M,K) × (K,N) → (M,N)
 *   N-D:  batched matmul on the last two axes
 *
 * Math: C[i,j] = Σ_k A[i,k] * B[k,j]
 *
 * This is the dominant compute operation in transformers —
 * every Q/K/V projection and FFN layer is a matMul.
 */
export function matMul(a: Tensor, b: Tensor): Tensor {
  throw new Error("matMul not implemented");
}

/**
 * Permute tensor axes. Default: reverse all axes.
 * Pass axes to specify a custom permutation.
 *
 * Used to swap head/seq dims in multi-head attention.
 */
export function transpose(t: Tensor, axes?: number[]): Tensor {
  throw new Error("transpose not implemented");
}

/**
 * Return a view with a new shape. Total element count must be unchanged.
 * Pass -1 for at most one dimension to infer its size automatically.
 */
export function reshape(t: Tensor, newShape: number[]): Tensor {
  throw new Error("reshape not implemented");
}

/** Collapse all axes from startAxis onward into a single dimension. */
export function flatten(t: Tensor, startAxis?: number): Tensor {
  throw new Error("flatten not implemented");
}

/** Remove a size-1 dimension at the given axis. */
export function squeeze(t: Tensor, axis: number): Tensor {
  throw new Error("squeeze not implemented");
}

/**
 * Insert a size-1 dimension at the given axis.
 * Used to go [batch, seq, dModel] → [batch, 1, seq, dModel] for masks.
 */
export function unsqueeze(t: Tensor, axis: number): Tensor {
  throw new Error("unsqueeze not implemented");
}
"""))

FILES.append(("tensor/linalg.test.ts", """\
/**
 * Tests for tensor/linalg.ts
 * Chapter 04 — Matrix Operations
 *
 * Run: bun test src/tensor/linalg.test.ts
 */
import { describe, it, expect } from "bun:test";
// import { matMul, transpose, reshape, flatten } from "./linalg.ts";
// import { createTensor } from "./types.ts";

describe("matMul", () => {
  it.todo("(M×K) × (K×N) produces shape (M×N)");
  it.todo("[[1,2],[3,4]] × [[1],[1]] === [[3],[7]]");
  it.todo("batched matMul (B,M,K) × (B,K,N) produces (B,M,N)");
  it.todo("throws when inner dimensions do not match");
});

describe("transpose", () => {
  it.todo("transpose of (3,4) produces shape (4,3)");
  it.todo("element at [i,j] moves to [j,i]");
  it.todo("double transpose is the identity");
});

describe("reshape", () => {
  it.todo("preserves total element count");
  it.todo("reshape to [-1] flattens to 1-D");
  it.todo("throws when shape has incompatible element count");
});

describe("flatten", () => {
  it.todo("collapses all dimensions into one 1-D tensor");
  it.todo("startAxis=1 keeps the first dim, flattens the rest");
});
"""))

FILES.append(("tensor/reduce.ts", """\
/**
 * tensor/reduce.ts
 * ══════════════════════════════════════════════════════════
 * Axis-wise reductions: sum, mean, max, min, argmax, softmax.
 * Equivalent to torch.sum, torch.mean, torch.softmax, torch.argmax.
 *
 * Chapter: 05 — Reductions & Statistical Ops
 * Doc:     docs/part-1-tensor-library/ch-05-reductions.md
 */
import type { Tensor } from "./types.ts";

/**
 * Sum along an axis (or all elements when axis is omitted).
 * keepDims=true inserts a size-1 axis at the reduced position.
 */
export function sum(t: Tensor, axis?: number, keepDims?: boolean): Tensor {
  throw new Error("sum not implemented");
}

/**
 * Mean along an axis. Used in LayerNorm to compute per-token mean.
 * Math: mean(x) = sum(x) / n
 */
export function mean(t: Tensor, axis?: number, keepDims?: boolean): Tensor {
  throw new Error("mean not implemented");
}

/**
 * Maximum value along an axis.
 * Used in numerically-stable softmax (subtract max before exp).
 */
export function max(t: Tensor, axis?: number, keepDims?: boolean): Tensor {
  throw new Error("max not implemented");
}

/** Minimum value along an axis. */
export function min(t: Tensor, axis?: number, keepDims?: boolean): Tensor {
  throw new Error("min not implemented");
}

/**
 * Index of the maximum value along an axis.
 * Used during greedy decoding: argmax(logits) → predicted token ID.
 */
export function argmax(t: Tensor, axis?: number): Tensor {
  throw new Error("argmax not implemented");
}

/**
 * Numerically stable softmax:
 *   softmax(x)_i = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
 *
 * Output is a probability distribution summing to 1.
 * Final step of every attention head.
 */
export function softmax(t: Tensor, axis?: number): Tensor {
  throw new Error("softmax not implemented");
}
"""))

FILES.append(("tensor/reduce.test.ts", """\
/**
 * Tests for tensor/reduce.ts
 * Chapter 05 — Reductions & Statistical Ops
 *
 * Run: bun test src/tensor/reduce.test.ts
 */
import { describe, it, expect } from "bun:test";
// import { sum, mean, max, argmax, softmax } from "./reduce.ts";
// import { createTensor } from "./types.ts";

const EPSILON = 1e-6;
const close = (a: number, b: number) => Math.abs(a - b) < EPSILON;

describe("sum", () => {
  it.todo("sum of all elements equals the scalar total");
  it.todo("sum along axis=0 of a (3,4) tensor produces shape (4,)");
  it.todo("keepDims=true preserves the reduced axis as size 1");
});

describe("mean", () => {
  it.todo("mean of [1,2,3,4] === 2.5");
  it.todo("mean along axis produces the correct per-row average");
});

describe("softmax", () => {
  it.todo("output sums to 1.0 along the target axis");
  it.todo("shift invariance: softmax(x) === softmax(x + c) for any constant c");
  it.todo("largest logit gets the highest probability");
  it.todo("identical logits produce a uniform distribution");
});

describe("argmax", () => {
  it.todo("returns the index of the maximum element");
  it.todo("argmax along axis=1 produces shape (batch,)");
});
"""))

FILES.append(("tensor/math.ts", """\
/**
 * tensor/math.ts
 * ══════════════════════════════════════════════════════════
 * Elementwise math functions applied to tensors.
 * Equivalent to torch.exp, torch.log, torch.tanh, etc.
 *
 * Chapter: 06 — Math Primitives
 * Doc:     docs/part-1-tensor-library/ch-06-math-primitives.md
 */
import type { Tensor } from "./types.ts";

/** Elementwise e^x. Used in softmax, cross-entropy, and GELU. */
export function exp(t: Tensor): Tensor {
  throw new Error("exp not implemented");
}

/**
 * Elementwise ln(x). Guard against log(0) with a small epsilon clamp.
 * Used in cross-entropy loss.
 */
export function log(t: Tensor): Tensor {
  throw new Error("log not implemented");
}

/**
 * Elementwise √x.
 * Used in LayerNorm (÷ std) and attention scaling (÷ √dHead).
 */
export function sqrt(t: Tensor): Tensor {
  throw new Error("sqrt not implemented");
}

/** Elementwise x^exponent. */
export function pow(t: Tensor, exponent: number): Tensor {
  throw new Error("pow not implemented");
}

/** Elementwise |x|. */
export function abs(t: Tensor): Tensor {
  throw new Error("abs not implemented");
}

/**
 * Clamp every element to [min, max].
 * Used to prevent log(0) and clip activations during training.
 */
export function clip(t: Tensor, min: number, max: number): Tensor {
  throw new Error("clip not implemented");
}

/**
 * Elementwise tanh(x) = (e^x - e^{-x}) / (e^x + e^{-x}).
 * Output in (-1, 1). Used in the GELU approximation.
 */
export function tanh(t: Tensor): Tensor {
  throw new Error("tanh not implemented");
}

/**
 * Elementwise σ(x) = 1 / (1 + e^{-x}).
 * Output in (0, 1). Gradient: σ(x)(1 − σ(x)).
 */
export function sigmoid(t: Tensor): Tensor {
  throw new Error("sigmoid not implemented");
}
"""))

FILES.append(("tensor/math.test.ts", """\
/**
 * Tests for tensor/math.ts
 * Chapter 06 — Math Primitives
 *
 * Run: bun test src/tensor/math.test.ts
 */
import { describe, it, expect } from "bun:test";
// import { exp, log, sqrt, pow, abs, clip, tanh, sigmoid } from "./math.ts";
// import { scalar } from "./types.ts";

const EPSILON = 1e-6;
const close = (a: number, b: number) => Math.abs(a - b) < EPSILON;

describe("exp", () => {
  it.todo("exp(0) === 1 for every element");
  it.todo("exp(1) ≈ 2.71828");
});

describe("log", () => {
  it.todo("log(exp(x)) ≈ x — round-trip identity");
  it.todo("exp(log(x)) ≈ x for positive x");
});

describe("sqrt", () => {
  it.todo("sqrt(4) === 2 elementwise");
  it.todo("sqrt(0) === 0");
});

describe("clip", () => {
  it.todo("values below min become min");
  it.todo("values above max become max");
  it.todo("values within [min, max] are unchanged");
});

describe("tanh", () => {
  it.todo("tanh(0) === 0");
  it.todo("output is in (-1, 1) for any finite input");
});

describe("sigmoid", () => {
  it.todo("sigmoid(0) === 0.5");
  it.todo("output is in (0, 1) for any finite input");
});
"""))

FILES.append(("tensor/index.ts", """\
/**
 * tensor/index.ts — public API for the tensor module.
 *
 * import { Tensor, zeros, matMul, softmax } from "../tensor/index.ts";
 */
export * from "./types.ts";
export * from "./creation.ts";
export * from "./ops.ts";
export * from "./linalg.ts";
export * from "./reduce.ts";
export * from "./math.ts";
"""))

# =============================================================================
# src/autograd/
# =============================================================================

FILES.append(("autograd/value.ts", """\
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
"""))

FILES.append(("autograd/value.test.ts", """\
/**
 * Tests for autograd/value.ts
 * Chapters 08a & 08b — Scalar Autograd
 *
 * Run: bun test src/autograd/value.test.ts
 */
import { describe, it, expect } from "bun:test";
// import { Value } from "./value.ts";

const EPSILON = 1e-5;
const close = (a: number, b: number) => Math.abs(a - b) < EPSILON;

describe("Value — forward pass", () => {
  it.todo("add produces the correct scalar sum");
  it.todo("mul produces the correct scalar product");
  it.todo("_inputs records both operands of a binary op");
  it.todo("chained ops build a graph of depth > 1");
  it.todo("exp(0).data === 1");
  it.todo("tanh(0).data === 0");
});

describe("Value — backward pass", () => {
  it.todo("z = x*y: backward gives dz/dx = y and dz/dy = x");
  it.todo("z = x^2: backward gives dz/dx = 2x");
  it.todo("z = exp(x): backward gives dz/dx = exp(x)");
  it.todo("gradients accumulate correctly through a chain of ops");
  it.todo("zeroGrad resets grad to 0");
});

describe("numerical gradient checks", () => {
  it.todo("add gradient matches finite differences");
  it.todo("mul gradient matches finite differences");
  it.todo("pow gradient matches finite differences");
  it.todo("exp gradient matches finite differences");
  it.todo("tanh gradient matches finite differences");
  it.todo("relu gradient matches finite differences (x ≠ 0)");
});
"""))

FILES.append(("autograd/engine.ts", """\
/**
 * autograd/engine.ts
 * ══════════════════════════════════════════════════════════
 * Topological sort of the scalar computation graph.
 * Called by Value.backward() to determine gradient propagation order.
 *
 * Chapter: 08b — Autograd Backward
 * Doc:     docs/part-2-autodiff/ch-08b-autograd-backward.md
 */
import type { Value } from "./value.ts";

/**
 * Return all nodes reachable from root in topological order
 * (inputs before outputs — ready for reversed gradient accumulation).
 *
 * Algorithm: depth-first post-order traversal + visited set, then reverse.
 */
export function topoSort(root: Value): Value[] {
  throw new Error("topoSort not implemented");
}
"""))

FILES.append(("autograd/grad.ts", """\
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
"""))

FILES.append(("autograd/grad.test.ts", """\
/**
 * Tests for autograd/grad.ts
 * Chapter 10 — Tensor Autograd Bridge
 *
 * Run: bun test src/autograd/grad.test.ts
 */
import { describe, it, expect } from "bun:test";
// import { TensorValue, sumToShape, checkTensorGradient } from "./grad.ts";
// import { zeros, ones, createTensor } from "../tensor/index.ts";

describe("sumToShape", () => {
  it.todo("sums over broadcast axes to recover the original shape");
  it.todo("is a no-op when grad already has the target shape");
  it.todo("[3,4] summed to [1,4] reduces axis 0");
});

describe("TensorValue — forward", () => {
  it.todo("add output shape matches the broadcast shape");
  it.todo("matMul output shape is (M,N) for (M,K) × (K,N)");
  it.todo("sum reduces to a scalar when no axis is given");
});

describe("TensorValue — backward", () => {
  it.todo("add backward: each input receives the upstream gradient");
  it.todo("add backward with broadcast: grad is summed to original shape");
  it.todo("mean backward distributes grad/n to each element");
  it.todo("matMul backward: dA = dZ @ Bᵀ, dB = Aᵀ @ dZ");
  it.todo("reshape backward: grad has the original pre-reshape shape");
  it.todo("transpose backward applies the inverse permutation");
});

describe("checkTensorGradient", () => {
  it.todo("passes for add");
  it.todo("passes for matMul");
  it.todo("passes for mean");
});
"""))

FILES.append(("autograd/index.ts", """\
/**
 * autograd/index.ts — public API for the autograd module.
 *
 * import { Value } from "../autograd/index.ts";
 * import { TensorValue, sumToShape } from "../autograd/index.ts";
 */
export * from "./value.ts";
export * from "./engine.ts";
export * from "./grad.ts";
"""))

# =============================================================================
# src/optim/
# =============================================================================

FILES.append(("optim/sgd.ts", """\
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
"""))

FILES.append(("optim/adam.ts", """\
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
"""))

FILES.append(("optim/sgd.test.ts", """\
/**
 * Tests for optim/sgd.ts
 * Chapters 09 & 14 — Gradient Descent / Optimizers
 *
 * Run: bun test src/optim/sgd.test.ts
 */
import { describe, it, expect } from "bun:test";
// import { SGD, SGDMomentum } from "./sgd.ts";
// import { Value } from "../autograd/value.ts";

describe("SGD", () => {
  it.todo("step moves a parameter in the direction that reduces loss");
  it.todo("on L=(w−3)², SGD converges w toward 3");
  it.todo("zeroGrad resets all parameter gradients to 0");
  it.todo("larger learning rate produces a larger parameter change");
});

describe("SGDMomentum", () => {
  it.todo("velocity accumulates across steps");
  it.todo("momentum=0 is equivalent to vanilla SGD");
  it.todo("converges faster than vanilla SGD on a quadratic");
});
"""))

FILES.append(("optim/adam.test.ts", """\
/**
 * Tests for optim/adam.ts
 * Chapter 14 — Optimizers
 *
 * Run: bun test src/optim/adam.test.ts
 */
import { describe, it, expect } from "bun:test";
// import { Adam } from "./adam.ts";
// import { Value } from "../autograd/value.ts";

describe("Adam", () => {
  it.todo("step reduces loss on a simple quadratic");
  it.todo("bias correction dominates at step t=1");
  it.todo("zeroGrad resets all gradients to 0");
  it.todo("converges faster than plain SGD on L=(w−3)²");
  it.todo("moment estimates are initialised to 0");
});
"""))

FILES.append(("optim/index.ts", """\
/**
 * optim/index.ts — public API for the optimizer module.
 *
 * import { SGD, Adam } from "../optim/index.ts";
 */
export * from "./sgd.ts";
export * from "./adam.ts";
"""))

# =============================================================================
# src/utils/
# =============================================================================

FILES.append(("utils/numerical.ts", """\
/**
 * utils/numerical.ts
 * ══════════════════════════════════════════════════════════
 * Numerical gradient checking via finite differences.
 * Use this to verify every backward pass you implement.
 *
 * Chapter: 07 — Calculus for ML
 * Doc:     docs/part-2-autodiff/ch-07-calculus-for-ml.md
 */
import type { Tensor } from "../tensor/index.ts";

/**
 * Symmetric finite-difference approximation of the scalar derivative:
 *   f′(x) ≈ ( f(x+h) − f(x−h) ) / 2h
 *
 * More accurate than the one-sided formula for smooth functions.
 */
export function numericalGradient(
  fn: (x: number) => number,
  x: number,
  h?: number
): number {
  throw new Error("numericalGradient not implemented");
}

/**
 * Compute the numerical gradient of a scalar-valued function w.r.t.
 * every element of a Tensor, perturbing one element at a time.
 */
export function numericalGradientTensor(
  fn: (t: Tensor) => number,
  t: Tensor,
  h?: number
): Tensor {
  throw new Error("numericalGradientTensor not implemented");
}

/**
 * Return true when |analytical − numerical| < tolerance.
 * Default tolerance: 1e-5.
 */
export function checkGradient(
  analytical: number,
  numerical: number,
  tolerance?: number
): boolean {
  throw new Error("checkGradient not implemented");
}
"""))

FILES.append(("utils/data.ts", """\
/**
 * utils/data.ts
 * ══════════════════════════════════════════════════════════
 * Dataset helpers: sequence batching, shifted targets, loss masking,
 * perplexity, and dataset splitting.
 *
 * Chapter: 28 — Sequence Data & Training Objectives
 * Doc:     docs/part-6-transformer/ch-28-sequence-data-objectives.md
 */
import type { Tensor } from "../tensor/index.ts";
import type { TensorValue } from "../autograd/grad.ts";

/**
 * Build decoder input by prepending BOS and dropping the last token.
 *   targets     = [w1, w2, w3, EOS]
 *   shiftRight  = [BOS, w1, w2, w3]
 */
export function shiftRight(targetIds: Tensor, bosId: number): Tensor {
  throw new Error("shiftRight not implemented");
}

/**
 * Cross-entropy averaged only over real (non-pad) token positions.
 * lossMask: 1 for real tokens, 0 for padding.
 */
export function maskedCrossEntropy(
  logits: TensorValue,
  targetIds: Tensor,
  lossMask: Tensor
): TensorValue {
  throw new Error("maskedCrossEntropy not implemented");
}

/**
 * Exponentiated average loss:  perplexity = e^loss
 * Perfect model → perplexity 1.  Random over V tokens → perplexity V.
 */
export function perplexity(loss: number): number {
  throw new Error("perplexity not implemented");
}

/**
 * Toy reversal dataset: target is the reversed source string.
 * Great for sanity-checking an encoder-decoder transformer.
 */
export function makeReversalDataset(
  count: number,
  minLen: number,
  maxLen: number,
  alphabet?: string
): Array<{ source: string; target: string }> {
  throw new Error("makeReversalDataset not implemented");
}

/**
 * Deterministic train/validation split using a seeded shuffle.
 */
export function splitDataset<T>(
  examples: T[],
  validFraction: number,
  seed?: number
): { train: T[]; valid: T[] } {
  throw new Error("splitDataset not implemented");
}
"""))

FILES.append(("utils/numerical.test.ts", """\
/**
 * Tests for utils/numerical.ts
 * Chapter 07 — Calculus for ML
 *
 * Run: bun test src/utils/numerical.test.ts
 */
import { describe, it, expect } from "bun:test";
// import { numericalGradient, checkGradient } from "./numerical.ts";

describe("numericalGradient", () => {
  it.todo("derivative of x² at x=3 ≈ 6");
  it.todo("derivative of sin(x) at x=0 ≈ 1");
  it.todo("derivative of exp(x) at x=0 ≈ 1");
  it.todo("smaller h gives a more accurate estimate for smooth functions");
});

describe("checkGradient", () => {
  it.todo("returns true when analytical and numerical agree within tolerance");
  it.todo("returns false when gradients differ beyond tolerance");
});
"""))

FILES.append(("utils/data.test.ts", """\
/**
 * Tests for utils/data.ts
 * Chapter 28 — Sequence Data & Training Objectives
 *
 * Run: bun test src/utils/data.test.ts
 */
import { describe, it, expect } from "bun:test";
// import { shiftRight, maskedCrossEntropy, perplexity, makeReversalDataset, splitDataset } from "./data.ts";

describe("shiftRight", () => {
  it.todo("prepends BOS and shifts all tokens right by one position");
  it.todo("output length equals input length");
});

describe("maskedCrossEntropy", () => {
  it.todo("ignores positions where lossMask === 0 (padding)");
  it.todo("equals regular cross-entropy when mask is all ones");
});

describe("perplexity", () => {
  it.todo("perplexity of loss=0 is 1 (perfect predictions)");
  it.todo("perplexity of loss=ln(V) equals V (random baseline)");
});

describe("makeReversalDataset", () => {
  it.todo("target equals the reversed source string");
  it.todo("source length is between minLen and maxLen");
});

describe("splitDataset", () => {
  it.todo("split is deterministic with the same seed");
  it.todo("valid fraction is approximately validFraction");
  it.todo("no example appears in both train and valid sets");
});
"""))

# =============================================================================
# src/tokenizer/
# =============================================================================

FILES.append(("tokenizer/char.ts", """\
/**
 * tokenizer/char.ts
 * ══════════════════════════════════════════════════════════
 * Character-level tokenizer with special tokens (PAD, UNK, BOS, EOS)
 * and batch encoding with padding + binary attention masks.
 *
 * Chapter: 16 — Character-Level Tokenizer
 * Doc:     docs/part-4-tokenizer-and-inputs/ch-16-char-tokenizer.md
 */
import type { Tensor } from "../tensor/index.ts";

/** Special token IDs reserved at the start of every vocabulary. */
export const PAD_ID = 0;
export const UNK_ID = 1;
export const BOS_ID = 2;
export const EOS_ID = 3;
export const SPECIAL_TOKEN_COUNT = 4;

/**
 * Scan text, collect unique characters, and assign integer IDs.
 * IDs 0–3 are reserved for special tokens.
 */
export function buildVocab(
  text: string
): { stoi: Map<string, number>; itos: Map<number, string>; vocabSize: number } {
  throw new Error("buildVocab not implemented");
}

/**
 * Character-level tokenizer.
 *
 * Usage:
 *   const tok = new CharTokenizer("hello world");
 *   tok.encode("hello")           // → [6, 3, 8, 8, 9]
 *   tok.decode([6, 3, 8, 8, 9])   // → "hello"
 */
export class CharTokenizer {
  readonly stoi: Map<string, number>;
  readonly itos: Map<number, string>;
  readonly vocabSize: number;

  constructor(text: string) {
    throw new Error("CharTokenizer constructor not implemented");
  }

  /** String → integer IDs.  Unknown chars → UNK_ID.  Optional truncation. */
  encode(text: string, maxLen?: number): number[] {
    throw new Error("CharTokenizer.encode not implemented");
  }

  /** Integer IDs → string.  Special tokens are skipped. */
  decode(ids: number[]): string {
    throw new Error("CharTokenizer.decode not implemented");
  }

  /**
   * Encode a batch, padding all sequences to maxLen.
   * Returns:
   *   ids  — Tensor [batch, maxLen]  integer token IDs
   *   mask — Tensor [batch, maxLen]  binary mask (1=real token, 0=pad)
   */
  encodeBatch(
    texts: string[],
    maxLen: number
  ): { ids: Tensor; mask: Tensor } {
    throw new Error("CharTokenizer.encodeBatch not implemented");
  }
}
"""))

FILES.append(("tokenizer/bpe.ts", """\
/**
 * tokenizer/bpe.ts
 * ══════════════════════════════════════════════════════════
 * Byte-Pair Encoding tokenizer.
 * Iteratively merges the most frequent adjacent symbol pair until
 * the vocabulary reaches the target size.  GPT-2 uses BPE with ~50k merges.
 *
 * Chapter: 17 — BPE Tokenizer
 * Doc:     docs/part-4-tokenizer-and-inputs/ch-17-bpe-tokenizer.md
 */

/**
 * Count adjacent symbol pairs across all tokenized sequences.
 *   [["h","e","l","l","o"]] → Map{ "h e": 1, "e l": 1, "l l": 1, "l o": 1 }
 */
export function countPairs(corpus: string[][]): Map<string, number> {
  throw new Error("countPairs not implemented");
}

/**
 * Replace every occurrence of (a, b) with the merged symbol "a+b".
 */
export function mergePair(
  corpus: string[][],
  pair: [string, string]
): string[][] {
  throw new Error("mergePair not implemented");
}

/**
 * BPE tokenizer.  Train on a corpus, then use encode/decode.
 */
export class BPETokenizer {
  /** Learned merge rules in order of application. */
  readonly merges: Array<[string, string]>;
  readonly stoi: Map<string, number>;
  readonly itos: Map<number, string>;
  readonly vocabSize: number;

  constructor() {
    throw new Error("BPETokenizer constructor not implemented");
  }

  /**
   * Run BPE training: start with character vocab, merge the most-frequent
   * adjacent pair until vocabulary reaches vocabSize.
   */
  train(text: string, vocabSize: number): void {
    throw new Error("BPETokenizer.train not implemented");
  }

  /** Apply learned merges to tokenize text into subword IDs. */
  encode(text: string): number[] {
    throw new Error("BPETokenizer.encode not implemented");
  }

  /** Convert subword IDs back to text. */
  decode(ids: number[]): string {
    throw new Error("BPETokenizer.decode not implemented");
  }
}
"""))

FILES.append(("tokenizer/masks.ts", """\
/**
 * tokenizer/masks.ts
 * ══════════════════════════════════════════════════════════
 * Attention mask utilities.
 *
 * Key insight: tokenizers produce BINARY masks (1=real, 0=pad), but
 * attention softmax needs ADDITIVE masks (0=attend, -1e9=block).
 * This module bridges the two representations and provides causal masks.
 *
 * Chapter: 21 — Attention Mask Cookbook
 * Doc:     docs/part-4-tokenizer-and-inputs/ch-21-mask-cookbook.md
 */
import type { Tensor } from "../tensor/index.ts";

/** Large negative value that drives a softmax probability to ≈ 0. */
export const MASK_VALUE = -1e9;

/**
 * Binary mask from token IDs: 1 for non-pad, 0 for pad.
 * Input shape: [batch, seq].  Output shape: [batch, seq].
 */
export function binaryMaskFromIds(ids: Tensor, padId: number): Tensor {
  throw new Error("binaryMaskFromIds not implemented");
}

/**
 * Convert 1/0 binary mask → 0/−1e9 additive mask.
 *   1 (real)    → 0      (logit unchanged)
 *   0 (padding) → −1e9   (logit → −∞ before softmax → prob ≈ 0)
 */
export function toAdditiveMask(binaryMask: Tensor): Tensor {
  throw new Error("toAdditiveMask not implemented");
}

/**
 * Reshape [batch, keyLen] → [batch, 1, 1, keyLen] so it broadcasts over
 * attention scores of shape [batch, heads, queryLen, keyLen].
 */
export function expandPaddingMask(mask: Tensor): Tensor {
  throw new Error("expandPaddingMask not implemented");
}

/**
 * Upper-triangular causal mask: position i cannot attend to j > i.
 *   shape:  [queryLen, keyLen]
 *   values: 0 for allowed positions, −1e9 for future positions.
 */
export function causalMask(queryLen: number, keyLen?: number): Tensor {
  throw new Error("causalMask not implemented");
}

/**
 * Expand causal mask [queryLen, keyLen] → [1, 1, queryLen, keyLen].
 */
export function expandCausalMask(mask: Tensor): Tensor {
  throw new Error("expandCausalMask not implemented");
}

/**
 * Sum a list of additive masks.  Any position blocked by any one mask
 * stays at −1e9 after softmax.
 */
export function combineMasks(masks: Tensor[]): Tensor {
  throw new Error("combineMasks not implemented");
}

/**
 * Build the combined decoder self-attention mask:
 *   target padding mask + causal mask
 * Shape: [batch, 1, tgtLen, tgtLen]
 */
export function makeDecoderSelfAttentionMask(
  targetIds: Tensor,
  padId: number
): Tensor {
  throw new Error("makeDecoderSelfAttentionMask not implemented");
}
"""))

FILES.append(("tokenizer/char.test.ts", """\
/**
 * Tests for tokenizer/char.ts
 * Chapter 16 — Character-Level Tokenizer
 *
 * Run: bun test src/tokenizer/char.test.ts
 */
import { describe, it, expect } from "bun:test";
// import { CharTokenizer, buildVocab, PAD_ID, UNK_ID } from "./char.ts";

describe("buildVocab", () => {
  it.todo("assigns a unique ID to each distinct character");
  it.todo("vocabSize = unique chars + SPECIAL_TOKEN_COUNT");
});

describe("CharTokenizer.encode / decode", () => {
  it.todo("encode then decode is a round trip for known characters");
  it.todo("unknown characters map to UNK_ID");
  it.todo("encode with maxLen truncates long strings");
});

describe("CharTokenizer.encodeBatch", () => {
  it.todo("shorter sequences are padded to maxLen with PAD_ID");
  it.todo("attention mask is 1 for real tokens and 0 for padding");
  it.todo("ids tensor shape is [batch, maxLen]");
  it.todo("mask tensor shape matches ids shape");
});
"""))

FILES.append(("tokenizer/bpe.test.ts", """\
/**
 * Tests for tokenizer/bpe.ts
 * Chapter 17 — BPE Tokenizer
 *
 * Run: bun test src/tokenizer/bpe.test.ts
 */
import { describe, it, expect } from "bun:test";
// import { BPETokenizer, countPairs, mergePair } from "./bpe.ts";

describe("countPairs", () => {
  it.todo("most frequent adjacent pair is identified correctly");
  it.todo("pair count equals the number of adjacent occurrences");
});

describe("mergePair", () => {
  it.todo("replaces every occurrence of the pair with the merged token");
  it.todo("non-matching pairs are unchanged");
});

describe("BPETokenizer", () => {
  it.todo("training reduces token count vs character-level");
  it.todo("most frequent pair is merged first");
  it.todo("encode then decode is a round trip");
  it.todo("merged tokens appear in vocabulary after training");
});
"""))

FILES.append(("tokenizer/masks.test.ts", """\
/**
 * Tests for tokenizer/masks.ts
 * Chapter 21 — Attention Mask Cookbook
 *
 * Run: bun test src/tokenizer/masks.test.ts
 */
import { describe, it, expect } from "bun:test";
// import { binaryMaskFromIds, toAdditiveMask, expandPaddingMask, causalMask, combineMasks } from "./masks.ts";

describe("binaryMaskFromIds", () => {
  it.todo("pad tokens produce 0");
  it.todo("non-pad tokens produce 1");
});

describe("toAdditiveMask", () => {
  it.todo("converts 1 to 0 (logit unchanged)");
  it.todo("converts 0 to −1e9 (position blocked)");
});

describe("expandPaddingMask", () => {
  it.todo("reshapes [batch, keyLen] to [batch, 1, 1, keyLen]");
});

describe("causalMask", () => {
  it.todo("upper triangle is −1e9 (future positions blocked)");
  it.todo("lower triangle and diagonal are 0 (past and present allowed)");
  it.todo("shape is [queryLen, keyLen]");
});

describe("combineMasks", () => {
  it.todo("combined mask blocks positions blocked by either input mask");
  it.todo("decoder mask blocks both future tokens and padding");
});
"""))

FILES.append(("tokenizer/index.ts", """\
/**
 * tokenizer/index.ts — public API for the tokenizer module.
 *
 * import { CharTokenizer } from "../tokenizer/index.ts";
 * import { causalMask, makeDecoderSelfAttentionMask } from "../tokenizer/index.ts";
 */
export * from "./char.ts";
export * from "./bpe.ts";
export * from "./masks.ts";
"""))

# =============================================================================
# src/nn/
# =============================================================================

FILES.append(("nn/activations.ts", """\
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
"""))

FILES.append(("nn/losses.ts", """\
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
"""))

FILES.append(("nn/linear.ts", """\
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
"""))

FILES.append(("nn/embedding.ts", """\
/**
 * nn/embedding.ts
 * ══════════════════════════════════════════════════════════
 * Learnable token embedding lookup table.
 * Equivalent to torch.nn.Embedding.
 *
 * Chapter: 18 — Token Embeddings
 * Doc:     docs/part-4-tokenizer-and-inputs/ch-18-token-embeddings.md
 */
import type { Tensor } from "../tensor/index.ts";
import { TensorValue } from "../autograd/grad.ts";

/**
 * Maps integer token IDs to dense dModel-dimensional vectors.
 *
 * Weight table shape: [vocabSize, dModel]
 * Forward: ids [batch, seq] → output [batch, seq, dModel]
 *
 * After training, semantically similar tokens will have similar vectors
 * (King − Man + Woman ≈ Queen).
 */
export class Embedding {
  readonly weight: TensorValue;
  readonly vocabSize: number;
  readonly dModel: number;

  constructor(vocabSize: number, dModel: number) {
    throw new Error("Embedding constructor not implemented");
  }

  /** Look up embedding vectors for each token ID. */
  forward(ids: Tensor): TensorValue {
    throw new Error("Embedding.forward not implemented");
  }

  /** Return [this.weight] — the full table is one parameter. */
  parameters(): TensorValue[] {
    throw new Error("Embedding.parameters not implemented");
  }
}
"""))

FILES.append(("nn/positional.ts", """\
/**
 * nn/positional.ts
 * ══════════════════════════════════════════════════════════
 * Sinusoidal positional encoding added to token embeddings.
 * Defined in §3.5 of 'Attention Is All You Need'.
 *
 * Chapter: 19 — Positional Encoding
 * Doc:     docs/part-4-tokenizer-and-inputs/ch-19-positional-encoding.md
 */
import { TensorValue } from "../autograd/grad.ts";

/**
 * Precomputed sinusoidal PE table.
 *
 *   PE[pos, 2i]   = sin( pos / 10000^(2i / dModel) )
 *   PE[pos, 2i+1] = cos( pos / 10000^(2i / dModel) )
 *
 * Rationale: PE[pos+k] is a linear function of PE[pos], so the model
 * can attend to relative positions by learning linear combinations.
 */
export class PositionalEncoding {
  readonly maxSeqLen: number;
  readonly dModel: number;

  constructor(maxSeqLen: number, dModel: number) {
    throw new Error("PositionalEncoding constructor not implemented");
  }

  /**
   * Add PE to token embeddings.
   * x shape: [batch, seqLen, dModel]
   * Slices the table to [:seqLen, :] and broadcasts over the batch dim.
   */
  forward(x: TensorValue): TensorValue {
    throw new Error("PositionalEncoding.forward not implemented");
  }
}
"""))

FILES.append(("nn/layernorm.ts", """\
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
"""))

FILES.append(("nn/dropout.ts", """\
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
"""))

FILES.append(("nn/attention.ts", """\
/**
 * nn/attention.ts
 * ══════════════════════════════════════════════════════════
 * Self-attention, multi-head attention, and cross-attention.
 * The core mechanism of every transformer model.
 *
 * Chapters: 22, 23, 24
 * Doc:      docs/part-5-attention/ch-22-self-attention.md
 *           docs/part-5-attention/ch-23-multi-head-attention.md
 *           docs/part-5-attention/ch-24-cross-attention.md
 */
import type { Tensor } from "../tensor/index.ts";
import { TensorValue } from "../autograd/grad.ts";
import { Linear } from "./linear.ts";

/**
 * Scaled dot-product attention — the inner loop of every attention head.
 *
 *   Attention(Q,K,V) = softmax( Q Kᵀ / √dHead  +  mask ) V
 *
 * Q shape: [batch, queryLen, dHead]
 * K shape: [batch, keyLen,   dHead]
 * V shape: [batch, keyLen,   dHead]
 * output:  [batch, queryLen, dHead]
 *
 * The √dHead scale prevents dot products from growing so large that
 * softmax saturates and gradients vanish.
 */
export function scaledDotProductAttention(
  Q: TensorValue,
  K: TensorValue,
  V: TensorValue,
  mask?: Tensor
): TensorValue {
  throw new Error("scaledDotProductAttention not implemented");
}

/**
 * Single-head self-attention.
 * Q, K, V are all linear projections of the same input x.
 */
export class SelfAttention {
  readonly Wq: Linear;
  readonly Wk: Linear;
  readonly Wv: Linear;
  readonly dHead: number;

  constructor(dModel: number, dHead: number) {
    throw new Error("SelfAttention constructor not implemented");
  }

  forward(x: TensorValue, mask?: Tensor): TensorValue {
    throw new Error("SelfAttention.forward not implemented");
  }

  parameters(): TensorValue[] {
    throw new Error("SelfAttention.parameters not implemented");
  }
}

/**
 * Multi-head attention.  Runs numHeads independent attention heads in parallel.
 *
 * Architecture:
 *   1. Project x → Q, K, V   [batch, seq, dModel]
 *   2. Split into heads       [batch, numHeads, seq, dHead]
 *   3. Attend per head
 *   4. Merge heads            [batch, seq, dModel]
 *   5. Output projection W_O
 *
 * Multiple heads let the model attend to different relationship types
 * (syntax, semantics, coreference, etc.) simultaneously.
 */
export class MultiHeadAttention {
  readonly Wq: Linear;
  readonly Wk: Linear;
  readonly Wv: Linear;
  readonly Wo: Linear;
  readonly dModel: number;
  readonly numHeads: number;
  readonly dHead: number;

  constructor(dModel: number, numHeads: number) {
    throw new Error(
      "MultiHeadAttention constructor not implemented — dModel must be divisible by numHeads"
    );
  }

  /** [batch, seq, dModel] → [batch, numHeads, seq, dHead] */
  splitHeads(x: TensorValue): TensorValue {
    throw new Error("MultiHeadAttention.splitHeads not implemented");
  }

  /** [batch, numHeads, seq, dHead] → [batch, seq, dModel] */
  mergeHeads(x: TensorValue): TensorValue {
    throw new Error("MultiHeadAttention.mergeHeads not implemented");
  }

  forward(x: TensorValue, mask?: Tensor): TensorValue {
    throw new Error("MultiHeadAttention.forward not implemented");
  }

  parameters(): TensorValue[] {
    throw new Error("MultiHeadAttention.parameters not implemented");
  }
}

/**
 * Cross-attention: Q from the decoder, K and V from the encoder.
 * This is the bridge that lets the decoder 'read' the encoded source.
 *
 * query:    decoder hidden states  [batch, tgtLen, dModel]
 * context:  encoder output         [batch, srcLen, dModel]
 * output:                          [batch, tgtLen, dModel]
 */
export class CrossAttention {
  readonly Wq: Linear;
  readonly Wk: Linear;
  readonly Wv: Linear;
  readonly Wo: Linear;
  readonly dModel: number;
  readonly numHeads: number;
  readonly dHead: number;

  constructor(dModel: number, numHeads: number) {
    throw new Error("CrossAttention constructor not implemented");
  }

  forward(
    query: TensorValue,
    context: TensorValue,
    mask?: Tensor
  ): TensorValue {
    throw new Error("CrossAttention.forward not implemented");
  }

  parameters(): TensorValue[] {
    throw new Error("CrossAttention.parameters not implemented");
  }
}
"""))

FILES.append(("nn/feedforward.ts", """\
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
"""))

FILES.append(("nn/transformer.ts", """\
/**
 * nn/transformer.ts
 * ══════════════════════════════════════════════════════════
 * Encoder block, Decoder block, and the full Encoder-Decoder Transformer.
 * Implements 'Attention Is All You Need' (Vaswani et al., 2017).
 *
 * Chapters: 26 (encoder block), 27 (decoder block), 29 (full transformer)
 * Doc:      docs/part-6-transformer/ch-26-encoder-block.md
 *           docs/part-6-transformer/ch-27-decoder-block.md
 *           docs/part-6-transformer/ch-29-full-transformer.md
 */
import type { Tensor } from "../tensor/index.ts";
import { TensorValue } from "../autograd/grad.ts";
import { MultiHeadAttention, CrossAttention } from "./attention.ts";
import { FFN } from "./feedforward.ts";
import { LayerNorm } from "./layernorm.ts";
import { Dropout } from "./dropout.ts";
import { Embedding } from "./embedding.ts";
import { PositionalEncoding } from "./positional.ts";
import { Linear } from "./linear.ts";

// ─── Encoder ─────────────────────────────────────────────────────────────────

/**
 * One encoder block.  Pre-norm self-attention + pre-norm FFN, both with residual.
 *
 *   x = x + Dropout( MHA( LN(x) ) )
 *   x = x + Dropout( FFN( LN(x) ) )
 */
export class EncoderBlock {
  readonly selfAttn: MultiHeadAttention;
  readonly ffn: FFN;
  readonly norm1: LayerNorm;
  readonly norm2: LayerNorm;
  readonly dropout: Dropout;

  constructor(
    dModel: number,
    numHeads: number,
    dFf: number,
    dropoutRate?: number
  ) {
    throw new Error("EncoderBlock constructor not implemented");
  }

  forward(x: TensorValue, mask?: Tensor): TensorValue {
    throw new Error("EncoderBlock.forward not implemented");
  }

  parameters(): TensorValue[] {
    throw new Error("EncoderBlock.parameters not implemented");
  }
}

/** Stack of N encoder blocks. */
export class Encoder {
  readonly layers: EncoderBlock[];

  constructor(
    numLayers: number,
    dModel: number,
    numHeads: number,
    dFf: number,
    dropoutRate?: number
  ) {
    throw new Error("Encoder constructor not implemented");
  }

  forward(x: TensorValue, mask?: Tensor): TensorValue {
    throw new Error("Encoder.forward not implemented");
  }

  parameters(): TensorValue[] {
    throw new Error("Encoder.parameters not implemented");
  }
}

// ─── Decoder ─────────────────────────────────────────────────────────────────

/**
 * One decoder block.  Masked self-attention + cross-attention + FFN, all pre-norm.
 *
 *   x = x + Dropout( MaskedMHA( LN(x), tgtMask ) )
 *   x = x + Dropout( CrossAttn( LN(x), encoderOut, srcMask ) )
 *   x = x + Dropout( FFN( LN(x) ) )
 */
export class DecoderBlock {
  readonly selfAttn: MultiHeadAttention;
  readonly crossAttn: CrossAttention;
  readonly ffn: FFN;
  readonly norm1: LayerNorm;
  readonly norm2: LayerNorm;
  readonly norm3: LayerNorm;
  readonly dropout: Dropout;

  constructor(
    dModel: number,
    numHeads: number,
    dFf: number,
    dropoutRate?: number
  ) {
    throw new Error("DecoderBlock constructor not implemented");
  }

  forward(
    x: TensorValue,
    encoderOut: TensorValue,
    srcMask?: Tensor,
    tgtMask?: Tensor
  ): TensorValue {
    throw new Error("DecoderBlock.forward not implemented");
  }

  parameters(): TensorValue[] {
    throw new Error("DecoderBlock.parameters not implemented");
  }
}

/** Stack of N decoder blocks. */
export class Decoder {
  readonly layers: DecoderBlock[];

  constructor(
    numLayers: number,
    dModel: number,
    numHeads: number,
    dFf: number,
    dropoutRate?: number
  ) {
    throw new Error("Decoder constructor not implemented");
  }

  forward(
    x: TensorValue,
    encoderOut: TensorValue,
    srcMask?: Tensor,
    tgtMask?: Tensor
  ): TensorValue {
    throw new Error("Decoder.forward not implemented");
  }

  parameters(): TensorValue[] {
    throw new Error("Decoder.parameters not implemented");
  }
}

// ─── Full Transformer ─────────────────────────────────────────────────────────

/** Configuration for the full encoder-decoder transformer. */
export interface TransformerConfig {
  srcVocabSize: number;
  tgtVocabSize: number;
  dModel: number;
  numHeads: number;
  numLayers: number;
  dFf: number;
  maxSeqLen: number;
  dropoutRate?: number;
}

/**
 * Full encoder-decoder Transformer (Vaswani et al., 2017).
 *
 * Forward pipeline:
 *   src → srcEmbed → scale(√dModel) → PE → dropout → Encoder → encoderOut
 *   tgt → tgtEmbed → scale(√dModel) → PE → dropout → Decoder(encoderOut)
 *       → finalNorm → outputProjection → logits [batch, tgtLen, tgtVocabSize]
 */
export class Transformer {
  readonly srcEmbed: Embedding;
  readonly tgtEmbed: Embedding;
  readonly pe: PositionalEncoding;
  readonly encoder: Encoder;
  readonly decoder: Decoder;
  readonly finalNorm: LayerNorm;
  readonly outputProjection: Linear;
  readonly cfg: TransformerConfig;

  constructor(cfg: TransformerConfig) {
    throw new Error("Transformer constructor not implemented");
  }

  encode(src: Tensor, srcMask?: Tensor): TensorValue {
    throw new Error("Transformer.encode not implemented");
  }

  decode(
    tgt: Tensor,
    encoderOut: TensorValue,
    srcMask?: Tensor,
    tgtMask?: Tensor
  ): TensorValue {
    throw new Error("Transformer.decode not implemented");
  }

  /** Returns logits [batch, tgtLen, tgtVocabSize]. */
  forward(src: Tensor, tgt: Tensor, srcMask?: Tensor): TensorValue {
    throw new Error("Transformer.forward not implemented");
  }

  parameters(): TensorValue[] {
    throw new Error("Transformer.parameters not implemented");
  }
}

/**
 * Greedy autoregressive generation.
 * Extends the target one token at a time until EOS or maxLen.
 */
export function generate(
  model: Transformer,
  srcTokens: number[],
  maxLen: number,
  bosId: number,
  eosId: number
): number[] {
  throw new Error("generate not implemented");
}
"""))

FILES.append(("nn/gpt.ts", """\
/**
 * nn/gpt.ts
 * ══════════════════════════════════════════════════════════
 * GPT-style decoder-only language model.
 *
 * No encoder.  No cross-attention.  Just N × (masked self-attention + FFN).
 * This is the architecture behind GPT-2, GPT-3, LLaMA, Mistral, etc.
 *
 * Chapter: 30 — Decoder-Only GPT
 * Doc:     docs/part-6-transformer/ch-30-decoder-only-gpt.md
 */
import type { Tensor } from "../tensor/index.ts";
import { TensorValue } from "../autograd/grad.ts";
import { MultiHeadAttention } from "./attention.ts";
import { FFN } from "./feedforward.ts";
import { LayerNorm } from "./layernorm.ts";
import { Dropout } from "./dropout.ts";
import { Embedding } from "./embedding.ts";
import { PositionalEncoding } from "./positional.ts";
import { Linear } from "./linear.ts";

/** Configuration for the GPT model. */
export interface GPTConfig {
  vocabSize: number;
  blockSize: number;    // maximum context (sequence) length
  dModel: number;
  numHeads: number;
  numLayers: number;
  dFf: number;
  dropoutRate?: number;
}

/**
 * One decoder-only transformer block.
 * Pre-norm masked self-attention + pre-norm FFN, both with residual.
 *
 *   x = x + Dropout( MHA( LN(x), causalMask ) )
 *   x = x + Dropout( FFN( LN(x) ) )
 */
export class DecoderOnlyBlock {
  readonly selfAttn: MultiHeadAttention;
  readonly ffn: FFN;
  readonly norm1: LayerNorm;
  readonly norm2: LayerNorm;
  readonly dropout: Dropout;

  constructor(
    dModel: number,
    numHeads: number,
    dFf: number,
    dropoutRate?: number
  ) {
    throw new Error("DecoderOnlyBlock constructor not implemented");
  }

  forward(x: TensorValue, causalMask: Tensor): TensorValue {
    throw new Error("DecoderOnlyBlock.forward not implemented");
  }

  parameters(): TensorValue[] {
    throw new Error("DecoderOnlyBlock.parameters not implemented");
  }
}

/**
 * GPT language model.
 *
 * Forward pipeline:
 *   inputIds → Embedding + PE → N × DecoderOnlyBlock → LayerNorm → Linear → logits
 *
 * inputIds shape: [batch, seq]
 * output logits:  [batch, seq, vocabSize]
 */
export class GPT {
  readonly embedding: Embedding;
  readonly pe: PositionalEncoding;
  readonly blocks: DecoderOnlyBlock[];
  readonly finalNorm: LayerNorm;
  readonly lmHead: Linear;
  readonly cfg: GPTConfig;

  constructor(cfg: GPTConfig) {
    throw new Error("GPT constructor not implemented");
  }

  forward(inputIds: Tensor): TensorValue {
    throw new Error("GPT.forward not implemented");
  }

  parameters(): TensorValue[] {
    throw new Error("GPT.parameters not implemented");
  }
}

/**
 * Sample random windows of length blockSize for language-modelling training.
 * target[t] = input[t+1]  (next-token prediction objective).
 */
export function makeLanguageModelingBatch(
  tokenIds: number[],
  blockSize: number,
  batchSize: number
): { inputs: Tensor; targets: Tensor } {
  throw new Error("makeLanguageModelingBatch not implemented");
}

/**
 * Autoregressive text generation with optional temperature sampling.
 *   temperature = 0  → greedy (argmax)
 *   temperature < 1  → sharper distribution (more confident)
 *   temperature > 1  → flatter distribution (more random)
 */
export function generateText(
  model: GPT,
  tokenizer: {
    encode: (s: string) => number[];
    decode: (ids: number[]) => string;
  },
  prompt: string,
  maxNewTokens: number,
  temperature?: number
): string {
  throw new Error("generateText not implemented");
}
"""))

# ── nn test files ─────────────────────────────────────────────────────────────

FILES.append(("nn/activations.test.ts", """\
/**
 * Tests for nn/activations.ts
 * Chapter 11 — Activation Functions
 *
 * Run: bun test src/nn/activations.test.ts
 */
import { describe, it, expect } from "bun:test";
// import { relu, gelu, sigmoid, softmax } from "./activations.ts";

describe("relu", () => {
  it.todo("relu(2) === 2 (positive input passes through)");
  it.todo("relu(−2) === 0 (negative input is zeroed)");
  it.todo("numerical gradient check passes (x ≠ 0)");
});

describe("gelu", () => {
  it.todo("gelu(0) ≈ 0");
  it.todo("gelu is smooth near x=0 (no kink unlike ReLU)");
  it.todo("numerical gradient check passes");
});

describe("sigmoid", () => {
  it.todo("sigmoid(0) === 0.5");
  it.todo("output is strictly in (0,1) for any finite input");
  it.todo("numerical gradient check passes");
});

describe("softmax", () => {
  it.todo("output sums to 1.0 along the target axis");
  it.todo("shift invariance: softmax(x) === softmax(x + c)");
  it.todo("numerical gradient check passes");
});
"""))

FILES.append(("nn/losses.test.ts", """\
/**
 * Tests for nn/losses.ts
 * Chapter 12 — Loss Functions
 *
 * Run: bun test src/nn/losses.test.ts
 */
import { describe, it, expect } from "bun:test";
// import { mseLoss, logSumExp, crossEntropyFromLogits } from "./losses.ts";

describe("mseLoss", () => {
  it.todo("loss is 0 when predictions exactly equal targets");
  it.todo("gradient points toward the targets");
  it.todo("numerical gradient check passes");
});

describe("logSumExp", () => {
  it.todo("shift invariance: logSumExp(x) === logSumExp(x+c) + c");
  it.todo("numerical gradient check passes");
});

describe("crossEntropyFromLogits", () => {
  it.todo("lower loss when the correct class has a higher logit");
  it.todo("gradient is softmax(logits) − one_hot(targets)");
  it.todo("numerical gradient check passes");
});
"""))

FILES.append(("nn/linear.test.ts", """\
/**
 * Tests for nn/linear.ts
 * Chapter 13 — Linear Layer
 *
 * Run: bun test src/nn/linear.test.ts
 */
import { describe, it, expect } from "bun:test";
// import { Linear } from "./linear.ts";

describe("Linear", () => {
  it.todo("forward output shape is [*, outputDim]");
  it.todo("forward with bias=true adds a learnable offset");
  it.todo("forward with bias=false has no bias parameter");
  it.todo("parameters() returns weight and bias");
  it.todo("He init: weight variance ≈ 2/inputDim for large layers");
  it.todo("numerical gradient check passes w.r.t. weight and bias");
});
"""))

FILES.append(("nn/attention.test.ts", """\
/**
 * Tests for nn/attention.ts
 * Chapters 22, 23, 24 — Attention Mechanisms
 *
 * Run: bun test src/nn/attention.test.ts
 */
import { describe, it, expect } from "bun:test";
// import { scaledDotProductAttention, MultiHeadAttention, CrossAttention } from "./attention.ts";

describe("scaledDotProductAttention", () => {
  it.todo("output shape is [batch, queryLen, dHead]");
  it.todo("attention weights sum to 1.0 along the key dimension");
  it.todo("scaling by 1/√dHead reduces logit magnitude");
  it.todo("causal mask prevents position i from attending to j > i");
  it.todo("numerical gradient check passes for Q, K, V");
});

describe("MultiHeadAttention", () => {
  it.todo("forward output shape is [batch, seq, dModel]");
  it.todo("splitHeads produces [batch, numHeads, seq, dHead]");
  it.todo("mergeHeads is the inverse of splitHeads");
  it.todo("dModel must be divisible by numHeads");
});

describe("CrossAttention", () => {
  it.todo("forward output shape matches query shape [batch, tgtLen, dModel]");
  it.todo("K and V are derived from context, not from query");
  it.todo("source padding mask blocks encoder pad positions");
});
"""))

FILES.append(("nn/transformer.test.ts", """\
/**
 * Tests for nn/transformer.ts
 * Chapters 26, 27, 29 — Encoder, Decoder, Full Transformer
 *
 * Run: bun test src/nn/transformer.test.ts
 */
import { describe, it, expect } from "bun:test";
// import { EncoderBlock, Encoder, DecoderBlock, Decoder, Transformer } from "./transformer.ts";

describe("EncoderBlock", () => {
  it.todo("output shape is [batch, seq, dModel]");
  it.todo("residual connection preserves input information");
  it.todo("pre-norm is applied before attention and before FFN");
});

describe("DecoderBlock", () => {
  it.todo("output shape is [batch, tgtLen, dModel]");
  it.todo("causal mask prevents attending to future target tokens");
  it.todo("cross-attention Q comes from decoder, K/V from encoder");
});

describe("Transformer", () => {
  it.todo("forward output shape is [batch, tgtLen, tgtVocabSize]");
  it.todo("encoder output shape is [batch, srcLen, dModel]");
  it.todo("model can overfit a single reversal example");
  it.todo("generate produces EOS for a trained model");
});
"""))

FILES.append(("nn/gpt.test.ts", """\
/**
 * Tests for nn/gpt.ts
 * Chapter 30 — Decoder-Only GPT
 *
 * Run: bun test src/nn/gpt.test.ts
 */
import { describe, it, expect } from "bun:test";
// import { GPT, makeLanguageModelingBatch, generateText } from "./gpt.ts";

describe("DecoderOnlyBlock", () => {
  it.todo("output shape is [batch, seq, dModel]");
  it.todo("causal mask is applied to self-attention");
});

describe("GPT", () => {
  it.todo("forward output shape is [batch, seq, vocabSize]");
  it.todo("model can overfit a tiny repeated string to near-zero loss");
  it.todo("greedy generation is deterministic for the same prompt");
});

describe("makeLanguageModelingBatch", () => {
  it.todo("target equals input shifted right by one token");
  it.todo("output shape is [batchSize, blockSize]");
});

describe("generateText", () => {
  it.todo("temperature=0 gives deterministic greedy output");
  it.todo("temperature < 1 concentrates probability on likely tokens");
});
"""))

FILES.append(("nn/index.ts", """\
/**
 * nn/index.ts — public API for the neural network module.
 *
 * import { Linear, MultiHeadAttention, Transformer, GPT } from "../nn/index.ts";
 */
export * from "./activations.ts";
export * from "./losses.ts";
export * from "./linear.ts";
export * from "./embedding.ts";
export * from "./positional.ts";
export * from "./layernorm.ts";
export * from "./dropout.ts";
export * from "./attention.ts";
export * from "./feedforward.ts";
export * from "./transformer.ts";
export * from "./gpt.ts";
"""))

# =============================================================================
# src/index.ts — main library entry point
# =============================================================================

FILES.append(("index.ts", """\
/**
 * neural-nets — A transformer library built from scratch in TypeScript.
 * ══════════════════════════════════════════════════════════════════════
 * Zero npm dependencies.  Every function you import was written by you.
 *
 * Import by module (recommended for large files):
 *
 *   import { Tensor, zeros, matMul }     from "./tensor/index.ts";
 *   import { TensorValue }               from "./autograd/index.ts";
 *   import { Linear, MultiHeadAttention, GPT } from "./nn/index.ts";
 *   import { Adam }                      from "./optim/index.ts";
 *   import { CharTokenizer, causalMask } from "./tokenizer/index.ts";
 *
 * Or import everything from here:
 *
 *   import { zeros, Linear, Adam, GPT, CharTokenizer } from "./index.ts";
 *
 * Implementation order follows the 30-chapter curriculum in docs/.
 */
export * from "./tensor/index.ts";
export * from "./autograd/index.ts";
export * from "./nn/index.ts";
export * from "./optim/index.ts";
export * from "./tokenizer/index.ts";
export * from "./utils/numerical.ts";
export * from "./utils/data.ts";
"""))

# ── 3. Write all files ────────────────────────────────────────────────────────
created = 0
for rel_path, content in FILES:
    full_path = os.path.join(SRC, rel_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w") as f:
        f.write(content)
    created += 1

print(f"Created {created} files")
print()
print("Library structure:")
for dirpath, dirnames, filenames in os.walk(SRC):
    dirnames.sort()
    level = dirpath.replace(SRC, "").count(os.sep)
    indent = "  " * level
    folder = os.path.basename(dirpath)
    print(f"{indent}{folder}/")
    sub = "  " * (level + 1)
    for fname in sorted(filenames):
        print(f"{sub}{fname}")

print()
print("Run:  bun test   (all tests should show as todo/pending, 0 failures)")
