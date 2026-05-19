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
