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
