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
