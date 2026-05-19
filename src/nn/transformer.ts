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
