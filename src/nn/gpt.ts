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
