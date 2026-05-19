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
