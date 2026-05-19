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
