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
