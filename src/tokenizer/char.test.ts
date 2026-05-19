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
