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
