/**
 * Tests for tokenizer/masks.ts
 * Chapter 21 — Attention Mask Cookbook
 *
 * Run: bun test src/tokenizer/masks.test.ts
 */
import { describe, it, expect } from "bun:test";
// import { binaryMaskFromIds, toAdditiveMask, expandPaddingMask, causalMask, combineMasks } from "./masks.ts";

describe("binaryMaskFromIds", () => {
  it.todo("pad tokens produce 0");
  it.todo("non-pad tokens produce 1");
});

describe("toAdditiveMask", () => {
  it.todo("converts 1 to 0 (logit unchanged)");
  it.todo("converts 0 to −1e9 (position blocked)");
});

describe("expandPaddingMask", () => {
  it.todo("reshapes [batch, keyLen] to [batch, 1, 1, keyLen]");
});

describe("causalMask", () => {
  it.todo("upper triangle is −1e9 (future positions blocked)");
  it.todo("lower triangle and diagonal are 0 (past and present allowed)");
  it.todo("shape is [queryLen, keyLen]");
});

describe("combineMasks", () => {
  it.todo("combined mask blocks positions blocked by either input mask");
  it.todo("decoder mask blocks both future tokens and padding");
});
