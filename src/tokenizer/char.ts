/**
 * tokenizer/char.ts
 * ══════════════════════════════════════════════════════════
 * Character-level tokenizer with special tokens (PAD, UNK, BOS, EOS)
 * and batch encoding with padding + binary attention masks.
 *
 * Chapter: 16 — Character-Level Tokenizer
 * Doc:     docs/part-4-tokenizer-and-inputs/ch-16-char-tokenizer.md
 */
import type { Tensor } from "../tensor/index.ts";

/** Special token IDs reserved at the start of every vocabulary. */
export const PAD_ID = 0;
export const UNK_ID = 1;
export const BOS_ID = 2;
export const EOS_ID = 3;
export const SPECIAL_TOKEN_COUNT = 4;

/**
 * Scan text, collect unique characters, and assign integer IDs.
 * IDs 0–3 are reserved for special tokens.
 */
export function buildVocab(
  text: string
): { stoi: Map<string, number>; itos: Map<number, string>; vocabSize: number } {
  throw new Error("buildVocab not implemented");
}

/**
 * Character-level tokenizer.
 *
 * Usage:
 *   const tok = new CharTokenizer("hello world");
 *   tok.encode("hello")           // → [6, 3, 8, 8, 9]
 *   tok.decode([6, 3, 8, 8, 9])   // → "hello"
 */
export class CharTokenizer {
  readonly stoi: Map<string, number>;
  readonly itos: Map<number, string>;
  readonly vocabSize: number;

  constructor(text: string) {
    throw new Error("CharTokenizer constructor not implemented");
  }

  /** String → integer IDs.  Unknown chars → UNK_ID.  Optional truncation. */
  encode(text: string, maxLen?: number): number[] {
    throw new Error("CharTokenizer.encode not implemented");
  }

  /** Integer IDs → string.  Special tokens are skipped. */
  decode(ids: number[]): string {
    throw new Error("CharTokenizer.decode not implemented");
  }

  /**
   * Encode a batch, padding all sequences to maxLen.
   * Returns:
   *   ids  — Tensor [batch, maxLen]  integer token IDs
   *   mask — Tensor [batch, maxLen]  binary mask (1=real token, 0=pad)
   */
  encodeBatch(
    texts: string[],
    maxLen: number
  ): { ids: Tensor; mask: Tensor } {
    throw new Error("CharTokenizer.encodeBatch not implemented");
  }
}
