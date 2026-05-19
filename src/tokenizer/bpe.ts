/**
 * tokenizer/bpe.ts
 * ══════════════════════════════════════════════════════════
 * Byte-Pair Encoding tokenizer.
 * Iteratively merges the most frequent adjacent symbol pair until
 * the vocabulary reaches the target size.  GPT-2 uses BPE with ~50k merges.
 *
 * Chapter: 17 — BPE Tokenizer
 * Doc:     docs/part-4-tokenizer-and-inputs/ch-17-bpe-tokenizer.md
 */

/**
 * Count adjacent symbol pairs across all tokenized sequences.
 *   [["h","e","l","l","o"]] → Map{ "h e": 1, "e l": 1, "l l": 1, "l o": 1 }
 */
export function countPairs(corpus: string[][]): Map<string, number> {
  throw new Error("countPairs not implemented");
}

/**
 * Replace every occurrence of (a, b) with the merged symbol "a+b".
 */
export function mergePair(
  corpus: string[][],
  pair: [string, string]
): string[][] {
  throw new Error("mergePair not implemented");
}

/**
 * BPE tokenizer.  Train on a corpus, then use encode/decode.
 */
export class BPETokenizer {
  /** Learned merge rules in order of application. */
  readonly merges: Array<[string, string]>;
  readonly stoi: Map<string, number>;
  readonly itos: Map<number, string>;
  readonly vocabSize: number;

  constructor() {
    throw new Error("BPETokenizer constructor not implemented");
  }

  /**
   * Run BPE training: start with character vocab, merge the most-frequent
   * adjacent pair until vocabulary reaches vocabSize.
   */
  train(text: string, vocabSize: number): void {
    throw new Error("BPETokenizer.train not implemented");
  }

  /** Apply learned merges to tokenize text into subword IDs. */
  encode(text: string): number[] {
    throw new Error("BPETokenizer.encode not implemented");
  }

  /** Convert subword IDs back to text. */
  decode(ids: number[]): string {
    throw new Error("BPETokenizer.decode not implemented");
  }
}
