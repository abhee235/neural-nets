/**
 * CHAPTER 17: The BPE Tokenizer
 * ════════════════════════════════════════
 * Part 4 of 6: Language Model Inputs
 *
 * WHAT WE'RE BUILDING:  countPairs, mergePair, BPETokenizer — a tokenizer that
 *                       learns its own vocabulary from a corpus instead of
 *                       being told what the tokens are.
 * WHY IT MATTERS:       Ch 16's character tokenizer emits one token per
 *                       character, and attention in Part 5 costs the SQUARE of
 *                       the sequence length. Fewer tokens is quadratically
 *                       cheaper. BPE also spells out words it has never seen,
 *                       instead of losing them to <unk>.
 * WHAT THIS UNLOCKS:    → Ch 18 (Token Embeddings) — one embedding row per
 *                       vocabulary entry, so vocabSize sets the model's size.
 *
 * REFERENCE: docs/part-4-tokenizer-and-inputs/ch-17-bpe-tokenizer.md
 *
 * ── THE ALGORITHM IN FIVE LINES ─────────────────────────────────────────────
 *
 *     1.  split the corpus into words, and each word into characters
 *     2.  count every adjacent pair across the whole corpus
 *     3.  take the pair with the highest count
 *     4.  replace that pair everywhere, as one new token; write the rule down
 *     5.  go back to 2, until the vocabulary is big enough
 *
 * No gradients, no optimisation. It is counting, in a loop.
 *
 * ── THE RUNNING CORPUS ──────────────────────────────────────────────────────
 * Every trace in this file uses the chapter's corpus:
 *
 *     "low low low low low lower lower newest newest newest widest widest"
 *
 * which is  low ×5   lower ×2   newest ×3   widest ×2.
 *
 * Trained to vocabSize 23, it learns exactly these eight rules, in this order:
 *
 *     1  ("l","o")   → lo        5  ("n","e")     → ne
 *     2  ("lo","w")  → low       6  ("ne","w")    → new
 *     3  ("e","s")   → es        7  ("new","est") → newest
 *     4  ("es","t")  → est       8  ("low","e")   → lowe
 */
import { UNK_ID, SPECIAL_TOKEN_COUNT } from "./char.ts";

/**
 * A corpus is a list of WORDS, and each word is a list of tokens.
 * Two levels, never one flat list — see countPairs for why that matters.
 *
 *     "low lower"   →   [ ["l","o","w"], ["l","o","w","e","r"] ]
 */
export type Corpus = string[][];

/** One learned merge: the two tokens that get joined, in order. */
export type MergeRule = [string, string];

/**
 * Count every adjacent pair of tokens across the whole corpus.
 *
 * ── WORKED TRACE ────────────────────────────────────────────────────────────
 *
 *     countPairs([ ["l","o","w"], ["l","o","w","e","r"] ])
 *
 *       word 1   l o w          pairs:  (l o)  (o w)
 *       word 2   l o w e r      pairs:  (l o)  (o w)  (w e)  (e r)
 *                                       ─────  ─────
 *                                       seen twice, across two words
 *
 *                               → Map { "l o" → 2, "o w" → 2,
 *                                       "w e" → 1, "e r" → 1 }
 *
 * ── WHY A LIST OF WORDS, NOT ONE FLAT LIST ──────────────────────────────────
 * A pair must never straddle a space. If the corpus above were flattened to
 * one array, this loop would also count the pair (w l) — the "w" ending the
 * first word next to the "l" starting the second. That pair is an accident of
 * two words sitting side by side and means nothing anywhere else.
 *
 * Taking Corpus rather than string[] is what prevents it: a loop over one
 * word's array cannot see past that array's end. The type does the work.
 *
 * ── THE KEY ─────────────────────────────────────────────────────────────────
 * A Map cannot be keyed by an array — two arrays with the same contents are
 * different objects — so the pair has to become a string. Use a space:
 *
 *     `${a} ${b}`         "l o"   "lo w"   "new est"
 *
 * and it can be split straight back with .split(" "). That is unambiguous
 * here because no token ever contains a space: the corpus was split ON
 * spaces, so a space cannot be inside a word, so it cannot be inside a token
 * built from a word. (Be aware this is the reason it is safe — the same trick
 * would be a bug in a tokenizer whose tokens could hold spaces.)
 *
 * ── WHAT TO WATCH FOR ───────────────────────────────────────────────────────
 * The loop bound is `i < seq.length - 1`, not `seq.length`. A word of n tokens
 * has n-1 adjacent pairs. A one-token word contributes none, and the loop must
 * survive that without reading past the end.
 */
export function countPairs(corpus: Corpus): Map<string, number> {
  throw new Error("Not implemented — read the chapter doc first");
}

/**
 * Replace every occurrence of one pair with the single joined token.
 *
 * ── WORKED TRACE ────────────────────────────────────────────────────────────
 *
 *     mergePair([ ["l","o","w"], ["l","o","w","e","r"] ], ["l","o"])
 *
 *       l o w        →   lo w
 *       l o w e r    →   lo w e r
 *
 * The merge happens in EVERY word at once. That is what makes the next call to
 * countPairs see a different corpus — round 2 of training counts (lo w), a
 * pair that did not exist before this call.
 *
 * ── THE BUG EVERYONE WRITES ─────────────────────────────────────────────────
 * After matching a pair at position i, the loop must step past BOTH tokens.
 * Step past only one and the second half gets emitted again:
 *
 *     mergePair on "banana" with ("a","n")
 *
 *       correct   b | an | an | a        rejoins to "banana"     ✓
 *       buggy     b | an | n | an | n | a    rejoins to "bannanna"   ✗
 *
 * Nothing throws. The corpus quietly stops being the corpus, and every count
 * after this point is wrong. Advance i by two on a match.
 *
 * ── PURE FUNCTION ───────────────────────────────────────────────────────────
 * Return a new corpus; do not mutate the one passed in. Training calls this in
 * a loop and reassigns, and tests check the input is untouched.
 */
export function mergePair(corpus: Corpus, pair: MergeRule): Corpus {
  throw new Error("Not implemented — read the chapter doc first");
}

/**
 * A tokenizer whose vocabulary is learned from text rather than declared.
 *
 * Unlike CharTokenizer, this one is built empty and then trained:
 *
 *     const tokenizer = new BPETokenizer();
 *     tokenizer.train(text, 23);
 *
 * The four fields below are filled in by train(), which is why none of them is
 * `readonly` — there is nothing to freeze at construction time.
 */
export class BPETokenizer {
  /**
   * The learned merge rules, IN THE ORDER THEY WERE LEARNED.
   * The order is not bookkeeping — encode replays them in this order and gets
   * a different (worse) answer in any other. See encodeWord.
   */
  merges: MergeRule[] = [];

  /** token string → ID. Same idea as CharTokenizer's, with subwords in it. */
  stoi = new Map<string, number>();

  /** ID → token string, for decode. */
  itos = new Map<number, string>();

  /**
   * Starts at SPECIAL_TOKEN_COUNT, not 0 — IDs 0 to 3 are already spent on
   * <pad>, <unk>, <bos> and <eos>, exactly as in Ch 16. Giving a real token
   * ID 0 would make it indistinguishable from padding.
   */
  vocabSize = SPECIAL_TOKEN_COUNT;

  /**
   * Learn the vocabulary and the merge rules from `text`.
   *
   * ── THE VOCABULARY IS BUILT IN THREE LAYERS ─────────────────────────────
   *
   *      0 – 3     the four specials             <pad> <unk> <bos> <eos>
   *      4 – 14    every character in the text   ' ' d e i l n o r s t w
   *     15 – 22    one entry per merge, in order lo low es est ne new
   *                                              newest lowe
   *                                              ──
   *                                     vocabSize 23
   *
   * The middle layer is exactly Ch 16's vocabulary. BPE does not replace the
   * character tokenizer — it starts from it and stacks tokens on top. Every
   * character stays in the vocabulary forever, which is what guarantees any
   * word can be spelled out even when no merge applies to it.
   *
   * Note the space is in that middle layer, at ID 4. It is a normal token; it
   * simply never merges with anything, because merges only happen inside a
   * word and the corpus was split on spaces.
   *
   * ── THE LOOP ────────────────────────────────────────────────────────────
   *
   *     while (vocabSize < target)
   *       count pairs in the corpus
   *       stop if no pair occurs more than once
   *       merge the winner everywhere, record the rule, add the new token
   *
   * ── TWO STOPPING CONDITIONS, AND BOTH ARE NEEDED ────────────────────────
   * The obvious one is reaching `vocabSize`. The other is running out of
   * repeated pairs: once every remaining pair occurs exactly once, any further
   * merge invents a token that will be used exactly once — noise in the
   * vocabulary and a wasted embedding row in Ch 18.
   *
   *     train(corpus, 40)   →  stops after 12 merges, vocabSize 27
   *
   * That is correct behaviour, not a failure. `vocabSize` is a ceiling, not a
   * promise, and the caller must read back the real size rather than assume.
   *
   * ── TIES MUST BREAK THE SAME WAY EVERY TIME ─────────────────────────────
   * Round 1 on this corpus is a tie:
   *
   *     (l o) = 7        (o w) = 7
   *
   * Something has to choose. Whatever it is, the SAME corpus must produce the
   * SAME rules on every run — otherwise two training runs give two different
   * vocabularies, and a model trained against one reads the other's output as
   * nonsense. This is the same hazard Ch 16 avoided by sorting characters
   * before numbering them.
   *
   * A Map iterates in insertion order, so scanning its entries and keeping a
   * strict improvement (`count > best`, not `>=`) takes the pair first seen
   * scanning the corpus left to right. That is deterministic. Just make sure
   * it is deterministic on purpose rather than by luck.
   */
  train(text: string, vocabSize: number): void {
    throw new Error("Not implemented — read the chapter doc first");
  }

  /**
   * Apply every merge rule, in order, to ONE word.
   *
   * ── WORKED TRACE: "lowest" ──────────────────────────────────────────────
   * Every rule is tried once, in order. Most find nothing, and that is fine.
   *
   *     start            l | o | w | e | s | t
   *     ("l","o")     →  lo | w | e | s | t
   *     ("lo","w")    →  low | e | s | t
   *     ("e","s")     →  low | es | t
   *     ("es","t")    →  low | est
   *     ("n","e")        low | est          nothing to do
   *     ("ne","w")       low | est          nothing to do
   *     ("new","est")    low | est          nothing to do
   *     ("low","e")      low | est          nothing to do
   *                                            → 2 tokens, was 6 characters
   *
   * ── WHY THE ORDER IS THE ALGORITHM ──────────────────────────────────────
   * Run the SAME eight rules from last to first and the answer changes:
   *
   *     in training order    low | est               2 tokens
   *     reversed             lo | w | es | t         4 tokens
   *
   * ("lo","w") can only fire on a sequence that already contains "lo", and the
   * only thing that produces "lo" is ("l","o"). Reversed, ("lo","w") is tried
   * while there is still no "lo" anywhere, so it does nothing; by the time
   * "lo" appears on the last line, the rule that wanted it is already spent.
   *
   * A merge rule consumes what an earlier rule produced. The rules are a chain,
   * not a set.
   *
   * ── UNSEEN WORDS STILL WORK ─────────────────────────────────────────────
   *     encodeWord("slowest")   →   s | low | est
   *
   * "slowest" is nowhere in the training corpus. A word-level tokenizer would
   * emit <unk> and lose it. Here it is spelled out of pieces that already
   * exist, and decode gets it back exactly. This is the whole reason Ch 17
   * follows Ch 16.
   */
  encodeWord(word: string): string[] {
    throw new Error("Not implemented — read the chapter doc first");
  }

  /**
   * Text → token IDs.
   *
   * ── WORKED TRACE ────────────────────────────────────────────────────────
   *
   *     encode("low lower")
   *       word "low"      →  low                →  [16]
   *       the space       →  ' '                →  [4]
   *       word "lower"    →  lowe | r           →  [22, 11]
   *                                             →  [16, 4, 22, 11]
   *
   * Split on the space, run encodeWord on each word, and emit the space token
   * BETWEEN consecutive words. That last part is what lets decode put the text
   * back together — without it the flat list of IDs has no record of where one
   * word ended and the next began.
   *
   * ── AWKWARD SPACING STILL ROUND-TRIPS ───────────────────────────────────
   * "low  lower" has two spaces, so splitting gives three words with an EMPTY
   * one in the middle. The empty word contributes no tokens, but two space
   * tokens are still emitted between the three, and the text comes back
   * exactly. Do not filter empty words out — they are load-bearing.
   *
   * ── A CHARACTER THE CORPUS NEVER HAD ────────────────────────────────────
   * Fall back to UNK_ID on a lookup miss, exactly as Ch 16 does. It is rarer
   * here but not gone; see decode.
   */
  encode(text: string): number[] {
    throw new Error("Not implemented — read the chapter doc first");
  }

  /**
   * Token IDs → text. Specials are skipped, everything else is concatenated.
   *
   * ── WORKED TRACE ────────────────────────────────────────────────────────
   *
   *     decode([16, 4, 22, 11])
   *       16 → "low"
   *        4 → " "
   *       22 → "lowe"
   *       11 → "r"
   *                                             → "low lower"
   *
   * Plain concatenation, with no separator inserted. The space is already a
   * token in the list, so adding one here would double every gap.
   *
   * ── THE ROUND TRIP IS BETTER THAN CH 16'S, BUT NOT PERFECT ──────────────
   *
   *     decode(encode("low lower"))   === "low lower"     ✓
   *     decode(encode("low  lower"))  === "low  lower"    ✓  spacing survives
   *     decode(encode("slowest"))     === "slowest"       ✓  never trained on
   *     decode(encode("the lowest"))  === "te lowest"     ✗  the 'h' is gone
   *
   * That last line is the honest limit. The corpus — low, lower, newest,
   * widest — contains no 'h' at all, so 'h' is not in the base vocabulary, so
   * it encodes to UNK_ID and is lost, exactly as in Ch 16.
   *
   * So: BPE removes <unk> for unseen WORDS, not for unseen CHARACTERS. An
   * unseen word can be decomposed; an unseen character has nothing to
   * decompose into.
   *
   * Real systems close that last gap by making the base vocabulary BYTES
   * instead of characters. There are only 256 possible bytes, all of them go
   * in the vocabulary up front, and every possible text is a sequence of
   * bytes — so <unk> becomes impossible rather than merely rare. That is the
   * "byte" in byte-pair encoding, and what GPT-2 does. This file works on
   * characters, which is the same algorithm with a smaller alphabet.
   */
  decode(ids: number[]): string {
    throw new Error("Not implemented — read the chapter doc first");
  }
}
