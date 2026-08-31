/**
 * CHAPTER 16: The Character Tokenizer
 * ════════════════════════════════════════
 * Part 4 of 6: Language Model Inputs
 *
 * WHAT WE'RE BUILDING:  buildVocab + class CharTokenizer — text becomes
 *                       integer IDs, and a batch of ragged sentences becomes
 *                       a rectangular tensor plus a mask saying which cells
 *                       are real.
 * WHY IT MATTERS:       every input from here to Ch 30 arrives through this
 *                       door. Ch 18's Embedding looks up the IDs; Part 5's
 *                       attention multiplies by the mask.
 * WHAT THIS UNLOCKS:    → Ch 17 (BPE Tokenizer) — same interface, a smarter
 *                       vocabulary learned from the data instead of fixed.
 *
 * REFERENCE: docs/part-4-tokenizer-and-inputs/ch-16-char-tokenizer.md
 *
 * ──────────────────────────────────────────────────────────────────────────
 * WHY THIS FILE EXISTS
 * ──────────────────────────────────────────────────────────────────────────
 * Part 3 fed the network numbers that were already numbers — pixels,
 * coordinates, the four rows of XOR. Language is not. `matMul` has nothing
 * to do with the character 'h'.
 *
 * There is no mathematics in this file and nothing here is ever
 * differentiated. It is bookkeeping, and getting it wrong produces a model
 * that trains happily and learns nonsense.
 *
 * ──────────────────────────────────────────────────────────────────────────
 * THE RUNNING EXAMPLE — the doc's corpus, used in every trace below
 * ──────────────────────────────────────────────────────────────────────────
 * Build everything from the string "hello world":
 *
 *     text        "hello world"                    11 characters
 *     unique      ' '  d  e  h  l  o  r  w          8 of them, sorted
 *     vocabSize   8 + 4 specials = 12
 *
 * The complete vocabulary this produces:
 *
 *      0  <pad>       4  ' '       8  'l'
 *      1  <unk>       5  'd'       9  'o'
 *      2  <bos>       6  'e'      10  'r'
 *      3  <eos>       7  'h'      11  'w'
 *
 * And everything the tests check:
 *
 *     encode("hello")          [7, 6, 8, 8, 9]
 *     encode("world")          [11, 9, 10, 8, 5]
 *     encode("hi")             [7, 1]        'i' unseen → <unk>
 *     encode("hello", 3)       [7, 6, 8]     truncated
 *     decode([7, 6, 8, 8, 9])  "hello"
 *
 * Every number above was produced by running this file, not by hand.
 */
import type { Tensor } from "../tensor/index.ts";

/**
 * Special token IDs, reserved at the start of every vocabulary.
 *
 * These four are spent BEFORE any character gets a number, which is why the
 * first real character below is 4 and not 0.
 *
 *   PAD  filler, so a short sentence can share a batch with a long one
 *   UNK  "a character I was never shown" — makes encode total, never failing
 *   BOS  beginning of sequence  (unused here; Ch 28 needs it)
 *   EOS  end of sequence        (unused here; Ch 30 needs it)
 *
 * BOS and EOS do nothing in this chapter and are reserved anyway. Adding them
 * later would shift every character's ID by two, and every weight trained
 * against the old numbering would be pointing at the wrong row.
 */
export const PAD_ID = 0;
export const UNK_ID = 1;
export const BOS_ID = 2;
export const EOS_ID = 3;
export const SPECIAL_TOKEN_COUNT = 4;

/**
 * Scan text, collect its distinct characters, and give each one an integer.
 *
 * ── WORKED TRACE — the running example ────────────────────────────────────
 *
 *     input   "hello world"
 *
 *     step 1  keep each character once
 *             h e l o ' ' w r d          (8 survive; 'l' appeared 3 times)
 *
 *     step 2  SORT them
 *             ' ' d e h l o r w
 *
 *     step 3  number from SPECIAL_TOKEN_COUNT, not from 0
 *             ' '→4  d→5  e→6  h→7  l→8  o→9  r→10  w→11
 *
 *     output  vocabSize = 8 + 4 = 12
 *
 * ── PITFALL: sorting is not cosmetic ──────────────────────────────────────
 * A Set preserves INSERTION order, so skipping the sort numbers characters
 * by the order they happen to appear:
 *
 *     unsorted   h→4 e→5 l→6 o→7 ' '→8 w→9 r→10 d→11
 *     sorted     ' '→4 d→5 e→6 h→7 l→8 o→9 r→10 w→11
 *
 * Both are internally consistent, so both "work" — until the same corpus is
 * read in a different order and produces different IDs. A model's weights
 * are indexed BY these numbers (Ch 18), so a tokenizer that renumbers is a
 * model that silently points at the wrong embedding rows. Sorting makes the
 * vocabulary a function of the text alone.
 *
 * ── PITFALL: numbering from 0 ─────────────────────────────────────────────
 * Then 'd' and <pad> are both 0, and padding becomes indistinguishable from
 * a real character. The mask in encodeBatch would still be correct, but any
 * code reading the IDs alone could not tell them apart.
 */
export function buildVocab(
  text: string
): { stoi: Map<string, number>; itos: Map<number, string>; vocabSize: number } {
  // Set removes duplicates; sort makes the result depend on the text alone.
  const unique = [...new Set(text)].sort();

  const stoi = new Map<string, number>();
  const itos = new Map<number, string>();
  unique.forEach((char, index) => {
    const id = index + SPECIAL_TOKEN_COUNT;   // 0-3 are already spoken for
    stoi.set(char, id);
    itos.set(id, char);
  });

  return { stoi, itos, vocabSize: unique.length + SPECIAL_TOKEN_COUNT };
}

/**
 * Character-level tokenizer: text in, integer IDs out, and back again.
 *
 *     const tok = new CharTokenizer("hello world");
 *     tok.vocabSize                  // 12
 *     tok.encode("hello")            // [7, 6, 8, 8, 9]
 *     tok.decode([7, 6, 8, 8, 9])    // "hello"
 */
export class CharTokenizer {
  readonly stoi: Map<string, number>;
  readonly itos: Map<number, string>;
  readonly vocabSize: number;

  /** Builds the vocabulary from `text` and keeps both directions of the map. */
  constructor(text: string) {
    const { stoi, itos, vocabSize } = buildVocab(text);
    this.stoi = stoi;
    this.itos = itos;
    this.vocabSize = vocabSize;
  }

  /**
   * String → integer IDs. One lookup per character.
   *
   * ── WORKED TRACE ──────────────────────────────────────────────────────────
   *
   *     encode("hello")
   *       'h' → 7
   *       'e' → 6
   *       'l' → 8      both l's give the SAME id — the map is one-to-one
   *       'l' → 8      on tokens, many-to-one on text
   *       'o' → 9
   *                                            → [7, 6, 8, 8, 9]
   *
   *     encode("hi")
   *       'h' → 7
   *       'i' → not in the vocabulary → UNK_ID
   *                                            → [7, 1]
   *
   * "hello world" never contained an 'i', so there is no entry to find. The
   * tokenizer does not fail; it records that something unrecognisable was
   * there. That is the whole job of UNK_ID, and it is why encode can accept
   * any string at all.
   *
   * ── TRUNCATION ────────────────────────────────────────────────────────────
   * Models have a fixed context length, so something must give when the input
   * is longer. `maxLen` cuts from the end:
   *
   *     encode("hello")     → [7, 6, 8, 8, 9]
   *     encode("hello", 3)  → [7, 6, 8]
   *
   * Cutting is crude, and it is what is used here.
   */
  encode(text: string, maxLen?: number): number[] {
    const ids: number[] = [];
    for (const char of text) {
      // ?? not ||, since a valid id of 0 would be falsy — not possible today
      // because 0 is PAD, but the habit costs nothing and survives changes.
      ids.push(this.stoi.get(char) ?? UNK_ID);
    }
    return maxLen === undefined ? ids : ids.slice(0, maxLen);
  }

  /**
   * Integer IDs → string. Special tokens are skipped.
   *
   * ── WORKED TRACE ──────────────────────────────────────────────────────────
   *
   *     decode([7, 6, 8, 8, 9])
   *       7 → 'h'
   *       6 → 'e'
   *       8 → 'l'
   *       8 → 'l'
   *       9 → 'o'
   *                                            → "hello"
   *
   *     decode([7, 1, 0, 0, 0, 0])       a padded row straight from a batch
   *       7 → 'h'
   *       1 → below SPECIAL_TOKEN_COUNT → skipped
   *       0 → below SPECIAL_TOKEN_COUNT → skipped   (four times)
   *                                            → "h"
   *
   * ── WHY SKIP, RATHER THAN DECODE, THE SPECIALS ────────────────────────────
   * A padded row is the normal thing to decode — it is what comes back out of
   * a batch. Skipping is what lets `decode(row)` give the sentence rather than
   * the sentence followed by "<pad><pad><pad>".
   *
   * ── THE ROUND TRIP HOLDS, EXCEPT THROUGH UNK ──────────────────────────────
   *     decode(encode("hello world"))  === "hello world"      ✓
   *     decode(encode("hi"))           === "h"                ✗ the 'i' is gone
   *
   * That second line is not a bug to fix. UNK_ID records THAT a character was
   * unknown, never WHICH one, so the information is destroyed at encode time.
   * It is the honest cost of a fixed vocabulary, and the reason Ch 17 builds
   * one that can cover text it was not built from.
   */
  decode(ids: number[]): string {
    let text = "";
    for (const id of ids) {
      if (id < SPECIAL_TOKEN_COUNT) continue;      // pad, unk, bos, eos
      text += this.itos.get(id) ?? "";
    }
    return text;
  }

  /**
   * Encode a batch, padding every sequence to `maxLen`.
   *
   * ── WHY THIS EXISTS ───────────────────────────────────────────────────────
   * Ch 15 fed the network [64, 784] because every MNIST image is the same
   * size. Sentences are not. A Tensor is a flat Float64Array with a shape
   * (Ch 01) and there is no shape meaning "rows of different lengths".
   *
   * So pad every row out to the same length, and return a second tensor
   * recording which cells were invented.
   *
   * ── WORKED TRACE — encodeBatch(["hello", "world", "hi"], 6) ───────────────
   *
   *     ids                            mask                    shape [3, 6]
   *     [  7   6   8   8   9   0 ]     [ 1  1  1  1  1  0 ]    "hello"
   *     [ 11   9  10   8   5   0 ]     [ 1  1  1  1  1  0 ]    "world"
   *     [  7   1   0   0   0   0 ]     [ 1  1  0  0  0  0 ]    "hi"
   *                       ↑                          ↑
   *                    <pad>                    invented cells
   *
   * Row 3 shows both jobs at once: 'i' was never in "hello world" so it
   * became UNK (1, a REAL token, mask 1), and the four cells after it are
   * padding (0, mask 0).
   *
   * ── PITFALL: the mask records a POSITION, not a value ─────────────────────
   * It is tempting to write:
   *
   *     mask[j] = ids[j] !== PAD_ID ? 1 : 0          // WRONG
   *
   * Two things break. Past the end of a short sequence `ids[j]` is
   * `undefined`, and `undefined !== 0` is TRUE — so every pad cell is marked
   * REAL, the exact opposite of the truth. And writing that `undefined` into
   * a Float64Array stores NaN, not 0, so the ids are wrong too:
   *
   *     ids  [4, 1, NaN, NaN, NaN, NaN]      mask [1, 1, 1, 1, 1, 1]
   *
   * Nothing throws. Part 5 would attend to padding while every shape looked
   * correct. Decide by POSITION — `j < ids.length` — and the value never
   * enters into it.
   *
   * ── PITFALL: Tensor is an interface, not a class ──────────────────────────
   * `new Tensor(data, shape)` throws "Tensor is not defined". Tensor is a
   * plain shape — { data, shape, ndim, size } — and the import above is
   * `import type`, which does not exist at runtime at all. Build the object
   * literally, the same way Ch 15's loader did.
   *
   * Returns ids and mask, both [texts.length, maxLen].
   */
  encodeBatch(
    texts: string[],
    maxLen: number
  ): { ids: Tensor; mask: Tensor } {
    const batchSize = texts.length;

    // Start fully padded and fully masked-out, then overwrite the real cells.
    // Anything never touched below is therefore correct by construction.
    const idsData = new Float64Array(batchSize * maxLen).fill(PAD_ID);
    const maskData = new Float64Array(batchSize * maxLen).fill(0);

    for (let row = 0; row < batchSize; row++) {
      const ids = this.encode(texts[row]!, maxLen);
      // loop to ids.length, NOT to maxLen — the tail is already padding
      for (let col = 0; col < ids.length; col++) {
        idsData[row * maxLen + col] = ids[col]!;
        maskData[row * maxLen + col] = 1;
      }
    }

    return {
      ids: { data: idsData, shape: [batchSize, maxLen], ndim: 2, size: idsData.length },
      mask: { data: maskData, shape: [batchSize, maxLen], ndim: 2, size: maskData.length },
    };
  }
}
