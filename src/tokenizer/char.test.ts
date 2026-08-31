/**
 * Tests for tokenizer/char.ts
 * Chapter 16 — Character-Level Tokenizer
 *
 * Run: bun test src/tokenizer/char.test.ts
 *
 * Every fixture is the chapter's own corpus, "hello world":
 *
 *      0  <pad>       4  ' '       8  'l'
 *      1  <unk>       5  'd'       9  'o'
 *      2  <bos>       6  'e'      10  'r'
 *      3  <eos>       7  'h'      11  'w'
 *
 *     encode("hello")  [7, 6, 8, 8, 9]      encode("hi")  [7, 1]
 *     encode("world")  [11, 9, 10, 8, 5]    vocabSize     12
 */
import { describe, it, expect } from "bun:test";
import {
  buildVocab,
  CharTokenizer,
  PAD_ID,
  UNK_ID,
  SPECIAL_TOKEN_COUNT,
} from "./char.ts";

const CORPUS = "hello world";
const tok = () => new CharTokenizer(CORPUS);
/** Pull one row out of a flat [batch, maxLen] tensor. */
const row = (t: { data: Float64Array }, index: number, maxLen: number) =>
  Array.from(t.data.slice(index * maxLen, (index + 1) * maxLen));

describe("buildVocab", () => {
  it("gives each distinct character exactly one ID", () => {
    // "hello world" has 11 characters but only 8 distinct ones —
    // 'l' appears three times and 'o' twice, each earning one entry.
    const { stoi } = buildVocab(CORPUS);
    expect(stoi.size).toBe(8);
  });

  it("vocabSize = distinct characters + SPECIAL_TOKEN_COUNT", () => {
    // 8 + 4 = 12. The four specials occupy IDs before any character.
    expect(buildVocab(CORPUS).vocabSize).toBe(8 + SPECIAL_TOKEN_COUNT);
  });

  it("numbers characters from SPECIAL_TOKEN_COUNT, never from 0", () => {
    // If a character were given ID 0 it would be indistinguishable from
    // <pad>, and a padded batch could not be told from real data.
    const { stoi } = buildVocab(CORPUS);
    for (const id of stoi.values()) expect(id).toBeGreaterThanOrEqual(SPECIAL_TOKEN_COUNT);
  });

  it("assigns IDs in sorted order", () => {
    // Sorted: ' ' d e h l o r w  ->  4 5 6 7 8 9 10 11.
    const { stoi } = buildVocab(CORPUS);
    expect([...stoi.entries()]).toEqual([
      [" ", 4], ["d", 5], ["e", 6], ["h", 7],
      ["l", 8], ["o", 9], ["r", 10], ["w", 11],
    ]);
  });

  it("depends on the text alone, not the order characters appear", () => {
    // THE reason for sorting. Two corpora with the same characters in a
    // different order must give the same numbering, or a saved model's
    // weights stop matching the tokenizer that trained it.
    expect([...buildVocab("abc").stoi.entries()])
      .toEqual([...buildVocab("cba").stoi.entries()]);
  });

  it("stoi and itos are inverses of each other", () => {
    // decode depends on this: itos.get(stoi.get(c)) must be c.
    const { stoi, itos } = buildVocab(CORPUS);
    for (const [char, id] of stoi) expect(itos.get(id)).toBe(char);
  });

  it("an empty corpus still reserves the special tokens", () => {
    // No characters, but the four IDs are spent regardless.
    const { stoi, vocabSize } = buildVocab("");
    expect(stoi.size).toBe(0);
    expect(vocabSize).toBe(SPECIAL_TOKEN_COUNT);
  });
});

describe("CharTokenizer.encode", () => {
  it("maps each character through the vocabulary", () => {
    expect(tok().encode("hello")).toEqual([7, 6, 8, 8, 9]);
  });

  it("gives repeated characters the same ID", () => {
    // Both 'l's are 8. The map is one-to-one on tokens and many-to-one on
    // text — this is what lets a fixed-size vocabulary cover any input.
    const ids = tok().encode("hello");
    expect(ids[2]).toBe(ids[3]!);
  });

  it("maps unseen characters to UNK_ID rather than failing", () => {
    // 'i' never appears in "hello world". encode must still return something,
    // which is the whole reason UNK_ID exists.
    expect(tok().encode("hi")).toEqual([7, UNK_ID]);
  });

  it("encodes any string at all, including one with no known characters", () => {
    expect(tok().encode("xyz")).toEqual([UNK_ID, UNK_ID, UNK_ID]);
  });

  it("truncates when maxLen is smaller than the input", () => {
    expect(tok().encode("hello", 3)).toEqual([7, 6, 8]);
  });

  it("leaves the result untouched when maxLen exceeds the input", () => {
    // maxLen means "at most this long", not "exactly this long" — encode
    // never pads. Padding belongs to encodeBatch, which is the only place
    // that needs rectangles.
    expect(tok().encode("hello", 99)).toEqual([7, 6, 8, 8, 9]);
  });

  it("returns an empty list for empty input", () => {
    expect(tok().encode("")).toEqual([]);
  });
});

describe("CharTokenizer.decode", () => {
  it("turns IDs back into the characters they stand for", () => {
    expect(tok().decode([7, 6, 8, 8, 9])).toBe("hello");
  });

  it("skips every ID below SPECIAL_TOKEN_COUNT", () => {
    // A padded row is the normal thing to decode, since it is what comes
    // back out of a batch. Without skipping, this would end in "<pad><pad>".
    expect(tok().decode([7, UNK_ID, PAD_ID, PAD_ID, PAD_ID, PAD_ID])).toBe("h");
  });

  it("is the inverse of encode for text the vocabulary has seen", () => {
    expect(tok().decode(tok().encode(CORPUS))).toBe(CORPUS);
  });

  it("cannot recover characters that became UNK", () => {
    // UNK_ID records THAT a character was unknown, never WHICH one, so the
    // information is destroyed at encode time. Not a bug — the cost of a
    // fixed vocabulary, and the reason Ch 17 exists.
    expect(tok().decode(tok().encode("hi"))).toBe("h");
  });
});

describe("CharTokenizer.encodeBatch", () => {
  const BATCH = ["hello", "world", "hi"];
  const MAXLEN = 6;

  it("returns ids of shape [batch, maxLen]", () => {
    expect(tok().encodeBatch(BATCH, MAXLEN).ids.shape).toEqual([3, 6]);
  });

  it("returns a mask of exactly the same shape as ids", () => {
    // Part 5 multiplies them together. A mismatch would broadcast into
    // something that runs and is silently wrong.
    const { ids, mask } = tok().encodeBatch(BATCH, MAXLEN);
    expect(mask.shape).toEqual(ids.shape);
  });

  it("pads short sequences with PAD_ID", () => {
    // "hi" is 2 tokens in a row of 6, so four cells are invented.
    expect(row(tok().encodeBatch(BATCH, MAXLEN).ids, 2, MAXLEN))
      .toEqual([7, UNK_ID, PAD_ID, PAD_ID, PAD_ID, PAD_ID]);
  });

  it("leaves each row's real tokens unchanged", () => {
    const { ids } = tok().encodeBatch(BATCH, MAXLEN);
    expect(row(ids, 0, MAXLEN)).toEqual([7, 6, 8, 8, 9, PAD_ID]);
    expect(row(ids, 1, MAXLEN)).toEqual([11, 9, 10, 8, 5, PAD_ID]);
  });

  it("marks the mask 1 for real tokens and 0 for padding", () => {
    expect(row(tok().encodeBatch(BATCH, MAXLEN).mask, 2, MAXLEN))
      .toEqual([1, 1, 0, 0, 0, 0]);
  });

  it("masks by POSITION, so an UNK token counts as real", () => {
    // The trap: deciding from the value (`ids[j] !== PAD_ID`) marks every
    // cell past the end as real, because ids[j] there is undefined and
    // undefined !== 0. UNK is a genuine token and must stay masked 1.
    const { ids, mask } = tok().encodeBatch(["hi"], MAXLEN);
    expect(ids.data[1]).toBe(UNK_ID);
    expect(mask.data[1]).toBe(1);
  });

  it("never writes NaN into the ids", () => {
    // Assigning an out-of-range array element into a Float64Array stores
    // NaN, not 0 — the same bug as above, seen from the ids side.
    const { ids } = tok().encodeBatch(BATCH, MAXLEN);
    expect(Array.from(ids.data).some(Number.isNaN)).toBe(false);
  });

  it("truncates sequences longer than maxLen", () => {
    // "hello" is 5 tokens capped at 3, and the mask has no zeros because
    // every cell in the row is real.
    const { ids, mask } = tok().encodeBatch(["hello"], 3);
    expect(Array.from(ids.data)).toEqual([7, 6, 8]);
    expect(Array.from(mask.data)).toEqual([1, 1, 1]);
  });

  it("the mask row sums to the number of real tokens", () => {
    // A property that holds for every row: 5, 5, 2 for this batch.
    const { mask } = tok().encodeBatch(BATCH, MAXLEN);
    const sums = [0, 1, 2].map((r) => row(mask, r, MAXLEN).reduce((a, b) => a + b, 0));
    expect(sums).toEqual([5, 5, 2]);
  });

  it("handles a row that is entirely padding", () => {
    // An empty string contributes no real tokens, so its mask is all zeros.
    const { ids, mask } = tok().encodeBatch([""], 4);
    expect(Array.from(ids.data)).toEqual([PAD_ID, PAD_ID, PAD_ID, PAD_ID]);
    expect(Array.from(mask.data)).toEqual([0, 0, 0, 0]);
  });
});
