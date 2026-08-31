/**
 * EXERCISES — Ch 16: Character Tokenizer
 * ════════════════════════════════════════
 * Prereq : src/tokenizer/char.ts implemented
 * Run    : bun run exercises/ch-16-char-tokenizer.ts
 *
 * REFERENCE: docs/part-4-tokenizer-and-inputs/ch-16-char-tokenizer.md
 *
 * Every language model starts here: text becomes integers, and a batch of
 * ragged sentences becomes a rectangle plus a mask admitting which cells
 * were invented.
 *
 * The corpus is the chapter's, so every number below can be checked against
 * the doc:
 *
 *      0  <pad>       4  ' '       8  'l'
 *      1  <unk>       5  'd'       9  'o'
 *      2  <bos>       6  'e'      10  'r'
 *      3  <eos>       7  'h'      11  'w'
 */
import {
  buildVocab,
  CharTokenizer,
  PAD_ID,
  UNK_ID,
  BOS_ID,
  EOS_ID,
  SPECIAL_TOKEN_COUNT,
} from "../src/tokenizer/char.ts";
import type { Tensor } from "../src/tensor/index.ts";

const CORPUS = "hello world";

/** Print one row of a flat [batch, maxLen] tensor. */
function row(t: Tensor, index: number, maxLen: number): number[] {
  return Array.from(t.data.slice(index * maxLen, (index + 1) * maxLen));
}

function stage(title: string, body: () => void): void {
  console.log(`\n─── ${title} ───`);
  try {
    body();
  } catch (error) {
    if (error instanceof Error && error.message.includes("not implemented")) {
      console.log("  pending —", error.message);
    } else throw error;
  }
}

// ─── E1: the vocabulary is just a lookup table ───────────────────────────────
stage("E1: build a vocabulary", () => {
  const { stoi, vocabSize } = buildVocab(CORPUS);
  console.log(`  corpus      ${JSON.stringify(CORPUS)}   ${CORPUS.length} characters`);
  console.log(`  distinct    ${stoi.size}   ('l' appears 3 times and earns one entry)`);
  console.log(`  vocabSize   ${stoi.size} + ${SPECIAL_TOKEN_COUNT} specials = ${vocabSize}`);
  console.log("  the table:", [...stoi.entries()].map(([c, i]) => `${JSON.stringify(c)}=${i}`).join(" "));
  console.log("  first real character is", Math.min(...stoi.values()), "— ids 0-3 were spent before it");
});

// ─── E2: sorting is what makes the vocabulary reproducible ───────────────────
stage("E2: why the characters are sorted", () => {
  // Same characters, opposite order. A vocabulary that depended on reading
  // order would number these differently — and a model trained with one
  // would be pointing at the wrong embedding rows under the other.
  const forward = [...buildVocab("abc").stoi.entries()];
  const reverse = [...buildVocab("cba").stoi.entries()];
  console.log('  buildVocab("abc") :', forward.map(([c, i]) => `${c}=${i}`).join(" "));
  console.log('  buildVocab("cba") :', reverse.map(([c, i]) => `${c}=${i}`).join(" "));
  console.log("  identical?         ", JSON.stringify(forward) === JSON.stringify(reverse));
});

// ─── E3: encode, decode, and the round trip that does not hold ───────────────
stage("E3: encode and decode", () => {
  const tok = new CharTokenizer(CORPUS);
  console.log("  encode('hello')  ", tok.encode("hello"), "  both l's give the same id");
  console.log("  encode('world')  ", tok.encode("world"));
  console.log("  decode back      ", JSON.stringify(tok.decode(tok.encode("hello"))));
  console.log("  round trip       ", JSON.stringify(tok.decode(tok.encode(CORPUS))), " === corpus");
  console.log();
  console.log("  encode('hello',3)", tok.encode("hello", 3), "  truncated");
  console.log("  encode('hello',99)", tok.encode("hello", 99), " maxLen means AT MOST — encode never pads");
});

// ─── E4: unknown characters, and what they cost ──────────────────────────────
stage("E4: a character the vocabulary never saw", () => {
  const tok = new CharTokenizer(CORPUS);
  console.log("  encode('hi')     ", tok.encode("hi"), `  'i' is not in the corpus → UNK_ID = ${UNK_ID}`);
  console.log("  decode of that   ", JSON.stringify(tok.decode(tok.encode("hi"))));
  console.log("  the 'i' is gone. UNK records THAT a character was unknown, never WHICH one,");
  console.log("  so encode destroys it. That is the cost of a fixed vocabulary — and Ch 17's reason to exist.");
});

// ─── E5: batching, padding, and the mask ─────────────────────────────────────
stage("E5: a batch has to be rectangular", () => {
  const tok = new CharTokenizer(CORPUS);
  const texts = ["hello", "world", "hi"];
  const maxLen = 6;
  const { ids, mask } = tok.encodeBatch(texts, maxLen);

  console.log(`  ids.shape [${ids.shape}]   mask.shape [${mask.shape}]`);
  console.log();
  console.log("     ids                        mask                 text");
  for (let i = 0; i < texts.length; i++) {
    const r = row(ids, i, maxLen).map((n) => String(n).padStart(2)).join(", ");
    const m = row(mask, i, maxLen).join(", ");
    console.log(`     [${r}]   [${m}]   ${JSON.stringify(texts[i])}`);
  }
  console.log();
  console.log("  row 3 does both jobs: 'i' became UNK (a REAL token, mask 1),");
  console.log("  and the four cells after it are padding (mask 0).");
  const real = row(mask, 2, maxLen).reduce((a, b) => a + b, 0);
  console.log(`  mask row 3 sums to ${real} — the number of real tokens in "hi"`);
});

// ─── E6: the special tokens nothing uses yet ─────────────────────────────────
stage("E6: BOS and EOS", () => {
  const tok = new CharTokenizer(CORPUS);
  const framed = [BOS_ID, ...tok.encode("hello"), EOS_ID];
  console.log("  [BOS] hello [EOS] =", framed);
  console.log("  decode ignores them:", JSON.stringify(tok.decode(framed)));
  console.log(`  they do nothing here. Ch 28 needs BOS=${BOS_ID} to start generating from nothing,`);
  console.log(`  and Ch 30's GPT needs EOS=${EOS_ID} to know when to stop.`);
  console.log("  reserving them now means no id ever has to move.");
});

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO 1: build a vocabulary from a whole paragraph of English and count it.
//         Roughly how many distinct characters does ordinary text contain,
//         and how does that compare to GPT-2's 50,257 tokens?
//
// TODO 2: encode "To be, or not to be, that is the question" with a vocabulary
//         built from that same line. Print token count against character
//         count — they are equal, and that is the character tokenizer's whole
//         problem. Sequence length is what attention costs in Part 5.
//
// TODO 3: build the vocabulary from lowercase text and encode something with
//         capitals. Every capital becomes UNK. Would lowercasing the input
//         fix it, and what does that throw away?
//
// TODO 4: pass a maxLen SMALLER than the longest sentence to encodeBatch and
//         look at the mask. Nothing is padded, so nothing is masked out —
//         predict what the mask looks like before running it.
