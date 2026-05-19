/**
 * EXERCISES — Ch 16: Character Tokenizer
 * ════════════════════════════════════════
 * Prereq : src/tokenizer/char.ts implemented
 * Run    : bun run exercises/ch-16-char-tokenizer.ts
 *
 * Every language model starts here: turning raw text into integer IDs.
 * The char tokenizer is the simplest possible vocabulary — one token per char.
 */
import { buildVocab, CharTokenizer, PAD_ID, BOS_ID, EOS_ID } from "../src/tokenizer/char.ts";

// ─── E1: Build vocabulary ─────────────────────────────────────────────────────
const text = "hello world! neural nets are fun.";
const vocab = buildVocab(text);
console.log("vocab size:", vocab.size, "  (special tokens + unique chars)");
console.log("contains 'h':", vocab.has("h"), "  expected: true");
console.log("contains 'z':", vocab.has("z"), "  expected: false");

// ─── E2: Encode & decode round-trip ──────────────────────────────────────────
const tokenizer = new CharTokenizer(vocab);
const encoded   = tokenizer.encode("hello");
const decoded   = tokenizer.decode(encoded);
console.log("\nencode('hello'):", encoded);
console.log("decode back    :", decoded, "  expected: 'hello'");
console.log("round-trip ok  :", decoded === "hello", "  expected: true");

// ─── E3: Special tokens ───────────────────────────────────────────────────────
// BOS (beginning of sequence) and EOS (end of sequence) frame every sequence.
const withSpecial = [BOS_ID, ...tokenizer.encode("hi"), EOS_ID];
console.log("\n[BOS] + 'hi' + [EOS]:", withSpecial);
// BOS=2, EOS=3 → [2, id_h, id_i, 3]

// ─── E4: Unknown characters ───────────────────────────────────────────────────
// Chars not in vocab should map to UNK_ID.
const unkEncoded = tokenizer.encode("héllo");   // é not in vocab
console.log("\nencode('héllo') with unk:", unkEncoded);
// 'é' should be UNK_ID=1

// ─── E5: Batch encode ────────────────────────────────────────────────────────
// Batch encode pads all sequences to the same length.
const sentences = ["hi", "hello", "hey!"];
const batch     = tokenizer.encodeBatch(sentences);
console.log("\nbatch shapes:");
for (const [i, seq] of batch.entries()) {
  console.log(\`  [\${i}]: length=\${seq.length}\`);
}
// All sequences should have the same length (padded with PAD_ID=0).

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: encode Shakespeare's first line, print token count vs character count.
//   "To be, or not to be, that is the question"
