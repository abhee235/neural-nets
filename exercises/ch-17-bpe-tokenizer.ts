/**
 * EXERCISES — Ch 17: BPE Tokenizer
 * ══════════════════════════════════
 * Prereq : src/tokenizer/bpe.ts + char.ts implemented
 * Run    : bun run exercises/ch-17-bpe-tokenizer.ts
 *
 * BPE (Byte Pair Encoding) builds a subword vocabulary by merging the most
 * frequent adjacent byte-pairs. GPT-2 uses 50,257 BPE tokens.
 */
import { BPETokenizer, countPairs, mergePair } from "../src/tokenizer/bpe.ts";

// ─── E1: Count pairs ─────────────────────────────────────────────────────────
// Before any merges, each word is split into chars.
const corpus = ["low", "low", "low", "lower", "newer", "wider"];
const tokens = corpus.map(w => w.split(""));
const pairs  = countPairs(tokens);
console.log("Most common pair:", [...pairs.entries()].sort((a,b) => b[1]-a[1])[0]);
// expected: ["l,o", 4]  or similar

// ─── E2: Merge a pair ────────────────────────────────────────────────────────
const merged = mergePair(tokens, ["l", "o"]);
console.log("\nAfter merge l+o:");
console.log(merged.slice(0, 3)); // ["lo", "w"], ["lo","w"], ["lo","w"]

// ─── E3: Train BPE on tiny corpus ────────────────────────────────────────────
const bpe = new BPETokenizer();
const trainText = "the cat sat on the mat. the cat is fat.";
bpe.train(trainText, 20);   // learn 20 merges
console.log("\nBPE vocab size after 20 merges:", bpe.vocabSize);

// ─── E4: Encode + decode ─────────────────────────────────────────────────────
const enc = bpe.encode("the cat");
const dec = bpe.decode(enc);
console.log("encode('the cat'):", enc);
console.log("decode back      :", dec);
console.log("round-trip ok    :", dec === "the cat");

// ─── E5: Token efficiency ────────────────────────────────────────────────────
// Compare token count for char tokenizer vs BPE on the same sentence.
import { CharTokenizer, buildVocab } from "../src/tokenizer/char.ts";
const charTok = new CharTokenizer(buildVocab(trainText));
const sentence = "the cat sat on the mat";
const charTokens = charTok.encode(sentence);
const bpeTokens  = bpe.encode(sentence);
console.log("\nChar tokens:", charTokens.length);
console.log("BPE tokens :", bpeTokens.length, "  (should be fewer)");

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: train BPE on a larger text (50+ merges) and print the 10 most useful merges.
