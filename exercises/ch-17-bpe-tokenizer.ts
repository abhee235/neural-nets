/**
 * EXERCISES — Ch 17: BPE Tokenizer
 * ════════════════════════════════════════
 * Prereq : src/tokenizer/bpe.ts implemented (and char.ts from Ch 16)
 * Run    : bun run exercises/ch-17-bpe-tokenizer.ts
 *
 * REFERENCE: docs/part-4-tokenizer-and-inputs/ch-17-bpe-tokenizer.md
 *
 * Ch 16's tokenizer emits one token per character and loses any character it
 * was not built from. BPE learns its own vocabulary by counting: glue together
 * whichever two neighbours occur most often, over and over. Common words end
 * up as one token, and a word it has never seen is spelled out of pieces.
 *
 * The corpus is the chapter's, so every number below can be checked against
 * the doc:
 *
 *      "low low low low low lower lower newest newest newest widest widest"
 *
 *      4 ' '   6 e    8 l   10 o   12 s   14 w    16 low   18 est   20 new     22 lowe
 *      5 d     7 i    9 n   11 r   13 t   15 lo   17 es    19 ne    21 newest
 */
import { countPairs, mergePair, BPETokenizer } from "../src/tokenizer/bpe.ts";
import { CharTokenizer, UNK_ID, SPECIAL_TOKEN_COUNT } from "../src/tokenizer/char.ts";

const CORPUS = "low low low low low lower lower newest newest newest widest widest";
const VOCAB_SIZE = 23;

/** Split text the way countPairs and mergePair expect it. */
const words = (text: string): string[][] => text.split(" ").map((word) => [...word]);
/** Render a corpus the way the chapter draws it. */
const show = (corpus: string[][]): string =>
  corpus.map((word) => word.join(" ")).join("  |  ");

function stage(title: string, body: () => void): void {
  console.log(`\n─── ${title} ───`);
  try {
    body();
  } catch (error) {
    if (error instanceof Error && /not implemented/i.test(error.message)) {
      console.log("  pending —", error.message);
    } else throw error;
  }
}

function trained(): BPETokenizer {
  const tokenizer = new BPETokenizer();
  tokenizer.train(CORPUS, VOCAB_SIZE);
  return tokenizer;
}

// ─── E1: counting is the whole first half of the algorithm ───────────────────
stage("E1: count the adjacent pairs", () => {
  const counts = countPairs(words(CORPUS));
  const ranked = [...counts.entries()].sort((a, b) => b[1] - a[1]);
  console.log(`  corpus       ${CORPUS.split(" ").length} words, ${words(CORPUS).flat().length} tokens`);
  console.log("  top pairs   ", ranked.slice(0, 5).map(([k, n]) => `(${k})=${n}`).join("  "));

  // The first round is a TIE — two pairs both occur 7 times. Something has to
  // choose, and it must choose the same way on every run, or two trainings on
  // the same text give two different vocabularies.
  const highest = ranked[0]![1];
  const tied = ranked.filter(([, n]) => n === highest);
  console.log(`  highest count ${highest}, tied between ${tied.length} pairs:`,
    tied.map(([k]) => `(${k})`).join(" "));
});

// ─── E2: a merge rewrites the corpus, so the next count differs ──────────────
stage("E2: merge the winner, everywhere at once", () => {
  const before = words("low lower newest");
  console.log("  before        ", show(before));
  const after = mergePair(before, ["l", "o"]);
  console.log('  after ("l","o")', show(after));
  console.log("  input untouched?", JSON.stringify(before) === JSON.stringify(words("low lower newest")));

  // Round 2 can now merge ("lo","w") — a pair whose first half did not exist
  // until round 1 created it. That is how BPE builds long tokens out of short.
  console.log('  then  ("lo","w")', show(mergePair(after, ["lo", "w"])));

  // The invariant: a merge changes how the text is CUT UP, never what it says.
  const rejoined = mergePair(words(CORPUS), ["e", "s"]).map((w) => w.join("")).join(" ");
  console.log("  text preserved? ", rejoined === CORPUS);

  // The bug everyone writes: after a match, step past BOTH tokens. Stepping
  // past one re-emits the second half and the text silently grows.
  console.log('  "banana" merge ("a","n") →', mergePair([[..."banana"]], ["a", "n"])[0]!.join(" | "),
    "  (buggy code gives b | an | n | an | n | a → \"bannanna\")");
});

// ─── E3: training is those two steps in a loop ───────────────────────────────
stage("E3: train, and watch the corpus shrink", () => {
  let corpus = words(CORPUS);
  const tokenizer = trained();
  console.log(`  ${corpus.flat().length} tokens before any merge`);
  for (let i = 0; i < tokenizer.merges.length; i++) {
    const [a, b] = tokenizer.merges[i]!;
    corpus = mergePair(corpus, [a, b]);
    console.log(`  merge ${i + 1}  ("${a}","${b}") → "${a + b}"`.padEnd(34) +
      `${String(corpus.flat().length).padStart(3)} tokens left`);
  }

  // The vocabulary is three layers: specials, then every character, then one
  // entry per merge. The middle layer is EXACTLY Ch 16's vocabulary — BPE
  // starts from the character tokenizer and stacks tokens on top.
  const characters = new Set(CORPUS).size;
  console.log();
  console.log(`  ${SPECIAL_TOKEN_COUNT} specials + ${characters} characters + ${tokenizer.merges.length} merges = ${tokenizer.vocabSize}`);
  console.log("  merged tokens:", tokenizer.merges
    .map(([a, b]) => `${a + b}=${tokenizer.stoi.get(a + b)}`).join("  "));

  // vocabSize is a ceiling, not a promise. Once no pair repeats, training
  // stops on its own rather than inventing tokens used exactly once.
  const greedy = new BPETokenizer();
  greedy.train(CORPUS, 40);
  console.log(`  asked for 40 → learned ${greedy.merges.length} merges, vocabSize ${greedy.vocabSize} (nothing repeats past that)`);
});

// ─── E4: encoding is the payoff ──────────────────────────────────────────────
stage("E4: characters vs BPE, same words", () => {
  const tokenizer = trained();
  console.log("     word       characters   BPE   pieces");
  for (const word of ["low", "lower", "newest", "lowest", "slowest"]) {
    const pieces = tokenizer.encodeWord(word);
    console.log(`     ${word.padEnd(10)} ${String(word.length).padStart(6)}   ${String(pieces.length).padStart(5)}   ${pieces.join(" | ")}`);
  }
  console.log();
  console.log("  'lowest' and 'slowest' are nowhere in the training corpus. A word-level");
  console.log("  tokenizer would emit <unk> and lose them; BPE spells them out of pieces.");
  console.log("  'est' is a real English suffix, and nothing told the algorithm about suffixes.");
});

// ─── E5: the rules are a chain, not a set ────────────────────────────────────
stage("E5: apply the same rules in the wrong order", () => {
  const tokenizer = trained();
  console.log('  in training order  encodeWord("lowest") =', tokenizer.encodeWord("lowest").join(" | "));

  tokenizer.merges = [...tokenizer.merges].reverse();
  console.log('  reversed           encodeWord("lowest") =', tokenizer.encodeWord("lowest").join(" | "));
  console.log();
  console.log('  ("lo","w") can only fire once ("l","o") has created "lo". Reversed, it is');
  console.log("  tried while no \"lo\" exists yet, finds nothing, and the chain never forms.");
  console.log("  Same rules, same word, twice as many tokens. Order IS the algorithm.");
});

// ─── E6: round trips, and the one that still fails ───────────────────────────
stage("E6: encode and decode", () => {
  const tokenizer = trained();
  for (const text of ["low lower", "low  lower", "slowest", "the lowest"]) {
    const ids = tokenizer.encode(text);
    const back = tokenizer.decode(ids);
    console.log(`  ${back === text ? "ok  " : "LOSS"}  ${JSON.stringify(text).padEnd(14)} → [${ids.join(", ")}]  → ${JSON.stringify(back)}`);
  }
  console.log();
  console.log(`  "low  lower" keeps both spaces: the space is an ordinary token (id ${tokenizer.stoi.get(" ")})`);
  console.log("  that simply never merges, because merges only happen inside a word.");
  console.log();
  console.log(`  "the lowest" loses its 'h'. No 'h' appears anywhere in low/lower/newest/widest,`);
  console.log(`  so it is not in the base vocabulary and encodes to UNK_ID = ${UNK_ID}.`);
  console.log("  BPE removes <unk> for unseen WORDS, not for unseen CHARACTERS. Real systems");
  console.log("  close that gap by starting from the 256 possible BYTES instead of characters.");
});

// ─── E7: what this buys Part 5 ───────────────────────────────────────────────
stage("E7: sequence length is what attention costs", () => {
  const bpe = trained();
  const chars = new CharTokenizer(CORPUS);
  const charLen = chars.encode(CORPUS).length;
  const bpeLen = bpe.encode(CORPUS).length;
  console.log(`  character tokenizer (Ch 16)  ${charLen} tokens`);
  console.log(`  BPE                 (Ch 17)  ${bpeLen} tokens   ${(charLen / bpeLen).toFixed(2)}x shorter`);
  // Attention compares every position with every other, so its work grows with
  // the SQUARE of the length — the saving there is the ratio squared.
  console.log(`  attention work scales with length², so that is ${((charLen / bpeLen) ** 2).toFixed(1)}x less work in Part 5`);
});

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO 1: train on a paragraph of real English with vocabSize 300 and print the
//         last 20 merges. How many are recognisable word fragments — prefixes,
//         suffixes, whole common words? Nothing taught it those categories.
//
// TODO 2: train the SAME text twice with two different vocabSize values, say
//         50 and 200. Is the shorter list of merge rules a prefix of the longer
//         one? Explain why that has to be true from how the loop works.
//
// TODO 3: encode a word that shares no characters with the training corpus.
//         Count the UNK ids. Then repeat after adding one sentence containing
//         those characters to the training text — the merges barely change, but
//         the UNKs vanish. Which layer of the vocabulary did that fix?
//
// TODO 4: measure where the tokens/characters ratio stops improving as vocabSize
//         grows. Plot it if you like. GPT-2 chose 50,257 and LLaMA 32,000 — and
//         since Ch 18 gives every vocabulary entry its own embedding row, a
//         bigger vocabulary is a bigger model before a single layer is added.
