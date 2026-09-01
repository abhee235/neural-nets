/**
 * Tests for tokenizer/bpe.ts
 * Chapter 17 — BPE Tokenizer
 *
 * Run: bun test src/tokenizer/bpe.test.ts
 *
 * Every fixture is the chapter's own corpus:
 *
 *     "low low low low low lower lower newest newest newest widest widest"
 *
 * Trained to vocabSize 23 it learns exactly eight rules, in this order:
 *
 *     ("l","o")  ("lo","w")  ("e","s")  ("es","t")
 *     ("n","e")  ("ne","w")  ("new","est")  ("low","e")
 *
 * and the vocabulary comes out as:
 *
 *      4 ' '   6 e    8 l   10 o   12 s   14 w    16 low   18 est   20 new     22 lowe
 *      5 d     7 i    9 n   11 r   13 t   15 lo   17 es    19 ne    21 newest
 */
import { describe, it, expect } from "bun:test";
import { countPairs, mergePair, BPETokenizer, type Corpus } from "./bpe.ts";
import { UNK_ID, SPECIAL_TOKEN_COUNT } from "./char.ts";

const CORPUS = "low low low low low lower lower newest newest newest widest widest";
const VOCAB_SIZE = 23;

/** The eight rules the chapter's corpus produces, in learned order. */
const RULES: Array<[string, string]> = [
  ["l", "o"], ["lo", "w"], ["e", "s"], ["es", "t"],
  ["n", "e"], ["ne", "w"], ["new", "est"], ["low", "e"],
];

const trained = () => {
  const tokenizer = new BPETokenizer();
  tokenizer.train(CORPUS, VOCAB_SIZE);
  return tokenizer;
};
/** Split a string into the shape countPairs/mergePair expect. */
const words = (text: string): Corpus => text.split(" ").map((word) => [...word]);

describe("countPairs", () => {
  it("counts each adjacent pair once per occurrence", () => {
    // "low" contributes (l o) and (o w); "lower" contributes those two plus
    // (w e) and (e r). A word of n tokens has exactly n-1 adjacent pairs.
    const counts = countPairs(words("low lower"));
    expect(counts.get("l o")).toBe(2);
    expect(counts.get("o w")).toBe(2);
    expect(counts.get("w e")).toBe(1);
    expect(counts.get("e r")).toBe(1);
  });

  it("finds the pair the chapter's corpus merges first", () => {
    // (l o) occurs 7 times: five in "low", two in "lower".
    expect(countPairs(words(CORPUS)).get("l o")).toBe(7);
  });

  it("round 1 of the chapter's corpus is a genuine tie", () => {
    // (l o) and (o w) both occur 7 times, so "the most frequent pair" is not
    // unique. This is why the tie-break has to be deterministic — see below.
    const counts = countPairs(words(CORPUS));
    const highest = Math.max(...counts.values());
    const tied = [...counts.entries()].filter(([, n]) => n === highest);
    expect(highest).toBe(7);
    expect(tied.map(([key]) => key).sort()).toEqual(["l o", "o w"]);
  });

  it("never counts a pair spanning two words", () => {
    // "low" then "newest" sit side by side, but (w n) is an accident of word
    // order and means nothing. The two-level Corpus type is what prevents it:
    // a loop over one word's array cannot see into the next.
    expect(countPairs(words("low newest")).has("w n")).toBe(false);
  });

  it("a single-token word contributes no pairs", () => {
    // n-1 pairs for n tokens, and n-1 is 0 here. The loop must not read past
    // the end of a one-element array.
    expect(countPairs([["low"]]).size).toBe(0);
  });

  it("an empty corpus has no pairs", () => {
    expect(countPairs([]).size).toBe(0);
  });

  it("keys can be split back into the pair they stand for", () => {
    // No token can contain a space, because the corpus was split ON spaces —
    // so a space is an unambiguous separator inside the key.
    for (const key of countPairs(words(CORPUS)).keys()) {
      expect(key.split(" ")).toHaveLength(2);
    }
  });
});

describe("mergePair", () => {
  it("replaces the pair in every word at once", () => {
    // A merge is global. This is what makes the next round of counting see a
    // different corpus from the one before it.
    expect(mergePair(words("low lower"), ["l", "o"]))
      .toEqual([["lo", "w"], ["lo", "w", "e", "r"]]);
  });

  it("leaves a corpus untouched when the pair is absent", () => {
    expect(mergePair(words("low"), ["z", "q"])).toEqual([["l", "o", "w"]]);
  });

  it("consumes both halves of a match, so the text is preserved", () => {
    // The classic bug: after matching at position i, advance by TWO. Advancing
    // by one re-emits the second half and "banana" comes back "bannanna" —
    // no error thrown, the corpus simply stops being the corpus.
    const merged = mergePair([[..."banana"]], ["a", "n"]);
    expect(merged[0]).toEqual(["b", "an", "an", "a"]);
    expect(merged[0]!.join("")).toBe("banana");
  });

  it("merges a pair of identical tokens without overlapping", () => {
    // "aaa" has two (a a) pairs but they overlap, so only the first can merge.
    expect(mergePair([[..."aaa"]], ["a", "a"])[0]).toEqual(["aa", "a"]);
    expect(mergePair([[..."aaaa"]], ["a", "a"])[0]).toEqual(["aa", "aa"]);
  });

  it("merges tokens that are themselves merges", () => {
    // Round 2 of training: "lo" is not a character. It exists only because
    // round 1 created it, which is what lets BPE build up long tokens.
    expect(mergePair([["lo", "w", "e", "r"]], ["lo", "w"])[0])
      .toEqual(["low", "e", "r"]);
  });

  it("does not mutate the corpus it was given", () => {
    // Training reassigns the result in a loop; a mutating merge would corrupt
    // any caller still holding the old corpus.
    const before = words("low lower");
    mergePair(before, ["l", "o"]);
    expect(before).toEqual([["l", "o", "w"], ["l", "o", "w", "e", "r"]]);
  });

  it("concatenating a merged corpus reproduces the original text", () => {
    // The invariant that holds after ANY merge: merging changes how the text
    // is cut up, never what the text says.
    const merged = mergePair(words(CORPUS), ["e", "s"]);
    expect(merged.map((word) => word.join("")).join(" ")).toBe(CORPUS);
  });
});

describe("BPETokenizer.train", () => {
  it("learns the chapter's eight merge rules in order", () => {
    expect(trained().merges).toEqual(RULES);
  });

  it("gives 4 specials + 11 characters + 8 merges = 23", () => {
    // The vocabulary is three layers: reserved specials, then every character
    // in the corpus, then one entry per merge.
    const tokenizer = trained();
    expect(new Set(CORPUS).size).toBe(11);
    expect(tokenizer.merges).toHaveLength(8);
    expect(tokenizer.vocabSize).toBe(SPECIAL_TOKEN_COUNT + 11 + 8);
  });

  it("numbers tokens from SPECIAL_TOKEN_COUNT, never from 0", () => {
    // ID 0 is <pad>. A real token there would be indistinguishable from
    // padding — the same rule Ch 16 established.
    for (const id of trained().stoi.values()) {
      expect(id).toBeGreaterThanOrEqual(SPECIAL_TOKEN_COUNT);
    }
  });

  it("keeps every character of the corpus in the vocabulary", () => {
    // This is the guarantee that makes <unk> rare: no merge ever removes a
    // character, so any word can still be spelled out one character at a time.
    const tokenizer = trained();
    for (const char of new Set(CORPUS)) expect(tokenizer.stoi.has(char)).toBe(true);
  });

  it("puts the space in the vocabulary even though it never merges", () => {
    // Merges only happen inside a word, so the space can never be half of a
    // pair. It is still a token — encode emits it between words.
    expect(trained().stoi.get(" ")).toBe(SPECIAL_TOKEN_COUNT);
  });

  it("adds one vocabulary entry per merge, holding the joined token", () => {
    const tokenizer = trained();
    for (const [a, b] of RULES) expect(tokenizer.stoi.has(a + b)).toBe(true);
  });

  it("stoi and itos are inverses of each other", () => {
    const { stoi, itos } = trained();
    for (const [token, id] of stoi) expect(itos.get(id)).toBe(token);
  });

  it("is deterministic — two runs on the same text agree", () => {
    // THE reason the tie-break must be fixed. If round 1's 7-7 tie resolved
    // differently between runs, two vocabularies would disagree and a model
    // trained against one would read the other's output as nonsense.
    const first = trained();
    const second = trained();
    expect(first.merges).toEqual(second.merges);
    expect([...first.stoi.entries()]).toEqual([...second.stoi.entries()]);
  });

  it("stops early once no pair occurs more than once", () => {
    // vocabSize is a ceiling, not a promise. Merging a pair seen once invents
    // a token used once — noise in the vocabulary and a wasted embedding row
    // in Ch 18. Asking for 40 on this corpus yields 12 merges and a size of 27.
    const tokenizer = new BPETokenizer();
    tokenizer.train(CORPUS, 40);
    expect(tokenizer.merges).toHaveLength(12);
    expect(tokenizer.vocabSize).toBe(27);
  });

  it("learns no merges at all when nothing repeats", () => {
    // "hello world" has 8 distinct adjacent pairs, all occurring exactly once,
    // so there is no most-frequent pair and nothing for BPE to learn. This is
    // why the chapter cannot use Ch 16's corpus.
    const tokenizer = new BPETokenizer();
    tokenizer.train("hello world", 50);
    expect(tokenizer.merges).toHaveLength(0);
    expect(tokenizer.vocabSize).toBe(SPECIAL_TOKEN_COUNT + new Set("hello world").size);
  });

  it("each merge shortens the corpus", () => {
    // The whole point of BPE: the same text carried in fewer tokens. Going
    // 55 → 48 → 41 → ... → 20 over the eight merges.
    let corpus = words(CORPUS);
    let previous = corpus.flat().length;
    expect(previous).toBe(55);
    for (const rule of RULES) {
      corpus = mergePair(corpus, rule);
      const now = corpus.flat().length;
      expect(now).toBeLessThan(previous);
      previous = now;
    }
    expect(previous).toBe(20);
  });
});

describe("BPETokenizer.encodeWord", () => {
  it("turns a frequent word into a single token", () => {
    // "low" was 3 characters and is now 1 token — merges 1 and 2 built it.
    expect(trained().encodeWord("low")).toEqual(["low"]);
    expect(trained().encodeWord("newest")).toEqual(["newest"]);
  });

  it("splits a word it never saw into pieces it knows", () => {
    // "lowest" is not in the corpus. It comes out as 2 tokens instead of 6
    // characters, using "low" and "est" — both invented by counting alone.
    expect(trained().encodeWord("lowest")).toEqual(["low", "est"]);
  });

  it("falls back to characters where no merge applies", () => {
    // "slowest" is unseen and starts with an unmergeable "s". A word-level
    // tokenizer would emit <unk>; BPE spells it out of pieces it has.
    expect(trained().encodeWord("slowest")).toEqual(["s", "low", "est"]);
  });

  it("applies every rule, including ones that find nothing", () => {
    // "lower" fires rules 1, 2 and 8 and ignores the other five.
    expect(trained().encodeWord("lower")).toEqual(["lowe", "r"]);
  });

  it("gives a worse answer when the rules are applied in reverse", () => {
    // Same word, same eight rules, wrong order: 4 tokens instead of 2.
    // ("lo","w") can only fire once ("l","o") has created "lo", so reversing
    // the order means it finds nothing and the chain never forms.
    const tokenizer = trained();
    expect(tokenizer.encodeWord("lowest")).toEqual(["low", "est"]);
    tokenizer.merges = [...tokenizer.merges].reverse();
    expect(tokenizer.encodeWord("lowest")).toEqual(["lo", "w", "es", "t"]);
  });

  it("never changes what the word says, only how it is cut up", () => {
    for (const word of ["low", "lowest", "slowest", "widest", "lower"]) {
      expect(trained().encodeWord(word).join("")).toBe(word);
    }
  });

  it("returns nothing for an empty word", () => {
    // An empty word is what a double space produces, and it must survive it.
    expect(trained().encodeWord("")).toEqual([]);
  });
});

describe("BPETokenizer.encode and decode", () => {
  it("maps a sentence to the IDs of its tokens", () => {
    // "low" = 16, the space = 4, "lowe" = 22, "r" = 11.
    expect(trained().encode("low lower")).toEqual([16, 4, 22, 11]);
  });

  it("emits the space token between words", () => {
    // Without it the flat list of IDs has no record of where one word ended,
    // and decode could not put the sentence back together.
    const tokenizer = trained();
    const spaceId = tokenizer.stoi.get(" ")!;
    expect(tokenizer.encode("low low")).toEqual([16, spaceId, 16]);
  });

  it("is a round trip for text built from known characters", () => {
    expect(trained().decode(trained().encode(CORPUS))).toBe(CORPUS);
  });

  it("round-trips a word that never appeared in training", () => {
    // The headline improvement over Ch 16: no <unk>, nothing lost.
    expect(trained().decode(trained().encode("slowest"))).toBe("slowest");
  });

  it("preserves repeated spaces", () => {
    // "low  lower" splits into three words, the middle one empty. It emits no
    // tokens, but both space tokens are still emitted, so spacing survives.
    expect(trained().decode(trained().encode("low  lower"))).toBe("low  lower");
  });

  it("loses a character the corpus never contained", () => {
    // The honest limit. There is no 'h' anywhere in low/lower/newest/widest,
    // so 'h' is not in the base vocabulary and encodes to UNK — exactly as in
    // Ch 16. BPE removes <unk> for unseen WORDS, not unseen CHARACTERS.
    const tokenizer = trained();
    expect(tokenizer.encode("the lowest")).toContain(UNK_ID);
    expect(tokenizer.decode(tokenizer.encode("the lowest"))).toBe("te lowest");
  });

  it("skips every ID below SPECIAL_TOKEN_COUNT when decoding", () => {
    // A padded row is the normal thing to decode. Without skipping it would
    // come back with "<pad><pad>" glued on the end.
    const tokenizer = trained();
    expect(tokenizer.decode([16, 0, 0, 0])).toBe("low");
  });

  it("decodes an empty list to an empty string", () => {
    expect(trained().decode([])).toBe("");
  });

  it("needs fewer tokens than the character tokenizer for the same text", () => {
    // 66 characters carried in 31 tokens. Attention in Part 5 costs the square
    // of the sequence length, so this is a 4.5x saving there, not 2.1x.
    expect(CORPUS.length).toBe(66);
    expect(trained().encode(CORPUS)).toHaveLength(31);
  });
});
