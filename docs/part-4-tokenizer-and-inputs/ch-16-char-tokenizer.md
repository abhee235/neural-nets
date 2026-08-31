# Chapter 16: The Character Tokenizer

**Part 4 of 6: Language Model Inputs**
**Build:** `src/tokenizer/char.ts` — `buildVocab`, `CharTokenizer`
**Run:** `bun run exercises/ch-16-char-tokenizer.ts`

---

## Where we left off

Part 3 finished with a network that reads 784 numbers and answers with 10. Every input it has ever seen was **already numeric** — pixel brightnesses, the coordinates of a point, the four rows of XOR.

Language is not numeric. `"hello"` is five characters, and `matMul` has nothing to do with a character.

So Part 4 begins with the least glamorous and most unavoidable job in the whole course: **turning text into numbers the network can multiply.** Nothing here is mathematics. It is bookkeeping, done exactly right, and getting it wrong produces a model that trains happily and learns nonsense.

---

## Words we'll use in this chapter

| word | meaning |
|---|---|
| **token** | the smallest unit the model sees. Here, one character. |
| **vocabulary** | the fixed list of every token the model knows, each with a number |
| **token ID** | the integer standing for one token — a row number in the vocabulary |
| **`stoi`** | "string to integer" — the map from character to ID |
| **`itos`** | "integer to string" — the map back, needed for `decode` |
| **encode** | text → list of IDs |
| **decode** | list of IDs → text |
| **special token** | an ID that stands for something other than a character (padding, unknown, start, end) |
| **padding** | filler added to short sequences so a batch becomes rectangular |
| **mask** | a parallel grid of 1s and 0s recording which cells are real |

---

## 1. The vocabulary is a lookup table

A **tokenizer** is a dictionary with two columns: a token, and the number that stands for it. That is the entire idea. Everything else in this chapter is consequences.

For a *character* tokenizer, the tokens are characters. Scan the text, keep each distinct character once, sort them, and number them.

<div align="center">
  <img src="../assets/ch-16/vocabulary.svg" alt="Building a vocabulary from the text hello world, in three stages. First the text itself, eleven characters in boxes, with the space shown as an open box symbol. Second, keeping each character once and sorting them, leaving eight unique characters: space, d, e, h, l, o, r, w, with a note that sorting is what makes the same text always produce the same IDs. Third, handing out the numbers, with four red boxes first for the reserved special tokens 0 pad, 1 unk, 2 bos and 3 eos, labelled as four slots spent before we start, and then the eight characters in blue boxes numbered 4 through 11. A footer states vocabSize equals 8 unique characters plus 4 special tokens equals 12, and that any character not in the table becomes unk, which is the whole reason id 1 exists." />
</div>

This is the example the rest of the chapter uses. The corpus is `"hello world"`:

```text
  text        "hello world"                        11 characters
  unique      ' '  d  e  h  l  o  r  w              8 of them
  vocabSize   8 + 4 special tokens = 12
```

Eleven characters, eight unique — `l` appears three times and `o` twice, and each gets **one** entry. A vocabulary records what *can* appear, not how often it does.

**Why sort?** Nothing forces it. But without sorting, the IDs depend on the order characters happen to appear, so re-reading the same corpus in a different order gives a different vocabulary — and a saved model's weights would no longer match its tokenizer. Sorting makes the mapping a function of the text alone.

---

## 2. Four IDs are spent before any character gets one

Look again at where the characters start: `' '` is **4**, not 0. IDs 0 to 3 are reserved.

```text
  0   <pad>    filler, so short sequences can share a batch with long ones
  1   <unk>    "a character I was never shown"
  2   <bos>    beginning of sequence
  3   <eos>    end of sequence
```

Two of these earn their place immediately, and two are for later.

**`<unk>` is what makes the tokenizer total.** A vocabulary built from `"hello world"` has never seen `z`. Without a fallback, `encode("zebra")` has no answer — it must either crash or drop the character silently, and silently dropping input is the worse of the two. `<unk>` gives every possible string an encoding.

**`<pad>` is what makes batching possible**, which is section 4.

`<bos>` and `<eos>` mark where a sequence starts and stops. They do nothing in this chapter — a character tokenizer has no use for them yet — but Chapter 28 needs `<bos>` to teach a decoder how to begin generating with nothing written yet, and Chapter 30's GPT needs `<eos>` to know when to stop. Reserving them now means the IDs do not shift later, which would invalidate every weight trained against the old numbering.

> **The rule to remember:** special tokens come first, so `SPECIAL_TOKEN_COUNT` is the offset of the first real character. Change that constant and every ID in the course moves.

---

## 3. Encode and decode

**Encode** is a lookup per character, with `<unk>` as the fallback:

```text
  encode("hello")   h→7  e→6  l→8  l→8  o→9      [7, 6, 8, 8, 9]
  encode("world")   w→11 o→9  r→10 l→8  d→5      [11, 9, 10, 8, 5]
  encode("hi")      h→7  i→?                     [7, 1]
```

That last line is the one to look at. `i` never appears in `"hello world"`, so it is not in the vocabulary, and it becomes `<unk>` — ID **1**. The tokenizer does not fail; it records that something unrecognisable was there.

Both `l`s in `"hello"` encode to the same `8`. **A vocabulary is many-to-one in text and one-to-one in tokens**: every occurrence of a character maps to the same number.

**Decode** walks the other way, through `itos`, and skips anything below `SPECIAL_TOKEN_COUNT`:

```text
  decode([7, 6, 8, 8, 9])  =  "hello"
```

Skipping the specials is what makes the round trip clean — decoding a padded sequence should give back the sentence, not the sentence plus a run of padding.

**The round trip holds for anything the vocabulary has seen:**

```text
  decode(encode("hello world"))  =  "hello world"
```

It does **not** hold once `<unk>` is involved. `encode("hi")` gives `[7, 1]`, and decoding that gives `"h"` — the `i` is gone for good. That is not a bug to fix; it is the honest cost of a fixed vocabulary, and the reason Chapter 17 builds a tokenizer whose vocabulary can cover text it was not built from.

**Truncation.** `encode` takes an optional `maxLen` and cuts the list short:

```text
  encode("hello", 3)  =  [7, 6, 8]
```

Models have a fixed context length, so something has to give when input exceeds it. Cutting is the crude answer, and it is the one used here.

---

## 4. A batch has to be rectangular. Sentences are not.

Chapter 15 fed the network `[64, 784]` — sixty-four images, each exactly 784 numbers. That worked because every MNIST image is the same size.

Sentences are not. `"hello"` is 5 characters and `"hi"` is 2. A `Tensor` is a flat `Float64Array` with a shape (Chapter 01), and there is no shape that means "rows of different lengths".

**So we lie, and then we admit it.**

<div align="center">
  <img src="../assets/ch-16/padding-and-mask.svg" alt="A diagram of encodeBatch on the three strings hello, world and hi with maxLen 6. Each string has a row of six id boxes with its mask directly beneath. Hello gives ids 7, 6, 8, 8, 9 then a red 0 padding cell, with mask 1 1 1 1 1 0. World gives 11, 9, 10, 8, 5 then a red 0, with the same mask. Hi gives 7 then 1 then four red padding zeros, with mask 1 1 0 0 0 0, and a note that the letter i is not in the vocabulary so it becomes unk equals 1. A footer states both tensors are shape 3 by 6, one holding the data and one holding whether each cell is real, and that without the mask attention in Part 5 would happily learn from the padding." />
</div>

**The lie is padding.** Pick a length — here 6 — and fill every sequence out to it with `<pad>`, ID 0. Now all three rows are the same length and the batch is a `[3, 6]` tensor.

**The admission is the mask.** A second tensor of the same shape, holding `1` where the cell is a real token and `0` where it was invented:

```text
  ids                          mask                text
  [ 7,  6,  8,  8,  9,  0]     [1, 1, 1, 1, 1, 0]  "hello"
  [11,  9, 10,  8,  5,  0]     [1, 1, 1, 1, 1, 0]  "world"
  [ 7,  1,  0,  0,  0,  0]     [1, 1, 0, 0, 0, 0]  "hi"
```

Both are `[3, 6]`. Read any column and the mask tells you whether the ID above it means anything.

**Why the mask is not optional.** Padding is data as far as the network is concerned. ID 0 will be looked up in the embedding table (Chapter 18) and produce a vector like any other, and attention (Part 5) will compute a score for it like any other position. Nothing about a zero announces itself as filler. Without the mask, a batch of mostly-short sentences trains the model largely on padding — and it will not error, it will just learn badly, in the way Chapter 15's un-scaled pixels learned badly.

The mask is how Chapter 21 builds the padding mask, and how Part 5's attention is told which positions to ignore. **This chapter's real output is not the IDs. It is the mask.**

---

## 5. One thing worth noticing about the types

`encodeBatch` returns `Tensor`, and a `Tensor` holds a `Float64Array`. So token IDs — which are integers, and are about to be used as **row numbers** into the embedding table in Chapter 18 — travel as floating-point numbers.

That is deliberate, and it has one benefit and one cost.

The benefit: everything downstream already speaks `Tensor`. The mask especially, since Part 5 multiplies attention scores by it, and that multiplication wants a tensor.

The cost: `Float64Array` can hold `7.5`, and nothing in the type stops it. Chapter 18 will have to round or truncate when it uses an ID as an index. Doubles represent every integer up to 2⁵³ exactly, so no ID will ever drift — but the *type* no longer says "integer", and you have to remember that it means one.

Worth knowing now rather than discovering it in two chapters.

---

## What to implement

`src/tokenizer/char.ts`, five pieces:

| | |
|---|---|
| `buildVocab(text)` | unique characters, sorted, numbered from `SPECIAL_TOKEN_COUNT`. Returns `stoi`, `itos`, `vocabSize`. |
| `constructor(text)` | build the vocabulary and keep both maps |
| `encode(text, maxLen?)` | one lookup per character, `<unk>` on a miss, truncate if `maxLen` given |
| `decode(ids)` | `itos` per ID, skipping anything below `SPECIAL_TOKEN_COUNT` |
| `encodeBatch(texts, maxLen)` | encode each, pad to `maxLen`, return `ids` and `mask` as `[texts.length, maxLen]` tensors |

There is no new mathematics and no gradients — nothing in this file is ever differentiated. It is the last chapter of which that is true.

---

## Common pitfalls

**Numbering characters from 0.** Then `'d'` and `<pad>` are both ID 0, and padding is indistinguishable from a real character. Start at `SPECIAL_TOKEN_COUNT`.

**Forgetting to sort.** The vocabulary then depends on the order characters appeared, so the same corpus can produce two different tokenizers, and a saved model no longer matches the one that trained it.

**Building the vocabulary from the wrong text.** It must come from the *training* corpus. Building it from the test set too is a way of leaking information about data the model is not supposed to have seen.

**Decoding padding.** `decode` must skip IDs below `SPECIAL_TOKEN_COUNT`, or a padded row comes back with trailing junk instead of the original sentence.

**Returning the mask with the wrong shape.** It must match `ids` exactly. Part 5 multiplies them together, and a shape mismatch there will broadcast into something that runs and is wrong.

---

## How to verify

```bash
bun test src/tokenizer/char.test.ts
bun run exercises/ch-16-char-tokenizer.ts
```

With the corpus `"hello world"`, a correct implementation gives exactly:

```text
  vocabSize                 12
  encode("hello")           [7, 6, 8, 8, 9]
  encode("world")           [11, 9, 10, 8, 5]
  encode("hi")              [7, 1]
  encode("hello", 3)        [7, 6, 8]
  decode([7, 6, 8, 8, 9])   "hello"
  decode(encode("hello world"))  "hello world"

  encodeBatch(["hello","world","hi"], 6)
    ids.shape   [3, 6]
    mask.shape  [3, 6]
    row 2 ids   [7, 1, 0, 0, 0, 0]
    row 2 mask  [1, 1, 0, 0, 0, 0]
```

Every number above was produced by running the implementation, not by hand.

---

## What you should be able to explain

1. Why the first real character has ID 4 and not 0.
2. What `encode("hi")` returns for a vocabulary built from `"hello world"`, and why nothing goes wrong.
3. Why `decode(encode(s))` returns `s` for some strings and not others.
4. Why a batch needs padding at all, given that `encode` handles any length.
5. What breaks if `encodeBatch` returns the IDs but not the mask.
6. Why `<bos>` and `<eos>` are reserved now when nothing uses them until Chapter 28.

---

## Next chapter

**Chapter 17 — the BPE tokenizer.** A character vocabulary is tiny and never truly unknown, but it makes sequences long: `"hello world"` is 11 tokens, and a model's cost grows with sequence length. Word-level vocabularies are short but explode in size and hit `<unk>` constantly. Byte-pair encoding sits between them, learning its tokens from the data rather than being told what they are — and it is what GPT actually uses.
