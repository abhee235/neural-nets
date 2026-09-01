# Chapter 17: The BPE Tokenizer

**Part 4 of 6: Language Model Inputs**
**Build:** `src/tokenizer/bpe.ts` — `countPairs`, `mergePair`, `BPETokenizer`
**Run:** `bun run exercises/ch-17-bpe-tokenizer.ts`

---

## Where we left off

Chapter 16 built a tokenizer whose vocabulary is a list of characters. It works, and it has one guarantee worth keeping: **every possible string encodes**, because any character the vocabulary has not seen becomes `<unk>`.

It also has one cost, and the cost is the length of what comes out.

```text
  "hello world"    →  11 tokens        one per character
```

Eleven tokens for eleven characters, and there is no way to do better — a character tokenizer emits exactly as many tokens as the text has characters. That matters because in Part 5, attention compares every position with every other position, so its work grows with the *square* of the sequence length. Halving the token count quarters that work; section 7 does that arithmetic on this chapter's own corpus.

The obvious fix is to make each token a whole word. That fails in the other direction. A word vocabulary large enough for English runs to hundreds of thousands of entries, most of them seen a handful of times, and it still meets a word it has never seen on almost every page — at which point it emits `<unk>` and the information is gone.

So the two obvious choices are:

| | vocabulary | sequence length | unseen input |
|---|---|---|---|
| characters (Ch 16) | tiny — 12 entries for `"hello world"` | longest possible | fine, unless the *character* is new |
| words | enormous | shortest | `<unk>`, constantly |

**Byte-pair encoding sits between them, and the reason it is interesting is that nobody chooses its tokens.** It is handed a corpus and it works out its own vocabulary, by repeatedly gluing together whichever two neighbouring pieces occur most often. Common words end up as one token. Rare words end up as a few pieces. Nothing is ever unseen, because the pieces bottom out at characters.

This is what GPT actually uses.

---

## A note on the example

The rest of Part 4 uses `"hello world"`. This chapter cannot, and it is worth seeing exactly why, because the reason *is* the algorithm.

BPE's first step is "find the most frequent adjacent pair". Here is every adjacent pair in `"hello world"`, counted:

```text
  h e   →  1        w o   →  1
  e l   →  1        o r   →  1
  l l   →  1        r l   →  1
  l o   →  1        l d   →  1
```

Eight distinct pairs, and **all eight are tied at one**. There is no most frequent pair. The text is too short and too varied for anything to repeat, so BPE has nothing to learn from it.

So this chapter uses the corpus the BPE literature uses, which is chosen to make the merges legible:

```text
  low ×5    lower ×2    newest ×3    widest ×2
```

Written out as one string, that is the twelve words the chapter trains on. Its most frequent pair occurs **seven** times, which is a fact rather than a tie.

---

## Words we'll use in this chapter

| word | meaning |
|---|---|
| **pair** | two tokens that sit next to each other, like `l` then `o` |
| **merge** | replacing every occurrence of one pair with a single new token |
| **merge rule** | the record of one merge, `("l","o")`, kept so encoding can repeat it |
| **subword** | a token that is more than one character but less than a word — `est` |
| **base vocabulary** | the characters BPE starts from, before any merge |
| **corpus** | the training text, held as a list of words, each a list of tokens |
| **greedy** | takes the best option available right now, never revisiting it |

---

## 1. Training: count, merge, repeat

The whole algorithm is five lines.

```text
  1.  split the corpus into words, and each word into characters
  2.  count every adjacent pair across the whole corpus
  3.  take the pair with the highest count
  4.  replace that pair everywhere, as one new token; write the rule down
  5.  go back to 2, until the vocabulary is big enough
```

That is it. There is no gradient here and nothing is learned by optimisation — it is counting, in a loop.

Watch it run. The animation below is one frame per merge, showing four of the twelve words; the token created each round is outlined in red.

<div align="center">
  <img src="../assets/ch-17/merge-rounds.svg" alt="An animation of BPE training on the corpus low times five, lower times two, newest times three and widest times two, cycling through nine frames. The first frame shows the four distinct words low, lower, newest and widest each broken into single-character boxes, labelled before any merge, with 55 tokens across the whole corpus. Each following frame is headed with the merge it performed and shows the corpus after it. Merge one joins l and o into lo, leaving 48 tokens. Merge two joins lo and w into low, leaving 41. Merge three joins e and s into es, leaving 36. Merge four joins es and t into est, leaving 31. Merge five joins n and e into ne, leaving 28. Merge six joins ne and w into new, leaving 25. Merge seven joins new and est into newest, leaving 22. Merge eight joins low and e into lowe, leaving 20. In every frame the newly created token is drawn in a red outlined box while all other tokens are blue, and a footer notes that each merge replaces a pair everywhere at once, so the corpus shortens and the next round counts different pairs." />
</div>

Two things in that animation are the point of the chapter.

**The corpus shrinks, every round.** 55 tokens, then 48, 41, 36, 31, 28, 25, 22, 20. That is what BPE is *for*: the same text, carried in fewer tokens.

**The counts change as you go.** A merge rewrites the corpus, so the next round is counting pairs in a different sequence from the one before. Here are the top pairs at the start of each round, alongside what was chosen:

```text
  round   the leading pairs                          merged        tokens after
    1     (l o)=7  (o w)=7  (w e)=5  (e s)=5         ("l","o")          48
    2     (lo w)=7  (w e)=5  (e s)=5  (s t)=5        ("lo","w")         41
    3     (e s)=5  (s t)=5  (n e)=3  (e w)=3         ("e","s")          36
    4     (es t)=5  (n e)=3  (e w)=3  (w es)=3       ("es","t")         31
    5     (n e)=3  (e w)=3  (w est)=3  (low e)=2     ("n","e")          28
    6     (ne w)=3  (w est)=3  (low e)=2  (e r)=2    ("ne","w")         25
    7     (new est)=3  (low e)=2  (e r)=2  (w i)=2   ("new","est")      22
    8     (low e)=2  (e r)=2  (w i)=2  (i d)=2       ("low","e")        20
```

Look at round 2. The pair `(lo w)` is counted 7 times — and `lo` **did not exist** when round 1 counted. Round 1 created it. This is the mechanism that makes BPE build up long tokens out of short ones, and it is the subject of the next section.

Look also at what came out. `est` is a real English suffix and `new` is a real English word, and nothing in the algorithm knows what a suffix or a word is. They fell out of counting.

**BPE is greedy.** At each round it takes the highest count available and never reconsiders. Round 8 merging `("low","e")` to make `lowe` is a slightly silly token, but it was the best count on offer at the time and BPE does not go back and trade it for something better. Nobody has a practical way to do better; the greedy answer is what every production tokenizer uses.

---

## 2. Merges compose — which is why order matters

Round 2 merged `("lo","w")`. That rule can only ever fire on a sequence that already contains `lo`, and the only thing that produces `lo` is round 1's rule.

So the merge rules are not eight independent facts. They are a **chain**, and each link depends on the ones before it.

<div align="center">
  <img src="../assets/ch-17/merge-tree.svg" alt="Two tree diagrams read from the bottom up, showing how merges compose. On the left, the characters l, o and w sit in blue boxes at the base. Merge one joins l and o into a red box lo one level up, and merge two joins that lo box with the w below it into a red box low at the top. On the right, the six characters n, e, w, e, s and t sit at the base. Merge three joins the second e with s into es, merge four joins es with t into est, merge five joins n with the first e into ne, merge six joins ne with w into new, and merge seven joins new with est at the top into a single box newest. Each composed token is drawn in a red outlined box with its merge number beside it, and lines connect every parent token down to the two tokens it consumed. A footer notes that est is a real English suffix and that nothing told the algorithm about suffixes, it counted pairs." />
</div>

Read either tree from the bottom. `newest` is one token by round 7, but getting there took five merges stacked on each other, and `new + est` at the top is only possible because rounds 3 to 6 built `est` and `new` first.

This is why **the merge rules are stored in order, and replayed in order.** They are not a set of rewrites you may apply in any sequence; they are a recipe whose steps depend on each other.

---

## 3. Encoding replays the rules

Training produces two things: a vocabulary, and the ordered list of merge rules. Encoding a new piece of text uses the rules, and it is simpler than training — there is no counting.

```text
  1.  split the text into words, and each word into characters
  2.  for each merge rule, in the order it was learned:
        replace every occurrence of that pair in the word
  3.  look up each surviving token's ID
```

Every rule is tried, once, in order. Most of them will find nothing to do, and that is fine.

Here is `"lowest"` going through all eight rules — on the left in the order they were learned, on the right the exact same eight rules in reverse:

<div align="center">
  <img src="../assets/ch-17/order-matters.svg" alt="A side-by-side comparison of encoding the word lowest with the same eight merge rules applied in two different orders. Each side lists nine rows: the starting sequence, then one row per merge rule, with the token sequence after that rule. A rule that changed the sequence is marked with a red arrow and its label is bold; a rule that found nothing to merge is marked with a faint dot. The left column, headed training order and labelled correct, starts from l o w e s t and fires four rules in turn: l and o become lo, then lo and w become low, then e and s become es, then es and t become est, leaving low and est. The remaining four rules find nothing. Its result is two tokens. The right column, headed reversed and labelled same rules wrong order, applies the rules from last to first. The first five rules all find nothing, then e and s become es, the next rule finds nothing, and finally l and o become lo, leaving lo, w, es and t. Its result is four tokens. A footer explains that the rule joining lo and w can only fire after the rule joining l and o has created lo, so a rule consumes what an earlier rule produced." />
</div>

In training order, four rules fire and `"lowest"` comes out as **2 tokens**, `low | est`. Reversed, only two rules fire and it comes out as **4 tokens**, `lo | w | es | t`.

The reason is visible in the right-hand column. `("lo","w")` is tried *before* `("l","o")` has run, so at that moment there is no `lo` anywhere in the word and the rule does nothing. By the time `lo` finally appears, on the last line, the rule that would have consumed it is already spent.

Same rules, same word, worse answer. **Order is not a convention here; it is the algorithm.**

---

## 4. Where the space goes

One decision has been quietly doing work throughout: step 1 splits the text into **words**, and pairs are only ever counted *inside* a word. A pair never straddles a space.

This matters. Without the split, BPE would happily learn a token like `w␣n` spanning the gap between `low` and `newest` — an artifact of two words that happened to sit next to each other, useless anywhere else.

But if the words are split apart, something has to put the spaces back, or `decode` cannot reproduce the input. The answer is the simplest one available:

**The space is an ordinary token in the vocabulary. It just never merges with anything, because merges only happen inside words.**

`encode` emits that token between consecutive words, and `decode` concatenates every token string it is given. The two are exact inverses:

```text
  encode("low lower")   →  low  ␣  lowe  r          4 ids
  decode(...)           →  "low lower"
```

It holds for awkward input too. `"low  lower"` has two spaces, which splits into three words where the middle one is empty and contributes no tokens at all — so two space tokens are emitted, and the round trip is exact:

```text
  decode(encode("low  lower"))  =  "low  lower"
```

Chapter 16's round trip broke as soon as `<unk>` was involved. This one does not break on spacing, which is a genuine improvement — and Chapter 30 depends on it, because a model that generates text has to be able to turn its output back into a string.

> **What real BPE does instead.** GPT-2 attaches the space to the *front of the following word*, so `"low"` at the start of a sentence and `" low"` in the middle are two different tokens. It is why token counts for the same word differ depending on where it sits, and why GPT tokenizers look strange when you inspect them. The effect on the algorithm is nil — it just changes what the words are — so this chapter uses the plainer split.

---

## 5. The vocabulary, and what `<unk>` still costs

The vocabulary is built in three layers:

```text
   0 – 3     the four special tokens          <pad> <unk> <bos> <eos>
   4 – 14    every character in the corpus    ' ' d e i l n o r s t w
  15 – 22    one entry per merge, in order    lo low es est ne new newest lowe
                                              ──
                                     vocabSize 23
```

The middle layer is worth a second look: **that is exactly Chapter 16's vocabulary.** BPE does not replace the character tokenizer, it starts from it and adds tokens on top. Every character stays in the vocabulary permanently, which is what guarantees a word can always be spelled out even if no merge applies to it.

That guarantee is the payoff:

```text
  encode("slowest")  →  s | low | est          3 tokens, round trip exact
```

`"slowest"` does not appear anywhere in the training corpus. A word tokenizer would emit `<unk>` and lose it. BPE spells it out of pieces it already has, and `decode` gets it back perfectly.

**But `<unk>` has not gone away.** It has only moved.

```text
  decode(encode("the lowest"))  =  "te lowest"
```

The `h` is missing. The training corpus — `low lower newest widest` — contains no `h` at all, so `h` is not in the base vocabulary, so it encodes to `<unk>` and is lost, exactly as in Chapter 16.

So the honest statement is: **BPE removes `<unk>` for unseen *words*, not for unseen *characters*. An unseen word is decomposed; an unseen character has nothing to decompose into.

The real systems close that last gap by making the base vocabulary bytes instead of characters. There are only 256 possible bytes, every one of them is put in the vocabulary up front, and any text whatsoever — any language, any emoji, any corrupted file — is a sequence of bytes. `<unk>` then becomes impossible rather than merely rare. That is the "byte" in byte-pair encoding, and it is what GPT-2 does. This chapter works on characters, which is the same algorithm with a smaller alphabet.

---

## 6. Ties, and why they must be broken the same way twice

Round 1 of the training run is a tie:

```text
  (l o) = 7      (o w) = 7
```

Both pairs occur seven times, and the algorithm as written has no opinion about which to take. Something in the implementation will decide — the order a `Map` happens to iterate, or the order the counts were inserted — and whatever it is, **it must give the same answer every time the same corpus is trained.**

If it does not, two training runs on the same text produce different merge rules, therefore different vocabularies, therefore different IDs. A model trained against the first will read the second's output as nonsense. This is the same failure Chapter 16 avoided by sorting the characters before numbering them, arriving from a different direction.

Taking the first pair encountered while scanning left to right is deterministic, and is all this needs. Just be sure it *is* deterministic rather than accidentally so.

---

## 7. Where these tokens actually go

A tokenizer is not part of the model. It is the thing that runs before the model can start, and the list of integers it produces is the model's **only** input. Here is the whole path from this chapter to the transformer, and which chapter builds each step:

```text
    "low lower"
        │   encode                                    ← this chapter
        ▼
    [16, 4, 22, 11]              4 token IDs
        │   embedding lookup                          ← Ch 18
        ▼
    [4, 16]                      4 rows of 16 numbers, one row per token
        │   + positional encoding                     ← Ch 19
        ▼
    [4, 16]                      same shape, now carrying position too
        │   attention                                 ← Ch 22
        ▼
    [4, 16]                      each row rewritten using the others
```

An ID on its own is useless to a network — ID 16 is not "sixteen times" anything, and the fact that `low` is 16 while `lowe` is 22 says nothing about how alike they are. [Chapter 18](ch-18-token-embeddings.md) fixes that by giving every vocabulary entry its own row of learned numbers. But it can only do that because this chapter decided **what the entries are**.

Two numbers settled here control that pipeline, and they pull in opposite directions.

**Sequence length sets what attention costs.** [Self-attention](../part-5-attention/ch-22-self-attention.md) compares every position with every other position, which means it builds a `seqLen × seqLen` grid of scores — one for each ordered pair of positions. That grid is the reason sequence length is not a minor detail:

```text
  characters (Ch 16)   66 tokens  →  66 × 66  =  4,356 scores
  BPE        (Ch 17)   31 tokens  →  31 × 31  =    961 scores
```

The sequence got 2.13× shorter and the work got **4.53× smaller**, because the saving is squared. This is the single biggest reason BPE exists.

**Vocabulary size sets how big the model is before a single layer is added.** One embedding row per entry, at Chapter 18's `dModel` of 16:

```text
  character vocabulary   15 × 16  =  240 parameters
  BPE vocabulary         23 × 16  =  368 parameters
```

And the same `vocabSize` appears again at the far end: to predict the next token, [Chapter 30's GPT](../part-6-transformer/ch-30-decoder-only-gpt.md) projects back out to one score per vocabulary entry, then softmaxes over them. Every entry you add is paid for twice.

**So the whole trade, on this corpus, is: BPE costs 128 extra parameters and saves 3,395 attention scores on every sequence.** That is not a close call, and it stays not-close as the model grows.

### The same trade at GPT-2's scale

```text
  vocabulary  50,257        dModel  768        context  1,024 tokens
  token embedding table     50,257 × 768  =  38,597,376 parameters
                                          =  31.1% of GPT-2's ~124M
```

Nearly a third of that model is the vocabulary. That is the pressure *against* making it bigger. The pressure *for* it is the context window: GPT-2 can hold 1,024 tokens, and at roughly four characters per token in ordinary English, that window carries about 4,000 characters. Character-level, the same 1,024 slots would carry 1,024 characters — the same compute budget holding a quarter of the text.

Those two pressures are why every production vocabulary lands in the same band:

| model | vocabulary | how it is built |
|---|---|---|
| byte-level base, no merges | 256 | every possible byte — the floor |
| LLaMA | 32,000 | SentencePiece BPE |
| GPT-2 | 50,257 | 50,000 merges + 256 bytes + 1 `<endoftext>` |
| GPT-4 (`cl100k`) | ~100,000 | tiktoken |

There is no principle that makes 50,000 correct. It is a chosen point on the curve between "sequences too long" and "embedding table too large", and the number of merges is the dial that sets it. In this chapter that dial is the `vocabSize` argument to `train`.

---

## What to implement

`src/tokenizer/bpe.ts`, five pieces:

| | |
|---|---|
| `countPairs(corpus)` | count every adjacent pair across all sequences; key them so the pair can be recovered |
| `mergePair(corpus, pair)` | replace every occurrence of the pair with the joined token, in every sequence |
| `train(text, vocabSize)` | base vocabulary from the characters, then merge the top pair until the vocabulary is big enough or nothing repeats |
| `encode(text)` | split to words and characters, replay every merge rule in order, look up IDs, emit the space token between words |
| `decode(ids)` | token string per ID, skipping anything below `SPECIAL_TOKEN_COUNT`, concatenated |

`PAD_ID`, `UNK_ID` and `SPECIAL_TOKEN_COUNT` are imported from `char.ts` — the two tokenizers share one numbering, so `<pad>` means the same thing to both.

Like Chapter 16, nothing in this file is ever differentiated. It is the last chapter of which that is true.

---

## Common pitfalls

**Not skipping the second half of a merged pair.** After matching a pair at position `i`, the loop must advance past *both* tokens. If it advances by one, the second token gets emitted a second time and the text silently grows:

```text
  merge ("a","n") on "banana"
     correct   b | an | an | a     →  "banana"
     buggy     b | an | n | an | n | a     →  "bannanna"
```

Nothing throws. The corpus just quietly stops being the corpus.

**Applying merge rules out of order at encode time.** Covered in section 3 — the result is still valid tokens, just more of them, so tests that only check the round trip will pass while the tokenizer silently does a worse job.

**Letting a merge cross a word boundary.** Count pairs *within* each sequence, never across the join between two of them. `countPairs` taking a `string[][]` rather than a flat `string[]` is what enforces this — the boundary between two words is the boundary between two arrays, and a loop over one array cannot see past its end.

**Breaking ties non-deterministically.** Section 6. Two runs must agree.

**No stopping rule.** Once no pair occurs more than once, every remaining "merge" is a token that will be used exactly once — pure noise in the vocabulary. Training must stop then, even if the requested `vocabSize` has not been reached. On this corpus, asking for 40 gives 12 merges and a vocabulary of 27, and that is correct behaviour, not a failure.

**Numbering the vocabulary from 0.** `vocabSize` starts at `SPECIAL_TOKEN_COUNT`, not at zero, or the first character collides with `<pad>` and padding becomes indistinguishable from real text. Same trap as Chapter 16, and the reason both tokenizers import the constant from the same place rather than each declaring its own.

---

## How to verify

```bash
bun test src/tokenizer/bpe.test.ts
bun run exercises/ch-17-bpe-tokenizer.ts
```

Training on `"low low low low low lower lower newest newest newest widest widest"` with `vocabSize` 23, a correct implementation gives exactly:

```text
  merge rules, in order
    ("l","o")  ("lo","w")  ("e","s")  ("es","t")
    ("n","e")  ("ne","w")  ("new","est")  ("low","e")

  vocabSize                     23        4 specials + 11 characters + 8 merges
  merged token IDs              lo=15  low=16  es=17  est=18
                                ne=19  new=20  newest=21  lowe=22

  encodeWord("low")             low                 1 token   (was 3 characters)
  encodeWord("lowest")          low | est           2 tokens  (was 6)
  encodeWord("newest")          newest              1 token   (was 6)
  encodeWord("slowest")         s | low | est       3 tokens  — never seen in training
  encodeWord("lower")           lowe | r            2 tokens

  decode(encode("low lower"))        "low lower"
  decode(encode("low  lower"))       "low  lower"     two spaces survive
  decode(encode("the lowest"))       "te lowest"      'h' is not in the corpus

  whole corpus, characters (Ch 16)   66 tokens
  whole corpus, BPE      (Ch 17)     31 tokens
```

Every number above was produced by running the implementation, not by hand.

---

## What you should be able to explain

1. Why `"hello world"` is a useless corpus for BPE, in terms of pair counts.
2. Why round 2 can merge `("lo","w")` when `lo` is not a character.
3. What `"lowest"` encodes to if the merge rules are applied in reverse, and why that is worse rather than wrong.
4. Why `"slowest"` needs no `<unk>` but `"the"` does.
5. What breaks if two training runs on the same corpus break a tie differently.
6. Why `countPairs` takes a list of sequences instead of one flat list of tokens.
7. Why halving the number of tokens cuts attention's work by four rather than two.
8. Why a bigger vocabulary is paid for twice — at the embedding table and again at the output.

---

## Further reading

- [Gage, 1994 — A New Algorithm for Data Compression](https://www.derczynski.com/papers/archive/BPE_Gage.pdf) — the original, about compressing files. Not about language at all.
- [Sennrich, Haddow, Birch, 2016 — Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — the paper that brought BPE to NLP, and the source of the `low / lower / newest / widest` corpus.
- [Karpathy — minbpe](https://github.com/karpathy/minbpe) — a minimal byte-level BPE, worth reading after this chapter to see what changes when the base vocabulary is bytes.
- [Google — SentencePiece](https://github.com/google/sentencepiece) — how production systems handle whitespace and languages that do not use spaces at all.

---

## Next chapter

**Chapter 18 — token embeddings.** Every chapter of Part 4 so far has produced integers, and an integer is a terrible thing to feed a network: ID 16 is not "sixteen times" anything, and the fact that `low` is 16 and `lowe` is 22 says nothing about how similar they are. Chapter 18 replaces each ID with a row of learned numbers — the first parameters in the course that exist purely to describe *meaning*.
