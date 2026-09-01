# Chapter 17: The BPE Tokenizer

**Part 4 of 6 — Language Model Inputs**
**Build:** [`src/tokenizer/bpe.ts`](../../src/tokenizer/bpe.ts)
**Tests:** [`src/tokenizer/bpe.test.ts`](../../src/tokenizer/bpe.test.ts)
**Exercise:** [`exercises/ch-17-bpe-tokenizer.ts`](../../exercises/ch-17-bpe-tokenizer.ts)

---

## We have a tokenization problem

Chapter 16 gave us a character tokenizer. It has a useful property: every character can be represented. For example, `"lowest"` becomes `l o w e s t`.

But there is a cost. Six characters become six tokens. For a long sentence, that can become a very long sequence. Later, attention will work with those positions, so fewer tokens can mean much less computation.

So perhaps we should make the tokens larger.

---

## 1. What if a token were a whole word?

That would make the sequence much shorter — `"the cat sat"` becomes `the | cat | sat`. But now imagine a word that was not in the vocabulary, like `"the cats sat"`. If `cats` is unknown, a word tokenizer may have to use `cats → <unk>`, and the original word is lost.

So we have two extremes:

```text
characters                     words
→ everything can be spelled    → sequences are short
→ but sequences are long       → but unseen words are a problem
```

We want something in between.

---

## 2. What if common pieces became tokens?

Look at these words:

```text
low  lower  lowest        all contain    low
new  newest               share          new
```

So instead of choosing a vocabulary by hand, we could let the training text show us which pieces are worth keeping together. The basic idea is simple:

> Find a frequent neighboring pair and merge it into one token. Then do it again.

That is the idea behind BPE — Byte Pair Encoding. Let's discover it before writing the algorithm.

---

## 3. Let's make a token ourselves

Start with five copies of `low`. At first, every word is characters:

```text
l o w
l o w
l o w
l o w
l o w
```

Look at the neighboring pairs. Both `l o` and `o w` occur repeatedly. Suppose we choose `l o`, and merge every occurrence:

```text
lo w        ← we have just created a new token: lo
lo w
lo w
lo w
lo w
```

Now count pairs again. `lo w` is now the repeated pair, so merge again:

```text
low
low
low
low
low
```

We started with three character tokens per word and created one useful token: `low`. The tokenizer did not know that `low` was an English word. It only knew that certain pieces occurred together often. That is the central idea of BPE.

---

## 4. Now give the tokenizer more interesting text

Use this small corpus: `low low low low low lower lower newest newest newest widest widest`. Start with characters.

```text
l o w          l o w e r         n e w e s t        w i d e s t
l o w          l o w e r         n e w e s t        w i d e s t
l o w                            n e w e s t
l o w
l o w
```

Now perform the same game: count pairs → choose the most frequent pair → merge it everywhere → count again → repeat. The animation below runs exactly that loop on this corpus. Each frame is one merge, the new token is outlined in red, and the token count falls underneath.

<div align="center">
  <img src="../assets/ch-17/merge-rounds.svg" alt="An animation of BPE training on the corpus low times five, lower times two, newest times three and widest times two, cycling through nine frames. The first frame shows the four distinct words low, lower, newest and widest each broken into single-character boxes, labelled before any merge, with 55 tokens across the whole corpus. Each following frame is headed with the merge it performed and shows the corpus after it. Merge one joins l and o into lo, leaving 48 tokens. Merge two joins lo and w into low, leaving 41. Merge three joins e and s into es, leaving 36. Merge four joins es and t into est, leaving 31. Merge five joins n and e into ne, leaving 28. Merge six joins ne and w into new, leaving 25. Merge seven joins new and est into newest, leaving 22. Merge eight joins low and e into lowe, leaving 20. In every frame the newly created token is drawn in a red outlined box while all other tokens are blue, and a footer notes that each merge replaces a pair everywhere at once, so the corpus shortens and the next round counts different pairs." />
</div>

One possible beginning is `l o → lo`, then `lo w → low`. Later we can build `e s → es`, then `es t → est`, and `n e → ne`, then `ne w → new`. Eventually, repeated pieces can combine into larger pieces such as `new + est → newest`.

So BPE gradually moves from characters toward useful subwords, and sometimes whole words.

---

## 5. Something important happened after every merge

When we merged `l o → lo`, the corpus changed. That means the pair counts changed too. Before the merge, we counted `l o`. After the merge, the next round might count `lo w`.

So BPE cannot count the pairs once and stop. It must repeatedly count, change the corpus, count again, change the corpus again. This is why BPE builds tokens step by step.

---

## 6. What exactly does BPE learn?

There are two different things — the token strings that exist, and the operations that created them:

```text
  vocabulary            merge rules

  l                     ("l",  "o")  → "lo"
  o                     ("lo", "w")  → "low"
  w                     ("e",  "s")  → "es"
  lo                    ("es", "t")  → "est"
  low                   ...
  es
  est
  ...
```

The merge rules are especially important because we will need them later when we encode text that the tokenizer has never seen. Think of them as a recipe: start with small pieces, apply rule 1, apply rule 2, apply rule 3, and so on.

---

## 7. Why does merge order matter?

Suppose we have `("l","o") → "lo"` and `("lo","w") → "low"`. To turn `l o w` into `low`, we must first create `lo`:

```text
l o w
  ↓
lo w
  ↓
low
```

If we try the second rule first, there is no `lo` yet. The diagram below shows the same thing for two words at once. Read it from the bottom up: every red box sits on top of the pieces an earlier merge produced.

<div align="center">
  <img src="../assets/ch-17/merge-tree.svg" alt="Two tree diagrams read from the bottom up, showing how merges compose. On the left, the characters l, o and w sit in blue boxes at the base. Merge one joins l and o into a red box lo one level up, and merge two joins that lo box with the w below it into a red box low at the top. On the right, the six characters n, e, w, e, s and t sit at the base. Merge three joins the second e with s into es, merge four joins es with t into est, merge five joins n with the first e into ne, merge six joins ne with w into new, and merge seven joins new with est at the top into a single box newest. Each composed token is drawn in a red outlined box with its merge number beside it, and lines connect every parent token down to the two tokens it consumed. A footer notes that est is a real English suffix and that nothing told the algorithm about suffixes, it counted pairs." />
</div>

So the merge rules are not just a set of replacements. They are an ordered sequence, and encoding must replay them in the same order in which they were learned.

---

## 8. BPE training in five steps

Now that we have seen it happen, the algorithm is easy to state.

```text
1. Start with the base vocabulary.
2. Count every neighboring pair.
3. Pick the most frequent pair.
4. Merge that pair everywhere and save the rule.
5. Repeat until the vocabulary is large enough.
```

If no useful repeated pair remains, stop. There is no gradient here, no loss, no optimizer. BPE training is simply `count → merge → count → merge → ...`

---

## 9. Now we have a tokenizer. Can it handle a new word?

Suppose training never contained `slowest`. A word tokenizer might not know it. Our BPE tokenizer can reuse pieces it already learned, giving `s | low | est`. The complete word was never seen, but its pieces were. That is the key advantage of subword tokenization:

> We do not need a token for every possible word. We need reusable pieces that can build many words.

---

## 10. Training and encoding are different

This distinction is important. During training, the tokenizer discovers rules: count → choose → merge → repeat. During encoding, the rules already exist, and we simply replay them:

```text
text  →  base pieces  →  merge rule 1  →  merge rule 2  →  ...  →  token IDs
```

There is no counting during encoding. There are no new merges. The tokenizer is only following the recipe it learned during training.

---

## 11. Encode `lowest` step by step

Suppose the learned rules include `("l","o") → "lo"`, `("lo","w") → "low"`, `("e","s") → "es"` and `("es","t") → "est"`. Starting from characters and applying them in order:

```text
l o w e s t
    ↓   rule 1   ("l","o")  → "lo"
lo w e s t
    ↓   rule 2   ("lo","w") → "low"
low e s t
    ↓   rule 3   ("e","s")  → "es"
low es t
    ↓   rule 4   ("es","t") → "est"
low est
```

So `lowest → low | est`. Six characters became two tokens.

The left column below is that same walk, with every learned rule tried in turn. The right column applies the identical rules in reverse order — which is what section 7 warned about.

<div align="center">
  <img src="../assets/ch-17/order-matters.svg" alt="A side-by-side comparison of encoding the word lowest with the same eight merge rules applied in two different orders. Each side lists nine rows: the starting sequence, then one row per merge rule, with the token sequence after that rule. A rule that changed the sequence is marked with a red arrow and its label is bold; a rule that found nothing to merge is marked with a faint dot. The left column, headed training order and labelled correct, starts from l o w e s t and fires four rules in turn: l and o become lo, then lo and w become low, then e and s become es, then es and t become est, leaving low and est. The remaining four rules find nothing. Its result is two tokens. The right column, headed reversed and labelled same rules wrong order, applies the rules from last to first. The first five rules all find nothing, then e and s become es, the next rule finds nothing, and finally l and o become lo, leaving lo, w, es and t. Its result is four tokens. A footer explains that the rule joining lo and w can only fire after the rule joining l and o has created lo, so a rule consumes what an earlier rule produced." />
</div>

Same rules, same word, four tokens instead of two.

---

## 12. What does the tokenizer do with spaces?

For this implementation, words are processed separately. So `low newest` is treated as two sequences, `low` and `newest`. A merge can happen inside a word, but not across the boundary between words. The space is kept in the token stream so that decoding can reconstruct supported input exactly.

Real tokenizers can handle spaces differently — GPT-2, for example, incorporates whitespace into tokens in a different way. That is an implementation detail. The BPE idea is unchanged: start with small pieces, merge frequent neighbors, replay the learned merges.

---

## 13. What if a character was never seen?

Our implementation starts from characters. That means BPE can decompose an unseen word into smaller pieces, but it cannot invent a character that was never in the base vocabulary. For example, if the training corpus contains no `h`, then `the` cannot be completely represented using the learned character vocabulary.

This is why real production tokenizers often use bytes as their base vocabulary instead. There are only 256 possible byte values, so any text can be represented at the base level. We use characters in this chapter because the merging algorithm is easier to see.

---

## 14. Why does fewer tokens matter?

BPE is partly a compression method. For a language model, that compression can also reduce computation. Later, attention compares positions with one another, so its work grows roughly like `seqLen²`.

Suppose a piece of text takes 60 tokens with character tokenization and 30 tokens with BPE. The sequence is only two times shorter, but the pairwise work changes from `60² = 3600` to `30² = 900` — four times less pairwise work.

So a tokenizer affects not just how text is represented, but also how much computation the model needs.

---

## 15. There is a trade-off

A larger vocabulary can create larger pieces, and so shorter sequences. But a larger vocabulary also means more entries in the embedding table, and later more possible output classes. So vocabulary size is a trade-off:

```text
small vocabulary          large vocabulary
→ longer sequences        → shorter sequences
                          → more vocabulary parameters
```

BPE gives us a practical way to choose a point between those two extremes.

---

## 16. A small implementation detail: ties

Sometimes two pairs have exactly the same frequency — for example `(l, o) = 7` and `(o, w) = 7`. There is no unique mathematical winner, but the implementation still needs to make a deterministic choice. Otherwise the same training text could produce different merge rules on different runs. So:

> Same corpus should produce the same vocabulary and merge rules.

The exact tie-breaking policy is less important than making it deterministic.

---

## 17. Another implementation detail: merging safely

Suppose we merge `(a, b)` inside `a b c`. We want `ab c`. The implementation must move past both original tokens after a match — otherwise the second half of the pair can be processed again, silently corrupting the sequence. This kind of bug is particularly dangerous because the program may still return strings that look almost correct.

---

## 18. Build the tokenizer

Implement these pieces in [`src/tokenizer/bpe.ts`](../../src/tokenizer/bpe.ts).

**`countPairs(corpus)`** — look at neighboring tokens inside each word and count them. For `l o w` the pairs are `(l, o)` and `(o, w)`.

**`mergePair(corpus, pair)`** — replace every occurrence of the selected pair with one combined token.

**`train(text, vocabSize)`** — start from the base vocabulary, repeatedly choose the most frequent pair, merge it, and save the rule.

**`encode(text)`** — start from base pieces and replay the learned merge rules in order. Convert the final token strings to IDs.

**`decode(ids)`** — map IDs back to token strings and reconstruct the text.

The important separation is that `train()` discovers the tokenizer, `encode()` uses it, and `decode()` reconstructs the text.

---

## 19. Test the whole idea

A useful small corpus is `low low low low low lower lower newest newest newest widest widest`. After training, you should have learned reusable pieces such as `lo`, `low`, `es`, `est`, `ne`, `new` and `newest`. Then examples such as these should show the main property we wanted:

```text
low       →  low
lowest    →  low | est
newest    →  newest
slowest   →  s | low | est
```

The complete word does not have to appear in the training corpus. Its pieces can be enough.

---

## 20. Put BPE into the language-model pipeline

The tokenizer runs before the neural network. The flow is:

```text
"low lower"  →  BPE  →  [token IDs]  →  embedding lookup  →  vectors  →  transformer
```

The tokenizer's job is to turn text into a sequence of discrete pieces. Chapter 18 will turn those token IDs into vectors.

---

## What to implement

| Function | Job |
|---|---|
| `countPairs(corpus)` | Count neighboring pairs inside each sequence. |
| `mergePair(corpus, pair)` | Replace every occurrence of the selected pair. |
| `train(text, vocabSize)` | Build the vocabulary and ordered merge rules. |
| `encode(text)` | Replay the merge rules and produce token IDs. |
| `decode(ids)` | Turn token IDs back into text. |

Nothing in this tokenizer is differentiated.

---

## Verify

```bash
bun test src/tokenizer/bpe.test.ts
bun run exercises/ch-17-bpe-tokenizer.ts
```

The most useful checks are that training the same corpus twice gives the same vocabulary, the same merge rules and the same token IDs — and that `decode(encode(text))` returns the original text, for the supported input.

---

## Checkpoint

You should now be able to answer:

1. Why are character tokens safe but often inefficient?
2. Why is a whole-word vocabulary difficult to use?
3. What does BPE do when it finds a frequent neighboring pair?
4. Why must pair counts be recomputed after every merge?
5. Why does merge order matter?
6. What is the difference between a vocabulary and the merge rules?
7. How can BPE represent a word it never saw during training?
8. Why does a shorter token sequence matter for attention?
9. Why must tie-breaking be deterministic?
10. What is the difference between BPE training and BPE encoding?

---

## The idea to carry forward

BPE is a simple process:

```text
start with small pieces  →  find a frequent neighboring pair  →  merge it  →  repeat
```

The result is a vocabulary containing pieces that the training data found useful. When new text arrives, the tokenizer does not learn anything new. It simply replays the learned rules: `text → base pieces → learned merges → token IDs`. That is the whole idea.

A good tokenizer gives the model a compact way to represent text without requiring one token for every possible word.

---

## Next Chapter

**[Chapter 18: Token Embeddings](ch-18-token-embeddings.md)**

We now have token IDs, but an ID is only a label. The number 22 does not mean the token is "twice as important" as token 11, and two nearby IDs are not necessarily similar. Next, we'll turn each token ID into a learned vector — the representation the network actually reads.
