# Chapter 16: Character Tokenizer

> **Part 4 of 6 — Tokenizer & Inputs**
> Source: [`src/tokenizer/char.ts`](../../src/tokenizer/char.ts)
> Tests: [`src/tokenizer/char.test.ts`](../../src/tokenizer/char.test.ts)
> Exercise: [`exercises/ch-16-char-tokenizer.ts`](../../exercises/ch-16-char-tokenizer.ts)

---

## Learning Goals

By the end of this chapter you can:

- Build `stoi`/`itos` vocab maps over a fixed set of characters.
- Reserve and use special tokens `<pad>`, `<unk>`, `<bos>`, `<eos>`.
- Implement `encode(text) → number[]` and `decode(ids) → string`.
- Implement `encodeBatch` with right-padding and a parallel `paddingMask`.
- Round-trip every test string: `decode(encode(s)) === s`.

---

## Intuition First

Neural networks cannot read strings; they can only read numbers. A tokenizer is a **dictionary** that assigns each character (or subword) a unique integer. A character tokenizer is the simplest possible: one ID per character. It is slow (long sequences) but it never sees an unknown character if the vocabulary is the full alphabet.

---

## Mental Model

```text
  text:      "hi"
                │
                ▼ stoi
  token ids: [12, 21]
                │
                ▼ encodeBatch (pad to length 4)
  ids:       [12, 21, 0,  0]
  mask:      [ 1,  1, 0,  0]
                │
                ▼ decode
  text:      "hi"
```

---

## Concepts

### What is Tokenization?

Tokenization splits text into discrete units (tokens) and maps each to an integer:

```
"Hello" → ['H', 'e', 'l', 'l', 'o'] → [32, 69, 76, 76, 79]
```

The mapping from token → integer is the **vocabulary** (or "vocab"). The vocabulary is built
from the training corpus — you scan all text, collect unique characters, and assign each an index.

### Vocabulary

A vocabulary is a bidirectional mapping:
- `stoi` (string-to-index): `{ 'H': 32, 'e': 69, ... }`
- `itos` (index-to-string): `{ 32: 'H', 69: 'e', ... }`

Vocabulary size (`vocabSize`) is the number of unique tokens.
For character-level English text: typically ~100 (lowercase + uppercase + punctuation + digits).
For BPE on a large corpus: 50,000–100,000+ tokens.

### Special Tokens

Real tokenizers have reserved tokens with special meanings:

| Token | Meaning | Use case |
|-------|---------|----------|
| `<pad>` | Padding | Fill short sequences to a fixed length |
| `<eos>` | End of sequence | Signal "generation is done" |
| `<bos>` | Beginning of sequence | Signal "start generating here" |
| `<unk>` | Unknown | Character not in vocabulary |

These are assigned the first few integer IDs (0, 1, 2, 3) so they're easy to identify.

### Encode and Decode

```
encode("Hello\n") → [32, 69, 76, 76, 79, 1]   (1 = '\n' in this vocab)
decode([32, 69, 76, 76, 79, 1]) → "Hello\n"
```

**Out-of-vocabulary (OOV):** If a character isn't in the vocabulary, return `<unk>` index.
This can't happen for character-level tokenizers if you build the vocab from your full dataset,
but it becomes important for BPE (Ch 17).

### Encoding a Batch of Sequences

Sequences in a batch must have the same length for matrix operations to work.
**Padding:** append `<pad>` tokens to shorter sequences until all are the same length.
**Truncation:** cut sequences longer than `maxLength`.

```
encode("Hi")     → [30, 73, 1, 1]    (padded with <pad>=1)
encode("Hello")  → [32, 69, 76, 76, 79]  (length 5, fits)
```

An **attention mask** accompanies padded batches: 1 where real tokens are, 0 where padding is.
The model's attention mechanism should ignore padded positions (Ch 22).

---

## What to Implement

| Symbol | Description |
|--------|-------------|
| `buildVocab(text)` | Scan text, collect unique characters, return `{stoi, itos, vocabSize}` |
| `class CharTokenizer` | Tokenizer class with `encode`, `decode`, `vocabSize`, and special tokens |
| `CharTokenizer.encode(text, maxLen?)` | String → number[]. Handles OOV with `<unk>`. |
| `CharTokenizer.decode(ids)` | number[] → string. Skips special tokens. |
| `CharTokenizer.encodeBatch(texts, maxLen)` | Encode multiple strings with padding. Returns `{ids, mask}`. |

---

## TypeScript Hints

```typescript
const SPECIAL_TOKENS = {
  PAD: "<pad>",
  UNK: "<unk>",
  BOS: "<bos>",
  EOS: "<eos>",
} as const;

export class CharTokenizer {
  stoi: Map<string, number>;
  itos: Map<number, string>;
  vocabSize: number;

  readonly padId: number;
  readonly unkId: number;
  readonly bosId: number;
  readonly eosId: number;

  constructor(text: string) {
    // Start with special tokens at the front of the vocabulary
    this.stoi = new Map();
    this.itos = new Map();

    let idx = 0;
    for (const [, token] of Object.entries(SPECIAL_TOKENS)) {
      this.stoi.set(token, idx);
      this.itos.set(idx, token);
      idx++;
    }

    // Add all unique characters from the corpus
    const chars = [...new Set(text)].sort();
    for (const ch of chars) {
      if (!this.stoi.has(ch)) {
        this.stoi.set(ch, idx);
        this.itos.set(idx, ch);
        idx++;
      }
    }

    this.vocabSize = idx;
    this.padId = this.stoi.get(SPECIAL_TOKENS.PAD)!;
    this.unkId = this.stoi.get(SPECIAL_TOKENS.UNK)!;
    this.bosId = this.stoi.get(SPECIAL_TOKENS.BOS)!;
    this.eosId = this.stoi.get(SPECIAL_TOKENS.EOS)!;
  }

  encode(text: string, maxLen?: number): number[] {
    const ids = [...text].map(ch => this.stoi.get(ch) ?? this.unkId);
    return maxLen ? ids.slice(0, maxLen) : ids;
  }

  decode(ids: number[]): string {
    const specialIds = new Set([this.padId, this.bosId, this.eosId]);
    return ids
      .filter(id => !specialIds.has(id))
      .map(id => this.itos.get(id) ?? "")
      .join("");
  }
}
```

---

## Common Pitfalls

- Skipping the `<unk>` token — any character outside the vocab silently breaks encoding.
- Padding on the *left* by accident; the rest of the course assumes right padding.
- Returning padded IDs without the matching padding mask; downstream attention will attend to pad.
- Including newline/whitespace inconsistently — pick a normalisation rule and document it.
- Letting `stoi` and `itos` drift out of sync after vocabulary edits.

---

## How to Verify

Run the tests and the exercise. Both should pass cleanly with no warnings:

```bash
bun test src/tokenizer/char.test.ts
```
```bash
bun run exercises/ch-16-char-tokenizer.ts
```

---

## Self-Check Questions

1. Build a `CharTokenizer` from the text "abcabc". What is `vocabSize`? What is `stoi`?
2. Encode "bca" then decode the result. Do you get back "bca"?
3. What happens when you encode a character not in the vocabulary? Verify with a test.
4. In the batch encoding, if one sequence has length 3 and another has length 7,
   with `maxLen=5`, what do both sequences look like after encoding?
5. A transformer with `vocabSize=100` has an embedding table of shape `[100, 512]`.
   Each row is the embedding for one token. Which row does the `<pad>` token use?
   Should the `<pad>` embedding be trained?

---

## Further Reading

- [Karpathy — Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) — video walkthrough that starts at char level and builds up to BPE.
- [HuggingFace — Tokenizers introduction](https://huggingface.co/docs/tokenizers/introduction) — industry tokenizer reference.
- [Wikipedia — Unicode normalisation forms](https://en.wikipedia.org/wiki/Unicode_equivalence) — decide once whether you NFC-normalise your inputs.
- [OpenAI — tiktoken](https://github.com/openai/tiktoken) — GPT's production tokenizer; useful comparison point.

---

## Next Chapter

**[BPE Tokenizer](ch-17-bpe-tokenizer.md)** — graduate from one-token-per-character to learned subword units.
