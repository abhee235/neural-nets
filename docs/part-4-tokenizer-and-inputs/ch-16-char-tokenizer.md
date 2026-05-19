# Chapter 16: Character-level Tokenizer

> **Part 4 of 6 — Language Model Inputs**
> `src/ch-16-char-tokenizer/`

---

## What You're Building

A character-level tokenizer: the bridge between raw text (strings) and numbers that a
neural network can process. Implements `encode` (text → token IDs) and `decode` (token IDs → text),
plus a vocabulary builder and special tokens.

---

## Why This Matters

Neural networks work with numbers, not text. Before any transformer can process the sentence
"Hello world", every character (or word piece) must be converted to an integer index.
That index then gets looked up in the embedding table (Ch 18) to produce a vector.

A character-level tokenizer is the simplest possible tokenizer — it treats every character as
a token. While modern LLMs use BPE (Ch 17) for efficiency, a character-level tokenizer is
sufficient to build and train a working transformer, and it's much simpler to understand.

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

## → Next Chapter

**Ch 17: BPE Tokenizer** — the Byte-Pair Encoding algorithm used by GPT-2/3/4.
Handles multi-character tokens for efficiency. (Optional: Ch 18–27 only need Ch 16.)
