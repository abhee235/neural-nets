# Chapter 17: BPE Tokenizer (Byte-Pair Encoding)

> **Part 4 of 6 — Language Model Inputs**
> `src/ch-17-bpe-tokenizer/`

> ⚠️ **Optional chapter.** Chapters 18–30 only require the character tokenizer from Ch 16.
> Come back to this after completing the full transformer if you want GPT-style tokenization.

---

## What You're Building

A Byte-Pair Encoding tokenizer from scratch: the algorithm that learns a vocabulary of
sub-word tokens from a corpus, then encodes text using that vocabulary. This is how
GPT-2, GPT-3, GPT-4, LLaMA, and virtually all modern LLMs tokenize text.

---

## Why This Matters

Character-level tokenization works but is inefficient: the word "transformer" takes 11
tokens. With BPE, common multi-character sequences get their own token. "transformer"
might be a single token, or two tokens "trans" + "former". This means:
- Fewer tokens per sentence → shorter sequences → faster training
- The model attends over fewer positions → attention is less expensive
- Common words are atomic units, which may help the model learn their meaning

---

## Concepts

### The Core Idea

BPE starts with a vocabulary of individual characters (like Ch 16), then iteratively
merges the most frequent adjacent pair of tokens into a new single token, until the
desired vocabulary size is reached.

```
Initial:  ["h", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d"]
Step 1:   Most frequent pair = ("l", "l") → merge → "ll"
          ["h", "e", "ll", "o", " ", "w", "o", "r", "l", "d"]
Step 2:   Most frequent pair = ("o", " ") → merge → "o " 
          ["h", "e", "ll", "o ", "w", "o ", "r", "l", "d"]
...
After many merges: entire common words become single tokens.
```

### The Training Algorithm

```
1. Start: vocabulary = all unique bytes/characters in corpus
2. Represent corpus as a list of token sequences

3. Repeat until vocabulary size == target:
   a. Count all adjacent pairs (bigrams) across the entire corpus
   b. Find the most frequent pair (e.g., ('e', 's'))
   c. Merge that pair everywhere in the corpus → new token 'es'
   d. Add 'es' to the vocabulary
   e. Record the merge rule: ('e', 's') → 'es'

4. Save: the vocabulary + the ordered list of merge rules
```

The order of merge rules matters for encoding — they must be applied in training order.

### Encoding with Trained BPE

Given a new piece of text and the learned merge rules:

```
1. Start with character-level tokenization of the text
2. Apply merge rules in order:
   for each merge rule (a, b) → ab:
     scan the token sequence for adjacent (a, b)
     replace all occurrences with ab
3. The result is the BPE-encoded token sequence
```

The key insight: each merge rule is applied globally to the entire token sequence before
moving to the next rule. This ensures consistent encoding.

### Why "Byte-Pair"?

The original BPE algorithm (1994, Gage) was used for data compression — it operated on bytes.
In NLP, Sennrich et al. (2016) adapted it for neural machine translation. OpenAI's GPT-2
uses a variant that operates on Unicode bytes first, then merges.

For our implementation, we'll work directly on characters (not raw bytes), which is simpler
and sufficient for learning.

### Vocabulary Size Trade-offs

| Vocab Size | Token length | Trade-off |
|-----------|-------------|-----------|
| 256 (byte-level) | ~1-2 chars | Shortest vocab, longest sequences |
| ~500–5000 | ~2-5 chars | Good for small models |
| 32,000–50,000 | ~4-6 chars | GPT-2 (50,257), LLaMA (32,000) |
| 100,000+ | ~5-8 chars | GPT-4 (cl100k: ~100k) |

---

## What to Implement

| Symbol | Description |
|--------|-------------|
| `countPairs(tokens)` | Count frequency of all adjacent pairs in a token sequence |
| `mergePair(tokens, pair, merged)` | Replace all occurrences of `pair` with `merged` token |
| `trainBPE(text, vocabSize)` | Run the full BPE training algorithm; return `{vocab, merges}` |
| `class BPETokenizer` | Tokenizer using trained vocab + merge rules |
| `BPETokenizer.encode(text)` | Apply merge rules in order; return token IDs |
| `BPETokenizer.decode(ids)` | Map IDs back to strings and concatenate |

---

## TypeScript Hints

```typescript
type MergeRule = [string, string];   // (tokenA, tokenB) → tokenA+tokenB

export function trainBPE(
  text: string,
  targetVocabSize: number
): { vocab: Map<string, number>; merges: MergeRule[] } {
  // Initialize with character-level tokens
  let tokens: string[] = [...text];  // split into individual chars
  const vocab = new Map<string, number>();
  const merges: MergeRule[] = [];

  // Build initial vocab from unique characters
  const uniqueChars = [...new Set(tokens)].sort();
  uniqueChars.forEach((ch, i) => vocab.set(ch, i));

  // BPE training loop
  while (vocab.size < targetVocabSize) {
    // Count all adjacent pairs
    const pairCounts = countPairs(tokens);
    if (pairCounts.size === 0) break;

    // Find the most frequent pair
    const bestPair = [...pairCounts.entries()]
      .reduce((best, curr) => curr[1] > best[1] ? curr : best);
    const [pairKey, ] = bestPair;
    const [a, b] = pairKey.split("\x00") as [string, string];  // internal separator

    // Merge this pair everywhere
    const merged = a + b;
    tokens = mergePair(tokens, [a, b], merged);
    vocab.set(merged, vocab.size);
    merges.push([a, b]);
  }

  return { vocab, merges };
}

// Count adjacent pair frequencies
function countPairs(tokens: string[]): Map<string, number> {
  const counts = new Map<string, number>();
  for (let i = 0; i < tokens.length - 1; i++) {
    const key = `${tokens[i]}\x00${tokens[i + 1]}`;  // \x00 as separator
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  return counts;
}
```

---

## Self-Check Questions

1. Run BPE training on "aababcabc" with target vocab size 8.
   What are the first 3 merge rules? What is the final vocabulary?
2. Why must merge rules be applied in training order during encoding?
   What goes wrong if you apply them in a different order?
3. GPT-2 uses a vocabulary of 50,257 tokens. If training text has ~10 billion characters,
   roughly how many merge steps were needed?
4. "Transformer" encoded character-level: 11 tokens. With BPE vocab of 50k, it might be
   1-2 tokens. Why does shorter sequences matter computationally for attention?
5. What's the key difference between BPE and WordPiece (used in BERT)?
   (Hint: how the best merge is chosen at each step.)

---

## → Next Chapter

**Ch 18: Token Embeddings** — map integer token IDs to dense vectors using an embedding table.
