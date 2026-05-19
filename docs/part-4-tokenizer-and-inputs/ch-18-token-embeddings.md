# Chapter 18: Token Embeddings

> **Part 4 of 6 — Language Model Inputs**
> `src/ch-18-token-embeddings/`

---

## What You're Building

An embedding layer: a lookup table that maps integer token IDs to dense floating-point vectors.
This is the first learnable layer in every language model.

---

## Why This Matters

A transformer can't operate on integers directly — it needs continuous vectors it can apply
matrix multiplications to. The embedding layer is the bridge: each token in the vocabulary
gets its own learnable vector in $\mathbb{R}^{d_{\text{model}}}$. Semantically similar tokens
(like "king" and "queen") end up with similar embeddings after training.

In the transformer, the embedding layer (Ch 18) + positional encoding (Ch 19) are the
*entire input processing pipeline* before any attention is computed.

---

## Concepts

### The Embedding Table

The embedding table is a matrix $E \in \mathbb{R}^{V \times d}$ where:
- $V$ = `vocabSize` — number of unique tokens
- $d$ = `dModel` — the dimension of each embedding vector

Row $i$ of $E$ is the embedding vector for token $i$.

```
vocabSize = 100
dModel = 16

E = [[0.1, -0.3, ..., 0.7],   ← embedding for token 0 ("<pad>")
     [0.2,  0.5, ..., -0.1],   ← embedding for token 1 ("<unk>")
     ...
     [0.8, -0.2, ..., 0.4]]    ← embedding for token 99
```

### Lookup as Matrix Multiplication

Conceptually, embedding is a matrix multiply with a one-hot vector:

$$e_i = E^T \cdot \text{onehot}(i)$$

But you would never implement it this way — it's equivalent to just indexing the row directly:

$$e_i = E[i]$$

Row indexing is O(d) (just copy a row). The one-hot matrix multiply is O(V×d) — V times
slower for no reason. This is why PyTorch's `nn.Embedding` is just a fancy array lookup.

### Initialization

The standard initialization for embeddings is the same as other weight matrices:
normal distribution with small standard deviation (e.g., $\mathcal{N}(0, 0.02)$).

Some implementations use `randn` then scale by $1/\sqrt{d_{\text{model}}}$ to prevent
the initial norms from being too large.

### Input Shape and Output Shape

Input: token IDs as a 2D tensor of shape `[batchSize, seqLen]` (integers).
Output: embeddings of shape `[batchSize, seqLen, dModel]` (floats).

Each token ID in the input is replaced by its corresponding row from $E$.

### Padding Mask

Padding tokens (ID = 0) should not affect training. Options:
1. Set the `<pad>` embedding row to all zeros and zero its gradient after each step.
2. Use an attention mask (Ch 22) to prevent the model from attending to padding positions.

In practice, option 2 (masking in attention) is standard.

### Weight Tying

In language models, the embedding table $E$ is often **shared** (tied) with the output
projection matrix (the matrix that maps from $d_{\text{model}}$ back to vocabulary logits).

Rationale: both matrices represent the same thing — a mapping between token space and
model space. Tying them reduces parameters significantly (saves $V \times d$ parameters)
and often improves performance because both objectives reinforce the same representations.

$$\text{output logits} = h \cdot E^T \quad \text{where } h \in \mathbb{R}^{d_{\text{model}}}$$

---

## What to Implement

| Symbol | Description |
|--------|-------------|
| `class Embedding` | Embedding lookup table. Constructor: `(vocabSize, dModel)`. |
| `Embedding.forward(ids)` | Input: `[batch, seq]` integer tensor. Output: `[batch, seq, dModel]` tensor wrapped in `Value`. |
| `Embedding.parameters()` | Returns `[this.weight]` — the embedding table itself. |
| `Embedding.weight` | The `[vocabSize, dModel]` `Value` matrix. |

---

## TypeScript Hints

```typescript
export class Embedding {
  weight: Value;    // shape: [vocabSize, dModel]
  vocabSize: number;
  dModel: number;

  constructor(vocabSize: number, dModel: number) {
    this.vocabSize = vocabSize;
    this.dModel = dModel;
    // Small normal init — same as other weight matrices
    // Scale by 1/sqrt(dModel) to keep initial norms reasonable
    const scale = 1 / Math.sqrt(dModel);
    this.weight = new Value(mulScalar(randn([vocabSize, dModel]), scale));
  }

  // ids: [batchSize, seqLen] integer tensor
  // returns: [batchSize, seqLen, dModel] Value
  forward(ids: Tensor): Value {
    const [batchSize, seqLen] = ids.shape as [number, number];

    // Gather rows from the embedding table for each token ID
    const outputData: number[] = [];
    for (let b = 0; b < batchSize; b++) {
      for (let s = 0; s < seqLen; s++) {
        const tokenId = ids.data[b * seqLen + s]!;
        // Row tokenId of the weight matrix: weight.data[tokenId * dModel ... (tokenId+1)*dModel]
        const rowStart = tokenId * this.dModel;
        for (let d = 0; d < this.dModel; d++) {
          outputData.push(this.weight.data.data[rowStart + d]!);
        }
      }
    }

    // The gradient needs to flow back to the embedding rows that were looked up
    const out = new Value(
      createTensor(outputData, [batchSize, seqLen, this.dModel]),
      [this.weight],
      "embedding-lookup"
    );
    // _backward: accumulate gradients only into the rows that were used
    out._backward = () => { /* ... */ };
    return out;
  }

  parameters(): Value[] {
    return [this.weight];
  }
}
```

---

## Self-Check Questions

1. An embedding layer with `vocabSize=1000, dModel=64`: how many learnable parameters?
   Compare to a `Linear(1000, 64)` layer.
2. `Embedding.forward([[3, 1, 4], [1, 5, 9]])` — what shape is the output?
3. Why is embedding lookup O(d) but naive matrix multiply is O(V×d)?
4. In weight tying, the output logits are $h \cdot E^T$. If $h$ is `[batch, seq, 512]`
   and $E$ is `[50000, 512]`, what is $E^T$'s shape? What is the output shape?
5. Why might semantically similar tokens end up with similar embeddings after training?
   What loss function signal would cause this?

---

## → Next Chapter

**Ch 19: Positional Encoding** — add position information to embeddings so the model
knows the order of tokens in the sequence.
