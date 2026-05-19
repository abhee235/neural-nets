# Chapter 25: Feed-Forward Block

> **Part 6 of 6 — Transformer**
> `src/ch-25-feedforward-block/`

---

## What You're Building

The position-wise feed-forward network (FFN) used inside every transformer block.
Two linear layers with GELU activation and a 4× expansion in the middle:
`dModel → 4×dModel → dModel`.

---

## Why This Matters

Every transformer block has two sub-layers: attention and FFN. You've just built attention.
The FFN is the other half — and it accounts for roughly two-thirds of a transformer's
total parameters.

While attention is about *where* to look (routing information between positions),
the FFN is about *what to do with* the information once it arrives. It applies the same
transformation independently at each position — like a per-token expert.

---

## Concepts

### The FFN Formula

$$\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2$$

Where:
- $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$ — expands to $d_{\text{ff}}$ dimensions
- $W_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$ — projects back to $d_{\text{model}}$
- $d_{\text{ff}} = 4 \times d_{\text{model}}$ by convention (e.g., 512 → 2048 → 512)
- GELU (from Ch 11) replaces the ReLU used in the original transformer

Applied **position-wise**: the same two matrices are applied to every token independently.
The FFN does NOT mix information between positions (attention does that).

### Why 4×?

The 4× expansion factor is an empirical choice from the original paper. It gives the FFN
enough capacity to act as a key-value memory: the first matrix writes into an "intermediate"
space where each neuron can activate for specific input patterns, and the second matrix reads
out the corresponding output. Larger = more capacity, but also more parameters and compute.

### GELU vs ReLU

The original transformer used ReLU. GPT-2 and most modern models use GELU:
$$\text{GELU}(x) \approx 0.5x \left(1 + \tanh\!\left(\sqrt{2/\pi} \cdot (x + 0.044715 x^3)\right)\right)$$

GELU is smoother than ReLU (no hard zero at $x=0$), which helps optimization.

### Position-Wise Means Shared Weights

"Position-wise" means the same $W_1, b_1, W_2, b_2$ are applied to all positions.
The FFN doesn't have different weights for position 0 vs position 47. This is exactly like
applying a Conv1D with kernel size 1: the same filter at every position.

---

## What to Implement

| Symbol | Description |
|--------|-------------|
| `class FFN` | Constructor: `(dModel, dFf?)`. Default `dFf = 4 * dModel`. |
| `FFN.forward(x)` | Apply two Linear layers with GELU: `[batch, seq, dModel]` → `[batch, seq, dModel]` |
| `FFN.parameters()` | All parameters from both Linear layers |

---

## TypeScript Hints

```typescript
export class FFN {
  linear1: Linear;   // dModel → dFf
  linear2: Linear;   // dFf → dModel
  dFf: number;

  constructor(dModel: number, dFf: number = 4 * dModel) {
    this.dFf = dFf;
    this.linear1 = new Linear(dModel, dFf);   // expansion
    this.linear2 = new Linear(dFf, dModel);   // projection back
  }

  forward(x: Value): Value {
    // x: [batch, seq, dModel]
    const h = this.linear1.forward(x);       // [batch, seq, dFf]
    const activated = gelu(h);               // element-wise GELU (Ch 11)
    return this.linear2.forward(activated);  // [batch, seq, dModel]
  }

  parameters(): Value[] {
    return [...this.linear1.parameters(), ...this.linear2.parameters()];
  }
}
```

---

## Self-Check Questions

1. For `dModel=512`, `dFf=2048`: how many parameters does the FFN have?
   (Count W1, b1, W2, b2.)
2. What does "position-wise" mean? Compare: FFN at position 5 vs position 15 for
   the same input vector. Do they give the same output?
3. Why is the FFN applied after attention, not before? What role do they play in the
   transformer's information processing pipeline?
4. Replace GELU with ReLU in your implementation. What changes? Train a small model with
   each and observe the difference in loss curves.
5. Why do we project back to `dModel` in the second linear layer?
   What would happen if we kept the `dFf` dimensions going into the next block?

---

## → Next Chapter

**Ch 26: Encoder Block** — combine LayerNorm, MultiHeadAttention, FFN, and residual
connections into a complete transformer encoder block.
