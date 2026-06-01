# Deep Dive: Why Subtracting the Max Makes Softmax Safe — and Identical

> Optional reading for Chapter 05. No new code required. Just understanding.
> After this, the "subtract the max" trick will feel obvious and provably correct,
> not like a magic line you copy from a textbook.

---

## The question worth asking

The chapter tells you to implement softmax like this:

```
softmax(x)_i = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
```

But the textbook *definition* of softmax has no `max` in it at all:

```
softmax(x)_i = exp(x_i) / Σ_j exp(x_j)
```

So two fair questions:

1. **Is the shifted version actually equal to the real thing, or just close?**
2. **Why bother? What breaks if I just use the plain definition?**

This doc answers both. The first is a three-line proof. The second is a concrete
floating-point failure you can reproduce in the REPL.

---

## Part 1: The proof — shifting the input changes nothing

**Claim.** For *any* constant `c`, softmax(x) = softmax(x − c). Subtracting the same number
from every logit leaves the output probabilities exactly unchanged.

**Proof.** Start from the definition and subtract `c` from every input:

$$
\text{softmax}(x - c)_i
= \frac{e^{\,x_i - c}}{\sum_j e^{\,x_j - c}}
$$

Use the rule `e^{a-b} = e^a · e^{-b}` on both top and bottom. The `e^{-c}` factor appears
in the numerator once and in *every* term of the denominator:

$$
= \frac{e^{x_i}\,e^{-c}}{\sum_j e^{x_j}\,e^{-c}}
= \frac{e^{-c}\;e^{x_i}}{e^{-c}\sum_j e^{x_j}}
$$

`e^{-c}` is a common factor — it cancels:

$$
= \frac{e^{x_i}}{\sum_j e^{x_j}}
= \text{softmax}(x)_i \qquad \blacksquare
$$

That is the entire argument. The output depends only on the **differences** between
logits, never on their absolute level. Add 5 to every score, subtract 1000 from every
score — the probabilities are identical.

Since the identity holds for *any* `c`, we are free to pick the most convenient one.
We pick `c = max(x)`. Why that choice is the safe one is Part 2.

---

## Part 2: What actually breaks without it

Softmax lives on top of `exp`, and `exp` grows *fast*. In IEEE-754 double precision
(JavaScript's only number type) the largest finite value is about `1.8 × 10³⁰⁸`.
`exp` blows past it surprisingly early:

```text
exp(700)  ≈ 1.01 × 10³⁰⁴     still finite
exp(709)  ≈ 8.22 × 10³⁰⁷     still finite, barely (MAX_VALUE ≈ 1.80 × 10³⁰⁸)
exp(710)  =  Infinity         overflow
```

So if a single logit reaches ~710 — and in a deep network, pre-softmax scores can — the
plain definition computes `Infinity / Infinity`, which IEEE-754 defines as `NaN`. One
`NaN` then contaminates the loss, the gradients, and every weight on the next update.
Training silently dies.

### The fix, seen through the math

After subtracting the max, the largest shifted logit is exactly `0`, because
`max(x) − max(x) = 0`. Therefore:

- The biggest exponential is `exp(0) = 1` — no overflow is even *possible*.
- Every other exponential is `exp(negative) ∈ (0, 1]` — all safe.
- The denominator is a sum of values in `(0, 1]`, and at least one term equals 1, so it
  is always `≥ 1`. No underflow-to-zero in the denominator, so no division by zero.

The shift converts an expression that *can* overflow into one that provably *cannot*,
while Part 1 guarantees the answer is bit-for-bit the mathematically correct one (up to
normal rounding).

### Reproduce it yourself

```bash
bun -e '
const x = [710, 711, 712];

// Naive softmax — overflows
const expNaive = x.map(Math.exp);              // [Infinity, Infinity, Infinity]
const sumNaive = expNaive.reduce((a, b) => a + b, 0);
console.log("naive:  ", expNaive.map(e => e / sumNaive));  // [NaN, NaN, NaN]

// Stable softmax — subtract the max first
const m = Math.max(...x);                       // 712
const expStable = x.map(v => Math.exp(v - m));  // [exp(-2), exp(-1), exp(0)]
const sumStable = expStable.reduce((a, b) => a + b, 0);
console.log("stable: ", expStable.map(e => e / sumStable));  // [0.090, 0.245, 0.665]
'
```

The naive column is all `NaN`; the stable column is a clean probability distribution
summing to 1. Same inputs, same intended math — only the stable form survives the
hardware.

> The `.reduce((a, b) => a + b, 0)` above is a simple running sum (start at 0, add each
> element) — fine to use for a one-line fold like this, with the comment to say so.

---

## Part 3: Why this keeps coming back

This is not a softmax-only trick — it is an instance of a general principle that recurs
all over numerical ML:

> **Compute in the domain where the numbers are well-behaved, using an identity that
> guarantees the same answer.**

You will meet the same idea again in:

- **Log-sum-exp** (Ch 12, cross-entropy): `log Σ exp(x)` is computed as
  `max(x) + log Σ exp(x − max(x))` — the *exact* same shift, for the *exact* same reason.
  Cross-entropy is really "log-softmax," so this trick is load-bearing for the loss
  function, not just the prediction.
- **LayerNorm** (Ch 20): dividing by `sqrt(variance + ε)` adds a tiny `ε` so a
  zero-variance row can't divide by zero — the same defensive instinct.

Recognizing "shift/scale into a safe range, justified by an identity" as a *pattern*
means you will reach for it automatically the next time an `exp`, a `log`, or a divide
threatens to overflow.

---

## Pen-and-paper exercise

1. Prove the scale analogue is **false**: show that `softmax(2x) ≠ softmax(x)` in general
   by computing both for `x = [0, 1]`. (Softmax is shift-invariant but *not*
   scale-invariant — this is exactly why attention divides by `√dₖ` before the softmax,
   to control the scale.)
2. For `x = [0, 0, 0]`, compute softmax by hand. What distribution do equal logits
   produce, and why does that match intuition?
3. Show that `softmax(x)` always lands strictly inside `(0, 1)` for every component — no
   probability is ever exactly 0 or exactly 1. (Hint: every `exp` term is strictly
   positive.) What does that imply about a model ever being "100% certain"?
