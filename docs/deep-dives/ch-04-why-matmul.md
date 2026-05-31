# Deep Dive: Why Does Matrix Multiplication Work This Way?

> Optional reading for Chapter 04. No new code. Just understanding.
> After this, the row × column rule will feel inevitable, not arbitrary.

---

## The question worth asking

The chapter tells you the formula: `C[i,j] = Σ_k A[i,k] * B[k,j]`.

You can memorize it. You can implement it. It works.

But *why* row times column? Why not row times row? Why does the inner dimension have to match? Why does it produce a sum and not something else? Who decided all this — and did they prove it was the only valid choice?

This doc answers all of that. No assumed background. No jargon that isn't explained.

---

## Part 1: The proof — in plain English

### Start with a simpler question: what is a matrix *for*?

A matrix is a compact way to write down a **recipe for transforming a list of numbers into another list of numbers**.

For example, here is a recipe that takes 2 numbers in and produces 3 numbers out:

```
Recipe (3×2 matrix):
  output[0] = 2 * input[0] + 1 * input[1]
  output[1] = 0 * input[0] + 3 * input[1]
  output[2] = 1 * input[0] + 0 * input[1]
```

Written as a matrix:

```
A = [[ 2,  1 ],
     [ 0,  3 ],
     [ 1,  0 ]]
```

Reading off: row 0 says "multiply input[0] by 2, input[1] by 1, add them." Row 1 says "skip input[0], multiply input[1] by 3." And so on.

This is the only constraint on what a "recipe" can be: **each output is a weighted sum of all the inputs**. No squaring, no `if` statements. Just multiply each input by some number and add. This class of recipe has a name — *linear* — but all it means is: no bending, only stretching, rotating, and squishing.

### Now the key insight: chaining recipes

Say you have two recipes:
- **Recipe A**: turns 2 numbers into 3 numbers
- **Recipe B**: turns 3 numbers into 4 numbers

You can chain them: first apply A, then apply B. The result is a new recipe: 2 numbers go in, 4 numbers come out.

**Question**: can you write this chained recipe as a single matrix — *without* running both recipes step by step every time?

**Yes. And there is exactly one way to do it.**

Here is why.

Call the combined recipe C. For C to be correct, it must give the same answer as "first A, then B" for *any* input you give it. Not just some inputs — all of them.

Pick the simplest possible input: a vector with a 1 in one slot and 0s everywhere else. Feed `[1, 0]` into the chain:

```
A applied to [1, 0]:
  output[0] = 2*1 + 1*0 = 2
  output[1] = 0*1 + 3*0 = 0
  output[2] = 1*1 + 0*0 = 1
  Result: [2, 0, 1]
```

Now apply B to `[2, 0, 1]`. Say B is:

```
B = [[ 1,  0,  2 ],
     [ 0,  1,  1 ],
     [ 3,  0,  0 ],
     [ 0,  2,  1 ]]
```

```
B applied to [2, 0, 1]:
  output[0] = 1*2 + 0*0 + 2*1 = 4
  output[1] = 0*2 + 1*0 + 1*1 = 1
  output[2] = 3*2 + 0*0 + 0*1 = 6
  output[3] = 0*2 + 2*0 + 1*1 = 1
  Result: [4, 1, 6, 1]
```

So: `[1, 0]` → chain → `[4, 1, 6, 1]`. This must be the **first column** of C (because C applied to `[1, 0]` is just reading column 0 of C).

Now do the same with `[0, 1]` → get second column of C. That's it. We have derived every entry of C.

**What we just did is matrix multiplication.** C's column j = B applied to (A's column j). Written element-by-element:

```
C[i, j] = B[i, 0]*A[0, j] + B[i, 1]*A[1, j] + ... 
         = Σ_k  B[i, k] * A[k, j]
```

That is the row × column formula — for rows of B and columns of A.

(Note: the chapter uses the convention `C = A @ B`, so it's rows of A times columns of B. Same thing — the positions of A and B swapped because here we're doing B after A.)

**The proof that this is the ONLY valid definition:**

If you defined multiplication any other way — say, element-wise, or row × row — then C would not represent the chain of the two recipes. It would compute something else. You could verify this by checking that `C * [1, 0]` no longer equals "A then B applied to [1, 0]". The chain would break.

The row × column formula is forced by requiring that "multiplying matrices = composing recipes."

---

## Part 2: Why the inner dimensions must match

When you chain A (2→3) and then B (3→4), the middle number — 3 — must be the same. A's output has 3 numbers. B expects 3 numbers as input. If they don't match, you're handing A's output to a recipe that expects a different number of inputs. It is simply undefined.

The shape rule `[M, K] × [K, N] → [M, N]` is just:
- M = how many inputs the first recipe produces slots for (its output size)
- K = the shared middle space — A's output count = B's input count
- N = how many outputs B produces

K disappears because it was the "communication channel" between the two recipes. Once the recipes are chained, only the external interface matters: M inputs, N outputs.

---

## Part 3: Who invented this, and when

### The build-up (before Cayley)

**Chinese mathematicians, ~100 BC**
The *Nine Chapters on the Mathematical Art* arranges equations as grids and performs column operations to find solutions. The first systematic use of rectangular number arrays. But no multiplication of arrays.

**Carl Friedrich Gauss, ~1800**
Developed the elimination method for solving systems of equations — what we now call Gaussian elimination. He worked entirely with grids of coefficients. Still no array multiplication.

**James Joseph Sylvester, 1850**
A British mathematician who coined the word **"matrix"** — from the Latin word for *womb*, because he saw it as the rectangular structure from which determinants (smaller numbers) are "born." He was close friends with the person who would define multiplication.

### The inventor: Arthur Cayley, 1858

Arthur Cayley was an English mathematician who published **"A Memoir on the Theory of Matrices"** in the *Philosophical Transactions of the Royal Society* in **1858** (not 1855 as sometimes cited).

Cayley's motivating problem was exactly the one above: if you apply one linear substitution to a set of variables, then apply another, what is the single substitution that does both? He worked out the formula for combining the coefficients and saw that it followed the row × column pattern.

In the paper he wrote (paraphrased): "I do not stop to consider the general case, but define the product of two matrices as..." — and then he wrote exactly `C[i,j] = Σ_k A[i,k] * B[k,j]`.

He also noticed — and found remarkable — that `A × B ≠ B × A` in general. For regular numbers, order doesn't matter: `3 × 5 = 5 × 3`. But for matrices it does. He could not have predicted that 160 years later this non-commutativity would be crucial for multi-head attention, where the order in which you apply projections is meaningful.

**One more contributor: William Rowan Hamilton, 1843**
Hamilton invented quaternions — a 4-dimensional number system — a few years before Cayley defined matrices. Quaternions are non-commutative too (the product depends on order), and Hamilton proved that this was consistent and useful. His work convinced Cayley that non-commutativity was not a flaw but a feature.

---

## Part 4: Why does the row-major (serial) storage exist?

Two separate things are called "serial" here. Let's separate them.

### Serial 1: the triple loop

```typescript
for i in 0..M      // each row of A → each row of C
  for j in 0..N    // each col of B → each col of C
    for k in 0..K  // the shared inner dimension
      C[i,j] += A[i,k] * B[k,j]
```

This looks "serial" — one cell at a time. But that's just because the *definition* says C[i,j] is a sum over k. The sum has to be computed one term at a time. What varies is *which* cells you compute in which order — and you can parallelize the outer loops (that's exactly what a GPU does: compute all C[i,j] cells at once, one per processor core).

### Serial 2: row-major memory layout

Internally, a 2D matrix is stored as a flat 1D array. There are two choices:

```
Matrix [[1, 2, 3],         Row-major (C order):     Column-major (Fortran order):
        [4, 5, 6]]         [1, 2, 3, 4, 5, 6]       [1, 4, 2, 5, 3, 6]
                            read left→right, then     read top→bottom, then
                            down to the next row      right to the next column
```

**Why row-major?** The C programming language (1972) stored 2D arrays row by row. This became the default for most software built after C. NumPy, PyTorch, our library — all default to row-major (called "C order").

**Why is row-major cache-friendly for our triple loop?**

In the inner loop (`k` increasing), we read `A[i, k]` for fixed `i` and increasing `k`. In row-major layout, `A[i, 0], A[i, 1], A[i, 2]...` are adjacent in memory — the CPU prefetches them into cache in a single cache line. One cache miss, then cache hits for the rest of the row.

If we used column-major and the same loop order, `A[i, k]` for increasing `k` would jump by M positions in memory each step — a cache miss every single time. A 10× slowdown for large matrices.

FORTRAN uses column-major, which is equally valid. BLAS (the industry-standard linear algebra library, underlying NumPy, SciPy, and PyTorch) was originally written in FORTRAN. NumPy accepts both orders with a flag. The point: **row-major is a convention, not a mathematical truth**. But it is the convention this course follows, and it matches C, Python, and most of what you'll encounter.

---

## Part 5: Why AI specifically chose this math for everything

### Matrices are the only scalable way to mix information

A neural network's job is to transform input features into output predictions. Each layer must combine every input feature with every output neuron's weights. For `N` inputs and `M` outputs, that's `N × M` multiplications plus `M` additions — exactly one matMul.

There is no faster exact way to do this. And hardware (GPUs, TPUs) was designed specifically to perform matMul efficiently: an H100 GPU does 2,000 **trillion** matMul operations per second (2 petaFLOP/s for fp16). The entire deep learning stack — PyTorch, JAX, TensorFlow, CUDA — is ultimately a wrapper around fast matMul.

### The connection to the transformer

In the attention mechanism (`Ch 22`):
```
Attention(Q, K, V) = softmax( Q @ Kᵀ / √dₖ ) @ V
```

Three matMuls. Each one is "the recipe-chaining idea" applied to token representations:
- `Q @ Kᵀ` chains the query recipe with the transposed key recipe → produces a score for every pair of tokens
- Then `@ V` chains the attention weights with the value recipe → mixes values according to those scores

Every linear projection (`Q = X @ Wq`) is just: apply the learned recipe Wq to the input X. That's Cayley's 1858 definition, running on silicon.

### Why not some other operation?

Several alternatives exist:
- **Element-wise multiply**: Can't connect features across different dimensions. An image pixel can't "see" another pixel
- **Convolution**: Great for local patterns (nearby pixels), terrible for long-range dependencies (words 100 apart in a sentence)
- **Polynomial features**: Explodes in size, not learnable at scale
- **matMul + non-linearity**: The transformer's chosen combination. matMul handles the mixing; non-linearities (ReLU, GELU) handle the "bending" that matMul alone can't do

The transformer uses matMul because:
1. It's the unique operation for composing linear recipes at any scale
2. It is fully parallelizable across every output cell
3. Its gradient is clean and computable (Ch 10 proves `dL/dA = dL/dZ @ Bᵀ`)
4. Hardware is optimized for it at the chip design level

---

## Summary: the three-sentence version

Matrix multiplication is the unique rule for combining two linear recipes into one. Cayley discovered it in 1858 by asking "what single substitution equals applying two substitutions in sequence?" — and derived row × column as the forced answer. AI uses it for everything because every neural network layer is a learned recipe (weights) applied to data, and hardware optimized for this specific operation makes learning from billions of examples feasible.

---

## Interactive simulation

Run the step-by-step terminal simulation to watch all of this in action:

```bash
bun run docs/assets/ch-04/matmul-demo.ts
```

Shows: step-by-step computation, the recommendation problem solved with matMul, and a one-layer neural network forward pass.
