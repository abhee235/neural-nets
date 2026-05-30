#!/usr/bin/env python3
"""Rewrite all 31 chapter docs into the 11-section template.

This script preserves the *math* content of each existing doc (the "Concepts"
and "What to Implement" tables) and injects the new sections required by
.github/instructions/chapter-doc.instructions.md:

  1. Title + Header (src / test / exercise links)
  2. Learning Goals
  3. Intuition First
  4. Mental Model
  5. Concepts                  ← preserved from existing doc
  6. What to Implement         ← preserved from existing doc
  7. Common Pitfalls
  8. How to Verify
  9. Self-Check Questions      ← preserved (or added) from existing doc
 10. Further Reading
 11. Next Chapter

Run from repo root:
    python3 scripts/scaffold_chapter_docs.py
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"


# ---------------------------------------------------------------------------
# Chapter metadata
# ---------------------------------------------------------------------------

@dataclass
class Chapter:
    doc_path: str          # relative to docs/
    part_num: int
    part_name: str
    num: str               # "01", "08a", "08b" …
    title: str
    src: List[str]
    tests: List[str]
    exercise: str
    next_doc: str          # relative to docs/, empty if last
    next_blurb: str
    goals: List[str]
    intuition: str         # paragraph(s); may contain bullets
    mental_model: str      # ASCII diagram (already fenced ```text)
    pitfalls: List[str]
    verify: List[str]      # exact `bun …` commands
    further: List[Tuple[str, str, str]]   # (title, url, annotation)
    extra_self_check: List[str] = field(default_factory=list)


# Standard external links re-used across chapters.
NUMPY_BROADCASTING = (
    "NumPy broadcasting",
    "https://numpy.org/doc/stable/user/basics.broadcasting.html",
    "the canonical rules our tensor ops follow.",
)
KARPATHY_ZTH = (
    "Karpathy — Neural Networks: Zero to Hero",
    "https://karpathy.ai/zero-to-hero.html",
    "video series that builds micrograd and nanoGPT from scratch.",
)
ATTN_PAPER = (
    "Vaswani et al. — Attention Is All You Need (2017)",
    "https://arxiv.org/abs/1706.03762",
    "the original transformer paper; every formula in Parts 5–6 comes from it.",
)
THREE_BLUE = (
    "3Blue1Brown — Essence of Linear Algebra",
    "https://www.3blue1brown.com/topics/linear-algebra",
    "geometric intuition for vectors, matrices, and linear maps.",
)
GOODFELLOW = (
    "Goodfellow, Bengio, Courville — Deep Learning",
    "https://www.deeplearningbook.org/",
    "the standard graduate textbook; chapters map cleanly to this course.",
)
DISTILL = (
    "Distill.pub",
    "https://distill.pub/",
    "interactive visual explanations of deep-learning concepts.",
)


CHAPTERS: List[Chapter] = [
    # ─── Part 1 ──────────────────────────────────────────────────────────
    Chapter(
        doc_path="part-1-tensor-library/ch-01-tensor-type-system.md",
        part_num=1, part_name="Tensor Library",
        num="01", title="Tensor Type System",
        src=["src/tensor/types.ts"],
        tests=["src/tensor/types.test.ts"],
        exercise="exercises/ch-01-tensor-type-system.ts",
        next_doc="part-1-tensor-library/ch-02-tensor-creation.md",
        next_blurb="Tensor Creation — turn this type into real objects: zeros, ones, random, and from-array constructors.",
        goals=[
            "Explain what a tensor is in terms of shape, strides, and a flat data buffer.",
            "Compute the flat index for any logical multi-dimensional index using strides.",
            "Predict the shape of every tensor that appears in a transformer forward pass.",
            "Decide whether a given operation requires a copy or can be done by changing strides.",
            "Write a `Tensor` interface in strict TypeScript with no `any`.",
        ],
        intuition=(
            "A tensor is just a flat `Float64Array` plus a recipe for how to read it as a "
            "multi-dimensional box. The recipe has two parts: `shape` says how big the box is "
            "along each axis, and `strides` says how many flat-array steps to take when you move "
            "one step along each axis. Everything we will build — embeddings, attention scores, "
            "logits — is the same flat buffer reinterpreted with a different shape.\n\n"
            "- A scalar is a `[]`-shaped tensor.\n"
            "- A vector is `[n]`.\n"
            "- A matrix is `[rows, cols]`.\n"
            "- An attention score block inside multi-head attention is `[batch, heads, q, k]`."
        ),
        mental_model=(
            "```text\n"
            "shape:   [2, 3]            strides: [3, 1]\n"
            "data:    [a, b, c, d, e, f]   ← one flat Float64Array\n"
            "\n"
            "logical view:\n"
            "  row 0: a b c\n"
            "  row 1: d e f\n"
            "\n"
            "flat_index(i, j) = i * strides[0] + j * strides[1]\n"
            "```\n"
        ),
        pitfalls=[
            "Mixing up `number[]` and `Float64Array` — pick one and stick to it (we use `Float64Array`).",
            "Forgetting that strides for a contiguous tensor are *products of trailing shape dims*, not arbitrary.",
            "Letting `shape` and `strides` lengths disagree; they must always match.",
            "Mutating `shape` or `strides` in place when you should have created a new view.",
            "Allowing `any` to creep in just to silence a generics error — fix the types instead.",
        ],
        verify=[
            "bun test src/tensor/types.test.ts",
            "bun run exercises/ch-01-tensor-type-system.ts",
        ],
        further=[
            ("NumPy — N-dimensional array internals",
             "https://numpy.org/doc/stable/reference/arrays.ndarray.html",
             "the design our `Tensor` type mirrors; read the section on strides."),
            ("Eli Bendersky — Strides in NumPy",
             "https://eli.thegreenplace.net/2015/memory-layout-of-multi-dimensional-arrays",
             "clearest single explainer of row-major layout and strides."),
            ("PyTorch — `torch.Tensor` reference",
             "https://pytorch.org/docs/stable/tensors.html",
             "names and conventions we will roughly follow."),
            THREE_BLUE,
        ],
    ),
    Chapter(
        doc_path="part-1-tensor-library/ch-02-tensor-creation.md",
        part_num=1, part_name="Tensor Library",
        num="02", title="Tensor Creation",
        src=["src/tensor/creation.ts"],
        tests=["src/tensor/creation.test.ts"],
        exercise="exercises/ch-02-tensor-creation.ts",
        next_doc="part-1-tensor-library/ch-03-elementwise-ops-broadcasting.md",
        next_blurb="Elementwise Ops — once we can create tensors, we want to add, multiply, and broadcast them.",
        goals=[
            "Create zero, one, full, and arange tensors with correct shapes and strides.",
            "Sample uniform random tensors with a fixed seed for reproducibility.",
            "Implement Box–Muller to sample normal random tensors without a math library.",
            "Convert nested JS arrays to flat `Float64Array` tensors safely.",
            "Spot when a constructor copies vs. aliases its input.",
        ],
        intuition=(
            "Tensors come from one of two places: code that fills them with a known pattern "
            "(zeros, ones, arange) or code that fills them with randomness (uniform, normal). "
            "Randomness is the seed of every neural network — weights must be initialised so "
            "that no two neurons start identical, otherwise gradient descent has nothing to break "
            "the symmetry between them.\n\n"
            "We get normal samples from uniform samples using the **Box–Muller transform**: pick "
            "two independent uniform numbers $u_1, u_2 \\in (0, 1]$ and emit two independent "
            "standard-normal samples."
        ),
        mental_model=(
            "```text\n"
            "  uniform u1, u2 ∈ (0,1]\n"
            "          │\n"
            "          ▼\n"
            "  z0 = √(-2 ln u1) · cos(2π u2)   ← return now\n"
            "  z1 = √(-2 ln u1) · sin(2π u2)   ← cache for next call\n"
            "          │\n"
            "          ▼\n"
            "  standard normal samples (mean 0, var 1)\n"
            "```\n"
        ),
        pitfalls=[
            "Using `Math.random()` directly — it has no seed, so tests become non-deterministic.",
            "Calling `Math.log(0)` in Box–Muller; clamp `u1` away from `0`.",
            "Forgetting to cache the second Box–Muller sample, wasting half the work.",
            "Sharing one `Float64Array` between two tensors so mutating one mutates the other.",
            "Using JS `number[]` for large tensors; allocate `Float64Array(size)` once.",
        ],
        verify=[
            "bun test src/tensor/creation.test.ts",
            "bun run exercises/ch-02-tensor-creation.ts",
        ],
        further=[
            ("Wikipedia — Box–Muller transform",
             "https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform",
             "derivation and proof that the output is standard normal."),
            ("NumPy — random number routines",
             "https://numpy.org/doc/stable/reference/random/index.html",
             "the API style our creation functions mirror."),
            ("Glorot & Bengio — Understanding the difficulty of training deep networks (2010)",
             "http://proceedings.mlr.press/v9/glorot10a.html",
             "why scaled initial randomness matters; relevant in Ch 13."),
            ("PCG Random",
             "https://www.pcg-random.org/",
             "a modern seeded PRNG family if you outgrow LCG."),
        ],
    ),
    Chapter(
        doc_path="part-1-tensor-library/ch-03-elementwise-ops-broadcasting.md",
        part_num=1, part_name="Tensor Library",
        num="03", title="Elementwise Ops & Broadcasting",
        src=["src/tensor/ops.ts"],
        tests=["src/tensor/ops.test.ts"],
        exercise="exercises/ch-03-elementwise-ops.ts",
        next_doc="part-1-tensor-library/ch-04-matrix-ops.md",
        next_blurb="Matrix Ops — once add/mul broadcast correctly, we need matmul and transpose for linear layers.",
        goals=[
            "State and apply the two broadcasting rules from memory.",
            "Compute the broadcast result shape of any two shapes (or detect that they are incompatible).",
            "Implement `add`, `sub`, `mul`, `div`, `neg` for any two broadcast-compatible tensors.",
            "Recognise where broadcasting will appear in a transformer (bias, mask, scaling).",
            "Avoid silent shape errors by validating shapes at every entry point.",
        ],
        intuition=(
            "Broadcasting is a way of pretending a small tensor is a big one without copying any "
            "memory. If you add a `[d]` bias to a `[batch, seq, d]` activation, broadcasting "
            "stretches the bias along the first two axes — conceptually — so the shapes match.\n\n"
            "The two rules:\n\n"
            "1. Line up the shapes from the **right**. Missing axes on the left are treated as size `1`.\n"
            "2. At each axis, the two sizes must be **equal** or **one of them must be `1`**. The `1` "
            "stretches to match the other size."
        ),
        mental_model=(
            "```text\n"
            "  A.shape:        [8, 1, 64]\n"
            "  B.shape:           [12, 64]\n"
            "  ----------------------------- align right, pad with 1s\n"
            "  A.shape:        [8,  1, 64]\n"
            "  B.shape:        [1, 12, 64]\n"
            "  result shape:   [8, 12, 64]   ← each axis = max\n"
            "```\n"
        ),
        pitfalls=[
            "Aligning shapes from the *left* — broadcasting always aligns from the **right**.",
            "Allowing two axes of size 2 and 3 to broadcast — they must be equal or one must be 1.",
            "Forgetting that broadcasting may produce a result *larger than either input*.",
            "Materialising the broadcast result for a tiny op when an in-place loop would do.",
            "Letting NaN/Infinity slip through `div` when the denominator broadcasts to zero.",
        ],
        verify=[
            "bun test src/tensor/ops.test.ts",
            "bun run exercises/ch-03-elementwise-ops.ts",
        ],
        further=[
            NUMPY_BROADCASTING,
            ("PyTorch — Broadcasting semantics",
             "https://pytorch.org/docs/stable/notes/broadcasting.html",
             "same rules, slightly different wording."),
            ("Jake VanderPlas — Python Data Science Handbook (broadcasting chapter)",
             "https://jakevdp.github.io/PythonDataScienceHandbook/02.05-computation-on-arrays-broadcasting.html",
             "the clearest visual treatment of the rules."),
            THREE_BLUE,
        ],
    ),
    Chapter(
        doc_path="part-1-tensor-library/ch-04-matrix-ops.md",
        part_num=1, part_name="Tensor Library",
        num="04", title="Matrix Operations",
        src=["src/tensor/linalg.ts"],
        tests=["src/tensor/linalg.test.ts"],
        exercise="exercises/ch-04-matrix-ops.ts",
        next_doc="part-1-tensor-library/ch-05-reductions.md",
        next_blurb="Reductions — once we can multiply matrices, we will collapse them along axes to compute norms, means, and losses.",
        goals=[
            "Implement 2-D matmul, then generalise to batched matmul over leading axes.",
            "Implement `transpose` as an axis permutation (no data copy when possible).",
            "Implement `reshape` and explain when it is O(1) vs. when it must copy.",
            "Implement `concat` and `split` along an arbitrary axis.",
            "Predict the matmul shape for every layer of a transformer.",
        ],
        intuition=(
            "A matrix multiplication is just \"for every row of A, dot it with every column of B\". "
            "A batched matmul is the same operation done in parallel for many independent pairs of "
            "matrices stacked along leading axes. In a transformer, the attention score `QKᵀ` is a "
            "batched matmul over `[batch, heads]`; the linear layer's `xW` is a batched matmul over "
            "`[batch, seq]`. Get one matmul right and the whole library scales up cleanly."
        ),
        mental_model=(
            "```text\n"
            "  A:  [..., m, k]            B:  [..., k, n]\n"
            "                ╲    matmul    ╱\n"
            "                 ▼            ▼\n"
            "              C:  [..., m, n]\n"
            "\n"
            "  C[..., i, j] = Σ_p  A[..., i, p] · B[..., p, j]\n"
            "```\n"
        ),
        pitfalls=[
            "Forgetting the inner-dim constraint: `A.shape[-1]` must equal `B.shape[-2]`.",
            "Transposing the wrong matrix — `(AB)ᵀ = BᵀAᵀ`, not `AᵀBᵀ`.",
            "Treating `reshape` as free when the source isn't contiguous; it may have to copy.",
            "Allocating a fresh output buffer inside the inner loop; allocate once outside.",
            "Mismatched batch axes between the two operands; broadcast or fail loudly.",
        ],
        verify=[
            "bun test src/tensor/linalg.test.ts",
            "bun run exercises/ch-04-matrix-ops.ts",
        ],
        further=[
            ("NumPy — `numpy.matmul`",
             "https://numpy.org/doc/stable/reference/generated/numpy.matmul.html",
             "specifies the batched-matmul broadcasting rules we follow."),
            ("Goto & van de Geijn — Anatomy of High-Performance Matrix Multiplication",
             "https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_revision.pdf",
             "the algorithms behind every fast BLAS; optional but eye-opening."),
            THREE_BLUE,
            ("Matrix Cookbook",
             "https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf",
             "an identity reference; you will reach for it during backward passes."),
        ],
    ),
    Chapter(
        doc_path="part-1-tensor-library/ch-05-reductions.md",
        part_num=1, part_name="Tensor Library",
        num="05", title="Reductions",
        src=["src/tensor/reduce.ts"],
        tests=["src/tensor/reduce.test.ts"],
        exercise="exercises/ch-05-reductions.ts",
        next_doc="part-1-tensor-library/ch-06-math-primitives.md",
        next_blurb="Math Primitives — combine reductions with elementwise math to get softmax, LayerNorm, and friends.",
        goals=[
            "Implement `sum`, `mean`, `max`, `min`, `argmax`, and `var` with an `axis` and `keepDims` option.",
            "Choose between collapsing an axis (`keepDims=false`) and keeping it as size 1.",
            "Compute mean/variance in a single pass that is numerically stable.",
            "Use reductions to compute losses (mean of per-sample errors) and normalisation statistics.",
            "Predict which reductions appear inside softmax, LayerNorm, and cross-entropy.",
        ],
        intuition=(
            "A reduction collapses one or more axes into a single value per leftover position. "
            "Mean of a `[batch, features]` tensor along axis `1` gives `[batch]`, the average feature "
            "value per sample. Reductions are how we get from millions of activations down to a "
            "single scalar loss that gradient descent can minimise."
        ),
        mental_model=(
            "```text\n"
            "  x.shape: [2, 3]            sum(axis=1)\n"
            "  [[1, 2, 3],                ──────────►   [6, 15]    keepDims=false → [2]\n"
            "   [4, 5, 6]]                              [[6], [15]] keepDims=true  → [2, 1]\n"
            "```\n"
        ),
        pitfalls=[
            "Computing variance as `mean(x²) - mean(x)²` instead of `mean((x - mean(x))²)`; the first is faster but loses precision.",
            "Reducing over the wrong axis because you forgot what shape the input has — print shapes.",
            "Returning a 0-d tensor when the caller wanted a JS `number`; pick one convention and stick to it.",
            "Forgetting `keepDims=true` and breaking a broadcast that comes later (LayerNorm and softmax need it).",
            "Iterating in C-order when the reduced axis is not the inner axis; you'll thrash the cache.",
        ],
        verify=[
            "bun test src/tensor/reduce.test.ts",
            "bun run exercises/ch-05-reductions.ts",
        ],
        further=[
            ("NumPy — Reduction operations",
             "https://numpy.org/doc/stable/reference/routines.math.html",
             "list of standard reductions and their `axis`/`keepdims` semantics."),
            ("Welford's algorithm for variance",
             "https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm",
             "a streaming, numerically-stable variance; useful for very large batches."),
            ("PyTorch — `torch.Tensor.sum`",
             "https://pytorch.org/docs/stable/generated/torch.Tensor.sum.html",
             "exact semantics for `dim` and `keepdim` we will roughly mirror."),
            GOODFELLOW,
        ],
    ),
    Chapter(
        doc_path="part-1-tensor-library/ch-06-math-primitives.md",
        part_num=1, part_name="Tensor Library",
        num="06", title="Math Primitives",
        src=["src/tensor/math.ts"],
        tests=["src/tensor/math.test.ts"],
        exercise="exercises/ch-06-math-primitives.ts",
        next_doc="part-2-autodiff/ch-07-calculus-for-ml.md",
        next_blurb="Calculus for ML — equipped with `exp`, `log`, and `sqrt`, we can move on to derivatives and the autograd engine.",
        goals=[
            "Implement elementwise `exp`, `log`, `sqrt`, `tanh`, `pow`, and `abs` on tensors.",
            "Identify the three classic numerical-stability traps: `log(0)`, `exp(big)`, and `sqrt(x + ε)`.",
            "Write a reusable `applyElementwise(fn)` helper so each primitive is one line.",
            "Verify each primitive against finite differences before depending on it.",
            "Know which math primitive shows up where in the transformer (softmax, LayerNorm, GELU).",
        ],
        intuition=(
            "Math primitives are tiny scalar functions stamped onto every cell of a tensor. The "
            "interesting work is not the math (JavaScript already has `Math.exp`); the interesting "
            "work is **numerical hygiene**. A naive softmax with a single large logit can produce "
            "`Infinity / Infinity = NaN` and silently poison your entire training run. We learn the "
            "stable forms once and re-use them everywhere."
        ),
        mental_model=(
            "```text\n"
            "  applyElementwise(fn):\n"
            "      for i in 0 .. data.length-1:\n"
            "          out[i] = fn(data[i])\n"
            "\n"
            "  guard rails to remember:\n"
            "      log(x)        →  log(max(x, ε))\n"
            "      exp(x)        →  exp(x - max(x))   (in softmax)\n"
            "      sqrt(x)       →  sqrt(x + ε)        (in LayerNorm)\n"
            "```\n"
        ),
        pitfalls=[
            "Calling `Math.log` on a probability that has rounded to 0; clip with a tiny `eps`.",
            "Computing `exp(x)` for `x` of size 1000+; subtract the max first whenever you can.",
            "Using `**` for integer powers but forgetting it works for floats too (it is `Math.pow`).",
            "Allocating a new output tensor inside a hot loop; pre-allocate and write in place.",
            "Trusting your op without finite-difference checks — they catch sign bugs early.",
        ],
        verify=[
            "bun test src/tensor/math.test.ts",
            "bun run exercises/ch-06-math-primitives.ts",
        ],
        further=[
            ("Goldberg — What Every Computer Scientist Should Know About Floating-Point",
             "https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html",
             "the classic; explains why `(a + b) + c ≠ a + (b + c)` for floats."),
            ("Nicholas Higham — Accuracy and Stability of Numerical Algorithms",
             "https://epubs.siam.org/doi/book/10.1137/1.9780898718027",
             "deeper treatment if you want to design stable kernels."),
            ("Wikipedia — LogSumExp",
             "https://en.wikipedia.org/wiki/LogSumExp",
             "the trick behind stable softmax and stable cross-entropy."),
            GOODFELLOW,
        ],
    ),
    # ─── Part 2 ──────────────────────────────────────────────────────────
    Chapter(
        doc_path="part-2-autodiff/ch-07-calculus-for-ml.md",
        part_num=2, part_name="Autodiff Engine",
        num="07", title="Calculus for Machine Learning",
        src=["src/autograd/engine.ts"],
        tests=["src/autograd/value.test.ts"],
        exercise="exercises/ch-07-calculus-for-ml.ts",
        next_doc="part-2-autodiff/ch-08a-autograd-forward.md",
        next_blurb="Autograd Foundations — turn the chain rule into a small graph data structure we can actually code.",
        goals=[
            "Define a derivative as the limit of a difference quotient and as a slope.",
            "Approximate a derivative with the centred-difference formula and bound its error.",
            "Apply the chain rule to a small composition of functions, by hand.",
            "Extend partial derivatives to the gradient of a multi-variable function.",
            "Connect the gradient to the parameter-update rule `θ ← θ − η ∇L`.",
        ],
        intuition=(
            "A derivative measures \"if I nudge the input by a tiny amount, how much does the output "
            "change?\". That single idea drives **every** parameter update in the course. We do not "
            "need calculus tricks; we need three things:\n\n"
            "- A reliable way to compute a derivative numerically (centred differences).\n"
            "- The chain rule, to compose derivatives through a network of operations.\n"
            "- The gradient as the multi-input generalisation of the derivative."
        ),
        mental_model=(
            "```text\n"
            "        f(x + h)\n"
            "          /\n"
            "         /  ← slope ≈ (f(x+h) − f(x−h)) / (2h)\n"
            "        /\n"
            "  f(x−h)\n"
            "       └────────── x − h     x     x + h\n"
            "```\n"
        ),
        pitfalls=[
            "Using forward differences `(f(x+h)-f(x))/h` when centred differences are nearly free and twice as accurate.",
            "Picking `h` too small (round-off blows up) or too large (truncation error dominates); `1e-5` is a safe default.",
            "Forgetting the chain rule's *multiplication*: `dy/dx = (dy/du)(du/dx)`, not addition.",
            "Treating the gradient as a scalar; it is a *vector* with one entry per input.",
            "Comparing two derivatives with absolute error when the values themselves are large — use **relative** error.",
        ],
        verify=[
            "bun run exercises/ch-07-calculus-for-ml.ts",
        ],
        further=[
            ("3Blue1Brown — Essence of Calculus",
             "https://www.3blue1brown.com/topics/calculus",
             "the visual intuition we lean on throughout autograd."),
            ("Stanford CS231n — Backpropagation notes",
             "https://cs231n.github.io/optimization-2/",
             "best short write-up of chain-rule-as-computational-graph."),
            ("Khan Academy — Multivariable derivatives",
             "https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives",
             "partials and gradients at undergrad pace."),
            KARPATHY_ZTH,
        ],
    ),
    Chapter(
        doc_path="part-2-autodiff/ch-08a-autograd-forward.md",
        part_num=2, part_name="Autodiff Engine",
        num="08a", title="Autograd — Forward Graph",
        src=["src/autograd/value.ts"],
        tests=["src/autograd/value.test.ts"],
        exercise="exercises/ch-08-autograd.ts",
        next_doc="part-2-autodiff/ch-08b-autograd-backward.md",
        next_blurb="Autograd Backward — wire up the `backward()` pass that walks the graph in reverse and accumulates gradients.",
        goals=[
            "Wrap a single number in a `Value` that knows its parents, its operation, and its gradient.",
            "Build a computation graph by overloading `add`, `mul`, `pow`, `exp`, `log`, and `tanh`.",
            "Explain why every `Value` stores a `_backward` closure instead of a giant switch statement.",
            "Spot the difference between the forward graph (built eagerly) and the backward pass (run on demand).",
            "Replicate Karpathy's micrograd API in strict TypeScript.",
        ],
        intuition=(
            "Autograd is bookkeeping. Every time you do `c = a * b`, we secretly record:\n\n"
            "- `c` came from `a` and `b`.\n"
            "- The operation was `*`.\n"
            "- The local derivatives are `∂c/∂a = b` and `∂c/∂b = a`.\n\n"
            "When you later call `c.backward()`, we walk the recorded graph backwards and multiply "
            "local derivatives together using the chain rule. The forward pass *builds* the graph; "
            "the backward pass *walks* it."
        ),
        mental_model=(
            "```text\n"
            "    a ───┐\n"
            "         ├── (*) ──► c ───┐\n"
            "    b ───┘                ├── (+) ──► L\n"
            "                     d ───┘\n"
            "\n"
            "  forward:  store data, parents, op on each node\n"
            "  backward: visit nodes in reverse topological order and accumulate `grad`\n"
            "```\n"
        ),
        pitfalls=[
            "Storing parents as a `Set` and losing iteration order — use an array.",
            "Capturing `this` inside `_backward` and confusing it with a different node; bind the locals you need.",
            "Forgetting that `Value` is *scalar* in 08a — tensor autograd waits until Ch 10.",
            "Reusing the same `Value` across two unrelated computations; build a fresh graph for each forward pass.",
            "Returning a primitive `number` from an operator when you should return a `Value`.",
        ],
        verify=[
            "bun test src/autograd/value.test.ts",
            "bun run exercises/ch-08-autograd.ts",
        ],
        further=[
            ("Karpathy — micrograd",
             "https://github.com/karpathy/micrograd",
             "the ~100-line Python autograd this chapter is modelled on."),
            KARPATHY_ZTH,
            ("Baydin et al. — Automatic Differentiation in Machine Learning: a Survey",
             "https://arxiv.org/abs/1502.05767",
             "the survey paper; covers forward, reverse, and tape-based AD."),
            ("Chris Olah — Calculus on Computational Graphs",
             "https://colah.github.io/posts/2015-08-Backprop/",
             "best intuition piece on graph-based backprop."),
        ],
    ),
    Chapter(
        doc_path="part-2-autodiff/ch-08b-autograd-backward.md",
        part_num=2, part_name="Autodiff Engine",
        num="08b", title="Autograd — Backward Pass",
        src=["src/autograd/value.ts"],
        tests=["src/autograd/value.test.ts"],
        exercise="exercises/ch-08-autograd.ts",
        next_doc="part-2-autodiff/ch-09-gradient-descent.md",
        next_blurb="Gradient Descent — now that gradients flow, use them to actually move parameters downhill.",
        goals=[
            "Implement `backward()` as a reverse topological traversal of the computation graph.",
            "Accumulate gradients with `+=` so that a `Value` used in two places sums its contributions.",
            "Derive and memorise the local-gradient table for `add`, `mul`, `pow`, `exp`, `log`, `tanh`.",
            "Verify each operator's gradient with finite differences from Ch 07.",
            "Identify when *not* zeroing gradients between steps causes runaway updates.",
        ],
        intuition=(
            "Backprop is one rule applied repeatedly: **the gradient flowing into a node is the sum "
            "of the gradients flowing out of it, each multiplied by the local derivative on that "
            "edge.** Topological order guarantees we never visit a node before all its downstream "
            "users have contributed."
        ),
        mental_model=(
            "```text\n"
            "  forward:        a ─► (*) ─► c ─► (+) ─► L\n"
            "                   ╲                ╱\n"
            "                    b              d\n"
            "\n"
            "  backward starts at L.grad = 1, then for each node in reverse-topo order:\n"
            "      for each parent p:\n"
            "          p.grad += local_∂node/∂p · node.grad\n"
            "```\n"
        ),
        pitfalls=[
            "Assigning `p.grad = …` instead of `p.grad += …` — a node reused twice will lose half its gradient.",
            "Forgetting to call `zeroGrad()` between training steps; gradients accumulate forever.",
            "Visiting nodes in forward order; you must run in **reverse** topological order.",
            "Re-running `backward()` on a graph that has already been backwarded — fresh forward each time.",
            "Implementing `tanh` backward as `1 - x²` instead of `1 - out²` (use the cached output).",
        ],
        verify=[
            "bun test src/autograd/value.test.ts",
            "bun run exercises/ch-08-autograd.ts",
        ],
        further=[
            ("Karpathy — micrograd walkthrough video",
             "https://www.youtube.com/watch?v=VMj-3S1tku0",
             "step-by-step build of the same backward pass we are coding."),
            ("Goodfellow et al. — Deep Learning, ch. 6.5 (Back-Propagation)",
             "https://www.deeplearningbook.org/contents/mlp.html",
             "the formal treatment of reverse-mode AD."),
            ("Justin Domke — Why is automatic differentiation efficient?",
             "https://justindomke.wordpress.com/2009/02/17/automatic-differentiation-the-most-criminally-underused-tool-in-the-potential-machine-learning-toolbox/",
             "explains why reverse mode is `O(forward)` for scalar loss."),
            ("Chris Olah — Calculus on Computational Graphs",
             "https://colah.github.io/posts/2015-08-Backprop/",
             "complements the 08a reading; focus on multivariate examples."),
        ],
    ),
    Chapter(
        doc_path="part-2-autodiff/ch-09-gradient-descent.md",
        part_num=2, part_name="Autodiff Engine",
        num="09", title="Gradient Descent",
        src=["src/autograd/value.ts"],
        tests=["src/autograd/value.test.ts"],
        exercise="exercises/ch-09-gradient-descent.ts",
        next_doc="part-2-autodiff/ch-10-tensor-autograd-bridge.md",
        next_blurb="Tensor Autograd — lift scalar autograd to tensors so we can train real neural networks.",
        goals=[
            "Apply the rule `θ ← θ − η ∇L` to a single parameter and a vector of parameters.",
            "Observe loss curves under three learning rates: too small, well-tuned, too large.",
            "Explain why gradient descent finds a *local* minimum, not necessarily a global one.",
            "Fit a one-parameter line and a two-parameter line to synthetic data using only autograd.",
            "Spot divergence (loss going up) and react by lowering the learning rate.",
        ],
        intuition=(
            "Imagine the loss as a landscape over the parameter space and the gradient as the "
            "compass needle pointing **uphill**. Gradient descent walks in the opposite direction. "
            "The learning rate `η` is the step size — too small and you crawl, too large and you "
            "stride past the valley and end up climbing the next hill."
        ),
        mental_model=(
            "```text\n"
            "    loss\n"
            "     ▲\n"
            "     │     ●            ← start\n"
            "     │       ╲\n"
            "     │        ╲   ← step: −η ∇L\n"
            "     │         ●\n"
            "     │           ╲\n"
            "     │            ● ── ● ── ●   ← convergence\n"
            "     └──────────────────────────► parameter\n"
            "```\n"
        ),
        pitfalls=[
            "Forgetting `zeroGrad()` before `backward()` — gradients accumulate across steps.",
            "Updating parameters *inside* the autograd graph; do the update without tracking.",
            "Picking a learning rate by guessing; sweep on a log scale (`1e-4, 1e-3, 1e-2`).",
            "Declaring victory after one step — plot the loss curve.",
            "Comparing different runs without fixing the random seed first.",
        ],
        verify=[
            "bun test src/autograd/value.test.ts",
            "bun run exercises/ch-09-gradient-descent.ts",
        ],
        further=[
            ("Sebastian Ruder — An overview of gradient descent optimization algorithms",
             "https://ruder.io/optimizing-gradient-descent/",
             "tour of SGD variants; we will implement several in Ch 14."),
            ("3Blue1Brown — Gradient descent, how neural networks learn",
             "https://www.3blue1brown.com/lessons/gradient-descent",
             "visual treatment of the loss landscape."),
            ("Bottou, Curtis & Nocedal — Optimization Methods for Large-Scale ML",
             "https://arxiv.org/abs/1606.04838",
             "more rigorous look at SGD and its analysis."),
            GOODFELLOW,
        ],
    ),
    Chapter(
        doc_path="part-2-autodiff/ch-10-tensor-autograd-bridge.md",
        part_num=2, part_name="Autodiff Engine",
        num="10", title="Tensor Autograd",
        src=["src/autograd/grad.ts", "src/autograd/engine.ts"],
        tests=["src/autograd/grad.test.ts"],
        exercise="exercises/ch-10-tensor-autograd.ts",
        next_doc="part-3-neural-net-primitives/ch-11-activation-functions.md",
        next_blurb="Activations — with tensor autograd in place, we can build the nonlinear layers that make networks expressive.",
        goals=[
            "Promote `Value` from scalar to tensor: data and grad are both `Tensor`s with matching shapes.",
            "Implement backward passes for tensor `add`, `mul`, `matMul`, `reshape`, and `transpose`.",
            "Implement `sumToShape` to un-broadcast a gradient back to its original tensor shape.",
            "Verify every tensor op against finite differences before depending on it.",
            "Confirm that all 19 tests pass; this is the foundation every later chapter relies on.",
        ],
        intuition=(
            "Tensor autograd is scalar autograd with two extra rules:\n\n"
            "1. **Matrix backward.** If `C = A @ B`, then `∂L/∂A = ∂L/∂C @ Bᵀ` and `∂L/∂B = Aᵀ @ ∂L/∂C`. "
            "These two formulas are the workhorses of every backward pass in this course.\n"
            "2. **Un-broadcasting.** If a forward op broadcast a small tensor up to a large shape, "
            "the backward op must sum the gradient back down to the small shape. Otherwise gradients "
            "have the wrong shape and the optimizer blows up."
        ),
        mental_model=(
            "```text\n"
            "  forward:  C = A @ B           (shapes: A:[m,k], B:[k,n] → C:[m,n])\n"
            "\n"
            "  backward (given dL/dC of shape [m,n]):\n"
            "      dL/dA = dL/dC @ Bᵀ       → [m, k]\n"
            "      dL/dB = Aᵀ @ dL/dC      → [k, n]\n"
            "\n"
            "  broadcast rule:\n"
            "      forward stretched [d] → [batch, seq, d]\n"
            "      backward must  Σ over batch and seq to get [d] again\n"
            "```\n"
        ),
        pitfalls=[
            "Forgetting to transpose in matmul backward — the shape error will be obvious if your shape checks are strict.",
            "Skipping `sumToShape` after a broadcasted op; the gradient comes back the wrong shape.",
            "Reusing a tensor's data buffer in two ops without cloning — backward will corrupt forward data.",
            "Implementing `reshape` backward as a fresh reshape; it should be `reshape(grad, original_shape)`.",
            "Trusting your op without a finite-difference check; this chapter *needs* checks because every later layer depends on it.",
        ],
        verify=[
            "bun test src/autograd/grad.test.ts",
            "bun run exercises/ch-10-tensor-autograd.ts",
        ],
        further=[
            ("PyTorch internals — autograd",
             "https://pytorch.org/blog/overview-of-pytorch-autograd-engine/",
             "production autograd; same ideas, more bookkeeping."),
            ("Bornschein — Matrix calculus you need for deep learning",
             "https://arxiv.org/abs/1802.01528",
             "derivations of matmul, sum, mean, and softmax backward."),
            ("Justin Domke — Reverse-mode AD",
             "https://people.cs.umass.edu/~domke/courses/sml/08autodiff_nnets.pdf",
             "clean lecture notes generalising 08b to tensors."),
            GOODFELLOW,
        ],
    ),
    # ─── Part 3 ──────────────────────────────────────────────────────────
    Chapter(
        doc_path="part-3-neural-net-primitives/ch-11-activation-functions.md",
        part_num=3, part_name="Neural Net Primitives",
        num="11", title="Activation Functions",
        src=["src/nn/activations.ts"],
        tests=["src/nn/activations.test.ts"],
        exercise="exercises/ch-11-activations.ts",
        next_doc="part-3-neural-net-primitives/ch-12-loss-functions.md",
        next_blurb="Losses — pick a scalar to minimise so we have something to differentiate.",
        goals=[
            "Implement ReLU, sigmoid, tanh, softmax (numerically stable), and GELU.",
            "Sketch each activation by hand and identify its saturation regions.",
            "Derive each activation's gradient and verify with finite differences.",
            "Explain why softmax needs `x - max(x)` before `exp`.",
            "Pick the right activation for the right job (hidden vs. output, regression vs. classification).",
        ],
        intuition=(
            "Without nonlinearities, every stack of linear layers collapses into a single linear "
            "layer — you cannot model curves with straight lines composed of straight lines. "
            "Activations bend the signal between layers. ReLU is the workhorse; sigmoid and tanh "
            "are historic but still useful at output heads; softmax turns logits into a probability "
            "distribution; GELU is the smooth ReLU used inside transformers."
        ),
        mental_model=(
            "```text\n"
            "  ReLU:     y = max(0, x)               (kinks at 0)\n"
            "  sigmoid:  y = 1 / (1 + exp(-x))       (saturates at 0 and 1)\n"
            "  tanh:     y = (eˣ − e⁻ˣ)/(eˣ + e⁻ˣ)   (saturates at ±1)\n"
            "  softmax:  yᵢ = exp(xᵢ) / Σⱼ exp(xⱼ)   (probability distribution)\n"
            "  GELU:     y ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715 x³)))\n"
            "```\n"
        ),
        pitfalls=[
            "Computing softmax without subtracting the max — overflow on a single big logit gives `NaN`.",
            "Treating ReLU as differentiable at exactly 0; pick a sub-gradient (we use 0) and document it.",
            "Using sigmoid in deep hidden layers; gradients vanish through saturation.",
            "Implementing GELU exactly with `erf` when the tanh approximation matches to 1e-3 and is faster.",
            "Forgetting that softmax operates along an axis — usually the last one.",
        ],
        verify=[
            "bun test src/nn/activations.test.ts",
            "bun run exercises/ch-11-activations.ts",
        ],
        further=[
            ("Hendrycks & Gimpel — Gaussian Error Linear Units (GELU)",
             "https://arxiv.org/abs/1606.08415",
             "the paper introducing GELU; transformers use this everywhere."),
            ("Nair & Hinton — Rectified Linear Units (2010)",
             "https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf",
             "the paper that made ReLU mainstream."),
            ("Stanford CS231n — activations",
             "https://cs231n.github.io/neural-networks-1/#actfun",
             "compact comparison of all the common activations."),
            GOODFELLOW,
        ],
    ),
    Chapter(
        doc_path="part-3-neural-net-primitives/ch-12-loss-functions.md",
        part_num=3, part_name="Neural Net Primitives",
        num="12", title="Loss Functions",
        src=["src/nn/losses.ts"],
        tests=["src/nn/losses.test.ts"],
        exercise="exercises/ch-12-losses.ts",
        next_doc="part-3-neural-net-primitives/ch-13-linear-layer.md",
        next_blurb="Linear Layer — connect inputs to outputs with `y = xW + b`, the fundamental building block.",
        goals=[
            "Implement MSE for regression, BCE for binary classification, and cross-entropy for multi-class.",
            "Combine softmax with cross-entropy using log-sum-exp for numerical stability.",
            "Use the closed-form gradient `p − y` for softmax+CE instead of going through two backward passes.",
            "Pick the right loss for the right task — regression vs. binary vs. multi-class.",
            "Verify each loss's gradient with finite differences.",
        ],
        intuition=(
            "A loss collapses everything — millions of activations, thousands of parameters — into "
            "**one scalar** that says \"how wrong are we?\". Gradient descent only knows how to push "
            "that scalar down. Cross-entropy is a measure of how surprised the true label is to see "
            "our predicted distribution; surprise of 0 means perfect, large surprise means very wrong."
        ),
        mental_model=(
            "```text\n"
            "  predictions ──┐\n"
            "                ├──► loss(scalar)  ──►  d loss / d predictions\n"
            "  targets     ──┘\n"
            "\n"
            "  MSE:                  L = (1/N) Σ (p − y)²        gradient: (2/N)(p − y)\n"
            "  Binary CE:            L = −[y log p + (1−y) log(1−p)]\n"
            "  Softmax + CE:         L = −log softmax(z)[y]      gradient: p − one_hot(y)\n"
            "```\n"
        ),
        pitfalls=[
            "Computing softmax then `log()` separately — use log-softmax / log-sum-exp instead.",
            "Forgetting to average over the batch; you'll end up tuning the LR for one batch size and breaking another.",
            "Using MSE for classification — gradients shrink as predictions become extreme and learning stalls.",
            "Passing class indices where the loss expected one-hot, or vice versa; the API must be explicit.",
            "Skipping the closed-form `p − y` for softmax+CE; the long path is correct but numerically worse.",
        ],
        verify=[
            "bun test src/nn/losses.test.ts",
            "bun run exercises/ch-12-losses.ts",
        ],
        further=[
            ("Wikipedia — Cross entropy",
             "https://en.wikipedia.org/wiki/Cross_entropy",
             "the information-theoretic motivation for the loss."),
            ("Chris Olah — Visual information theory",
             "https://colah.github.io/posts/2015-09-Visual-Information/",
             "best intuition piece on entropy and KL divergence."),
            ("Stanford CS231n — losses",
             "https://cs231n.github.io/neural-networks-2/#losses",
             "a survey of standard losses and when to use them."),
            GOODFELLOW,
        ],
    ),
    Chapter(
        doc_path="part-3-neural-net-primitives/ch-13-linear-layer.md",
        part_num=3, part_name="Neural Net Primitives",
        num="13", title="Linear Layer",
        src=["src/nn/linear.ts"],
        tests=["src/nn/linear.test.ts"],
        exercise="exercises/ch-13-linear-layer.ts",
        next_doc="part-3-neural-net-primitives/ch-14-optimizers.md",
        next_blurb="Optimizers — go beyond plain SGD with momentum, Adam, and AdamW.",
        goals=[
            "Implement `Linear(inDim, outDim)` as `y = x @ W + b`.",
            "Initialise weights with Xavier (for tanh) and He (for ReLU) scaling.",
            "Initialise biases to zero and explain why that is safe.",
            "Confirm the forward shape is `[..., outDim]` for any leading shape on `x`.",
            "Read your parameter count and predict it for the full transformer.",
        ],
        intuition=(
            "A linear layer is a learned matrix transform. Geometrically it can rotate, stretch, "
            "and project the input; statistically it learns which combinations of inputs predict the "
            "output. Initialisation matters: weights too small and the signal dies; too large and "
            "it explodes. Xavier and He are calibrated so the variance of activations stays roughly "
            "constant through many layers."
        ),
        mental_model=(
            "```text\n"
            "  x : [..., in]  ──►  x @ W  ──►  + b  ──►  y : [..., out]\n"
            "                       W: [in, out]      b: [out]\n"
            "\n"
            "  Xavier (tanh):   W ~ U(±√(6 / (in + out)))\n"
            "  He (ReLU):       W ~ N(0, √(2 / in))\n"
            "  bias:            zeros\n"
            "```\n"
        ),
        pitfalls=[
            "Using zero initialisation for `W` — all neurons compute the same function and gradients are identical.",
            "Using a huge std-dev so activations saturate or explode after a few layers.",
            "Forgetting to broadcast the bias across the leading axes of `x`.",
            "Hard-coding the weight shape as `[out, in]` then transposing in forward; pick one convention.",
            "Mis-counting parameters; total = `in * out + out`.",
        ],
        verify=[
            "bun test src/nn/linear.test.ts",
            "bun run exercises/ch-13-linear-layer.ts",
        ],
        further=[
            ("Glorot & Bengio — Understanding the difficulty of training deep networks (2010)",
             "http://proceedings.mlr.press/v9/glorot10a.html",
             "the Xavier paper; derives the initialisation."),
            ("He et al. — Delving Deep into Rectifiers (2015)",
             "https://arxiv.org/abs/1502.01852",
             "He initialisation tuned for ReLU networks."),
            ("Stanford CS231n — weight initialisation",
             "https://cs231n.github.io/neural-networks-2/#init",
             "intuitive variance analysis."),
            GOODFELLOW,
        ],
    ),
    Chapter(
        doc_path="part-3-neural-net-primitives/ch-14-optimizers.md",
        part_num=3, part_name="Neural Net Primitives",
        num="14", title="Optimizers",
        src=["src/optim/sgd.ts", "src/optim/adam.ts"],
        tests=["src/optim/sgd.test.ts", "src/optim/adam.test.ts"],
        exercise="exercises/ch-14-optimizers.ts",
        next_doc="part-3-neural-net-primitives/ch-15-training-loop.md",
        next_blurb="Training Loop — assemble layers, loss, optimizer, and data into a working training routine.",
        goals=[
            "Implement plain SGD: `θ ← θ − η ∇L`.",
            "Implement SGD with momentum: `v ← μ v + ∇L; θ ← θ − η v`.",
            "Implement Adam with bias-corrected first and second moments.",
            "Implement AdamW (Adam with decoupled weight decay).",
            "Pick the right optimizer for a task and tune `lr`/`β₁`/`β₂` sanely.",
        ],
        intuition=(
            "Plain SGD treats every parameter the same. Real loss surfaces have steep, narrow "
            "valleys; SGD bounces off the walls. **Momentum** smooths updates by averaging recent "
            "gradients. **Adam** goes further: it gives each parameter its own learning rate based "
            "on its recent gradient magnitudes (parameters with big gradients get smaller steps, "
            "parameters with tiny gradients get bigger ones)."
        ),
        mental_model=(
            "```text\n"
            "  SGD:        θ ← θ − η · g\n"
            "\n"
            "  Momentum:   v ← μ v + g\n"
            "              θ ← θ − η · v\n"
            "\n"
            "  Adam:       m ← β₁ m + (1−β₁) g            ← first moment (mean)\n"
            "              v ← β₂ v + (1−β₂) g²           ← second moment (uncentred variance)\n"
            "              m̂ ← m / (1 − β₁ᵗ),  v̂ ← v / (1 − β₂ᵗ)\n"
            "              θ ← θ − η · m̂ / (√v̂ + ε)\n"
            "```\n"
        ),
        pitfalls=[
            "Mutating the parameter tensor in place inside the autograd graph; do updates outside.",
            "Forgetting bias correction in Adam — early steps will be tiny because `m`, `v` start at 0.",
            "Initialising `v` (Adam's second moment) at 0 and then doing `1/√v` on step 1 without `eps`.",
            "Using Adam's high learning rate (1e-3) with SGD — SGD wants 1e-2 or higher.",
            "Letting state (`m`, `v`) drift between training runs; reset it whenever you reset the model.",
        ],
        verify=[
            "bun test src/optim/sgd.test.ts src/optim/adam.test.ts",
            "bun run exercises/ch-14-optimizers.ts",
        ],
        further=[
            ("Kingma & Ba — Adam: A Method for Stochastic Optimization (2014)",
             "https://arxiv.org/abs/1412.6980",
             "the original Adam paper."),
            ("Loshchilov & Hutter — Decoupled Weight Decay Regularization (AdamW)",
             "https://arxiv.org/abs/1711.05101",
             "why AdamW differs from Adam + L2 and when it matters."),
            ("Sebastian Ruder — gradient descent algorithms",
             "https://ruder.io/optimizing-gradient-descent/",
             "compares every optimizer in one place."),
            ("Distill — Why Momentum Really Works",
             "https://distill.pub/2017/momentum/",
             "interactive treatment of the momentum dynamics."),
        ],
    ),
    Chapter(
        doc_path="part-3-neural-net-primitives/ch-15-training-loop.md",
        part_num=3, part_name="Neural Net Primitives",
        num="15", title="Training Loop",
        src=["src/nn/linear.ts", "src/optim/sgd.ts", "src/optim/adam.ts"],
        tests=["src/nn/linear.test.ts"],
        exercise="exercises/ch-15-training-loop.ts",
        next_doc="part-4-tokenizer-and-inputs/ch-16-char-tokenizer.md",
        next_blurb="Char Tokenizer — switch from synthetic data to text, the input format every language model needs.",
        goals=[
            "Build a small MLP from `Linear` + activation + `Linear`.",
            "Wire forward → loss → zeroGrad → backward → step into one training step.",
            "Iterate over batches, log loss, and confirm the loss curve goes down.",
            "Fit a non-trivial dataset (spiral, XOR, or moons) and visualise the decision boundary.",
            "Detect divergence and recover by lowering the learning rate.",
        ],
        intuition=(
            "Every neural-network training routine — from a 2-layer MLP to GPT-4 — is the same "
            "five-line ritual:\n\n"
            "1. `predictions = model.forward(inputs)`\n"
            "2. `loss = lossFn(predictions, targets)`\n"
            "3. `optimizer.zeroGrad()`\n"
            "4. `loss.backward()`\n"
            "5. `optimizer.step()`\n\n"
            "The rest is data plumbing, logging, and validation."
        ),
        mental_model=(
            "```text\n"
            "  ┌──────────────────────────────────────────┐\n"
            "  │ for epoch in 0..E:                      │\n"
            "  │   for batch in shuffle(dataset):        │\n"
            "  │     y_hat = model.forward(batch.x)      │\n"
            "  │     loss  = lossFn(y_hat, batch.y)      │\n"
            "  │     optimizer.zeroGrad()                │\n"
            "  │     loss.backward()                     │\n"
            "  │     optimizer.step()                    │\n"
            "  │   log(epoch, mean_loss, val_loss)       │\n"
            "  └──────────────────────────────────────────┘\n"
            "```\n"
        ),
        pitfalls=[
            "Forgetting `zeroGrad()` — gradients accumulate and the model explodes after a few steps.",
            "Computing loss on the *training* set and never on a held-out set — you cannot detect overfitting.",
            "Logging the last batch's loss instead of the epoch mean — too noisy to read.",
            "Shuffling the labels but not the inputs (or vice versa) by accident.",
            "Letting `dropout`/`train` mode leak into evaluation — turn it off for validation.",
        ],
        verify=[
            "bun run exercises/ch-15-training-loop.ts",
        ],
        further=[
            ("Karpathy — A Recipe for Training Neural Networks",
             "https://karpathy.github.io/2019/04/25/recipe/",
             "the *exact* habits this chapter trains you to develop."),
            ("Smith — Cyclical Learning Rates",
             "https://arxiv.org/abs/1506.01186",
             "an early but practical look at LR scheduling."),
            ("Stanford CS231n — practical training notes",
             "https://cs231n.github.io/neural-networks-3/",
             "babysitting the learning process; how to read loss curves."),
            ("PyTorch — training loop tutorial",
             "https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html",
             "the same ritual in a production library."),
        ],
    ),
    # ─── Part 4 ──────────────────────────────────────────────────────────
    Chapter(
        doc_path="part-4-tokenizer-and-inputs/ch-16-char-tokenizer.md",
        part_num=4, part_name="Tokenizer & Inputs",
        num="16", title="Character Tokenizer",
        src=["src/tokenizer/char.ts"],
        tests=["src/tokenizer/char.test.ts"],
        exercise="exercises/ch-16-char-tokenizer.ts",
        next_doc="part-4-tokenizer-and-inputs/ch-17-bpe-tokenizer.md",
        next_blurb="BPE Tokenizer — graduate from one-token-per-character to learned subword units.",
        goals=[
            "Build `stoi`/`itos` vocab maps over a fixed set of characters.",
            "Reserve and use special tokens `<pad>`, `<unk>`, `<bos>`, `<eos>`.",
            "Implement `encode(text) → number[]` and `decode(ids) → string`.",
            "Implement `encodeBatch` with right-padding and a parallel `paddingMask`.",
            "Round-trip every test string: `decode(encode(s)) === s`.",
        ],
        intuition=(
            "Neural networks cannot read strings; they can only read numbers. A tokenizer is a "
            "**dictionary** that assigns each character (or subword) a unique integer. A character "
            "tokenizer is the simplest possible: one ID per character. It is slow (long sequences) "
            "but it never sees an unknown character if the vocabulary is the full alphabet."
        ),
        mental_model=(
            "```text\n"
            "  text:      \"hi\"\n"
            "                │\n"
            "                ▼ stoi\n"
            "  token ids: [12, 21]\n"
            "                │\n"
            "                ▼ encodeBatch (pad to length 4)\n"
            "  ids:       [12, 21, 0,  0]\n"
            "  mask:      [ 1,  1, 0,  0]\n"
            "                │\n"
            "                ▼ decode\n"
            "  text:      \"hi\"\n"
            "```\n"
        ),
        pitfalls=[
            "Skipping the `<unk>` token — any character outside the vocab silently breaks encoding.",
            "Padding on the *left* by accident; the rest of the course assumes right padding.",
            "Returning padded IDs without the matching padding mask; downstream attention will attend to pad.",
            "Including newline/whitespace inconsistently — pick a normalisation rule and document it.",
            "Letting `stoi` and `itos` drift out of sync after vocabulary edits.",
        ],
        verify=[
            "bun test src/tokenizer/char.test.ts",
            "bun run exercises/ch-16-char-tokenizer.ts",
        ],
        further=[
            ("Karpathy — Let's build the GPT Tokenizer",
             "https://www.youtube.com/watch?v=zduSFxRajkE",
             "video walkthrough that starts at char level and builds up to BPE."),
            ("HuggingFace — Tokenizers introduction",
             "https://huggingface.co/docs/tokenizers/introduction",
             "industry tokenizer reference."),
            ("Wikipedia — Unicode normalisation forms",
             "https://en.wikipedia.org/wiki/Unicode_equivalence",
             "decide once whether you NFC-normalise your inputs."),
            ("OpenAI — tiktoken",
             "https://github.com/openai/tiktoken",
             "GPT's production tokenizer; useful comparison point."),
        ],
    ),
    Chapter(
        doc_path="part-4-tokenizer-and-inputs/ch-17-bpe-tokenizer.md",
        part_num=4, part_name="Tokenizer & Inputs",
        num="17", title="BPE Tokenizer",
        src=["src/tokenizer/bpe.ts"],
        tests=["src/tokenizer/bpe.test.ts"],
        exercise="exercises/ch-17-bpe-tokenizer.ts",
        next_doc="part-4-tokenizer-and-inputs/ch-18-token-embeddings.md",
        next_blurb="Embeddings — turn token IDs into dense vectors the network can actually learn from.",
        goals=[
            "Implement BPE training: count pairs, merge the most frequent pair, repeat to vocab size.",
            "Encode text by greedily applying learned merges in order.",
            "Decode back to text from BPE token IDs.",
            "Reason about the trade-off between vocab size, sequence length, and `<unk>` rate.",
            "Reproduce a tiny end-to-end train→encode→decode round-trip.",
        ],
        intuition=(
            "BPE compresses common letter pairs into single tokens. Start with characters; find the "
            "most common adjacent pair (say `t` + `h`); merge them into a new token `th`; repeat. "
            "After a few thousand merges, common words become one token, rare words become a "
            "handful of subwords, and unknown words never appear because every character is still a "
            "fallback."
        ),
        mental_model=(
            "```text\n"
            "  initial:   l   o   w   e   r   _   l   o   w   _   l   o   w   e   s   t\n"
            "  step 1:    lo  w   e   r   _   lo  w   _   lo  w   e   s   t            (merged \"l\"+\"o\")\n"
            "  step 2:    low e   r   _   low _   low e   s   t                          (merged \"lo\"+\"w\")\n"
            "  ...\n"
            "```\n"
        ),
        pitfalls=[
            "Re-counting pairs from scratch after every merge — keep an incremental counter.",
            "Forgetting that merges must be applied in **learn-order** at encode time.",
            "Letting whitespace be part of merges in subtle ways; pick a clear word-boundary rule.",
            "Setting `vocabSize` too high relative to corpus size; merges become noise.",
            "Skipping the round-trip test — silent off-by-one bugs in `encode` are common.",
        ],
        verify=[
            "bun test src/tokenizer/bpe.test.ts",
            "bun run exercises/ch-17-bpe-tokenizer.ts",
        ],
        further=[
            ("Sennrich, Haddow, Birch — Neural Machine Translation of Rare Words with Subword Units",
             "https://arxiv.org/abs/1508.07909",
             "the BPE paper."),
            ("HuggingFace — BPE tutorial",
             "https://huggingface.co/learn/nlp-course/chapter6/5",
             "walks through the same algorithm with diagrams."),
            ("Karpathy — minbpe",
             "https://github.com/karpathy/minbpe",
             "minimal reference implementation in Python."),
            ("Google — SentencePiece",
             "https://github.com/google/sentencepiece",
             "BPE + Unigram in one library; what most production models use."),
        ],
    ),
    Chapter(
        doc_path="part-4-tokenizer-and-inputs/ch-18-token-embeddings.md",
        part_num=4, part_name="Tokenizer & Inputs",
        num="18", title="Token Embeddings",
        src=["src/nn/embedding.ts"],
        tests=["src/nn/transformer.test.ts"],
        exercise="exercises/ch-18-embeddings.ts",
        next_doc="part-4-tokenizer-and-inputs/ch-19-positional-encoding.md",
        next_blurb="Positional Encoding — give the model a sense of *where* each token sits in the sequence.",
        goals=[
            "Implement `Embedding(vocabSize, dModel)` as a learned lookup table.",
            "Lookup turns `[batch, seq]` integer IDs into `[batch, seq, dModel]` dense vectors.",
            "Initialise the embedding table with small random values (e.g. `N(0, 0.02)`).",
            "Optionally share weights between the embedding lookup and the output projection (weight tying).",
            "Explain how the embedding table is updated during backprop.",
        ],
        intuition=(
            "An embedding table is a dictionary where each token ID maps to a learned vector. The "
            "vector lives in `ℝ^{dModel}` and is shaped by training: words used in similar contexts "
            "end up near each other. The lookup is just \"pick row `id` from the table\" — no matrix "
            "multiplication needed."
        ),
        mental_model=(
            "```text\n"
            "  table W: [vocabSize, dModel]\n"
            "\n"
            "      id = 3  ───►  W[3, :]   ─── one row, length dModel\n"
            "      id = 7  ───►  W[7, :]\n"
            "\n"
            "  Embedding(ids: [batch, seq]) → [batch, seq, dModel]\n"
            "```\n"
        ),
        pitfalls=[
            "Using `matmul` with a one-hot matrix — correct but absurdly slow; gather rows directly.",
            "Forgetting that backward must scatter-add into the embedding rows that were looked up.",
            "Sharing the embedding table with the output projection but transposing it inconsistently.",
            "Initialising with a huge std-dev; embeddings then dominate positional encodings.",
            "Including the `<pad>` row in the loss when it should be ignored.",
        ],
        verify=[
            "bun test src/nn/transformer.test.ts",
            "bun run exercises/ch-18-embeddings.ts",
        ],
        further=[
            ("Mikolov et al. — Efficient Estimation of Word Representations (word2vec)",
             "https://arxiv.org/abs/1301.3781",
             "the paper that made dense word embeddings famous."),
            ("Press & Wolf — Using the Output Embedding to Improve Language Models",
             "https://arxiv.org/abs/1608.05859",
             "the weight-tying paper we will optionally apply in Ch 29."),
            ("Distill — The Building Blocks of Interpretability",
             "https://distill.pub/2018/building-blocks/",
             "what learned embeddings actually represent."),
            ("Karpathy — makemore (videos)",
             "https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ",
             "embedding tables in action, from scratch."),
        ],
    ),
    Chapter(
        doc_path="part-4-tokenizer-and-inputs/ch-19-positional-encoding.md",
        part_num=4, part_name="Tokenizer & Inputs",
        num="19", title="Positional Encoding",
        src=["src/nn/positional.ts"],
        tests=["src/nn/transformer.test.ts"],
        exercise="exercises/ch-19-positional-encoding.ts",
        next_doc="part-4-tokenizer-and-inputs/ch-20-layernorm-dropout.md",
        next_blurb="LayerNorm & Dropout — make activations stable and add the only regulariser inside transformer blocks.",
        goals=[
            "Implement sinusoidal positional encoding with the original Transformer formula.",
            "Add positional encodings to embeddings so token + position information mix.",
            "Plot a few PE rows and recognise the wavelength pattern.",
            "Compare sinusoidal PE to learned PE and to RoPE / ALiBi at a high level.",
            "Predict PE output shape for any `[batch, seq, dModel]` input.",
        ],
        intuition=(
            "Self-attention is permutation-invariant: it treats a sequence like a *set*. Without "
            "positional information, \"the cat sat\" and \"sat the cat\" produce identical "
            "representations. Positional encoding injects a deterministic, position-dependent "
            "vector that the network can latch onto. Sinusoids give the model a smooth, "
            "extrapolatable signal across many wavelengths."
        ),
        mental_model=(
            "```text\n"
            "  PE(pos, 2i)   = sin(pos / 10000^(2i / dModel))\n"
            "  PE(pos, 2i+1) = cos(pos / 10000^(2i / dModel))\n"
            "\n"
            "      pos = 0  → [sin(0),    cos(0),    sin(0),    cos(0), …]\n"
            "      pos = 1  → [sin(1/1),  cos(1/1),  sin(1/10000^(2/d)), …]\n"
            "      pos = 2  → [sin(2/1),  cos(2/1),  …]\n"
            "```\n"
        ),
        pitfalls=[
            "Forgetting that PE is added to embeddings, not concatenated — shapes must match.",
            "Using base 10000 with `i` in the wrong units; the exponent is `2i / dModel`.",
            "Computing PE inside the training loop every step; cache it once.",
            "Allowing `pos` to exceed `maxSeqLen`; crop or extend the table explicitly.",
            "Scaling embeddings by `√dModel` but forgetting to leave PE unscaled (or vice versa).",
        ],
        verify=[
            "bun test src/nn/transformer.test.ts",
            "bun run exercises/ch-19-positional-encoding.ts",
        ],
        further=[
            ATTN_PAPER,
            ("Su et al. — RoPE: Rotary Position Embedding",
             "https://arxiv.org/abs/2104.09864",
             "the modern alternative used by LLaMA and others."),
            ("Press et al. — ALiBi: Train Short, Test Long",
             "https://arxiv.org/abs/2108.12409",
             "another modern alternative; bias-based, no learned PE."),
            ("Jay Alammar — The Illustrated Transformer",
             "https://jalammar.github.io/illustrated-transformer/",
             "visual treatment that includes positional encodings."),
        ],
    ),
    Chapter(
        doc_path="part-4-tokenizer-and-inputs/ch-20-layernorm-dropout.md",
        part_num=4, part_name="Tokenizer & Inputs",
        num="20", title="LayerNorm & Dropout",
        src=["src/nn/layernorm.ts", "src/nn/dropout.ts"],
        tests=["src/nn/transformer.test.ts"],
        exercise="exercises/ch-20-layernorm-dropout.ts",
        next_doc="part-4-tokenizer-and-inputs/ch-21-mask-cookbook.md",
        next_blurb="Masks — express \"don't look at padding\" and \"don't peek at the future\" with a single tensor primitive.",
        goals=[
            "Implement LayerNorm: per-token mean/variance with learnable `γ`, `β`.",
            "Implement inverted Dropout: zero a fraction `p` and scale the rest by `1/(1−p)`.",
            "Switch Dropout between training (active) and evaluation (no-op).",
            "Place LayerNorm in pre-norm position inside a transformer block.",
            "Identify which axes LayerNorm reduces over (the last one) versus BatchNorm.",
        ],
        intuition=(
            "Activations inside a deep network drift in scale over training; LayerNorm forces every "
            "token's feature vector back to mean 0 and variance 1, then lets the network re-stretch "
            "with learnable `γ`/`β`. It is the thermostat of the network.\n\n"
            "Dropout is the opposite: it deliberately injects noise during training by zeroing "
            "random units. The network learns redundant pathways; at evaluation time we use the "
            "full network and trust the redundancy."
        ),
        mental_model=(
            "```text\n"
            "  LayerNorm (per token):\n"
            "      μ = mean(x, axis=-1)\n"
            "      σ² = var(x, axis=-1)\n"
            "      x̂ = (x − μ) / √(σ² + ε)\n"
            "      y  = γ · x̂ + β\n"
            "\n"
            "  Dropout (inverted, training):\n"
            "      mask = bernoulli(1 − p)\n"
            "      y    = (x · mask) / (1 − p)\n"
            "  Dropout (eval):\n"
            "      y    = x\n"
            "```\n"
        ),
        pitfalls=[
            "Normalising over the batch axis (that's BatchNorm) instead of the feature axis.",
            "Forgetting `ε` inside `√` — variance can be exactly 0 for a constant token.",
            "Scaling dropout output by `(1 − p)` instead of dividing — back-to-front; inverted dropout *divides*.",
            "Leaving dropout active during validation; loss looks much worse than it really is.",
            "Skipping `γ`/`β` parameters; the network needs to be able to undo the normalisation.",
        ],
        verify=[
            "bun test src/nn/transformer.test.ts",
            "bun run exercises/ch-20-layernorm-dropout.ts",
        ],
        further=[
            ("Ba, Kiros, Hinton — Layer Normalization (2016)",
             "https://arxiv.org/abs/1607.06450",
             "the LayerNorm paper."),
            ("Srivastava et al. — Dropout: A Simple Way to Prevent Overfitting",
             "https://jmlr.org/papers/v15/srivastava14a.html",
             "the Dropout paper."),
            ("Xiong et al. — On Layer Normalization in the Transformer Architecture",
             "https://arxiv.org/abs/2002.04745",
             "pre-norm vs. post-norm; the analysis behind modern transformers."),
            ("Distill — Norm matters",
             "https://distill.pub/2019/visualizing-memorization/",
             "visualisations of how normalisation shapes training dynamics."),
        ],
    ),
    Chapter(
        doc_path="part-4-tokenizer-and-inputs/ch-21-mask-cookbook.md",
        part_num=4, part_name="Tokenizer & Inputs",
        num="21", title="Mask Cookbook",
        src=["src/tokenizer/masks.ts"],
        tests=["src/tokenizer/masks.test.ts"],
        exercise="exercises/ch-21-masks.ts",
        next_doc="part-5-attention/ch-22-self-attention.md",
        next_blurb="Self-Attention — combine Q, K, V, scaling, and masks into the heart of the transformer.",
        goals=[
            "Distinguish binary masks (`1` allowed, `0` blocked) from additive masks (`0` allowed, `-∞` blocked).",
            "Convert a binary mask to an additive mask with `(1 − mask) * −∞`.",
            "Build a causal (lower-triangular) mask of shape `[seq, seq]`.",
            "Expand a padding mask of shape `[batch, seq]` to `[batch, 1, 1, seq]` for multi-head attention.",
            "Combine padding mask + causal mask with element-wise min (or sum, in additive form).",
        ],
        intuition=(
            "Attention computes a softmax over scores. To **forbid** a position, set its score to "
            "`-∞` before the softmax — `exp(-∞) = 0`. Masks are the language we use to express "
            "\"please don't look here\". Two reasons to mask:\n\n"
            "- **Padding**: positions filled with `<pad>` carry no real information.\n"
            "- **Causality**: when generating, position `t` must not see positions `> t`."
        ),
        mental_model=(
            "```text\n"
            "  binary mask                       additive mask\n"
            "  ─────────────                     ─────────────\n"
            "  1 = allowed                       0    = allowed\n"
            "  0 = blocked                       −∞   = blocked\n"
            "\n"
            "  scores = QKᵀ / √d_k\n"
            "  scores = scores + additive_mask\n"
            "  weights = softmax(scores)      ← blocked positions contribute 0\n"
            "```\n"
        ),
        pitfalls=[
            "Multiplying scores by a binary mask — the softmax then renormalises and the blocked positions still get weight.",
            "Using `−∞` literally; use a large negative finite number (`-1e9`) to avoid `NaN` from `0 * ∞`.",
            "Forgetting to expand the padding mask to `[batch, 1, 1, seq]` so it broadcasts over heads and query positions.",
            "Building the causal mask once per step inside the loop; build it once per max sequence length.",
            "Combining binary and additive masks without converting one to the other first.",
        ],
        verify=[
            "bun test src/tokenizer/masks.test.ts",
            "bun run exercises/ch-21-masks.ts",
        ],
        further=[
            ATTN_PAPER,
            ("Jay Alammar — The Illustrated Transformer",
             "https://jalammar.github.io/illustrated-transformer/",
             "diagrams of where masks plug into attention."),
            ("Andrej Karpathy — Let's build GPT (video)",
             "https://www.youtube.com/watch?v=kCc8FmEb1nY",
             "live-codes the causal mask and explains the additive trick."),
            ("PyTorch — `nn.Transformer` source",
             "https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html",
             "reference for how the same masks are wired in production."),
        ],
    ),
    # ─── Part 5 ──────────────────────────────────────────────────────────
    Chapter(
        doc_path="part-5-attention/ch-22-self-attention.md",
        part_num=5, part_name="Attention Mechanism",
        num="22", title="Self-Attention",
        src=["src/nn/attention.ts"],
        tests=["src/nn/attention.test.ts"],
        exercise="exercises/ch-22-self-attention.ts",
        next_doc="part-5-attention/ch-23-multi-head-attention.md",
        next_blurb="Multi-Head Attention — run several attention heads in parallel and concatenate their outputs.",
        goals=[
            "Project the input to Q, K, V with three learned linear layers.",
            "Compute scaled dot-product attention: `softmax(QKᵀ / √d_k) V`.",
            "Apply a mask before the softmax to forbid certain positions.",
            "Explain *why* we divide by `√d_k` (gradient flow through softmax).",
            "Predict the output shape: same as the input `[batch, seq, d_model]`.",
        ],
        intuition=(
            "Self-attention lets every token decide which other tokens are relevant. Each token "
            "publishes a **query** (\"what am I looking for?\"), a **key** (\"what do I match?\"), "
            "and a **value** (\"if you match me, here's what I provide\"). Compare every query to "
            "every key, turn the scores into weights with softmax, and average the values weighted "
            "by those weights. The token then becomes a context-aware mix of its neighbours."
        ),
        mental_model=(
            "```text\n"
            "  x : [batch, seq, d_model]\n"
            "      │\n"
            "      ├── @ W_Q ──► Q : [batch, seq, d_k]\n"
            "      ├── @ W_K ──► K : [batch, seq, d_k]\n"
            "      └── @ W_V ──► V : [batch, seq, d_v]\n"
            "\n"
            "  scores  = Q @ Kᵀ / √d_k          shape: [batch, seq_q, seq_k]\n"
            "  scores += additive_mask\n"
            "  weights = softmax(scores, axis=-1)\n"
            "  output  = weights @ V             shape: [batch, seq, d_v]\n"
            "```\n"
        ),
        pitfalls=[
            "Forgetting the `/√d_k` scaling — softmax saturates and gradients vanish.",
            "Applying the mask *after* the softmax — too late, the weights have already normalised.",
            "Using one linear layer for QKV concatenated, then forgetting to split correctly.",
            "Mixing up `d_k` (per-head key dim) with `d_model` when there is only one head.",
            "Returning `weights` (the attention map) when the caller asked for the *output*.",
        ],
        verify=[
            "bun test src/nn/attention.test.ts",
            "bun run exercises/ch-22-self-attention.ts",
        ],
        further=[
            ATTN_PAPER,
            ("Jay Alammar — The Illustrated Transformer",
             "https://jalammar.github.io/illustrated-transformer/",
             "the canonical visual explainer."),
            ("Karpathy — Let's build GPT (video)",
             "https://www.youtube.com/watch?v=kCc8FmEb1nY",
             "live build of self-attention from scratch in PyTorch."),
            ("Lilian Weng — The Transformer Family",
             "https://lilianweng.github.io/posts/2020-04-07-the-transformer-family/",
             "comprehensive survey with all the variants."),
        ],
    ),
    Chapter(
        doc_path="part-5-attention/ch-23-multi-head-attention.md",
        part_num=5, part_name="Attention Mechanism",
        num="23", title="Multi-Head Attention",
        src=["src/nn/attention.ts"],
        tests=["src/nn/attention.test.ts"],
        exercise="exercises/ch-23-multi-head-attention.ts",
        next_doc="part-5-attention/ch-24-cross-attention.md",
        next_blurb="Cross-Attention — let the decoder query the encoder, the glue between source and target.",
        goals=[
            "Split `d_model` into `H` heads of size `d_head = d_model / H`.",
            "Implement projection → reshape → permute to get `[batch, H, seq, d_head]`.",
            "Run scaled dot-product attention per head, in parallel.",
            "Concatenate heads back to `[batch, seq, d_model]` and project with `W_O`.",
            "Predict the parameter count: `4 · d_model²` (for `W_Q`, `W_K`, `W_V`, `W_O`).",
        ],
        intuition=(
            "One attention head captures one kind of relationship — say, \"which token is the "
            "subject?\". Multi-head attention runs several smaller heads in parallel so the model "
            "can attend to different relationships at the same time (syntax, coreference, semantic "
            "similarity). After each head has done its work, we concatenate their outputs and let a "
            "final projection mix the heads together."
        ),
        mental_model=(
            "```text\n"
            "  x : [batch, seq, d_model]\n"
            "       │\n"
            "       │  packed projections\n"
            "       ├── W_Q ──► Q : [batch, seq, d_model]  ──► reshape ──► [batch, H, seq, d_head]\n"
            "       ├── W_K ──► K : [batch, seq, d_model]  ──► reshape ──► [batch, H, seq, d_head]\n"
            "       └── W_V ──► V : [batch, seq, d_model]  ──► reshape ──► [batch, H, seq, d_head]\n"
            "\n"
            "  per-head:    scaled-dot-product-attention(Q_h, K_h, V_h)\n"
            "  concat heads back ──► [batch, seq, d_model]\n"
            "  W_O projection    ──► [batch, seq, d_model]\n"
            "```\n"
        ),
        pitfalls=[
            "Wrong reshape order — must split *then* permute; permute-then-split breaks per-head identity.",
            "Forgetting the final `W_O` projection; without it heads stay disjoint.",
            "Letting `d_model` not divide by `H`; assert this on construction.",
            "Computing scaling with `√d_model` instead of `√d_head`.",
            "Materialising 4 separate parameters when one packed `[d_model, 3 · d_model]` works fine.",
        ],
        verify=[
            "bun test src/nn/attention.test.ts",
            "bun run exercises/ch-23-multi-head-attention.ts",
        ],
        further=[
            ATTN_PAPER,
            ("Voita et al. — Analyzing Multi-Head Self-Attention",
             "https://arxiv.org/abs/1905.09418",
             "what individual heads learn; pruning experiments."),
            ("Jay Alammar — The Illustrated Transformer",
             "https://jalammar.github.io/illustrated-transformer/",
             "the multi-head section is the clearest visual we know."),
            ("Lilian Weng — Attention? Attention!",
             "https://lilianweng.github.io/posts/2018-06-24-attention/",
             "history of attention from seq2seq to multi-head."),
        ],
    ),
    Chapter(
        doc_path="part-5-attention/ch-24-cross-attention.md",
        part_num=5, part_name="Attention Mechanism",
        num="24", title="Cross-Attention",
        src=["src/nn/attention.ts"],
        tests=["src/nn/attention.test.ts"],
        exercise="exercises/ch-24-cross-attention.ts",
        next_doc="part-6-transformer/ch-25-feedforward-block.md",
        next_blurb="Feed-Forward Block — the per-token MLP that follows every attention layer.",
        goals=[
            "Implement cross-attention: Q from one sequence, K and V from another.",
            "Distinguish self-attention (Q, K, V same source) from cross-attention.",
            "Recognise that cross-attention does **not** use a causal mask — the decoder may see the entire source.",
            "Pass the encoder's *final* output as K, V to every decoder block.",
            "Predict the output shape: `[batch, seq_q, d_model]` (length follows the queries).",
        ],
        intuition=(
            "Self-attention answers \"how should this sequence look at itself?\". Cross-attention "
            "answers \"how should sequence A look at sequence B?\". In a translation model, the "
            "decoder (sequence A, partial target) uses cross-attention to query the encoder "
            "(sequence B, full source) — \"I've written so far; which source words do I need next?\""
        ),
        mental_model=(
            "```text\n"
            "  decoder state    ──► W_Q ──► Q : [batch, seq_dec, d_model]\n"
            "  encoder output   ──► W_K ──► K : [batch, seq_enc, d_model]\n"
            "  encoder output   ──► W_V ──► V : [batch, seq_enc, d_model]\n"
            "\n"
            "  scores  = Q @ Kᵀ / √d_head        shape: [batch, H, seq_dec, seq_enc]\n"
            "  scores += encoder_padding_mask    (no causal mask!)\n"
            "  output  = softmax(scores) @ V     shape: [batch, seq_dec, d_model]\n"
            "```\n"
        ),
        pitfalls=[
            "Passing a causal mask to cross-attention — the decoder *should* see all of the source.",
            "Re-running the encoder on every decoder step instead of caching its output.",
            "Mixing up which sequence's padding mask applies; cross-attention uses the **encoder** mask.",
            "Letting decoder and encoder have different `d_model` and forgetting the projection.",
            "Returning attention weights to the caller and forgetting which axis is the source.",
        ],
        verify=[
            "bun test src/nn/attention.test.ts",
            "bun run exercises/ch-24-cross-attention.ts",
        ],
        further=[
            ATTN_PAPER,
            ("Bahdanau, Cho, Bengio — Neural Machine Translation by Jointly Learning to Align and Translate (2014)",
             "https://arxiv.org/abs/1409.0473",
             "the paper that introduced attention; cross-attention's direct ancestor."),
            ("Jay Alammar — The Illustrated Transformer",
             "https://jalammar.github.io/illustrated-transformer/",
             "the encoder-decoder attention diagrams."),
            ("Karpathy — Let's build GPT (video)",
             "https://www.youtube.com/watch?v=kCc8FmEb1nY",
             "decoder-only, but the contrast with cross-attention is useful."),
        ],
    ),
    # ─── Part 6 ──────────────────────────────────────────────────────────
    Chapter(
        doc_path="part-6-transformer/ch-25-feedforward-block.md",
        part_num=6, part_name="Transformer",
        num="25", title="Feed-Forward Block",
        src=["src/nn/feedforward.ts"],
        tests=["src/nn/transformer.test.ts"],
        exercise="exercises/ch-25-feedforward-block.ts",
        next_doc="part-6-transformer/ch-26-encoder-block.md",
        next_blurb="Encoder Block — wrap attention + FFN with LayerNorm and residual connections.",
        goals=[
            "Implement FFN: `W₂ · GELU(W₁ x + b₁) + b₂`.",
            "Use the 4× expansion factor: `d_ff = 4 · d_model` by default.",
            "Confirm the FFN is *position-wise* — it does not mix tokens, only features.",
            "Count parameters: `d_model · d_ff + d_ff + d_ff · d_model + d_model ≈ 8 · d_model²`.",
            "Explain why GELU (not ReLU) became the standard activation here.",
        ],
        intuition=(
            "Attention mixes tokens. The FFN mixes **features within each token**. After attention, "
            "every token has a context-aware vector; the FFN gives the network space to compute "
            "something nonlinear with that vector — \"if this combination of features is present, "
            "emit this other combination\". The FFN runs the same MLP on every position "
            "independently."
        ),
        mental_model=(
            "```text\n"
            "  x : [batch, seq, d_model]\n"
            "       │\n"
            "       ├── @ W_1 + b_1 ──► [batch, seq, d_ff]       (d_ff = 4 · d_model)\n"
            "       ├── GELU\n"
            "       └── @ W_2 + b_2 ──► [batch, seq, d_model]\n"
            "```\n"
        ),
        pitfalls=[
            "Skipping the 4× expansion; with `d_ff = d_model` the FFN has almost no capacity.",
            "Using ReLU in transformer FFNs — GELU is the convention and the gradients are smoother.",
            "Forgetting the biases (some implementations skip them; ours keeps them for clarity).",
            "Sharing FFN weights across blocks — every block has its own.",
            "Letting the FFN mix tokens by accident (e.g., applying it on the wrong axis).",
        ],
        verify=[
            "bun test src/nn/transformer.test.ts",
            "bun run exercises/ch-25-feedforward-block.ts",
        ],
        further=[
            ATTN_PAPER,
            ("Hendrycks & Gimpel — GELU paper",
             "https://arxiv.org/abs/1606.08415",
             "why the FFN uses GELU specifically."),
            ("Shazeer — GLU Variants Improve Transformer (2020)",
             "https://arxiv.org/abs/2002.05202",
             "modern FFN variants (SwiGLU, GeGLU) — used in LLaMA and PaLM."),
            ("Anthropic — A Mathematical Framework for Transformer Circuits",
             "https://transformer-circuits.pub/2021/framework/index.html",
             "treats attention + FFN as the two basic primitives."),
        ],
    ),
    Chapter(
        doc_path="part-6-transformer/ch-26-encoder-block.md",
        part_num=6, part_name="Transformer",
        num="26", title="Encoder Block",
        src=["src/nn/transformer.ts"],
        tests=["src/nn/transformer.test.ts"],
        exercise="exercises/ch-26-encoder-block.ts",
        next_doc="part-6-transformer/ch-27-decoder-block.md",
        next_blurb="Decoder Block — add masked self-attention and cross-attention to make a generating layer.",
        goals=[
            "Combine multi-head self-attention, LayerNorm, residual connection, FFN, LayerNorm, residual into one block.",
            "Place LayerNorm in **pre-norm** position: `x + Sublayer(LN(x))`.",
            "Stack `N` blocks to form the full encoder.",
            "Confirm output shape equals input shape: `[batch, seq, d_model]`.",
            "Explain why residual connections make deep stacks trainable.",
        ],
        intuition=(
            "An encoder block is two passes through the standard \"normalise → transform → add\" "
            "ritual. The residual connection lets the gradient flow backwards through the identity "
            "shortcut even when the inner transform is hard to learn — this is what makes 12-, 24-, "
            "even 96-layer transformers actually train.\n\n"
            "Pre-norm (normalise *before* the sublayer) is now standard; it is more stable than the "
            "original post-norm formulation."
        ),
        mental_model=(
            "```text\n"
            "  x ─► LN ─► MHA ─►(+)─► LN ─► FFN ─►(+)─► out\n"
            "  │              ↑    │             ↑\n"
            "  └──────────────┘    └─────────────┘\n"
            "       residual 1          residual 2\n"
            "```\n"
        ),
        pitfalls=[
            "Using post-norm by accident (`LN(x + Sublayer(x))`); pre-norm is `x + Sublayer(LN(x))`.",
            "Forgetting the residual connection; deep stacks become untrainable.",
            "Sharing LayerNorm parameters across blocks; each block has its own `γ`, `β`.",
            "Letting dropout fire outside training mode.",
            "Mis-counting parameters; the FFN dominates (about `8 · d_model²` per block).",
        ],
        verify=[
            "bun test src/nn/transformer.test.ts",
            "bun run exercises/ch-26-encoder-block.ts",
        ],
        further=[
            ATTN_PAPER,
            ("Xiong et al. — On Layer Normalization in the Transformer",
             "https://arxiv.org/abs/2002.04745",
             "pre-norm vs. post-norm analysis."),
            ("He et al. — Deep Residual Learning (2015)",
             "https://arxiv.org/abs/1512.03385",
             "the residual-connection paper that made deep nets possible."),
            ("Jay Alammar — The Illustrated Transformer",
             "https://jalammar.github.io/illustrated-transformer/",
             "block-level diagrams."),
        ],
    ),
    Chapter(
        doc_path="part-6-transformer/ch-27-decoder-block.md",
        part_num=6, part_name="Transformer",
        num="27", title="Decoder Block",
        src=["src/nn/transformer.ts"],
        tests=["src/nn/transformer.test.ts"],
        exercise="exercises/ch-27-decoder-block.ts",
        next_doc="part-6-transformer/ch-28-sequence-data-objectives.md",
        next_blurb="Sequence Data & Objectives — build the data pipeline that feeds the full Transformer.",
        goals=[
            "Add a masked self-attention sublayer with a causal mask.",
            "Add a cross-attention sublayer with K, V from the encoder output.",
            "Keep the FFN sublayer; total of three pre-norm + residual passes.",
            "Stack `N` decoder blocks to form the full decoder.",
            "Trace the teacher-forcing training loop: shifted inputs, full-sequence parallel processing.",
        ],
        intuition=(
            "The decoder is a writer with one eye on what it has written so far (masked "
            "self-attention) and another on the source it is translating from (cross-attention). "
            "During training we cheat with **teacher forcing** — we feed the true shifted target as "
            "input and rely on the causal mask to keep each position from peeking at its answer."
        ),
        mental_model=(
            "```text\n"
            "  x ─► LN ─► Masked MHA ─►(+)─► LN ─► Cross-Attn ─►(+)─► LN ─► FFN ─►(+)─► out\n"
            "                    ↑                  ↑                          ↑\n"
            "                    │                  │                          │\n"
            "                    │       encoder_out                          │\n"
            "  └─ residual 1 ────┘       └─ residual 2 ────┘     └─ residual 3 ┘\n"
            "```\n"
        ),
        pitfalls=[
            "Forgetting the causal mask in the *masked* self-attention; the model trivially copies its target.",
            "Passing the encoder output of the wrong layer to cross-attention — use the encoder's final output.",
            "Adding a causal mask to cross-attention; only self-attention needs it.",
            "Mismatched padding masks for source vs. target; track both separately.",
            "Sharing weights between encoder and decoder; almost never what you want.",
        ],
        verify=[
            "bun test src/nn/transformer.test.ts",
            "bun run exercises/ch-27-decoder-block.ts",
        ],
        further=[
            ATTN_PAPER,
            ("Harvard NLP — The Annotated Transformer",
             "http://nlp.seas.harvard.edu/annotated-transformer/",
             "the original Transformer re-implemented with line-by-line commentary."),
            ("Jay Alammar — The Illustrated Transformer",
             "https://jalammar.github.io/illustrated-transformer/",
             "decoder-block diagrams."),
            ("Karpathy — Let's build GPT (video)",
             "https://www.youtube.com/watch?v=kCc8FmEb1nY",
             "decoder-only but illustrates the masking logic exactly."),
        ],
    ),
    Chapter(
        doc_path="part-6-transformer/ch-28-sequence-data-objectives.md",
        part_num=6, part_name="Transformer",
        num="28", title="Sequence Data & Training Objectives",
        src=["src/utils/data.ts"],
        tests=["src/utils/data.test.ts"],
        exercise="exercises/ch-28-sequence-data.ts",
        next_doc="part-6-transformer/ch-29-full-transformer.md",
        next_blurb="Full Transformer — train the complete encoder-decoder model on a real toy task.",
        goals=[
            "Define a `Seq2SeqExample` type and a tiny toy dataset (e.g. string reversal).",
            "Build `shiftRight(target, BOS)` so decoder input = `[BOS, t1, …, t_{n-1}]`.",
            "Build a masked cross-entropy loss that ignores `<pad>` positions.",
            "Report perplexity = `exp(loss)` alongside cross-entropy.",
            "Use the one-batch overfit test as a debugging tool *before* training on the full set.",
        ],
        intuition=(
            "Most transformer bugs live in the data pipeline, not in the model. Mis-shifted "
            "targets, padding tokens in the loss, missing causal masks — each of these silently "
            "kills training. This chapter is the data-engineering chapter that makes the next one "
            "(the full transformer) a controlled experiment instead of a guessing game.\n\n"
            "The **overfit-one-batch** test is the single most valuable habit in this part: if the "
            "model cannot drive loss near zero on one tiny batch, something is broken upstream."
        ),
        mental_model=(
            "```text\n"
            "  raw pair:        source=\"hello\"           target=\"olleh\"\n"
            "  tokenize:        src_ids =[BOS, h, e, l, l, o, EOS]\n"
            "                   tgt_ids =[BOS, o, l, l, e, h, EOS]\n"
            "  shiftRight:      dec_in  =[BOS, o, l, l, e, h]\n"
            "                   labels  =[o,   l, l, e, h, EOS]\n"
            "  loss mask:       [1, 1, 1, 1, 1,   1]    ← pad positions get 0\n"
            "  loss:            masked cross-entropy → scalar\n"
            "  metric:          perplexity = exp(loss)\n"
            "```\n"
        ),
        pitfalls=[
            "Training the decoder to copy the token it already sees instead of predicting the next token.",
            "Including `<pad>` tokens in the loss average; perplexity becomes nonsense.",
            "Forgetting EOS, so generation has no natural stopping condition.",
            "Testing only training loss; without validation you cannot detect overfitting.",
            "Moving to the full dataset before passing the one-batch overfit test.",
        ],
        verify=[
            "bun test src/utils/data.test.ts",
            "bun run exercises/ch-28-sequence-data.ts",
        ],
        further=[
            ("Harvard NLP — The Annotated Transformer",
             "http://nlp.seas.harvard.edu/annotated-transformer/",
             "shows the same data pipeline end-to-end."),
            ("Karpathy — A Recipe for Training Neural Networks",
             "https://karpathy.github.io/2019/04/25/recipe/",
             "the overfit-one-batch habit comes from here."),
            ("Wikipedia — Perplexity",
             "https://en.wikipedia.org/wiki/Perplexity",
             "the metric and its information-theoretic meaning."),
            ("Stanford CS224n — Lecture notes",
             "https://web.stanford.edu/class/cs224n/",
             "standard reference for sequence-modelling data pipelines."),
        ],
    ),
    Chapter(
        doc_path="part-6-transformer/ch-29-full-transformer.md",
        part_num=6, part_name="Transformer",
        num="29", title="Full Transformer",
        src=["src/nn/transformer.ts"],
        tests=["src/nn/transformer.test.ts"],
        exercise="exercises/ch-29-full-transformer.ts",
        next_doc="part-6-transformer/ch-30-decoder-only-gpt.md",
        next_blurb="Decoder-Only GPT — strip the encoder away and train next-token prediction directly.",
        goals=[
            "Assemble embeddings, positional encoding, encoder stack, decoder stack, final norm, and output projection.",
            "Run a full forward pass: `forward(src, tgt) → logits [batch, tgt_len, vocab_size]`.",
            "Scale embeddings by `√d_model` before adding positional encoding.",
            "Optionally tie embedding weights to the output projection.",
            "Train on the string-reversal task and achieve loss < 0.1 within ~100 epochs.",
        ],
        intuition=(
            "Every previous chapter has been one ingredient. This chapter is the **recipe**. The "
            "encoder reads the source into a context tensor; the decoder, guided by cross-attention "
            "on that tensor, writes the target token by token. Once you can train this on a toy "
            "task, you can train it on translation — only the dataset changes."
        ),
        mental_model=(
            "```text\n"
            "  source ids   ──► Embed + √d_model ──► +PE ──► Dropout ──► Encoder × N ──► enc_out\n"
            "                                                                            │\n"
            "  target ids   ──► Embed + √d_model ──► +PE ──► Dropout ──► Decoder × N ◄───┘\n"
            "                                                            │\n"
            "                                                            ▼\n"
            "                                                       Final LN\n"
            "                                                            │\n"
            "                                                            ▼\n"
            "                                                  Linear(d_model → vocab)\n"
            "                                                            │\n"
            "                                                            ▼\n"
            "                                                          logits\n"
            "```\n"
        ),
        pitfalls=[
            "Forgetting to scale embeddings by `√d_model`; the magnitudes don't match the positional encoding.",
            "Building one `nn.LayerNorm` and using it everywhere — each location needs its own.",
            "Feeding raw target tokens (not shifted) into the decoder; trivial copy task results.",
            "Letting the decoder maxSeqLen exceed the trained PE table; either crop or extend.",
            "Declaring victory at low train loss without checking generation on unseen examples.",
        ],
        verify=[
            "bun test src/nn/transformer.test.ts",
            "bun run exercises/ch-29-full-transformer.ts",
        ],
        further=[
            ATTN_PAPER,
            ("Harvard NLP — The Annotated Transformer",
             "http://nlp.seas.harvard.edu/annotated-transformer/",
             "the closest published cousin of what we are building."),
            ("Sasha Rush — The Annotated Encoder-Decoder",
             "https://bastings.github.io/annotated_encoder_decoder/",
             "more notes on training dynamics."),
            ("Jay Alammar — The Illustrated Transformer",
             "https://jalammar.github.io/illustrated-transformer/",
             "still the best single visual reference for the full model."),
        ],
    ),
    Chapter(
        doc_path="part-6-transformer/ch-30-decoder-only-gpt.md",
        part_num=6, part_name="Transformer",
        num="30", title="Decoder-Only GPT",
        src=["src/nn/gpt.ts"],
        tests=["src/nn/gpt.test.ts"],
        exercise="exercises/ch-30-decoder-only-gpt.ts",
        next_doc="",
        next_blurb="",
        goals=[
            "Strip the encoder and cross-attention; keep only masked self-attention + FFN blocks.",
            "Train on next-token prediction with a sliding-window language-modelling batch.",
            "Generate text autoregressively with greedy decoding, then with temperature sampling.",
            "Recognise the architecture family behind GPT-2, GPT-3, LLaMA, and Mistral.",
            "Identify natural extensions: KV cache, MoE, LoRA, quantization.",
        ],
        intuition=(
            "If you can already train an encoder-decoder transformer on string reversal, GPT is "
            "*simpler*. There is only one sequence, only one stack of blocks (decoder-only), and "
            "only one objective: predict the next token. The same self-attention you wrote in "
            "Ch 22, the same FFN from Ch 25, the same LayerNorm from Ch 20 — composed into a "
            "language model."
        ),
        mental_model=(
            "```text\n"
            "  token ids ──► Embed + PE ──► Dropout ──► DecoderOnlyBlock × N ──► LN ──► Linear(d→V)\n"
            "                                                                                │\n"
            "                                                                                ▼\n"
            "                                                                       next-token logits\n"
            "\n"
            "  DecoderOnlyBlock:  x = x + MaskedMHA(LN(x))\n"
            "                     x = x + FFN(LN(x))\n"
            "```\n"
        ),
        pitfalls=[
            "Accidentally importing the full decoder block (with cross-attention) instead of the decoder-only block.",
            "Forgetting the causal mask during training — the model trivially copies inputs.",
            "Generating from all logit positions instead of only the final one at inference.",
            "Letting the context length exceed `maxSeqLen` without cropping.",
            "Expecting meaningful language from a tiny dataset before it has memorised patterns.",
        ],
        verify=[
            "bun test src/nn/gpt.test.ts",
            "bun run exercises/ch-30-decoder-only-gpt.ts",
        ],
        further=[
            ("Radford et al. — Language Models are Unsupervised Multitask Learners (GPT-2)",
             "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
             "the GPT-2 paper; this is the architecture you just built."),
            ("Karpathy — nanoGPT",
             "https://github.com/karpathy/nanoGPT",
             "minimal reference implementation in PyTorch."),
            ("Karpathy — Let's build GPT (video)",
             "https://www.youtube.com/watch?v=kCc8FmEb1nY",
             "the live build you can compare your code against."),
            ("Anthropic — Transformer Circuits",
             "https://transformer-circuits.pub/",
             "for the next steps in interpretability."),
        ],
    ),
]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

CONCEPTS_RE = re.compile(r"^##\s+Concepts\s*$", re.MULTILINE)
NEXT_HEADING_RE = re.compile(
    r"^##\s+(?:Self-Check|Self Check|→\s*Next|Next Chapter|Common Pitfalls|"
    r"Required Tests|Course Milestone|Core Transformer Complete|Later Extensions|"
    r"Further Reading|How to Verify)\b",
    re.MULTILINE,
)
SELF_CHECK_RE = re.compile(r"^##\s+Self-Check Questions\s*$", re.MULTILINE)


def extract_concepts_and_implement(md: str) -> str:
    """Return the slice from `## Concepts` up to (but not including) the
    first of: Self-Check, → Next, Next Chapter, Required Tests, Common Pitfalls,
    Course Milestone, Core Transformer Complete, or Later Extensions. This
    captures the Concepts and What-to-Implement (and any TypeScript Hints / Required
    Tests intermediates we want to keep) verbatim from the existing doc."""
    m = CONCEPTS_RE.search(md)
    if not m:
        return ""
    start = m.start()
    rest = md[m.end():]
    n = NEXT_HEADING_RE.search(rest)
    end = m.end() + n.start() if n else len(md)
    return md[start:end].rstrip() + "\n"


def extract_self_check(md: str) -> str:
    """Return the Self-Check Questions block (heading + body) from the existing doc.
    Empty string if missing."""
    m = SELF_CHECK_RE.search(md)
    if not m:
        return ""
    start = m.start()
    rest = md[m.end():]
    n = NEXT_HEADING_RE.search(rest)
    end = m.end() + n.start() if n else len(md)
    return md[start:end].rstrip() + "\n"


def render(ch: Chapter, existing_md: str) -> str:
    # Header link block
    src_links = " · ".join(f"[`{p}`]({'../../' + p})" for p in ch.src)
    test_links = " · ".join(f"[`{p}`]({'../../' + p})" for p in ch.tests)
    exercise_link = f"[`{ch.exercise}`](../../{ch.exercise})"

    header = (
        f"# Chapter {ch.num}: {ch.title}\n\n"
        f"> **Part {ch.part_num} of 6 — {ch.part_name}**\n"
        f"> Source: {src_links}\n"
        f"> Tests: {test_links}\n"
        f"> Exercise: {exercise_link}\n\n"
        "---\n\n"
    )

    goals_md = "## Learning Goals\n\nBy the end of this chapter you can:\n\n" + \
        "\n".join(f"- {g}" for g in ch.goals) + "\n\n---\n\n"

    intuition_md = "## Intuition First\n\n" + ch.intuition.strip() + "\n\n---\n\n"

    mental_md = "## Mental Model\n\n" + ch.mental_model.strip() + "\n\n---\n\n"

    concepts_block = extract_concepts_and_implement(existing_md)
    if not concepts_block:
        concepts_block = "## Concepts\n\n_(to be filled in)_\n"
    # Ensure trailing separator
    concepts_block = concepts_block.rstrip() + "\n\n---\n\n"

    pitfalls_md = "## Common Pitfalls\n\n" + \
        "\n".join(f"- {p}" for p in ch.pitfalls) + "\n\n---\n\n"

    verify_lines = "\n".join(f"```bash\n{c}\n```" for c in ch.verify)
    verify_md = (
        "## How to Verify\n\n"
        "Run the tests and the exercise. Both should pass cleanly with no warnings:\n\n"
        f"{verify_lines}\n\n"
        "---\n\n"
    )

    self_check_block = extract_self_check(existing_md)
    if not self_check_block:
        self_check_block = (
            "## Self-Check Questions\n\n"
            "1. Restate the central definition of this chapter in one sentence.\n"
            "2. Where does this primitive appear in the full transformer (Ch 29) and GPT (Ch 30)?\n"
            "3. What is the most likely shape bug you could hit, and how would you detect it?\n"
            "4. Which previous chapter does this one depend on most heavily?\n"
        )
    self_check_block = self_check_block.rstrip() + "\n\n---\n\n"

    further_lines = []
    for title, url, note in ch.further:
        further_lines.append(f"- [{title}]({url}) — {note}")
    further_md = "## Further Reading\n\n" + "\n".join(further_lines) + "\n\n---\n\n"

    if ch.next_doc:
        # Compute relative link from this doc to next
        next_filename = ch.next_doc.split("/")[-1]
        same_part = ch.doc_path.split("/")[0] == ch.next_doc.split("/")[0]
        rel = next_filename if same_part else "../" + ch.next_doc
        next_md = (
            "## Next Chapter\n\n"
            f"**[{ch.next_blurb.split(' — ')[0]}]({rel})** — {ch.next_blurb.split(' — ', 1)[1] if ' — ' in ch.next_blurb else ch.next_blurb}\n"
        )
    else:
        next_md = (
            "## Next Chapter\n\n"
            "You have reached the end of the core course. Natural follow-ons: **KV cache** for fast inference, "
            "**Mixture of Experts** for sparse FFNs, **LoRA** for parameter-efficient fine-tuning, and "
            "**quantization** for low-precision inference.\n"
        )

    return (
        header + goals_md + intuition_md + mental_md +
        concepts_block + pitfalls_md + verify_md + self_check_block +
        further_md + next_md
    )


def main() -> None:
    for ch in CHAPTERS:
        path = DOCS / ch.doc_path
        if not path.exists():
            print(f"WARN: missing {path}")
            continue
        existing = path.read_text()
        new = render(ch, existing)
        path.write_text(new)
        print(f"rewrote {ch.doc_path}")


if __name__ == "__main__":
    main()
