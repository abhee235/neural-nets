---
description: "Use when writing, editing, or reviewing chapter documentation in docs/**. Defines the standard chapter doc template every chapter must follow: learning goals, intuition before math, mental model, concepts, what to implement, common pitfalls, verification, self-check, further reading."
applyTo: "docs/part-*/ch-*.md"
---

# Chapter Documentation Standard

Every chapter doc in `docs/part-N-name/ch-NN-name.md` is a **tutor-written textbook page**.
It must be readable from top to bottom in one sitting, build intuition before reaching for
math, and end with a clear pointer to the next chapter.

This file is the source of truth. When a user asks to add, rewrite, or review a chapter doc,
follow this structure exactly.

---

## Required Sections (in this order)

Every chapter doc must contain these 11 sections, in this exact order:

```
1.  Title + Header block (Part X of 6, cross-links)
2.  Learning Goals
3.  Intuition First
4.  Mental Model
5.  Concepts
6.  What to Implement
7.  Common Pitfalls
8.  How to Verify
9.  Self-Check Questions
10. Further Reading
11. Next Chapter
```

Do **not** rename sections. Do **not** reorder them. Do **not** omit any section. If a
section genuinely does not apply (rare), explain why in one sentence and leave the heading
in place.

---

## Section-by-Section Specification

### 1. Title + Header

The H1 is the chapter title. Immediately below it, a blockquote with:

- Part identifier ("Part X of 6 — Part Name")
- Markdown link to the code file (`src/...`)
- Markdown link to the test file (`src/.../*.test.ts`)
- Markdown link to the exercise file (`exercises/ch-NN-name.ts`)

Example:

```markdown
# Chapter 01: Tensor Type System

> **Part 1 of 6 — Tensor Library (NumPy-like Foundation)**
> Code: [`src/tensor/types.ts`](../../src/tensor/types.ts)
> Tests: [`src/tensor/types.test.ts`](../../src/tensor/types.test.ts)
> Exercise: [`exercises/ch-01-tensor-type-system.ts`](../../exercises/ch-01-tensor-type-system.ts)
```

### 2. Learning Goals

A `## Learning Goals` heading followed by 4-6 bullets, each starting with an action verb
("explain", "implement", "convert", "describe", "verify"). Each goal must be testable —
something a reader could either do or not do after finishing the chapter.

### 3. Intuition First

A `## Intuition First — <hook question>` heading.

This is the most important section. It explains what the chapter is about **using analogies,
not math**. Use:

- A vivid analogy (spreadsheet, library shelf, conveyor belt, etc.)
- Plain language a non-technical reader could follow
- A "Why we care for transformers" mini-table when applicable

This section earns the reader's attention before any formula is introduced.

### 4. Mental Model

A `## The Mental Model` heading.

A diagram (ASCII art is fine) or step-by-step picture that the reader can hold in their
head. Show how the abstract idea maps to a concrete data layout, shape, or graph.

### 5. Concepts

A `## Concepts` heading with sub-headings (`###`) for each concept.

This is where math lives. Rules:

- Define every symbol before using it.
- Use KaTeX math blocks (`$$ ... $$`) for important formulas.
- After every formula, add one plain-English sentence explaining what it computes.
- Prefer bullet points over walls of prose.
- Add a `> **Why this matters for transformers**` callout when a concept connects
  to a later transformer chapter.

### 6. What to Implement

A `## What to Implement` heading.

A markdown table with columns: Symbol, Description.
Each row is one function, class, or constant the student must implement.
Below the table, list validation rules and edge cases.

### 7. Common Pitfalls

A `## Common Pitfalls` heading with 3-6 bullets.

Each bullet describes a specific bug a student is likely to hit. Bullets must be:

- Concrete (not "be careful with shapes" — instead "stride is the product of *later*
  dimensions, not earlier ones")
- Actionable (the student knows what to check)
- Drawn from real experience implementing this material

### 8. How to Verify

A `## How to Verify` heading.

Two exact shell commands:

```bash
bun test <path to test file>
bun run <path to exercise file>
```

Plus one sentence stating what success looks like ("all tests green and the exercise prints
the expected outputs").

### 9. Self-Check Questions

A `## Self-Check Questions` heading with 4-6 numbered questions.

- At least half should have answers shown in italics: `*(Answer: 6)*`.
- Questions should test conceptual understanding, not just code recall.
- Connect at least one question to a future chapter ("In Ch 23, ...").

### 10. Further Reading

A `## Further Reading` heading with 4-6 external links, each:

- Bolded link title
- One-sentence annotation explaining *why* the student should read it
- The URL on a separate line in angle brackets: `<https://...>`

Preferred sources:

- Official docs (NumPy, PyTorch, Hugging Face)
- 3Blue1Brown videos
- Andrej Karpathy's lectures and blog
- Distill.pub
- "Deep Learning" by Goodfellow et al. (free online)
- Original papers (Attention Is All You Need, GPT, etc.) when relevant
- MDN for JS/TS API references

Avoid: dead links, paywalled content, low-quality Medium articles, content older than 2018
unless it is a primary source.

### 11. Next Chapter

A `## Next Chapter` heading with one paragraph that:

- Links to the next chapter doc
- Names the next chapter's main concept
- States how it builds on this one

---

## Math Formatting Conventions

Treat block-level math the way a textbook treats numbered equations: **key results get a
visible frame, intermediate steps do not.** This gives the reader a scannable visual
landmark when scrolling and a clear hierarchy between "the rule" and "the work".

Rules:

- **Key results** (definitions, named rules, the formula a section is *about*) must use
  KaTeX's `\boxed{...}` macro and the `\Large` size step:

  ```markdown
  $$\Large \boxed{\text{offset} = \text{row} \times \text{cols} + \text{col}}$$
  ```

- **Worked examples** that arrive at a single concrete answer may box just the answer:

  ```markdown
  $$\Large 1 \times 3 + 2 = \boxed{5}$$
  ```

- **Intermediate derivation steps** (the algebra leading to a result) stay un-boxed and
  may be inline (`$...$`) or plain block (`$$...$$`) without `\Large`. Boxing every line
  destroys the hierarchy.

- Use `\Large` (one step above default block size), not `\LARGE` or `\huge` — those
  overflow the page width on longer sums and products.

- `\boxed{}` and `\Large` are standard KaTeX macros and render correctly in GitHub,
  VS Code Markdown preview, and any KaTeX-aware viewer. Do not substitute non-KaTeX
  alternatives like `\fbox{}` or raw HTML `<div style="border:...">`.

- For matching visual treatment in **chapter media** (SVG diagrams under
  `docs/assets/ch-NN/`), use the shared "answer box" pattern: AMBER_L (`#fef3c7`) fill,
  AMBER (`#b45309`) stroke at 1.4–1.5 px, rounded corners `rx=8–10`, monospace formula
  font 17–22 pt. See `.github/instructions/chapter-media.instructions.md`.

Aim for **one to three boxed equations per chapter**. If everything is boxed, nothing is.

**Build formulas by progression, not by reveal.** When a formula generalises one
the reader already knows (e.g. 2-D offset → 3-D offset → N-D offset), stack the two
versions in an `aligned` block so the eye can spot the *single change*. Highlight
the new term with `\textcolor{#b45309}{...}` (the AMBER accent) and use `\phantom{...}`
on the lower-rank line so columns line up:

```markdown
$$
\Large
\begin{aligned}
\text{2D} \;\; [r, c] \;\;\,&:\;\; \text{offset} = \phantom{d \times (R \cdot C) + {}} r \times C + c \\
\text{3D} \;\; [d, r, c] &:\;\; \text{offset} = \textcolor{#b45309}{d \times (R \cdot C)} + r \times C + c
\end{aligned}
$$
```

This is the textbook "learning by comparison" move: the reader memorises the
general rule by seeing exactly what the next rank *adds* to the previous one,
never by reading a new formula in isolation.

**Climb to the general case, then translate to code.** When the chapter has an
implementation target (almost always true in this course), extend the staircase
all the way up — typically four rows: rank 2, rank 3, rank 4, and a final N-D row
written with `\sum` — so the pattern becomes visually inevitable. Immediately
after the staircase, show the **loop that implements the sum**, with a small
worked-trace table that walks one concrete index through the loop iteration by
iteration. This closes the math → code gap in one page: the reader sees the
formula, sees the staircase, sees the loop, and sees the loop's variables take
the values that produced the boxed answer from the worked example.

**Always pair a pattern with a pen-and-paper callout.** Reading a formula or a
staircase produces *recognition*, not *recall*. The student will not be able to
implement the algorithm from a blank editor unless their hand has produced the
pattern at least once. Whenever a chapter introduces a non-trivial formula,
staircase, or recurrence the student must later implement, add a blockquote
callout (use the 📝 emoji as the visual anchor) telling them to:

1. Copy the formula or staircase by hand, no peeking.
2. Invent their own small concrete example (different shape, different indices,
   different dimensions) and work it through twice — once symbolically by
   plugging into the formula, once procedurally by running the loop/algorithm in
   their head and filling in a trace table.
3. Only then open the exercise file and type the implementation from memory.

Place the callout at the *pattern climax* (right after the staircase + loop, or
right after the boxed key result), not at the end of the chapter where it gets
skipped. Keep the language direct ("Stop and write this down", "Do not skip the
pen-and-paper step") — gentle suggestions get ignored.

---

## Writing Voice

- Address the reader as "you".
- Speak as a patient tutor, not as a textbook author.
- Prefer short sentences. Cut filler words.
- Use bold for the first introduction of every technical term.
- Use backticks for: file paths, type names, function names, variable names, shapes.
- Use math blocks for formulas, not inline code.

---

## Forbidden Patterns

- Emojis in section headers (the H2 line). Inline emojis in body text are fine sparingly.
- Walls of prose longer than 5 lines without a break (list, code block, or sub-heading).
- "Obviously", "simply", "just" — they alienate beginners.
- Linking to `src/ch-NN-name/` (legacy path). Always use the new library paths:
  `src/tensor/...`, `src/nn/...`, `src/autograd/...`, etc.
- Inconsistencies with the actual code stubs (e.g. claiming `data: number[]` when the stub
  uses `Float64Array`).
- Code blocks without language tag (`typescript`, `bash`, `python`, or `text`).

---

## When User Asks to Add a New Chapter

1. Confirm the chapter number, part, and title.
2. Confirm which `src/` module the implementation lives in.
3. Generate the doc from this template, filled in with chapter-specific content.
4. Cross-link from the previous chapter's "Next Chapter" section.
5. Verify all internal markdown links resolve.

---

## When User Asks to Review a Chapter

Check, in this order:

1. All 11 sections present and in order.
2. Cross-links at the top resolve to existing files.
3. Path references match the actual library layout (no `src/ch-NN-name/`).
4. Type annotations in code blocks match the actual stub file.
5. At least one analogy in "Intuition First".
6. At least 3 specific pitfalls in "Common Pitfalls".
7. At least 4 external links in "Further Reading", each with an annotation.
8. Next-chapter link exists and resolves.

If any check fails, fix it without asking — these are non-negotiable standards.
