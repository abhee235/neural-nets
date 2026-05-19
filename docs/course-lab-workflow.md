# Course Lab Workflow: One Chapter, One Branch

This course is meant to be used like a real engineering project, not like a folder of disconnected notes. Each chapter should become a focused Git branch where the student implements one small slice, runs the tests, commits the result, and then moves forward.

The goal is not only to understand transformers. The goal is to practice the workflow of an AI engineer: isolate a task, implement it, verify it, review the diff, and preserve a clean history.

---

## Branch Model

Use one branch per chapter:

```bash
main
├── ch-01-tensor-type-system
├── ch-02-tensor-creation
├── ch-03-elementwise-broadcasting
├── ...
├── ch-29-full-transformer
└── ch-30-decoder-only-gpt
```

Each branch starts from the previous completed chapter branch. That keeps the course cumulative: Chapter 12 can rely on working code from Chapters 01–11, and Chapter 26 can rely on every primitive built before it.

---

## Student Workflow

For each chapter:

1. Read the chapter document in `docs/`.
2. Create a branch for that chapter from the previous completed branch.
3. Implement only the functions/classes listed in that chapter.
4. Run the chapter tests.
5. Run all tests when the chapter passes locally.
6. Commit the finished chapter.
7. Write a short note about what broke, what fixed it, and what still feels unclear.

Example:

```bash
git checkout ch-03-elementwise-broadcasting
git checkout -b ch-04-matrix-ops

bun test src/ch-04-matrix-ops
bun test

git add src/ch-04-matrix-ops src/ch-01-tensor-type-system src/ch-03-elementwise-ops-broadcasting
git commit -m "Implement chapter 04 matrix operations"
```

If a later chapter reveals a bug in an earlier chapter, fix the root cause in the earliest chapter where the bug belongs. This mirrors real ML engineering: many transformer bugs are actually tensor-shape bugs from earlier infrastructure.

---

## Forking Model

Students should fork the repository and keep their own implementation branches. The instructor repository can keep three branch families:

| Branch Family | Purpose |
|---------------|---------|
| `main` | Docs, scaffolding, and course instructions. No completed solutions. |
| `starter/ch-NN-*` | Function stubs and pending tests for a chapter. |
| `solution/ch-NN-*` | Reference solution after the chapter is complete. |

Students should work from `starter/ch-NN-*`, not from `solution/ch-NN-*`.

---

## Commit Standard

Each completed chapter commit should include:

- Implementation code for that chapter.
- Tests for all public functions introduced in that chapter.
- At least one shape/invariant test.
- At least one numerical test when gradients are involved.
- No unrelated refactors.

Good commit messages:

```text
Implement chapter 05 reductions
Add tensor autograd broadcast backward
Implement causal and padding attention masks
```

Bad commit messages:

```text
stuff
fix
many changes
```

---

## Verification Gates

Do not move to the next chapter until the current gate passes.

| Gate | Required Check |
|------|----------------|
| Tensor chapters | Shape invariants, flat-index round trip, broadcasting compatibility. |
| Autograd chapters | Numerical gradient checks with relative error below tolerance. |
| Neural net chapters | One-batch overfit test and decreasing loss curve. |
| Tokenizer/input chapters | Encode/decode round trip and mask shape checks. |
| Attention chapters | Attention weights sum to 1 and masked positions receive zero probability. |
| Transformer chapters | Overfit a tiny sequence task before attempting generalization. |

These gates are non-negotiable. If a gate fails, the correct lesson is to debug, not to continue.

---

## What Students Should Submit

For each chapter branch, submit:

- The Git branch name.
- The final commit hash.
- Test output from the chapter test.
- A short explanation of one bug encountered and how it was fixed.
- One numerical experiment, such as changing the learning rate, dropout rate, hidden size, or sequence length.

This makes the course more than code completion. It trains the habit of validating claims with experiments, which is central to AI engineering.
