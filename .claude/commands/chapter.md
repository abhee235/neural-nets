Scaffold a new chapter stub for the neural-nets course.

Arguments: $ARGUMENTS
Expected: `<chapter-number> <folder-slug>` — e.g. `05 reductions` or `08a autograd-forward`

Steps:
1. Find and read the chapter doc at `docs/` matching the chapter number provided. If you can't find it, list what's in docs/ and ask.
2. Create **two files** following the protocol in `AGENTS.md`:

**`src/ch-$ARGUMENTS-slug/index.ts`**
- Full chapter header block comment:
  ```
  /**
   * CHAPTER N: Title
   * ════════════════════════════════════════
   * Part X of 6: Part Name
   *
   * WHAT WE'RE BUILDING:  [what this exports]
   * WHY IT MATTERS:       [which later chapter calls this and why]
   * WHAT THIS UNLOCKS:    → Ch N+1 (Next Title)
   *
   * REFERENCE: docs/part-X-.../ch-NN-....md
   */
  ```
- Import section (commented out, pointing to previous chapter's index.ts)
- All exported functions/classes from the chapter doc, each with:
  - JSDoc block: what it does + where it appears in the transformer
  - Inline math comment for any formula
  - Body: `throw new Error("Not implemented — read the chapter doc first")`
- Commented-out demo block at the bottom

**`src/ch-$ARGUMENTS-slug/index.test.ts`**
- Import from `bun:test` only
- `const EPSILON = 1e-6; const close = (a: number, b: number) => Math.abs(a - b) < EPSILON;`
- One `describe` block per exported function
- `it.todo(...)` tests named after mathematical properties (not "returns a number")
- For Ch 07+: include `it.todo("numerical gradient check passes for ...")`

3. Print the created file paths and list every function stub with its signature.

Always follow the naming conventions from `.github/instructions/chapter-code.instructions.md`.
