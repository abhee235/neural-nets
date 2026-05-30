Implement a chapter for the neural-nets course, replacing all stubs with working code.

Arguments: $ARGUMENTS
Expected: `<chapter-number>` — e.g. `04` or `08b`

You are acting as a patient tutor. Before writing any code:

**Step 1 — Orient**
Read the chapter doc at `docs/` matching the chapter number. Then read the existing stub file at `src/ch-$ARGUMENTS-*/index.ts`. List all functions that still throw "Not implemented".

**Step 2 — Explain first**
For each function you are about to implement, briefly state:
- What it computes (plain English, one sentence)
- The key formula or algorithm
- Which later chapter will call it and why

**Step 3 — Implement**
Replace each `throw new Error(...)` with a correct implementation. Follow all conventions:
- Comments ARE documentation — keep/expand the existing JSDoc blocks
- Named intermediate variables (never chain complex expressions)
- Explicit loops, not `.reduce()` hiding an algorithm
- No `as any`, no `as unknown`
- `noUncheckedIndexedAccess` is on — narrow array accesses before use

**Step 4 — Verify**
Run `bun test src/ch-$ARGUMENTS-*` and report results. If tests fail, diagnose and fix.

**Step 5 — Gate check**
Based on the chapter part, confirm the verification gate passes:
- Part 1 (tensor): shape invariants, flat-index round trip, broadcasting
- Part 2 (autograd): numerical gradient check relative error < 1e-5
- Part 3 (nn): one-batch overfit, loss decreases
- Part 4 (tokenizer): encode/decode round trip, mask shapes
- Part 5 (attention): weights sum to 1, masked positions get zero probability
- Part 6 (transformer): overfit a tiny sequence

Point out any remaining pitfalls specific to this chapter.
