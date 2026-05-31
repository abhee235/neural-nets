Run tests for a chapter and give a structured report.

Arguments: $ARGUMENTS
Expected: `<chapter-number>` — e.g. `04` or `08a`. Omit to run all tests.

If a chapter number is provided:
```
bun test src/ch-$ARGUMENTS-*
```
Otherwise:
```
bun test
```

Then report:
- Total tests: passed / failed / todo / skipped
- Each failing test: test name + error message + likely cause
- Each `it.todo` test: the mathematical property it's supposed to verify (these are future work)
- Whether the **verification gate** for this chapter part passes:
  - Part 1: shape invariants, flat-index round trip
  - Part 2: relative error < 1e-5 on numerical gradient check
  - Part 3: one-batch overfit with decreasing loss
  - Part 4: encode/decode round trip, mask shape checks
  - Part 5: attention weights sum to 1, masked positions zero
  - Part 6: overfit a tiny sequence

If there are failures, suggest the most likely root cause for each (shape mismatch? gradient accumulation? broadcasting?) and ask a leading question to help debug rather than just handing over the fix.
