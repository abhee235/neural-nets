Check whether a chapter meets its verification gate before moving to the next chapter.

Arguments: $ARGUMENTS
Expected: `<chapter-number>` — e.g. `04` or `10`

Steps:

1. Run the chapter tests: `bun test src/ch-$ARGUMENTS-*`
2. Run all tests to check no regressions: `bun test`
3. Based on the chapter's part, apply the verification gate:

| Part | Gate requirements |
|---|---|
| 1 (Ch 01–06, tensors) | All shape-invariant tests pass. flatIndex round-trips correctly. Broadcasting rejects incompatible shapes with a clear error. |
| 2 (Ch 07–10, autograd) | Numerical gradient check passes with relative error < 1e-5 for every differentiable op. zeroGrad resets correctly. |
| 3 (Ch 11–15, nn) | One-batch overfit: loss decreases monotonically for 50+ steps on a tiny dataset. |
| 4 (Ch 16–21, tokenizer) | Encode → decode round-trips the original string. Mask shapes match `[1, 1, seqLen, seqLen]`. |
| 5 (Ch 22–24, attention) | Attention weights sum to 1 along the key axis. Masked positions receive exactly zero weight after softmax. |
| 6 (Ch 25–30, transformer) | Model overfits a toy sequence (e.g., string reversal) within 500 steps — training loss reaches < 0.1. |

4. Report:
   - PASS / FAIL for the gate
   - Any `it.todo` tests that are still pending (list them — these are gaps)
   - One concrete experiment to try before moving on (e.g., "change learning rate from 0.01 to 0.1 and watch the loss")
   - Whether it's safe to start the next chapter

Do not declare the gate passed if any non-todo test is failing.
