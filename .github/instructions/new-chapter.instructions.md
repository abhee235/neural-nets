---
description: "Use when the user asks to scaffold, create, or add a new chapter. Covers the folder naming convention, index.ts stub format with chapter header, and index.test.ts stub with pending tests."
---

# New Chapter Scaffolding Guide

When asked to scaffold a new chapter, create two files following the structure below.

## Folder Name Convention

```
src/ch-{NN}-{kebab-case-title}/
```

Examples:
- `src/ch-01-tensor-type-system/`
- `src/ch-08a-autograd-forward/`
- `src/ch-22-self-attention/`

## `index.ts` Stub Template

```typescript
/**
 * CHAPTER {N}: {Title}
 * ════════════════════════════════════════
 * Part {X} of 6: {Part Name}
 *
 * WHAT WE'RE BUILDING:
 *   [Describe what functions/types this chapter exports]
 *
 * WHY IT MATTERS:
 *   [Connect to the transformer — where will this code be used?]
 *
 * WHAT THIS UNLOCKS:
 *   → Ch {N+1} ({Next Chapter Title})
 *
 * REFERENCE: docs/part-{X}-{partname}/ch-{NN}-{chaptername}.md
 */

// ─── Imports from previous chapters ────────────────────────────────────────
// import { Tensor } from "../ch-01-tensor-type-system/index.ts";

// ─── Types ──────────────────────────────────────────────────────────────────

// ─── Functions ──────────────────────────────────────────────────────────────

/**
 * [Function description: what it does, why it exists in ML]
 *
 * Math: [formula]
 */
export function exampleFunction(): void {
  throw new Error("Not implemented — read docs/part-.../ch-{NN}-....md first");
}

// ─── Demo (run with: bun run src/ch-{NN}-{name}/index.ts) ───────────────────

// Uncomment to run a quick demo:
// console.log("Chapter {N} demo:");
```

## `index.test.ts` Stub Template

```typescript
/**
 * Tests for Chapter {N}: {Title}
 *
 * Each test verifies a specific mathematical property.
 * Read the test names — they teach you what the functions should do.
 */
import { describe, it, expect } from "bun:test";
// import { exampleFunction } from "./index.ts";

describe("exampleFunction", () => {
  it.todo("correct output shape for ...");
  it.todo("mathematical property: ...");
  it.todo("edge case: ...");
  it.todo("numerical gradient check passes");
});
```

## Sub-chapter Convention

If a chapter is large (e.g. Autograd), split into `ch-08a-autograd-forward/` and
`ch-08b-autograd-backward/`. Both follow the same template. The `a` chapter should not
import from the `b` chapter — the dependency only flows forward.
