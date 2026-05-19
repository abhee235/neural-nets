---
description: "Scaffold a new chapter: creates the src folder, index.ts stub with chapter header, and index.test.ts with pending tests. Use when starting a new chapter."
---

Create a new chapter scaffold for this project.

Chapter number: ${input:chapterNumber}
Chapter title: ${input:chapterTitle}
Part number: ${input:partNumber}
Part name: ${input:partName}
Folder slug (kebab-case): ${input:folderSlug}
Doc path: docs/part-${input:partNumber}-${input:partSlug}/ch-${input:chapterNumber}-${input:folderSlug}.md

Follow the scaffolding rules in `.github/instructions/new-chapter.instructions.md` exactly.

Steps:
1. Create `src/ch-${input:chapterNumber}-${input:folderSlug}/index.ts` with:
   - The full chapter header block comment
   - Import section (commented out, pointing to the previous chapter)
   - All function stubs with JSDoc comments describing the math and ML purpose
   - Each function body: `throw new Error("Not implemented — read the chapter doc first")`
   - A commented-out demo block at the bottom

2. Create `src/ch-${input:chapterNumber}-${input:folderSlug}/index.test.ts` with:
   - Import from `bun:test`
   - One `describe` block per function
   - `it.todo(...)` tests with names that describe the mathematical property being verified

After creating the files, print the folder path and list the function stubs created.
