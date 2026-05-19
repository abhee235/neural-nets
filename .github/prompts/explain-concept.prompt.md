---
description: "Get a deep tutor-style explanation of any concept from the course: intuition first, then math, then connection to the transformer, worked example, and common pitfalls."
---

Explain the concept: **${input:concept}**

Act as a patient tutor for a student who knows basic calculus and linear algebra.
Structure your explanation in exactly this order:

---

### 1. Intuition First
Plain English. No equations yet. Use an analogy if it helps.
What is this thing? What problem does it solve?

### 2. The Math
Write the equation(s). Then explain every symbol:
- What does it represent physically or geometrically?
- What are the units / dimensions?
- What happens when you change it?

### 3. Why Neural Networks Need This
Where does this concept appear in the transformer (Ch 25–26)?
What would break if we removed or ignored this?

### 4. Worked Example
Use the smallest numbers that make the math visible.
Walk through each step. Show intermediate values.
(Prefer 2×2 or 3×3 examples.)

### 5. Common Pitfalls
What mistakes do people make with this?
Any numerical instability issues? Shape mismatch traps? Off-by-one errors?

### 6. Connection to Our Code
Which chapter and which function(s) in `src/` implement this?
What should the student look for when they open that file?

---

Context: this project is building a transformer from scratch in TypeScript (Bun).
Reference the course structure in `.github/copilot-instructions.md` if needed.
