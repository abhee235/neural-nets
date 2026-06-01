# Deep Dive: The Rest of the Reduction Family

> Optional reading for Chapter 05. No new code beyond what the chapter already asks for.
> The main chapter explains the reductions you'll lean on most — `sum`, `mean`,
> `variance`/`std`, `argmax`, and `softmax`. This doc rounds out the family with the
> remaining members (`max`, `min`, `argmin`) so every function you implement has its
> "what and why" written down somewhere, without bloating the main chapter.

---

## The family at a glance

Every reduction folds an axis away. They differ only in *what* they compute as they fold:

| Function | Folds to… | Returns | Main chapter? |
|----------|-----------|---------|---------------|
| `sum` | the total | value | ✅ (running example) |
| `mean` | the average (center) | value | ✅ |
| `variance` / `std` | the spread | value | ✅ |
| `max` | the largest **value** | value | ➡️ here |
| `min` | the smallest **value** | value | ➡️ here |
| `argmax` | the **position** of the largest | index | ✅ |
| `argmin` | the **position** of the smallest | index | ➡️ here |
| `softmax` | scores → probabilities | distribution | ✅ |

Two natural pairs run through this table:
- **value vs. position:** `max`/`min` tell you *what* the extreme value is; `argmax`/`argmin` tell you *where* it is.
- **largest vs. smallest:** `max`/`argmax` look for the biggest; `min`/`argmin` are their mirror images.

---

## `max` and `min` — the extremes ("order statistics")

`max` returns the single largest value along an axis; `min` the smallest. Statisticians call these *order statistics* because they're the values you'd find at the ends after sorting — but you never actually sort; you just sweep once and keep the running best.

```
max along axis=1:           min along axis=1:
[[1, 5, 3],   → [5,          [[1, 5, 3],   → [1,
 [4, 2, 6]]      6]           [4, 2, 6]]      2]
```

**Why a network needs `max`.** Two distinct jobs:
1. **Confidence read-out.** The largest score in a classifier's output is *how strongly* it favours its top choice. (The *position* of that score — the actual label — is `argmax`'s job; see below.)
2. **Numerical safety.** `max` is the stabiliser inside `softmax`: subtract it before `exp` so nothing overflows. This is the single most important use of `max` in the whole course, and it's why the chapter implements `max` *before* `softmax`. (Full reasoning: [ch-05-why-subtract-the-max.md](ch-05-why-subtract-the-max.md).)

**Why a network needs `min`.** Less common than `max`, but it shows up wherever you care about the *worst* or *closest* value — e.g. clamping/clipping ranges, finding the nearest item by distance, or monitoring the smallest activation to catch values collapsing toward zero. It's mostly here for completeness and as the mirror that makes `max` easier to reason about.

**Implementation note — don't use `Math.max(...array)`.** Spreading a typed array into `Math.max` passes every element as a separate function argument, and JS engines cap the argument count (you'll hit `RangeError: too many arguments` on a real-sized tensor). Sweep with an explicit loop tracking the running best instead — exactly the loop `argmax` already uses.

---

## `argmin` — the mirror of `argmax`

`argmax` (covered in the main chapter) returns the *index* of the largest value; `argmin` returns the index of the smallest. Identical loop, flipped comparison (`<` instead of `>`).

```
argmin along axis=1:
[[1, 5, 3],   row 0: smallest is 1 at index 0  → 0
 [4, 2, 6]]   row 1: smallest is 2 at index 1  → 1
              result [0, 1]
```

**Why a network needs it.** Anywhere the *best* option is the *lowest-scoring* one. The clearest example: picking the closest match by **distance**. If you have distances `[2.1, 0.4, 1.7]` from a query to three candidates, `argmin` returns `1` — the nearest neighbour. (Embedding-similarity search and clustering both reduce to "argmin over distances.") Whenever lower = better, reach for `argmin`; whenever higher = better, reach for `argmax`.

**A subtlety worth knowing — ties.** If two positions hold the same extreme value, which index wins? The convention (and NumPy's) is **first occurrence wins**, which falls out naturally from a strict comparison (`>` for argmax, `<` for argmin): a later equal value doesn't beat the one already stored. Just be consistent so results are predictable.

---

## The one mental model behind all of them

Notice that the *only* thing that changes across the whole family is the per-fold rule:

- `sum` → keep a running **total**
- `max`/`min` → keep a running **best value**
- `argmax`/`argmin` → keep a running **best value _and its index_**
- `mean` → `sum` then divide
- `variance` → `mean`, then `mean` of squared gaps

The axis-walking machinery — figure out the output shape, iterate the output positions, sweep the reduced axis — is *identical* for all of them. Write it once for `sum`, and every other reduction is "the same loop with a different accumulator." That's why the chapter has you implement them together: they are one algorithm wearing different hats.
