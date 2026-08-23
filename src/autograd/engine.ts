/**
 * autograd/engine.ts
 * ══════════════════════════════════════════════════════════
 * Topological sort of the scalar computation graph.
 * Called by Value.backward() to determine gradient propagation order.
 *
 * Chapter: 08b — Autograd Backward
 * Doc:     docs/part-2-autodiff/ch-08b-autograd-backward.md
 */
import type { Value } from "./value.ts";
// Type-only import, so this does NOT create a runtime cycle with grad.ts —
// TypeScript erases it entirely. Ch 10 uses the tensor sort at the bottom.
import type { TensorValue } from "./grad.ts";

/**
 * Return all nodes reachable from root in topological order
 * (inputs before outputs — ready for reversed gradient accumulation).
 *
 * Algorithm: depth-first post-order traversal + visited set.
 *
 * ── WHAT "TOPOLOGICAL ORDER" BUYS US ──────────────────────────────────────
 * An ordering where every node appears AFTER all of its inputs. `backward()`
 * then walks that list in reverse, which visits every node only after all of
 * its CHILDREN have already pushed gradient into it. That is the correctness
 * condition for backprop: a node must hold its complete gradient before it
 * passes anything on. See 08b, "Why reverse topological order".
 *
 * ── THE RECIPE (the doc gives this one in full, under TypeScript Hints) ────
 * 1. A `visited` set of nodes, and an `order` array to build up.
 * 2. A recursive `dfs(node)`:
 *      - if already visited, return immediately;
 *      - mark visited FIRST, before recursing;
 *      - recurse into every node in `node._inputs`;
 *      - push `node` onto `order` LAST — after its inputs. This "push on the
 *        way back up" is what post-order means, and it is the whole trick.
 * 3. Run it from `root`, return `order`.
 *
 * ── DO NOT REVERSE HERE ───────────────────────────────────────────────────
 * The header comment above used to say "…and then reverse"; it doesn't, and
 * the doc's What-to-Implement table is explicit: "inputs-before-outputs (don't
 * reverse here)". Post-order DFS already produces inputs-before-outputs, and
 * `Value.backward()` does the reversing at the point of use. Reverse in both
 * places and you get forward order — every node fires before it is fully
 * accumulated, gradients come out wrong-but-plausible, and nothing throws.
 *
 * ── WHY THE VISITED SET IS LOAD-BEARING ───────────────────────────────────
 * It is not just a speed optimization. A node reused by several children (the
 * `x` in `x.mul(x)`) is reachable by more than one path; without the set it
 * would be appended to `order` more than once, its `_backward` would fire more
 * than once, and its parents would receive duplicated gradient. Mark visited
 * BEFORE recursing, not after.
 *
 * ── WHY A DAG, AND WHY THIS TERMINATES ────────────────────────────────────
 * Edges only ever point from inputs to the output they produced, and a node's
 * parents already existed when it was created — so the graph cannot contain a
 * cycle. A cycle would mean a value defined in terms of itself, which is not
 * something the forward pass can produce. Recursion depth tracks graph depth,
 * so deep chains could in principle blow the JS stack; scalar graphs in this
 * course are nowhere near that, and an explicit stack is the fix if it ever
 * matters.
 *
 * ── WORKED EXAMPLE — Figure 3's graph, traced call by call ────────────────
 * `L = (a·b) + d` with a = 2, b = -3, d = 10, so `c = a·b` and `L = c + d`.
 * Call `topoSort(L)` and follow the recursion:
 *
 *     dfs(L)   visit L,  L._inputs = [c, d]
 *       dfs(c)   visit c,  c._inputs = [a, b]
 *         dfs(a)   visit a,  no inputs  →  push a      order: [a]
 *         dfs(b)   visit b,  no inputs  →  push b      order: [a, b]
 *                                       →  push c      order: [a, b, c]
 *       dfs(d)   visit d,  no inputs    →  push d      order: [a, b, c, d]
 *                                       →  push L      order: [a, b, c, d, L]
 *
 *     returns  [a, b, c, d, L]        ← every node AFTER its own inputs
 *
 * `backward()` then walks that reversed — L, d, c, b, a — which is the only
 * order in which `c` is fully accumulated (by L) before its own closure fires.
 *
 * Now the reuse case, `z = x.mul(x)`, where `z._inputs = [x, x]`:
 *
 *     dfs(z)   visit z
 *       dfs(x)   visit x, push x                       order: [x]
 *       dfs(x)   ALREADY VISITED → return immediately  order: [x]
 *                                       →  push z      order: [x, z]
 *
 *     returns  [x, z]                 ← 2 nodes, not 3
 *
 * Without the visited set you would get `[x, x, z]`, `x._backward()` would
 * fire twice, and x's parents would receive double gradient. Note this is
 * separate from the `+=` inside the mul closure, which is what correctly gives
 * `x.grad = 6` here — one closure running once, accumulating twice.
 *
 * ✅ CHECKPOINT: `topoSort(L)` → 5 nodes, the three leaves before `c`, `c`
 *    before `L`. A lone leaf → just that node. `x.mul(x)` → 2 nodes.
 */
export function topoSort(root: Value): Value[] {
  const visited = new Set<Value>();
  const order: Value[] = [];

  function dfs(node: Value): void {
    if (visited.has(node)) return;
    visited.add(node);
    for (const input of node._inputs) {
      dfs(input);
    }
    order.push(node);
  }

  dfs(root);
  return order;
}

/* ══════════════════════════════════════════════════════════════════════════
 * ▲ ABOVE — Chapter 08b, the SCALAR version.  Leave it exactly as it is.
 * ▼ BELOW — Chapter 10, the TENSOR version.
 *
 * The two are kept side by side on purpose. Read the scalar one first; it is
 * the version you already understand. Then read the tensor one and look for
 * what changed.
 *
 * The answer, this time, is: nothing. Not "a little" — nothing at all.
 * ══════════════════════════════════════════════════════════════════════════ */

/**
 * Same topological sort, for the tensor graph (Ch 10).
 *
 * ── WHAT TO WRITE ─────────────────────────────────────────────────────────
 * Copy `topoSort` above and change `Value` to `TensorValue`. That is the whole
 * task, and the fact that it IS the whole task is the point of this chapter.
 *
 * ── WHY IT IS AN EXACT COPY ───────────────────────────────────────────────
 * Look at what the algorithm above actually touches: `node._inputs`, a visited
 * set, and an output array. It never reads `.data`. It never reads `.grad`. It
 * has no idea whether a node holds a single number or a 768×768 matrix, and it
 * would work just as well on nodes holding strings.
 *
 * Topological order is a property of the GRAPH, not of what is stored in it.
 * Chapter 10 changes the contents of the nodes and leaves the wiring alone —
 * so every piece of code that only looks at the wiring carries over untouched.
 * This function is the cleanest possible demonstration of that.
 *
 * ── WHY DUPLICATE IT RATHER THAN GENERALISE IT? ───────────────────────────
 * A production library would write the algorithm once, over any node type that
 * exposes `_inputs`:
 *
 *     function topoSort<T extends { _inputs: T[] }>(root: T): T[]
 *
 * That is genuinely better engineering, and one line. We are not doing it here
 * for one reason: the scalar version is the one you built while learning what
 * a computation graph is, and it is worth being able to re-read it later
 * without decoding a generic signature first. Clarity beats DRY in a codebase
 * whose purpose is to be read.
 *
 * Worth knowing the generic form exists, though — it is what you would reach
 * for outside a course.
 *
 * ✅ CHECKPOINT: identical in spirit to the scalar one. For grad.ts's
 *    GRAPH 1 — `C = A.mul(B); Z = C.add(d); L = Z.sum()` — calling
 *    `topoSortTensor(L)` returns 6 nodes ordered `[A, B, C, d, Z, L]`:
 *    every node after its own inputs, ready for `backward()` to walk it
 *    reversed, exactly as the scalar `topoSort` returned `[a, b, c, d, L]`
 *    for the same graph in Ch 08.
 */
export function topoSortTensor(root: TensorValue): TensorValue[] {
  const visited = new Set<TensorValue>();
  const order: TensorValue[] = [];

  function dfs(node: TensorValue): void {
    if (visited.has(node)) return;
    visited.add(node);
    for (const input of node._inputs) {
      dfs(input);
    }
    order.push(node);
  }

  dfs(root);
  return order;
}
