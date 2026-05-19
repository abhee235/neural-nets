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

/**
 * Return all nodes reachable from root in topological order
 * (inputs before outputs — ready for reversed gradient accumulation).
 *
 * Algorithm: depth-first post-order traversal + visited set, then reverse.
 */
export function topoSort(root: Value): Value[] {
  throw new Error("topoSort not implemented");
}
