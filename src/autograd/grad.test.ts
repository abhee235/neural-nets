/**
 * Tests for autograd/grad.ts
 * Chapter 10 — Tensor Autograd Bridge
 *
 * Run: bun test src/autograd/grad.test.ts
 *
 * The fixtures are the chapter's own running examples, so every expected
 * number below is one derived by hand in the doc:
 *
 *   GRAPH 1 (doc sections 2-6):  L = sum((A×B) + d)
 *       A all 2 [2,3], B all -3 [2,3], d = [10,10,10] [1,3]
 *       → C all -6, Z all 4, L = 24
 *       → A.grad all -3, B.grad all 2, d.grad = [2,2,2]
 *
 *   GRAPH 2 (doc section 8):     L = sum(A @ B)
 *       A = 1..6 [2,3], B = 1..12 [3,4]  →  L = 610
 *       → A.grad rows [10,26,42], B.grad rows of 5s/7s/9s
 */
import { describe, it, expect } from "bun:test";
import { TensorValue, sumToShape, checkTensorGradient } from "./grad.ts";
import { topoSortTensor } from "./engine.ts";
import { createTensor } from "../tensor/types.ts";
import { ones } from "../tensor/creation.ts";
import { transpose } from "../tensor/linalg.ts";

const EPSILON = 1e-6;
const close = (a: number, b: number) => Math.abs(a - b) < EPSILON;

/** Fresh GRAPH 1 leaves — a fresh graph per backward pass, as Ch 08 taught. */
function graph1() {
  return {
    A: new TensorValue(createTensor([2, 2, 2, 2, 2, 2], [2, 3])),
    B: new TensorValue(createTensor([-3, -3, -3, -3, -3, -3], [2, 3])),
    d: new TensorValue(createTensor([10, 10, 10], [1, 3])),
  };
}

/** Fresh GRAPH 2 leaves. */
function graph2() {
  return {
    A: new TensorValue(createTensor([1, 2, 3, 4, 5, 6], [2, 3])),
    B: new TensorValue(
      createTensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [3, 4]),
    ),
  };
}

describe("sumToShape", () => {
  it("sums over broadcast axes to recover the original shape", () => {
    // The exercise's four-row bias: each of d's 3 entries was copied into
    // 4 rows, so each collects 4 gradients of 1 — the doc's [4,4,4].
    const tall = sumToShape(ones([4, 3]), [1, 3]);
    expect(tall.shape).toEqual([1, 3]);
    expect(Array.from(tall.data)).toEqual([4, 4, 4]);

    // GRAPH 1's own d: two rows → [2,2,2] (doc section 4, entry by entry).
    const g1 = sumToShape(ones([2, 3]), [1, 3]);
    expect(g1.shape).toEqual([1, 3]);
    expect(Array.from(g1.data)).toEqual([2, 2, 2]);

    // Case 2, the keepDims trap (doc section 6): [3,4] → [3,1] must keep the
    // size-1 axis — same numbers as [3] but only [3,1] preserves the shape.
    const stretched = sumToShape(
      createTensor([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], [3, 4]),
      [3, 1],
    );
    expect(stretched.shape).toEqual([3, 1]);
    expect(Array.from(stretched.data)).toEqual([4, 8, 12]);
  });

  it("is a no-op when grad already has the target shape", () => {
    // Every caller relies on this being safe to call unconditionally.
    const grad = createTensor([1, 2, 3, 4, 5, 6], [2, 3]);
    const result = sumToShape(grad, [2, 3]);
    expect(result.shape).toEqual([2, 3]);
    expect(Array.from(result.data)).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it("[3,4] summed to [1,4] reduces axis 0", () => {
    // Distinct values so the reduced AXIS is unambiguous: the result must be
    // the column sums (1+5+9, 2+6+10, …), not the row sums.
    const grad = createTensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [3, 4]);
    const result = sumToShape(grad, [1, 4]);
    expect(result.shape).toEqual([1, 4]);
    expect(Array.from(result.data)).toEqual([15, 18, 21, 24]);
  });
});

describe("TensorValue — forward", () => {
  it("add output shape matches the broadcast shape", () => {
    // GRAPH 1's Z = C + d: [2,3] + [1,3] broadcasts to [2,3], all 4s.
    const C = new TensorValue(createTensor([-6, -6, -6, -6, -6, -6], [2, 3]));
    const { d } = graph1();
    const Z = C.add(d);
    expect(Z.data.shape).toEqual([2, 3]);
    expect(Array.from(Z.data.data)).toEqual([4, 4, 4, 4, 4, 4]);
    // The graph records both parents, in operand order — backward needs it.
    expect(Z._inputs[0]).toBe(C);
    expect(Z._inputs[1]).toBe(d);
  });

  it("matMul output shape is (M,N) for (M,K) × (K,N)", () => {
    // GRAPH 2: [2,3] @ [3,4] → [2,4], with section 8's exact values.
    const { A, B } = graph2();
    const Z = A.matMul(B);
    expect(Z.data.shape).toEqual([2, 4]);
    expect(Array.from(Z.data.data)).toEqual([38, 44, 50, 56, 83, 98, 113, 128]);
  });

  it("sum reduces to a scalar when no axis is given", () => {
    // GRAPH 1's L = sum(Z) = 24, shape [] — the single number backward()
    // requires as its root.
    const Z = new TensorValue(createTensor([4, 4, 4, 4, 4, 4], [2, 3]));
    const L = Z.sum();
    expect(L.data.shape).toEqual([]);
    expect(L.data.size).toBe(1);
    expect(L.data.data[0]).toBe(24);
  });
});

describe("TensorValue — backward", () => {
  it("add backward: each input receives the upstream gradient", () => {
    // Addition is the router: local derivative 1, so with same shapes each
    // parent receives the upstream gradient unchanged.
    const x = new TensorValue(createTensor([1, 2, 3, 4], [2, 2]));
    const y = new TensorValue(createTensor([10, 20, 30, 40], [2, 2]));
    x.add(y).sum().backward();
    expect(x.grad?.shape).toEqual([2, 2]);
    expect(Array.from(x.grad!.data)).toEqual([1, 1, 1, 1]);
    expect(Array.from(y.grad!.data)).toEqual([1, 1, 1, 1]);
  });

  it("add backward with broadcast: grad is summed to original shape", () => {
    // GRAPH 1's d: one row used in TWO rows collects 1+1 per entry — and the
    // shape invariant holds: d.grad has d's own shape [1,3], not Z's [2,3].
    const C = new TensorValue(createTensor([-6, -6, -6, -6, -6, -6], [2, 3]));
    const { d } = graph1();
    C.add(d).sum().backward();
    expect(d.grad?.shape).toEqual([1, 3]);
    expect(Array.from(d.grad!.data)).toEqual([2, 2, 2]);
    // C needed no summing — the no-op sumToShape case.
    expect(Array.from(C.grad!.data)).toEqual([1, 1, 1, 1, 1, 1]);
  });

  it("mean backward distributes grad/n to each element", () => {
    // Forward must BE mean: 24/6 = 4 (a sum-forward bug would give 24).
    const Z = new TensorValue(createTensor([4, 4, 4, 4, 4, 4], [2, 3]));
    const m = Z.mean();
    expect(m.data.data[0]).toBe(4);
    m.backward();
    // Each of the 6 elements contributed 1/6 of the mean.
    expect(Z.grad!.data.every((g) => close(g, 1 / 6))).toBe(true);

    // The which-n pitfall: with an axis, n is the AXIS length (3), not size.
    const R = new TensorValue(createTensor([3, 1, 5, 4, 2, 0], [2, 3]));
    R.mean(1).sum().backward();
    expect(R.grad!.data.every((g) => close(g, 1 / 3))).toBe(true);
  });

  it("matMul backward: dA = dZ @ Bᵀ, dB = Aᵀ @ dZ", () => {
    // GRAPH 2, exactly as section 8 derives it cell by cell: each A.grad
    // entry is the sum of the matching ROW of B (10 = 1+2+3+4), each B.grad
    // row is the sum of the matching COLUMN of A (5 = 1+4).
    const { A, B } = graph2();
    A.matMul(B).sum().backward();
    expect(A.grad?.shape).toEqual([2, 3]);
    expect(Array.from(A.grad!.data)).toEqual([10, 26, 42, 10, 26, 42]);
    expect(B.grad?.shape).toEqual([3, 4]);
    expect(Array.from(B.grad!.data)).toEqual([5, 5, 5, 5, 7, 7, 7, 7, 9, 9, 9, 9]);
  });

  it("reshape backward: grad has the original pre-reshape shape", () => {
    // GRAPH 2's A flattened to [6]: same six numbers, and the gradient comes
    // back wearing the ORIGINAL [2,3] label.
    const { A } = graph2();
    const flat = A.reshape([6]);
    expect(flat.data.shape).toEqual([6]);
    flat.sum().backward();
    expect(A.grad?.shape).toEqual([2, 3]);
    expect(Array.from(A.grad!.data)).toEqual([1, 1, 1, 1, 1, 1]);
  });

  it("transpose backward applies the inverse permutation", () => {
    // A 3-D permutation, because 2-D cannot catch the classic bug: for
    // [1,0] the inverse IS [1,0], so a wrong implementation passes every
    // 2-D test. Here axes [1,2,0] inverts to [2,0,1] — different arrays.
    const t = new TensorValue(
      createTensor(Array.from({ length: 24 }, (_, i) => i), [2, 3, 4]),
    );
    const W = new TensorValue(
      createTensor(Array.from({ length: 24 }, (_, i) => i + 1), [3, 4, 2]),
    );
    // L = sum(transpose(t) ⊙ W)  →  t.grad must be W carried back through
    // the INVERSE permutation: transpose(W, [2,0,1]). W's values are all
    // distinct, so applying the wrong permutation cannot match.
    t.transpose([1, 2, 0]).mul(W).sum().backward();
    const expected = transpose(W.data, [2, 0, 1]);
    expect(t.grad?.shape).toEqual([2, 3, 4]);
    expect(Array.from(t.grad!.data)).toEqual(Array.from(expected.data));
  });
});

describe("backward — the engine", () => {
  it("one sweep fills every gradient in GRAPH 1 — the section 2 table", () => {
    // The chapter's first end-to-end check: three gradients element-for-
    // element equal to Ch 08's scalars, and d summed down the broadcast.
    const { A, B, d } = graph1();
    const C = A.mul(B);
    const Z = C.add(d);
    const L = Z.sum();
    // Six nodes reachable, every node after its own inputs.
    expect(topoSortTensor(L)).toHaveLength(6);
    L.backward();
    // The seed: ∂L/∂L = 1.
    expect(L.grad?.data[0]).toBe(1);
    // The multiply routes each operand its sibling's values — the switch.
    expect(Array.from(A.grad!.data)).toEqual([-3, -3, -3, -3, -3, -3]);
    expect(Array.from(B.grad!.data)).toEqual([2, 2, 2, 2, 2, 2]);
    // The bias, summed over its two uses per entry.
    expect(Array.from(d.grad!.data)).toEqual([2, 2, 2]);
  });

  it("refuses a non-scalar root", () => {
    // Seeding a [2,3] node with ones would silently compute the gradient of
    // an implicit sum — the guard turns that surprise into an error.
    const Z = new TensorValue(createTensor([4, 4, 4, 4, 4, 4], [2, 3]));
    expect(() => Z.backward()).toThrow();
  });
});

describe("checkTensorGradient", () => {
  it("passes for add", () => {
    // The broadcast case, deliberately — it exercises sumToShape inside the
    // analytical path against a numerical referee that knows nothing of it.
    const C = new TensorValue(createTensor([-6, -6, -6, -6, -6, -6], [2, 3]));
    const d = new TensorValue(createTensor([10, 10, 10], [1, 3]));
    expect(checkTensorGradient((ins) => ins[0]!.add(ins[1]!), [C, d])).toBe(true);
  });

  it("passes for matMul", () => {
    // Non-square GRAPH 2 shapes: a wrong transpose cannot hide here.
    const { A, B } = graph2();
    expect(checkTensorGradient((ins) => ins[0]!.matMul(ins[1]!), [A, B])).toBe(true);
  });

  it("passes for mean", () => {
    // Axis mean: the 1/n scaling with n = axis length, verified numerically.
    const R = new TensorValue(createTensor([3, 1, 5, 4, 2, 0], [2, 3]));
    expect(checkTensorGradient((ins) => ins[0]!.mean(1), [R])).toBe(true);
  });
});
