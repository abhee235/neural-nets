/**
 * Tests for autograd/value.ts
 * Chapters 08a & 08b — Scalar Autograd
 *
 * Run: bun test src/autograd/value.test.ts
 *
 * Read the test names top to bottom and they teach the chapter: first that the
 * forward pass RECORDS a graph (08a), then that one reverse sweep over that
 * graph reproduces the chain rule (08b). The last block is the gate — every
 * analytical derivative is checked against Ch 07's finite differences.
 */
import { describe, it, expect } from "bun:test";
import { Value } from "./value.ts";
import { topoSort } from "./engine.ts";
import { numericalGradient } from "../utils/numerical.ts";

const EPSILON = 1e-5;
const close = (a: number, b: number) => Math.abs(a - b) < EPSILON;

/**
 * Differentiate the same function two ways at the same point:
 * analytically (build a graph, run one backward sweep) and numerically
 * (centered differences from Ch 07). Agreement between the two is what
 * "the backward pass is correct" actually means.
 */
function bothGradients(
  build: (v: Value) => Value,
  plain: (x: number) => number,
  at: number,
): { analytical: number; numerical: number } {
  const v = new Value(at);
  build(v).backward();
  return { analytical: v.grad, numerical: numericalGradient(plain, at) };
}

describe("Value — forward pass", () => {
  it("add produces the correct scalar sum", () => {
    const out = new Value(2).add(new Value(-3));
    // Arithmetic on the graph must agree with arithmetic on plain numbers.
    expect(out.data).toBe(-1);
    // …and unlike a plain number, the result remembers what produced it.
    expect(out._op).toBe("+");
  });

  it("mul produces the correct scalar product", () => {
    const out = new Value(2).mul(new Value(-3));
    // 2 × (-3) = -6 — the interior node `c` of the chapter's running example.
    expect(out.data).toBe(-6);
    expect(out._op).toBe("*");
  });

  it("_inputs records both operands of a binary op", () => {
    const a = new Value(2);
    const b = new Value(-3);
    const c = a.mul(b);
    // Both parents are recorded, in operand order. Order is load-bearing: the
    // backward pass sends each parent its SIBLING's value, so swapping them
    // swaps the gradients.
    expect(c._inputs).toHaveLength(2);
    expect(c._inputs[0]).toBe(a);
    expect(c._inputs[1]).toBe(b);
    // A leaf is not a special kind of node — it is just one nothing produced.
    expect(a._inputs).toHaveLength(0);
    expect(a._op).toBe("");
  });

  it("chained ops build a graph of depth > 1", () => {
    const a = new Value(2);
    const b = new Value(-3);
    const d = new Value(10);
    const L = a.mul(b).add(d); // L = (a·b) + d
    expect(L.data).toBe(4);
    // One level down: L's first parent is the product node, not a leaf.
    expect(L._inputs[0]!._op).toBe("*");
    // Two levels down: that product's own parent is the leaf `a`. The graph
    // nests without anyone assembling it — that nesting is what lets the chain
    // rule compose across arbitrarily deep expressions.
    expect(L._inputs[0]!._inputs[0]).toBe(a);
  });

  it("exp(0).data === 1", () => {
    // e⁰ = 1.
    expect(new Value(0).exp().data).toBe(1);
  });

  it("tanh(0).data === 0", () => {
    // tanh is an odd function, so it passes through the origin.
    expect(new Value(0).tanh().data).toBe(0);
  });
});

describe("topoSort", () => {
  it("orders every node after its own inputs", () => {
    const a = new Value(2);
    const b = new Value(-3);
    const d = new Value(10);
    const c = a.mul(b);
    const L = c.add(d);
    const order = topoSort(L);
    // Five distinct nodes are reachable from L: three leaves and two interior.
    expect(order).toHaveLength(5);
    // Post-order DFS yields inputs first; backward() consumes this reversed.
    expect(order).toEqual([a, b, c, d, L]);
    // The defining property, stated directly: no node precedes its own input.
    for (const node of order) {
      for (const input of node._inputs) {
        expect(order.indexOf(input)).toBeLessThan(order.indexOf(node));
      }
    }
  });

  it("visits a reused node exactly once", () => {
    const x = new Value(3);
    // x.mul(x) records _inputs = [x, x], so x is reachable by two paths.
    // Without a visited set it would be listed twice and its _backward would
    // fire twice, double-counting gradient into its parents.
    expect(topoSort(x.mul(x))).toHaveLength(2);
  });

  it("returns a lone leaf unchanged", () => {
    const x = new Value(5);
    // A graph of one node is still a valid graph — backward() must not special-case it.
    expect(topoSort(x)).toEqual([x]);
  });
});

describe("Value — backward pass", () => {
  it("z = x*y: backward gives dz/dx = y and dz/dy = x", () => {
    const x = new Value(3);
    const y = new Value(4);
    x.mul(y).backward();
    // ∂(xy)/∂x = y and ∂(xy)/∂y = x — each operand receives the OTHER's value.
    expect(x.grad).toBe(4);
    expect(y.grad).toBe(3);
  });

  it("z = x^2: backward gives dz/dx = 2x", () => {
    const x = new Value(3);
    x.pow(2).backward();
    // Power rule: d(x²)/dx = 2x = 6 at x = 3.
    expect(x.grad).toBe(6);
    // The same 6 must arrive by a different mechanism when x is reused as both
    // operands of a product: two accumulations of 3, rather than one of 6.
    const y = new Value(3);
    y.mul(y).backward();
    expect(y.grad).toBe(6);
  });

  it("z = exp(x): backward gives dz/dx = exp(x)", () => {
    const x = new Value(1);
    const z = x.exp();
    z.backward();
    // exp is its own derivative — uniquely, the gradient equals the forward output.
    expect(close(x.grad, Math.E)).toBe(true);
    expect(x.grad).toBe(z.data);
  });

  it("one sweep fills every gradient in the graph", () => {
    // The chapter's canonical example: L = (a·b) + d at a=2, b=-3, d=10.
    const a = new Value(2);
    const b = new Value(-3);
    const d = new Value(10);
    const c = a.mul(b);
    const L = c.add(d);
    L.backward();
    // The seed: a value's derivative with respect to itself is 1.
    expect(L.grad).toBe(1);
    // "+ is a router" — it copies its incoming gradient to both parents.
    expect(c.grad).toBe(1);
    expect(d.grad).toBe(1);
    // "× is a switch" — each parent is scaled by the other's forward value,
    // giving ∂L/∂a = b and ∂L/∂b = a, exactly as differentiating a·b + d by hand.
    expect(a.grad).toBe(-3);
    expect(b.grad).toBe(2);
  });

  it("gradients accumulate correctly through a chain of ops", () => {
    // f = a·b + b². `b` reaches the output along TWO paths, so its true
    // derivative is the SUM of both: ∂f/∂b = a + 2b = 1 + 6 = 7.
    const a = new Value(1);
    const b = new Value(3);
    const f = a.mul(b).add(b.pow(2));
    f.backward();
    expect(f.data).toBe(12);
    // Single path: ∂f/∂a = b = 3.
    expect(a.grad).toBe(3);
    // Two paths meeting and adding — this is precisely why accumulation is
    // `+=`. With `=`, one contribution would overwrite the other and this
    // would read 1 or 6 instead of 7, with nothing raising an error.
    expect(b.grad).toBe(7);
  });

  it("x.add(x) accumulates both contributions", () => {
    const x = new Value(5);
    x.add(x).backward();
    // d(x + x)/dx = 2. The add's two `+=` lines land on the same node.
    expect(x.grad).toBe(2);
  });

  it("zeroGrad resets grad to 0", () => {
    const x = new Value(3);
    const y = new Value(4);
    x.mul(y).backward();
    expect(x.grad).toBe(4);
    x.zeroGrad();
    // The reset is per-node: x is cleared, y is untouched.
    expect(x.grad).toBe(0);
    expect(y.grad).toBe(3);
  });

  it("re-running backward without zeroGrad contaminates the graph", () => {
    // Documents the trap rather than endorsing it: because accumulation is
    // `+=`, a second sweep adds to the first. It does not merely double —
    // backward() re-seeds the ROOT by assignment, but interior nodes keep
    // accumulating, so c.grad is already 2 before it multiplies into a:
    //   run 1: a.grad = -3      run 2: -3 + (-3×2) = -9
    // The contamination compounds with the depth of the graph.
    const a = new Value(2);
    const b = new Value(-3);
    const L = a.mul(b).add(new Value(10));
    L.backward();
    expect(a.grad).toBe(-3);
    L.backward();
    expect(a.grad).toBe(-9);
    expect(b.grad).toBe(6);
  });
});

describe("numerical gradient checks", () => {
  // The verification gate for Part 2: every analytical backward must agree with
  // Ch 07's centered finite differences. numericalGradient is the referee here
  // precisely because it was built without reference to any of these rules.

  it("add gradient matches finite differences", () => {
    const { analytical, numerical } = bothGradients(
      (v) => v.add(new Value(4)),
      (x) => x + 4,
      2.3,
    );
    expect(close(analytical, numerical)).toBe(true);
    // ∂(x + c)/∂x = 1 exactly — addition never scales the gradient.
    expect(analytical).toBe(1);
  });

  it("mul gradient matches finite differences", () => {
    const { analytical, numerical } = bothGradients(
      (v) => v.mul(new Value(4)),
      (x) => x * 4,
      2.3,
    );
    expect(close(analytical, numerical)).toBe(true);
    // ∂(cx)/∂x = c — the constant operand passes straight through.
    expect(analytical).toBe(4);
  });

  it("pow gradient matches finite differences", () => {
    const cubed = bothGradients((v) => v.pow(3), (x) => x ** 3, 2.3);
    expect(close(cubed.analytical, cubed.numerical)).toBe(true);
    // d(x³)/dx = 3x² = 15.87 at x = 2.3. Checked away from x = 1, where the
    // n and n−1 exponents would coincide and hide an off-by-one.
    expect(close(cubed.analytical, 15.87)).toBe(true);

    // A negative exponent too, since div is built as mul(other.pow(-1)):
    // d(x⁻¹)/dx = −x⁻² — a case a positive-only test would never exercise.
    const inverse = bothGradients((v) => v.pow(-1), (x) => x ** -1, 2.3);
    expect(close(inverse.analytical, inverse.numerical)).toBe(true);
  });

  it("exp gradient matches finite differences", () => {
    const { analytical, numerical } = bothGradients(
      (v) => v.exp(),
      (x) => Math.exp(x),
      1.0,
    );
    expect(close(analytical, numerical)).toBe(true);
  });

  it("log gradient matches finite differences", () => {
    const { analytical, numerical } = bothGradients(
      (v) => v.log(),
      (x) => Math.log(x),
      4.0,
    );
    expect(close(analytical, numerical)).toBe(true);
    // d(ln x)/dx = 1/x = 0.25 at x = 4 — the rule that reads the INPUT.
    expect(close(analytical, 0.25)).toBe(true);
  });

  it("tanh gradient matches finite differences", () => {
    // Checked at 1.0, NOT 0: the correct rule (1 − out²) and the common wrong
    // one (1 − in²) both give 1 at the origin, so a test at 0 cannot tell them
    // apart. At x = 1 they differ, and this test fails if the input is squared.
    const atOne = bothGradients((v) => v.tanh(), (x) => Math.tanh(x), 1.0);
    expect(close(atOne.analytical, atOne.numerical)).toBe(true);

    const atSeven = bothGradients((v) => v.tanh(), (x) => Math.tanh(x), 0.7);
    expect(close(atSeven.analytical, atSeven.numerical)).toBe(true);
  });

  it("relu gradient matches finite differences (x ≠ 0)", () => {
    // Active side: relu is the identity, so the gradient passes untouched.
    const positive = bothGradients((v) => v.relu(), (x) => Math.max(0, x), 2.0);
    expect(close(positive.analytical, positive.numerical)).toBe(true);
    expect(positive.analytical).toBe(1);

    // Inactive side: the gate is shut and the gradient is exactly 0, not small.
    const negative = bothGradients((v) => v.relu(), (x) => Math.max(0, x), -2.0);
    expect(close(negative.analytical, negative.numerical)).toBe(true);
    expect(negative.analytical).toBe(0);

    // x = 0 is deliberately untested: the derivative does not exist at the
    // kink, and a centered difference straddling it averages the two one-sided
    // slopes to 0.5, which disagrees with every convention a library may pick.
  });

  it("composed operations match finite differences", () => {
    // The rules above are only useful if they COMPOSE. This nests four node
    // types — pow, exp, add, tanh — so a sign or operand error anywhere in the
    // chain shows up here even when each rule passes in isolation.
    const { analytical, numerical } = bothGradients(
      (v) => v.pow(2).add(v.exp()).tanh(),
      (x) => Math.tanh(x ** 2 + Math.exp(x)),
      0.6,
    );
    expect(close(analytical, numerical)).toBe(true);
  });
});
