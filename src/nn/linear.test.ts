/**
 * Tests for nn/linear.ts
 * Chapter 13 — The Linear Layer
 *
 * Run: bun test src/nn/linear.test.ts
 *
 * The fixture is the chapter's hand-set layer (doc sections 3–4), which
 * reproduces Chapter 12's logits from Chapter 12's input:
 *
 *   x  = [1, 2]                 W = [ 1    0   ]     b = [0, 0, 1.5]
 *                                   [ 0    1   ]
 *   y  = [1, 2, 3]                  [ 0.5  0.5 ]
 *
 * With Ch 12's cross-entropy on top (truth "sat"):
 *   b.grad = p − y      = [-0.909969, 0.244728, 0.665241]
 *   W.grad row i        = (p − y)ᵢ · x
 *
 * Dimensions are deliberately uneven (2 → 3) so a transposed weight is a
 * loud shape error rather than a silent one.
 */
import { describe, it, expect } from "bun:test";
import { Linear } from "./linear.ts";
import { TensorValue } from "../autograd/grad.ts";
import { crossEntropyFromLogits } from "./losses.ts";
import { createTensor } from "../tensor/types.ts";

const EPSILON = 1e-6;

function expectRow(actual: number[], expected: number[], tol = EPSILON): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    expect(Math.abs(actual[i]! - expected[i]!)).toBeLessThan(tol);
  }
}

/** Standard deviation of a tensor's values — for checking init scale. */
function std(values: Float64Array): number {
  const arr = Array.from(values);
  const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
  return Math.sqrt(arr.reduce((a, v) => a + (v - mean) ** 2, 0) / arr.length);
}

/** The chapter's hand-set layer: weights chosen so the output is [1, 2, 3]. */
function handSetLayer(): Linear {
  const layer = new Linear(2, 3);
  layer.weight.data = createTensor([1, 0, 0, 1, 0.5, 0.5], [3, 2]);
  layer.bias!.data = createTensor([0, 0, 1.5], [3]);
  return layer;
}

describe("Linear — construction", () => {
  it("weight is [outputDim, inputDim], one row per output unit", () => {
    // Row i holds everything unit i knows. Reversed would be [2, 3] here.
    expect(new Linear(2, 3).weight.data.shape).toEqual([3, 2]);
  });

  it("bias is [outputDim] and starts at zero", () => {
    // Zero is safe for the bias: the random weights already broke symmetry.
    const layer = new Linear(2, 3);
    expect(layer.bias!.data.shape).toEqual([3]);
    expectRow(Array.from(layer.bias!.data.data), [0, 0, 0]);
  });

  it("bias is null — not a zeros tensor — when bias is false", () => {
    // A zeros tensor would build a dead graph node on every forward pass.
    expect(new Linear(2, 3, false).bias).toBe(null);
  });

  it("records inputDim and outputDim", () => {
    const layer = new Linear(2, 3);
    expect(layer.inputDim).toBe(2);
    expect(layer.outputDim).toBe(3);
  });

  it("he init scales by √(2/inputDim) — and by INPUT dim, not output", () => {
    // 400 -> 100 is deliberately uneven: √(2/400) = 0.0707 while a mixup
    // giving √(2/100) = 0.1414 is a factor of two away and cannot hide.
    const layer = new Linear(400, 100, false, "he");
    expect(Math.abs(std(layer.weight.data.data) - Math.sqrt(2 / 400))).toBeLessThan(0.005);
  });

  it("xavier init scales by √(1/inputDim)", () => {
    const layer = new Linear(400, 100, false, "xavier");
    expect(Math.abs(std(layer.weight.data.data) - Math.sqrt(1 / 400))).toBeLessThan(0.005);
  });

  it("normal init scales by a flat 0.02, ignoring the dimensions", () => {
    // GPT-2's choice: unlike the others, it does not depend on inputDim.
    const layer = new Linear(400, 100, false, "normal");
    expect(Math.abs(std(layer.weight.data.data) - 0.02)).toBeLessThan(0.002);
  });

  it("defaults to he init when none is given", () => {
    const layer = new Linear(400, 100, false);
    expect(Math.abs(std(layer.weight.data.data) - Math.sqrt(2 / 400))).toBeLessThan(0.005);
  });

  it("does not start at zero — the randomness is load-bearing", () => {
    // Zero weights make a layer unreachable from above (doc section 15).
    const layer = new Linear(2, 3);
    expect(Array.from(layer.weight.data.data).some((v) => v !== 0)).toBe(true);
  });
});

describe("Linear — forward", () => {
  it("reproduces Chapter 12's logits from the chapter's hand-set weights", () => {
    // 1·1 + 2·0 + 0 = 1,  1·0 + 2·1 + 0 = 2,  1·0.5 + 2·0.5 + 1.5 = 3
    const y = handSetLayer().forward(new TensorValue(createTensor([1, 2], [1, 2])));
    expect(y.data.shape).toEqual([1, 3]);
    expectRow(Array.from(y.data.data), [1, 2, 3]);
  });

  it("maps [*, inputDim] to [*, outputDim]", () => {
    const y = new Linear(2, 3).forward(new TensorValue(createTensor([1, 2], [1, 2])));
    expect(y.data.shape).toEqual([1, 3]);
  });

  it("handles a batch with no change to the layer", () => {
    // The batch dimension comes free from matMul — this is why Ch 04 built it.
    const x = new TensorValue(createTensor([1, 2, 3, 4, 5, 6, 7, 8], [4, 2]));
    expect(handSetLayer().forward(x).data.shape).toEqual([4, 3]);
  });

  it("every row of a batch is computed independently", () => {
    // Row 0 of a batched call equals the same input passed on its own.
    const batched = handSetLayer().forward(
      new TensorValue(createTensor([1, 2, 5, 6], [2, 2])),
    );
    const alone = handSetLayer().forward(new TensorValue(createTensor([5, 6], [1, 2])));
    expectRow(Array.from(batched.data.data).slice(3), Array.from(alone.data.data));
  });

  it("the bias is what lets a zero input produce a non-zero score", () => {
    // Without a bias, x = 0 could only ever give 0 (doc section 1).
    const y = handSetLayer().forward(new TensorValue(createTensor([0, 0], [1, 2])));
    expectRow(Array.from(y.data.data), [0, 0, 1.5]);
  });

  it("omits the bias entirely when bias is false", () => {
    const layer = new Linear(2, 3, false);
    layer.weight.data = createTensor([1, 0, 0, 1, 0.5, 0.5], [3, 2]);
    const y = layer.forward(new TensorValue(createTensor([1, 2], [1, 2])));
    // Same as the hand-set layer minus its [0, 0, 1.5] bias.
    expectRow(Array.from(y.data.data), [1, 2, 1.5]);
  });
});

describe("Linear — backward", () => {
  it("bias gradient is the output gradient, arriving untouched", () => {
    // ∂y/∂b = 1, so whatever reaches the output reaches the bias. With Ch 12's
    // loss that is exactly p − y.
    const layer = handSetLayer();
    const y = layer.forward(new TensorValue(createTensor([1, 2], [1, 2])));
    crossEntropyFromLogits(y, createTensor([1, 0, 0], [1, 3])).backward();
    expectRow(Array.from(layer.bias!.grad!.data), [-0.909969, 0.244728, 0.665241], 1e-6);
  });

  it("weight gradient row i is (output gradient)ᵢ × x", () => {
    // ∂y/∂wᵢ = xᵢ, so a weight's blame is scaled by the input it multiplied.
    // Feature 2 is twice feature 1, so it gets twice the gradient.
    const layer = handSetLayer();
    const y = layer.forward(new TensorValue(createTensor([1, 2], [1, 2])));
    crossEntropyFromLogits(y, createTensor([1, 0, 0], [1, 3])).backward();
    expectRow(
      Array.from(layer.weight.grad!.data),
      [-0.909969, -1.819938, 0.244728, 0.489456, 0.665241, 1.330482],
      1e-5,
    );
  });

  it("weight gradient keeps W's own shape — the transpose is undone", () => {
    // If forward transposed .data by hand instead of using the graph method,
    // W would be severed and this gradient would be null.
    const layer = handSetLayer();
    const y = layer.forward(new TensorValue(createTensor([1, 2], [1, 2])));
    y.sum().backward();
    expect(layer.weight.grad!.shape).toEqual([3, 2]);
    expect(layer.bias!.grad!.shape).toEqual([3]);
  });

  it("bias gradient sums over the batch, one number per unit", () => {
    // Ch 10's sumToShape collecting a broadcast bias back down. Four rows,
    // upstream all ones, so each unit's bias gradient is 4.
    const layer = handSetLayer();
    const x = new TensorValue(createTensor([1, 2, 3, 4, 5, 6, 7, 8], [4, 2]));
    layer.forward(x).sum().backward();
    expect(layer.bias!.grad!.shape).toEqual([3]);
    expectRow(Array.from(layer.bias!.grad!.data), [4, 4, 4]);
  });

  it("passes a gradient back to its input, so layers can stack", () => {
    // x.grad is unused when x is data, but it is how blame reaches the layer
    // below — the whole reason a deep network trains in one backward pass.
    const x = new TensorValue(createTensor([1, 2], [1, 2]));
    handSetLayer().forward(x).sum().backward();
    expect(x.grad).not.toBe(null);
    expect(x.grad!.shape).toEqual([1, 2]);
  });

  it("two stacked layers both receive gradients", () => {
    const first = new Linear(2, 4);
    const second = new Linear(4, 3);
    const x = new TensorValue(createTensor([1, 2], [1, 2]));
    second.forward(first.forward(x)).sum().backward();
    expect(first.weight.grad).not.toBe(null);
    expect(second.weight.grad).not.toBe(null);
    expect(first.weight.grad!.shape).toEqual([4, 2]);
  });

  it("numerical gradient check passes for the weight", () => {
    const layer = handSetLayer();
    const x = createTensor([1, 2], [1, 2]);
    const truth = createTensor([1, 0, 0], [1, 3]);
    const base = Array.from(layer.weight.data.data);

    const lossAt = (w: number[]) => {
      const probe = new Linear(2, 3);
      probe.weight.data = createTensor(w, [3, 2]);
      probe.bias!.data = createTensor([0, 0, 1.5], [3]);
      return crossEntropyFromLogits(probe.forward(new TensorValue(x)), truth).data.data[0]!;
    };

    crossEntropyFromLogits(layer.forward(new TensorValue(x)), truth).backward();

    const h = 1e-6;
    for (let i = 0; i < base.length; i++) {
      const up = [...base], down = [...base];
      up[i]! += h;
      down[i]! -= h;
      const numeric = (lossAt(up) - lossAt(down)) / (2 * h);
      expect(Math.abs(numeric - layer.weight.grad!.data[i]!)).toBeLessThan(1e-5);
    }
  });
});

describe("Linear — parameters", () => {
  it("returns the SAME objects, not copies", () => {
    // An optimizer given copies would faithfully update tensors nothing reads.
    const layer = new Linear(2, 3);
    const params = layer.parameters();
    expect(params[0]).toBe(layer.weight);
    expect(params[1]).toBe(layer.bias!);
  });

  it("returns [weight, bias] when there is a bias", () => {
    expect(new Linear(2, 3).parameters().length).toBe(2);
  });

  it("returns [weight] with no hole when bias is false", () => {
    // Never [weight, null] — an optimizer would crash walking the list.
    const layer = new Linear(2, 3, false);
    const params = layer.parameters();
    expect(params.length).toBe(1);
    expect(params[0]).toBe(layer.weight);
  });

  it("hands the optimizer tensors whose gradients are actually filled", () => {
    // The contract end to end: forward, backward, and every listed parameter
    // has a gradient waiting for Ch 14's step().
    const layer = new Linear(2, 3);
    layer.forward(new TensorValue(createTensor([1, 2], [1, 2]))).sum().backward();
    for (const p of layer.parameters()) expect(p.grad).not.toBe(null);
  });
});
