/**
 * EXERCISES — Ch 20: LayerNorm & Dropout
 * ════════════════════════════════════════
 * Prereq : src/nn/layernorm.ts + dropout.ts implemented
 * Run    : bun run exercises/ch-20-layernorm-dropout.ts
 *
 * LayerNorm stabilises training — without it, deep transformers diverge.
 * Dropout prevents co-adaptation (overfitting) during training.
 */
import { LayerNorm }    from "../src/nn/layernorm.ts";
import { Dropout }      from "../src/nn/dropout.ts";
import { TensorValue }  from "../src/autograd/grad.ts";
import { createTensor } from "../src/tensor/creation.ts";

// ─── E1: LayerNorm normalises each row ────────────────────────────────────────
// After LayerNorm, each row should have mean ≈ 0 and std ≈ 1.
const ln = new LayerNorm(4);
const x  = new TensorValue(createTensor([10, 20, 30, 40,  1, 2, 3, 4], [2, 4]));
const n  = ln.forward(x);

function rowStats(data: Float64Array, start: number, len: number) {
  const vals = Array.from(data).slice(start, start + len);
  const mean = vals.reduce((a, b) => a + b, 0) / len;
  const std  = Math.sqrt(vals.reduce((a, v) => a + (v - mean) ** 2, 0) / len);
  return { mean, std };
}
const row0 = rowStats(n.data.data, 0, 4);
const row1 = rowStats(n.data.data, 4, 4);
console.log("row 0 mean:", row0.mean.toFixed(6), "  expected: ≈ 0");
console.log("row 0 std :", row0.std.toFixed(6),  "  expected: ≈ 1");
console.log("row 1 mean:", row1.mean.toFixed(6), "  expected: ≈ 0");
console.log("row 1 std :", row1.std.toFixed(6),  "  expected: ≈ 1");

// ─── E2: Learnable gamma and beta ────────────────────────────────────────────
// gamma starts at 1 (no scaling), beta at 0 (no shift).
const [gamma, beta] = ln.parameters();
console.log("\ngamma initial:", Array.from(gamma!.data.data), "  expected: [1,1,1,1]");
console.log("beta  initial:", Array.from(beta!.data.data),  "  expected: [0,0,0,0]");

// ─── E3: LayerNorm gradient ───────────────────────────────────────────────────
n.backward();
console.log("\ngamma.grad:", Array.from(gamma!.grad!.data));
console.log("beta.grad :", Array.from(beta!.grad!.data));

// ─── E4: Dropout — train vs eval mode ────────────────────────────────────────
const drop = new Dropout(0.5);
drop.train();
const xDrop  = new TensorValue(createTensor(Array.from({length: 1000}, () => 1.0), [1000]));
const outTrain = drop.forward(xDrop);
const zeros1k  = Array.from(outTrain.data.data).filter(v => v === 0).length;
console.log("\nDropout (train, p=0.5) zeros:", zeros1k, "  expected: ≈ 500 (±50)");
// Inverted dropout scales surviving activations by 1/(1-p) to preserve expected value.
const meanTrain = Array.from(outTrain.data.data).reduce((a, b) => a + b, 0) / 1000;
console.log("Mean after dropout:", meanTrain.toFixed(3), "  expected: ≈ 1.0 (inverted scaling)");

drop.eval();
const outEval = drop.forward(xDrop);
const zerosEval = Array.from(outEval.data.data).filter(v => v === 0).length;
console.log("Dropout (eval) zeros:", zerosEval, "  expected: 0");

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: verify LayerNorm is invariant to input scale.
//   ln.forward(x) and ln.forward(10*x) should give identical outputs.
