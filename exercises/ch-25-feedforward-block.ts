/**
 * EXERCISES — Ch 25: Feed-Forward Block
 * ═══════════════════════════════════════
 * Prereq : src/nn/feedforward.ts + layernorm + attention implemented
 * Run    : bun run exercises/ch-25-feedforward-block.ts
 *
 * Every transformer block applies attention THEN a 2-layer FFN:
 *   FFN(x) = GELU(x @ W1 + b1) @ W2 + b2
 * The inner dimension (dFf) is typically 4× dModel.
 */
import { FFN }           from "../src/nn/feedforward.ts";
import { TensorValue }   from "../src/autograd/grad.ts";
import { randn }         from "../src/tensor/creation.ts";

// ─── E1: Basic forward pass ───────────────────────────────────────────────────
const ffn = new FFN(32, 128);   // dModel=32, dFf=128
const x   = new TensorValue(randn([4, 10, 32]), "x");  // [batch, seqLen, dModel]
const out = ffn.forward(x);
console.log("FFN output shape:", out.data.shape, "  expected: [4, 10, 32]");

// ─── E2: Default dFf = 4 × dModel ───────────────────────────────────────────
// GPT-2 uses dFf = 4 * dModel.  The default constructor should enforce this.
const ffnDefault = new FFN(64);  // dFf defaults to 256
const xDef = new TensorValue(randn([2, 8, 64]));
const outDef = ffnDefault.forward(xDef);
console.log("FFN(64) default output shape:", outDef.data.shape, "  expected: [2, 8, 64]");

// ─── E3: Backward pass ────────────────────────────────────────────────────────
out.backward();
console.log("\nInput grad shape:", x.grad?.shape, "  expected: [4, 10, 32]");
const paramShapes = ffn.parameters().map(p => p.data.shape);
console.log("Param shapes:", paramShapes);
// expected: [[128,32], [128], [32,128], [32]]  (W1, b1, W2, b2)

// ─── E4: Parameter count ─────────────────────────────────────────────────────
// W1: [dFf, dModel] + b1: [dFf] + W2: [dModel, dFf] + b2: [dModel]
// = 128*32 + 128 + 32*128 + 32 = 8320
const totalFFN = ffn.parameters().reduce((s, p) => s + p.data.size, 0);
const expected = 128 * 32 + 128 + 32 * 128 + 32;
console.log("\nTotal params:", totalFFN, "  expected:", expected);

// ─── E5: Residual connection ─────────────────────────────────────────────────
// In the transformer, FFN is used with a residual: output = x + FFN(LN(x))
import { LayerNorm } from "../src/nn/layernorm.ts";
const ln  = new LayerNorm(32);
const res = x.add(ffn.forward(ln.forward(x)));
console.log("\nResidual output shape:", res.data.shape, "  expected: [4, 10, 32]");

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: try dFf = dModel (no expansion). Compare final loss on Ch 15 spiral task.
