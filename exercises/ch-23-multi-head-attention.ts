/**
 * EXERCISES — Ch 23: Multi-Head Attention
 * ═════════════════════════════════════════
 * Prereq : src/nn/attention.ts (MultiHeadAttention) implemented
 * Run    : bun run exercises/ch-23-multi-head-attention.ts
 *
 * Multiple heads let the model attend to different representation subspaces
 * simultaneously. Each head sees a [seqLen, dHead] slice of the full dModel.
 */
import { MultiHeadAttention } from "../src/nn/attention.ts";
import { TensorValue }        from "../src/autograd/grad.ts";
import { randn }              from "../src/tensor/creation.ts";

// ─── E1: Shape contract ───────────────────────────────────────────────────────
// dModel=32, numHeads=4 → dHead = 32/4 = 8
const mha     = new MultiHeadAttention(32, 4);
const batchSz = 2, seqLen = 6;
const x       = new TensorValue(randn([batchSz, seqLen, 32]), "x");
const out     = mha.forward(x);
console.log("MHA output shape:", out.data.shape, "  expected: [2, 6, 32]");

// ─── E2: With causal mask ─────────────────────────────────────────────────────
import { expandCausalMask, causalMask, toAdditiveMask } from "../src/tokenizer/masks.ts";
const cm   = toAdditiveMask(causalMask(seqLen));
const mask = new TensorValue(expandCausalMask(cm, batchSz, 4), "mask");
const maskedOut = mha.forward(x, mask);
console.log("Masked MHA output shape:", maskedOut.data.shape, "  expected: [2, 6, 32]");

// ─── E3: Backward pass ────────────────────────────────────────────────────────
maskedOut.backward();
console.log("\nInput grad shape:", x.grad?.shape, "  expected: [2, 6, 32]");

// ─── E4: Parameter count ─────────────────────────────────────────────────────
// MHA has 4 Linear projections: Wq, Wk, Wv, Wo
// Each [dModel, dModel] + bias [dModel] → 4*(dModel² + dModel) params
const params = mha.parameters();
const totalParams = params.reduce((s, p) => s + p.data.size, 0);
const expected    = 4 * (32 * 32 + 32);   // Wq+Wk+Wv+Wo weights + biases
console.log("\nTotal params:", totalParams, "  expected:", expected);

// ─── E5: splitHeads / mergeHeads ─────────────────────────────────────────────
// These reshape operations convert between:
//   [batch, seq, dModel]  ↔  [batch, numHeads, seq, dHead]
// Verify the shapes are correct by running a forward pass with known input sizes.
const mha2 = new MultiHeadAttention(64, 8);
const x2   = new TensorValue(randn([3, 10, 64]), "x2");
const out2 = mha2.forward(x2);
console.log("\nMHA(dModel=64, heads=8) output shape:", out2.data.shape, "  expected: [3, 10, 64]");

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: verify that numHeads must divide dModel evenly.
//       Try MultiHeadAttention(32, 5) — should throw at construction.
