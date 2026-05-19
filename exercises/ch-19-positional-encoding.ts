/**
 * EXERCISES — Ch 19: Positional Encoding
 * ════════════════════════════════════════
 * Prereq : src/nn/positional.ts + embedding.ts implemented
 * Run    : bun run exercises/ch-19-positional-encoding.ts
 *
 * Self-attention is permutation-invariant — it sees a SET of tokens, not a SEQUENCE.
 * Positional encodings inject position information so the model knows word order.
 *
 * PE(pos, 2i)   = sin(pos / 10000^(2i/dModel))
 * PE(pos, 2i+1) = cos(pos / 10000^(2i/dModel))
 */
import { PositionalEncoding } from "../src/nn/positional.ts";
import { TensorValue }        from "../src/autograd/grad.ts";
import { createTensor, zeros } from "../src/tensor/creation.ts";

// ─── E1: Shape check ─────────────────────────────────────────────────────────
const pe    = new PositionalEncoding(512, 100);  // dModel=512, maxLen=100
const input = new TensorValue(createTensor(
  Array.from({length: 10 * 512}, () => 0), [10, 512]
));
const out = pe.forward(input);
console.log("PE output shape:", out.data.shape, "  expected: [10, 512]");

// ─── E2: Sinusoidal pattern at position 0 ────────────────────────────────────
// At position 0, all sin terms = 0 and all cos terms = 1.
// So PE[0, 0] = sin(0) = 0, PE[0, 1] = cos(0) = 1.
const pe2  = new PositionalEncoding(8, 20);
const zero = new TensorValue(zeros([1, 8]));
const p0   = pe2.forward(zero);
console.log("\nPE at pos 0, dim 0:", p0.data.data[0]!.toFixed(6), "  expected: 0.000000 (sin(0))");
console.log("PE at pos 0, dim 1:", p0.data.data[1]!.toFixed(6), "  expected: 1.000000 (cos(0))");

// ─── E3: Different positions → different encodings ───────────────────────────
// PE at pos 5 should differ from PE at pos 6.
const seq = new TensorValue(zeros([10, 8]));
const enc = pe2.forward(seq);
const pos5 = Array.from(enc.data.data.slice(5 * 8, 6 * 8));
const pos6 = Array.from(enc.data.data.slice(6 * 8, 7 * 8));
const distinct = pos5.some((v, i) => Math.abs(v - pos6[i]!) > 1e-6);
console.log("\npos 5 ≠ pos 6:", distinct, "  expected: true");

// ─── E4: High-frequency vs low-frequency dimensions ─────────────────────────
// Dimension 0 uses frequency 1/10000^0 = 1 (fast oscillation).
// Dimension dModel-2 uses frequency 1/10000^1 ≈ very slow oscillation.
// Print dim 0 and dim dModel-2 for positions 0..7.
const pe3 = new PositionalEncoding(16, 50);
const longSeq = new TensorValue(zeros([8, 16]));
const longEnc = pe3.forward(longSeq);
console.log("\nDim 0 values (fast freq):", Array.from({length:8}, (_, p) => longEnc.data.data[p*16]!.toFixed(3)));
console.log("Dim 14 values (slow freq):", Array.from({length:8}, (_, p) => longEnc.data.data[p*16+14]!.toFixed(3)));

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: verify that PE values are bounded in [-1, 1] for any sequence length.
//       Hint: sin and cos are always in [-1,1], so the output should be too.
