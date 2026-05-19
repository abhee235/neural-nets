/**
 * EXERCISES — Ch 22: Self-Attention
 * ═══════════════════════════════════
 * Prereq : src/nn/attention.ts + linear.ts + masks.ts implemented
 * Run    : bun run exercises/ch-22-self-attention.ts
 *
 * Self-attention lets every token look at every other token in the sequence.
 * Scores = softmax(Q @ Kᵀ / √dk) · V
 */
import { scaledDotProductAttention, SelfAttention } from "../src/nn/attention.ts";
import { TensorValue }  from "../src/autograd/grad.ts";
import { createTensor, randn } from "../src/tensor/creation.ts";
import { causalMask, toAdditiveMask } from "../src/tokenizer/masks.ts";

// ─── E1: Scaled dot-product attention — shapes ───────────────────────────────
// seqLen=5, dk=8
const seqLen = 5, dk = 8;
const Q = new TensorValue(randn([seqLen, dk]), "Q");
const K = new TensorValue(randn([seqLen, dk]), "K");
const V = new TensorValue(randn([seqLen, dk]), "V");

const attnOut = scaledDotProductAttention(Q, K, V);
console.log("attn output shape:", attnOut.data.shape, "  expected: [5, 8]");

// ─── E2: Attention with causal mask ──────────────────────────────────────────
// With a causal mask, position 0 only sees itself, position 2 sees [0,1,2], etc.
const mask    = toAdditiveMask(causalMask(seqLen));
const maskTV  = new TensorValue(mask, "mask");
const maskedOut = scaledDotProductAttention(Q, K, V, maskTV);
console.log("masked attn output shape:", maskedOut.data.shape, "  expected: [5, 8]");

// ─── E3: Attention weights sum to 1 ──────────────────────────────────────────
// After softmax, each row of the attention weight matrix sums to 1.
// We can't read the internal weights unless SelfAttention exposes them.
// Instead: a * Q + b * (1-a) * Q should equal Q when a=1.
// Simpler check: feed Q=K=V=identity and verify output ≈ input.
const I = createTensor([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1], [4, 4]);
const Qid = new TensorValue(I, "Qid");
const Kid = new TensorValue(I, "Kid");
const Vid = new TensorValue(I, "Vid");
const outId = scaledDotProductAttention(Qid, Kid, Vid);
console.log("\nWith Q=K=V=I, row 0:", Array.from(outId.data.data.slice(0, 4)).map(v => v.toFixed(4)));
// Should be near [0.25, 0.25, 0.25, 0.25] (uniform distribution over all rows)

// ─── E4: SelfAttention layer ─────────────────────────────────────────────────
const sa = new SelfAttention(16);   // dModel=16
const x  = new TensorValue(randn([6, 16]), "x");
const saOut = sa.forward(x);
console.log("\nSelfAttention output shape:", saOut.data.shape, "  expected: [6, 16]");

// ─── E5: Gradient through attention ──────────────────────────────────────────
saOut.backward();
console.log("Input grad shape:", x.grad?.shape, "  expected: [6, 16]");

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: verify that attention is scale-invariant in the following sense:
//   scaledDotProductAttention(Q, K, V) ≈ scaledDotProductAttention(2Q, 2K, V)
//   because the scaling factor cancels.
