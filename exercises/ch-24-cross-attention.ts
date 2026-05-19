/**
 * EXERCISES — Ch 24: Cross-Attention
 * ════════════════════════════════════
 * Prereq : src/nn/attention.ts (CrossAttention) implemented
 * Run    : bun run exercises/ch-24-cross-attention.ts
 *
 * Cross-attention lets the decoder query the encoder's output.
 * Q comes from the decoder; K, V come from the encoder.
 * This is the bridge between encoder and decoder (Ch 27).
 */
import { CrossAttention }   from "../src/nn/attention.ts";
import { TensorValue }      from "../src/autograd/grad.ts";
import { randn }            from "../src/tensor/creation.ts";

// ─── E1: Shape contract ───────────────────────────────────────────────────────
// Encoder output: [batch, srcLen, dModel]
// Decoder state : [batch, tgtLen, dModel]
// Cross-attention output: [batch, tgtLen, dModel]  ← query length preserved
const dModel  = 32, numHeads = 4;
const srcLen  = 10, tgtLen = 6;
const batch   = 2;

const crossAttn  = new CrossAttention(dModel, numHeads);
const encOut     = new TensorValue(randn([batch, srcLen, dModel]), "encOut");
const decState   = new TensorValue(randn([batch, tgtLen, dModel]), "decState");
const out        = crossAttn.forward(decState, encOut);
console.log("cross-attn output shape:", out.data.shape, "  expected: [2, 6, 32]");

// ─── E2: Query length drives output length ────────────────────────────────────
// Changing tgtLen (decoder sequence length) changes output rows.
// Changing srcLen (encoder sequence length) does NOT change output rows.
const decState2 = new TensorValue(randn([batch, 4, dModel]));   // tgtLen=4
const out2      = crossAttn.forward(decState2, encOut);          // srcLen still 10
console.log("\noutput shape with tgtLen=4:", out2.data.shape, "  expected: [2, 4, 32]");

// ─── E3: Backward pass ────────────────────────────────────────────────────────
out.backward();
console.log("decState grad shape:", decState.grad?.shape, "  expected: [2, 6, 32]");
console.log("encOut   grad shape:", encOut.grad?.shape,   "  expected: [2, 10, 32]");

// ─── E4: Cross-attention vs self-attention ───────────────────────────────────
// In self-attention, Q=K=V (all come from the same sequence).
// In cross-attention, Q≠K=V (Q from decoder, K/V from encoder).
// TODO: demonstrate this by passing encOut as both query and memory —
//       it should behave identically to SelfAttention.
import { MultiHeadAttention } from "../src/nn/attention.ts";
const selfAttn = new MultiHeadAttention(dModel, numHeads);
const crossSelf = new CrossAttention(dModel, numHeads);
// Use same weights ... this is conceptual; just verify shapes match.
const selfOut  = selfAttn.forward(encOut);
const crossOut = crossAttn.forward(encOut, encOut);
console.log("\nself-attn  output shape:", selfOut.data.shape);
console.log("cross-attn output shape:", crossOut.data.shape);
// Both should be [2, 10, 32].

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: in translation, encoder sees English (srcLen tokens),
//       decoder generates French (tgtLen tokens).
//       Model: srcLen=15, tgtLen=12, dModel=64, numHeads=8.
//       Verify all shapes.
