/**
 * EXERCISES — Ch 26: Encoder Block
 * ══════════════════════════════════
 * Prereq : src/nn/transformer.ts (EncoderBlock) implemented
 * Run    : bun run exercises/ch-26-encoder-block.ts
 *
 * Encoder block = LayerNorm → MultiHeadAttention → residual
 *               + LayerNorm → FFN → residual
 * (Pre-LN variant used by most modern transformers.)
 */
import { EncoderBlock }  from "../src/nn/transformer.ts";
import { TensorValue }   from "../src/autograd/grad.ts";
import { randn }         from "../src/tensor/creation.ts";

// ─── E1: Forward pass shape ───────────────────────────────────────────────────
const enc    = new EncoderBlock(32, 4, 128);   // dModel, numHeads, dFf
const x      = new TensorValue(randn([2, 8, 32]), "x");
const out    = enc.forward(x);
console.log("encoder block output shape:", out.data.shape, "  expected: [2, 8, 32]");

// ─── E2: With padding mask ────────────────────────────────────────────────────
import { binaryMaskFromIds, toAdditiveMask, expandPaddingMask } from "../src/tokenizer/masks.ts";
const paddedIds   = [1, 2, 3, 0, 0, 0, 0, 0];   // 3 real tokens, 5 padding
const binaryMask  = binaryMaskFromIds(paddedIds);
const additiveMask = toAdditiveMask(binaryMask);
const expandedMask = expandPaddingMask(additiveMask, 2, 4);   // batch=2, heads=4
const maskTV = new TensorValue(expandedMask, "mask");
const maskedOut = enc.forward(x, maskTV);
console.log("\nmasked encoder output shape:", maskedOut.data.shape, "  expected: [2, 8, 32]");

// ─── E3: Backward through the entire block ───────────────────────────────────
maskedOut.backward();
console.log("Input grad shape:", x.grad?.shape, "  expected: [2, 8, 32]");

// ─── E4: Stack two encoder blocks ─────────────────────────────────────────────
const enc2 = new EncoderBlock(32, 4, 128);
const out2 = enc2.forward(enc.forward(x));
console.log("\nStacked encoder output shape:", out2.data.shape, "  expected: [2, 8, 32]");

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: count total parameters in one EncoderBlock with dModel=512, numHeads=8, dFf=2048.
//   Expected: 2 * LayerNorm + MultiHeadAttention + FFN
//           = 2*(512+512) + 4*(512²+512) + 2*(512*2048+2048+2048*512+512)
