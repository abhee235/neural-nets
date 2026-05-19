/**
 * EXERCISES — Ch 21: Mask Cookbook
 * ══════════════════════════════════
 * Prereq : src/tokenizer/masks.ts + char.ts implemented
 * Run    : bun run exercises/ch-21-masks.ts
 *
 * Masks are boolean tensors that control which tokens can attend to which.
 * Two types: causal masks (decoder self-attention) + padding masks (variable-length inputs).
 */
import {
  causalMask, expandCausalMask,
  binaryMaskFromIds, toAdditiveMask, expandPaddingMask,
  combineMasks, makeDecoderSelfAttentionMask,
  MASK_VALUE
} from "../src/tokenizer/masks.ts";
import { PAD_ID } from "../src/tokenizer/char.ts";

// ─── E1: Causal mask ─────────────────────────────────────────────────────────
// A 4×4 causal mask should be lower-triangular: position i can only see j ≤ i.
const cm = causalMask(4);
console.log("causal mask (4×4):");
for (let i = 0; i < 4; i++) {
  console.log(Array.from(cm.data.slice(i * 4, i * 4 + 4)));
}
// row 0: [true, false, false, false]
// row 1: [true, true,  false, false]
// etc.

// ─── E2: Additive mask ────────────────────────────────────────────────────────
// Before softmax, masked positions get -1e9 added → exp(-1e9) ≈ 0.
const additive = toAdditiveMask(cm);
console.log("\nadditive mask row 0:", Array.from(additive.data.slice(0, 4)));
// [0, -1e9, -1e9, -1e9]

// ─── E3: Padding mask ─────────────────────────────────────────────────────────
// Token IDs with PAD_ID should be masked out.
const ids = [5, 3, PAD_ID, PAD_ID];   // 2 real tokens, 2 padding
const bm  = binaryMaskFromIds(ids);
console.log("\nbinary padding mask:", Array.from(bm.data));
// [true, true, false, false]

// ─── E4: Combine masks ────────────────────────────────────────────────────────
// Decoder self-attention uses causal + padding mask combined.
const paddingIds  = [5, 3, PAD_ID, PAD_ID];
const decoderMask = makeDecoderSelfAttentionMask(paddingIds);
console.log("\ndecoder self-attn mask (4×4):");
for (let i = 0; i < 4; i++) {
  console.log(Array.from(decoderMask.data.slice(i * 4, i * 4 + 4)));
}
// Position 2 and 3 are padding — no token should attend to them.

// ─── E5: Expand for batch + heads ────────────────────────────────────────────
// Attention expects mask shape [batch, numHeads, seqLen, seqLen].
const expanded = expandCausalMask(cm, 2, 4);   // batch=2, numHeads=4
console.log("\nexpanded mask shape:", expanded.data.shape, "  expected: [2,4,4,4]");

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: verify that after applying the causal mask and softmax,
//   position 0 can only attend to position 0 (attention weight ≈ 1.0 for pos 0→0).
