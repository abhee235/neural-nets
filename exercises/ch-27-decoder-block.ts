/**
 * EXERCISES — Ch 27: Decoder Block
 * ══════════════════════════════════
 * Prereq : src/nn/transformer.ts (DecoderBlock) implemented
 * Run    : bun run exercises/ch-27-decoder-block.ts
 *
 * Decoder block = masked self-attention + cross-attention + FFN (each with LN + residual).
 * The causal mask in self-attention prevents the decoder from seeing future tokens.
 */
import { DecoderBlock }  from "../src/nn/transformer.ts";
import { TensorValue }   from "../src/autograd/grad.ts";
import { randn }         from "../src/tensor/creation.ts";
import { causalMask, toAdditiveMask, expandCausalMask } from "../src/tokenizer/masks.ts";

// ─── E1: Forward pass shape ───────────────────────────────────────────────────
const dec     = new DecoderBlock(32, 4, 128);
const srcLen  = 10, tgtLen = 6, batch = 2;
const encOut  = new TensorValue(randn([batch, srcLen, 32]), "encOut");
const decIn   = new TensorValue(randn([batch, tgtLen, 32]), "decIn");
const out     = dec.forward(decIn, encOut);
console.log("decoder block output shape:", out.data.shape, "  expected: [2, 6, 32]");

// ─── E2: With causal mask on decoder self-attention ──────────────────────────
const cm        = toAdditiveMask(causalMask(tgtLen));
const causalTV  = new TensorValue(expandCausalMask(cm, batch, 4), "causalMask");
const maskedOut = dec.forward(decIn, encOut, causalTV);
console.log("masked decoder output:", maskedOut.data.shape, "  expected: [2, 6, 32]");

// ─── E3: Encoder output gradient ─────────────────────────────────────────────
maskedOut.backward();
console.log("encOut grad shape:", encOut.grad?.shape, "  expected: [2, 10, 32]");
console.log("decIn  grad shape:", decIn.grad?.shape,  "  expected: [2, 6, 32]");

// ─── E4: Auto-regressive generation preview ──────────────────────────────────
// In inference we run the decoder one token at a time:
// step 1: tgtLen=1, step 2: tgtLen=2, …
for (let t = 1; t <= 4; t++) {
  const tgt   = new TensorValue(randn([1, t, 32]));
  const enc_o = new TensorValue(randn([1, srcLen, 32]));
  const o     = dec.forward(tgt, enc_o);
  console.log(\`  tgtLen=\${t} → output shape:\`, o.data.shape);
}

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: verify that the decoder block has 3 sub-layers:
//   masked self-attention, cross-attention, FFN.
//   Count parameters and compare to the formula.
