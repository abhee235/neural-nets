/**
 * EXERCISES — Ch 29: Full Transformer
 * ═════════════════════════════════════
 * Prereq : src/nn/transformer.ts (Transformer) + utils/data.ts implemented
 * Run    : bun run exercises/ch-29-full-transformer.ts
 *
 * The complete encoder-decoder transformer: encode source → decode to target.
 * We train it on the sequence reversal task from Ch 28.
 */
import { Transformer, TransformerConfig } from "../src/nn/transformer.ts";
import { TensorValue }   from "../src/autograd/grad.ts";
import { createTensor }  from "../src/tensor/creation.ts";
import { Adam }          from "../src/optim/adam.ts";
import { makeReversalDataset, splitDataset, maskedCrossEntropy } from "../src/utils/data.ts";
import { CharTokenizer, buildVocab } from "../src/tokenizer/char.ts";

// ─── E1: Build and inspect the model ─────────────────────────────────────────
const config: TransformerConfig = {
  vocabSize: 20, dModel: 32, numHeads: 4,
  numEncoderLayers: 2, numDecoderLayers: 2,
  dFf: 64, maxSeqLen: 20, dropout: 0.0,
};
const model  = new Transformer(config);
const params = model.parameters();
const nParams = params.reduce((s, p) => s + p.data.size, 0);
console.log("Total parameters:", nParams);

// ─── E2: Forward pass ─────────────────────────────────────────────────────────
const srcIds = createTensor([3, 4, 5, 6, 7], [1, 5]);
const tgtIds = createTensor([3, 7, 6, 5, 4], [1, 5]);   // reversed
const logits = model.forward(srcIds, tgtIds);
console.log("\nlogits shape:", logits.data.shape, "  expected: [1, 5, 20]");

// ─── E3: Loss on one example ──────────────────────────────────────────────────
const targets = [7, 6, 5, 4, 0];   // shifted right, last = PAD
const loss = maskedCrossEntropy(logits, targets);
console.log("initial loss:", loss.data.data[0].toFixed(4), "  (random ≈ log(20) ≈ 3.0)");

// ─── E4: Train on reversal task ───────────────────────────────────────────────
const dataset = makeReversalDataset(200, 5);
const { train } = splitDataset(dataset, 0.8, 0.1, 0.1);
const opt = new Adam(model.parameters(), 3e-4);

for (let epoch = 0; epoch < 5; epoch++) {
  let totalLoss = 0;
  for (const ex of train.slice(0, 20)) {   // mini-batch of 20 for speed
    const src   = createTensor(ex.src, [1, ex.src.length]);
    const tgt   = createTensor(ex.tgt, [1, ex.tgt.length]);
    const logit = model.forward(src, tgt);
    const tgtShift = [...ex.tgt.slice(1), 0];
    const l = maskedCrossEntropy(logit, tgtShift);
    l.backward();
    opt.step();
    opt.zeroGrad();
    totalLoss += l.data.data[0]!;
  }
  console.log(\`epoch \${epoch+1}: avg loss = \${(totalLoss/20).toFixed(4)}\`);
}

// ─── E5: generate() ──────────────────────────────────────────────────────────
import { generate } from "../src/nn/transformer.ts";
const srcTest = [3, 4, 5, 6, 7];
const prediction = generate(model, srcTest, 5, 0);   // maxLen=5, BOS=0
console.log("\nSource  :", srcTest);
console.log("Predicted:", prediction, "  (target: [7,6,5,4,3])");

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: train for 50 epochs and compute accuracy on the test set.
//       Does the model learn to reverse sequences perfectly?
