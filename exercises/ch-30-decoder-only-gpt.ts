/**
 * EXERCISES — Ch 30: Decoder-Only GPT
 * ═════════════════════════════════════
 * Prereq : src/nn/gpt.ts + all prior modules implemented
 * Run    : bun run exercises/ch-30-decoder-only-gpt.ts
 *
 * GPT removes the encoder entirely. It's trained on a single objective:
 * predict the next token.  The causal mask ensures it cannot cheat by
 * looking at future tokens during training.
 */
import { GPT, GPTConfig, makeLanguageModelingBatch, generateText } from "../src/nn/gpt.ts";
import { CharTokenizer, buildVocab } from "../src/tokenizer/char.ts";
import { Adam }         from "../src/optim/adam.ts";
import { maskedCrossEntropy } from "../src/utils/data.ts";

// ─── E1: Build a tiny GPT ─────────────────────────────────────────────────────
const config: GPTConfig = {
  vocabSize: 50, dModel: 32, numHeads: 4,
  numLayers: 2, dFf: 64, maxSeqLen: 64, dropout: 0.0,
};
const gpt    = new GPT(config);
const nParam = gpt.parameters().reduce((s, p) => s + p.data.size, 0);
console.log("GPT parameter count:", nParam);

// ─── E2: Forward pass ─────────────────────────────────────────────────────────
import { createTensor } from "../src/tensor/creation.ts";
const ids    = createTensor([1, 5, 12, 7, 3], [1, 5]);   // [batch=1, seqLen=5]
const logits = gpt.forward(ids);
console.log("logits shape:", logits.data.shape, "  expected: [1, 5, 50]");

// ─── E3: Language modeling batch ─────────────────────────────────────────────
// The language modeling objective uses right-shifted targets.
const tokenSequence = [1, 2, 3, 4, 5, 6, 7, 8];
const { input: lmInput, target: lmTarget } = makeLanguageModelingBatch(tokenSequence, 0);
console.log("\nLM input :", lmInput, "  expected: [1,2,3,4,5,6,7,8]");
console.log("LM target:", lmTarget, "  expected: [2,3,4,5,6,7,8,0]");

// ─── E4: Train on "hello world" repeated ─────────────────────────────────────
const trainText = "hello world! hello world! hello world!";
const vocab     = buildVocab(trainText);
const tokenizer = new CharTokenizer(vocab);
const gptSmall: GPTConfig = {
  vocabSize: vocab.size, dModel: 32, numHeads: 4,
  numLayers: 2, dFf: 64, maxSeqLen: 40, dropout: 0.0,
};
const model = new GPT(gptSmall);
const opt   = new Adam(model.parameters(), 1e-3);

const trainIds = tokenizer.encode(trainText);
for (let epoch = 0; epoch < 100; epoch++) {
  const { input: inp, target: tgt } = makeLanguageModelingBatch(trainIds, 0);
  const inpT   = createTensor(inp, [1, inp.length]);
  const logit  = model.forward(inpT);
  const l      = maskedCrossEntropy(logit, tgt);
  l.backward();
  opt.step();
  opt.zeroGrad();
  if (epoch % 20 === 0)
    console.log(\`epoch \${epoch}: loss = \${l.data.data[0]!.toFixed(4)}\`);
}

// ─── E5: Text generation (autoregressive) ────────────────────────────────────
const prompt    = tokenizer.encode("hello");
const generated = generateText(model, prompt, 20, { temperature: 0.8 });
console.log("\nGenerated:", tokenizer.decode(generated));

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: after training, measure perplexity on a held-out sentence.
//       Try greedy decoding (temperature=0) vs temperature=1.2.
//       What is the difference in output diversity?
