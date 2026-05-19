/**
 * EXERCISES — Ch 18: Token Embeddings
 * ═════════════════════════════════════
 * Prereq : src/nn/embedding.ts + tokenizer/char.ts implemented
 * Run    : bun run exercises/ch-18-embeddings.ts
 *
 * Embeddings map discrete token IDs to continuous vectors.
 * The embedding table is a [vocabSize, dModel] learned matrix —
 * lookup = matMul(one-hot, W_embed).
 */
import { Embedding }    from "../src/nn/embedding.ts";
import { TensorValue }  from "../src/autograd/grad.ts";
import { createTensor } from "../src/tensor/creation.ts";

// ─── E1: Basic lookup ────────────────────────────────────────────────────────
const embed = new Embedding(10, 8);    // vocabSize=10, dModel=8
const ids   = createTensor([0, 2, 5, 9], [4]);
const vecs  = embed.forward(ids);
console.log("embedding output shape:", vecs.data.shape, "  expected: [4, 8]");

// ─── E2: Same ID → same vector ───────────────────────────────────────────────
// Token 3 should always map to the same embedding vector.
const ids2  = createTensor([3, 3, 3], [3]);
const vecs2 = embed.forward(ids2);
const row0  = Array.from(vecs2.data.data.slice(0, 8));
const row1  = Array.from(vecs2.data.data.slice(8, 16));
const same  = row0.every((v, i) => v === row1[i]);
console.log("same ID → same vector:", same, "  expected: true");

// ─── E3: Gradient flows through embedding ────────────────────────────────────
// Backprop should update only the rows that were looked up.
const ids3  = createTensor([1, 3], [2]);
const vecs3 = embed.forward(ids3);
vecs3.backward();
console.log("\nembedding weight grad shape:", embed.parameters()[0]!.grad?.shape);
// expected: [vocabSize, dModel]

// ─── E4: Cosine similarity of embeddings ─────────────────────────────────────
// Randomly initialised embeddings have no structure.
// After training, semantically similar tokens should be closer.
// Here we verify the math of cosine similarity.
function cosineSim(a: Float64Array, b: Float64Array): number {
  const dot  = Array.from(a).reduce((s, v, i) => s + v * b[i]!, 0);
  const normA = Math.sqrt(Array.from(a).reduce((s, v) => s + v * v, 0));
  const normB = Math.sqrt(Array.from(b).reduce((s, v) => s + v * v, 0));
  return dot / (normA * normB);
}
const v0 = embed.forward(createTensor([0], [1])).data.data;
const v1 = embed.forward(createTensor([1], [1])).data.data;
console.log("\ncos(embed[0], embed[1]):", cosineSim(v0, v1).toFixed(4), "  (random ≈ near 0)");

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: encode "hello" with CharTokenizer, pass through Embedding, print shape.
//   Verify output shape: [len("hello"), dModel].
