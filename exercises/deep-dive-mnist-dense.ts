/**
 * DEEP DIVE — A three-hidden-layer dense network on MNIST
 * ════════════════════════════════════════════════════════
 * Prereq : everything through Ch 15
 * Run    : bun run exercises/deep-dive-mnist-dense.ts
 *
 * REFERENCE: docs/deep-dives/ch-15-mnist-dense-network.md
 *
 * WHY THIS EXISTS: every chapter so far has trained on a toy — four rows of
 * XOR, a thousand points in a circle. This is the test of whether the
 * library we built is a real one: 784 inputs, 10 classes, three hidden
 * layers, 111,146 parameters, and a dataset nobody can solve by inspection.
 *
 * NOTHING NEW IS IMPORTED. Every function called here was written in
 * Chapters 01-15. If this runs, the library is finished enough to be useful.
 *
 * THE DATA: a stratified subset — 200 training and 50 test images per digit
 * — vendored at data/mnist/subset.bin.gz and produced by
 * scripts/make_mnist_subset.py. Small on purpose: it trains in about a
 * minute, and the point is the machinery, not the leaderboard.
 */
import { Linear } from "../src/nn/linear.ts";
import { relu } from "../src/nn/activations.ts";
import { crossEntropyFromLogits } from "../src/nn/losses.ts";
import { Adam } from "../src/optim/adam.ts";
import { TensorValue } from "../src/autograd/grad.ts";
import { createTensor, type Tensor } from "../src/tensor/types.ts";
import { argmax } from "../src/tensor/reduce.ts";

const PIXELS = 784;
const CLASSES = 10;

// ─── loading ─────────────────────────────────────────────────────────────────

interface Split {
  /** [count, 784], pixels scaled to 0..1 */
  images: Tensor;
  /** [count, 10], one-hot — the shape crossEntropyFromLogits expects */
  oneHot: Tensor;
  /** the plain digit for each row, for accuracy and the confusion matrix */
  labels: Uint8Array;
  count: number;
}

/**
 * Read the vendored blob. Format is documented in scripts/make_mnist_subset.py:
 * a 12-byte header, then train images, train labels, test images, test labels.
 *
 * Pixels arrive as bytes 0..255 and are divided by 255. That scaling is not
 * cosmetic — Ch 13's initialisation argument assumed inputs of roughly unit
 * size, and feeding raw 0..255 values into `he` init makes the first
 * pre-activations ~255x too large, straight into the flat region of
 * everything downstream.
 */
function load(): { train: Split; test: Split } {
  const gz = new Uint8Array(require("node:fs").readFileSync(
    new URL("../data/mnist/subset.bin.gz", import.meta.url)
  ));
  const blob = Bun.gunzipSync(gz);

  const magic = String.fromCharCode(...blob.subarray(0, 4));
  if (magic !== "MNSB") throw new Error(`bad magic ${magic} — is the data file intact?`);
  const view = new DataView(blob.buffer, blob.byteOffset, blob.byteLength);
  const nTrain = view.getUint32(4, true);
  const nTest = view.getUint32(8, true);

  let at = 12;
  const cut = (count: number): Split => {
    const pixels = blob.subarray(at, at + count * PIXELS); at += count * PIXELS;
    const labels = blob.subarray(at, at + count); at += count;

    const scaled = new Float64Array(count * PIXELS);
    for (let i = 0; i < scaled.length; i++) scaled[i] = pixels[i]! / 255;

    // one-hot: a 1 in the column of the true digit, 0 everywhere else
    const oneHot = new Float64Array(count * CLASSES);
    for (let i = 0; i < count; i++) oneHot[i * CLASSES + labels[i]!] = 1;

    return {
      images: { data: scaled, shape: [count, PIXELS], ndim: 2, size: scaled.length },
      oneHot: { data: oneHot, shape: [count, CLASSES], ndim: 2, size: oneHot.length },
      labels: new Uint8Array(labels),
      count,
    };
  };
  return { train: cut(nTrain), test: cut(nTest) };
}

// ─── batching ────────────────────────────────────────────────────────────────

/** Seeded shuffle, so a rerun reproduces the same curve. Fisher-Yates. */
function shuffledIndices(count: number, random: () => number): Uint32Array {
  const order = new Uint32Array(count);
  for (let i = 0; i < count; i++) order[i] = i;
  for (let i = count - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    const t = order[i]!; order[i] = order[j]!; order[j] = t;
  }
  return order;
}

function makeRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

/** Gather the given rows out of a [n, width] tensor into a [rows.length, width] one. */
function gatherRows(source: Tensor, rows: Uint32Array, from: number, size: number): Tensor {
  const width = source.shape[1]!;
  const out = new Float64Array(size * width);
  for (let r = 0; r < size; r++) {
    const src = rows[from + r]! * width;
    for (let c = 0; c < width; c++) out[r * width + c] = source.data[src + c]!;
  }
  return { data: out, shape: [size, width], ndim: 2, size: out.length };
}

// ─── the model ───────────────────────────────────────────────────────────────

/**
 * 784 → 128 → 64 → 32 → 10, with `relu` between each pair.
 *
 * The funnel is conventional and the reasoning is in the doc: each layer has
 * to describe its input with fewer numbers than it received, so it must keep
 * what distinguishes digits and discard the rest. The last layer emits ten
 * RAW LOGITS — no softmax, because crossEntropyFromLogits applies it
 * internally (Ch 12's log-sum-exp).
 *
 * Default init is `he`, which is what Ch 13 derived for relu layers.
 */
function makeModel() {
  const layers = [
    new Linear(PIXELS, 128),
    new Linear(128, 64),
    new Linear(64, 32),
    new Linear(32, CLASSES),
  ];
  return {
    layers,
    forward(x: TensorValue): TensorValue {
      let h = x;
      for (let i = 0; i < layers.length - 1; i++) h = relu(layers[i]!.forward(h));
      return layers[layers.length - 1]!.forward(h);   // logits, deliberately bare
    },
    // Ch 13's contract, four layers deep now. The optimizer still sees a flat list.
    parameters: () => layers.flatMap((l) => l.parameters()),
  };
}

// ─── evaluation ──────────────────────────────────────────────────────────────

/** Fraction correct: the highest logit is the prediction. */
function accuracy(logits: Tensor, labels: Uint8Array, from = 0): number {
  const predicted = argmax(logits, 1);
  let correct = 0;
  for (let i = 0; i < predicted.size; i++) {
    if (predicted.data[i] === labels[from + i]) correct++;
  }
  return correct / predicted.size;
}

/** counts[actual][predicted] — where the mistakes actually go. */
function confusion(logits: Tensor, labels: Uint8Array): number[][] {
  const predicted = argmax(logits, 1);
  const counts = Array.from({ length: CLASSES }, () => new Array(CLASSES).fill(0));
  for (let i = 0; i < predicted.size; i++) {
    counts[labels[i]!]![predicted.data[i]!]! += 1;
  }
  return counts;
}

function printConfusion(counts: number[][]): void {
  console.log("            predicted");
  console.log("        " + Array.from({ length: CLASSES }, (_, d) => String(d).padStart(4)).join(""));
  for (let actual = 0; actual < CLASSES; actual++) {
    const row = counts[actual]!;
    const cells = row.map((n, predictedDigit) =>
      (n === 0 ? "   ." : String(n).padStart(4)) + (predictedDigit === actual ? "" : "")).join("");
    console.log(`  ${actual} |` + cells + `   ${((row[actual]! / 50) * 100).toFixed(0)}%`);
  }
  console.log("  ^actual                                          per-digit recall");
}

/** Draw one image as characters, so a misclassification can be judged by eye. */
function printDigit(images: Tensor, index: number): void {
  const ramp = " .:-=+*#%@";
  for (let row = 0; row < 28; row += 2) {
    let line = "    ";
    for (let col = 0; col < 28; col++) {
      const v = images.data[index * PIXELS + row * 28 + col]!;
      line += ramp[Math.min(ramp.length - 1, Math.floor(v * ramp.length))];
    }
    console.log(line);
  }
}

// ─── run ─────────────────────────────────────────────────────────────────────

const { train, test } = load();
console.log(`  loaded ${train.count} training and ${test.count} test images, ${PIXELS} pixels each`);

const model = makeModel();
const params = model.parameters();
console.log(`  ${params.reduce((s, p) => s + p.data.size, 0).toLocaleString()} parameters in ${params.length} tensors`);
console.log(`  shapes: ${params.map((p) => `[${p.data.shape}]`).join(" ")}\n`);

const optimizer = new Adam(params, 1e-3);
const random = makeRandom(7);
const BATCH = 64;
const EPOCHS = 30;

console.log("  epoch     loss   train acc   test acc");
const started = performance.now();
for (let epoch = 1; epoch <= EPOCHS; epoch++) {
  const order = shuffledIndices(train.count, random);
  let lossSum = 0, batches = 0;

  for (let start = 0; start + BATCH <= train.count; start += BATCH) {
    const x = gatherRows(train.images, order, start, BATCH);
    const y = gatherRows(train.oneHot, order, start, BATCH);

    optimizer.zeroGrad();                                    // 1. forget
    const logits = model.forward(new TensorValue(x));        // 2. guess
    const loss = crossEntropyFromLogits(logits, y);          // 3. score
    loss.backward();                                         // 4. blame
    optimizer.step();                                        // 5. move

    lossSum += loss.data.data[0]!; batches++;
  }

  if (epoch % 5 === 0 || epoch === 1) {
    const trainLogits = model.forward(new TensorValue(train.images)).data;
    const testLogits = model.forward(new TensorValue(test.images)).data;
    console.log(`  ${String(epoch).padStart(5)}   ${(lossSum / batches).toFixed(4)}     ` +
      `${(accuracy(trainLogits, train.labels) * 100).toFixed(1)}%      ` +
      `${(accuracy(testLogits, test.labels) * 100).toFixed(1)}%`);
  }
}
console.log(`\n  trained in ${((performance.now() - started) / 1000).toFixed(1)} s\n`);

const finalTest = model.forward(new TensorValue(test.images)).data;
// Baselines measured on this exact subset, same optimizer, same 30 epochs:
//   chance                                     10.0%
//   linear only, 784 -> 10, 7,850 params       88.0%
//   one hidden layer, 784 -> 128 -> 10         90.0%
// Three hidden layers buy about three points over no hidden layer at all.
// Depth is not doing the heavy lifting here — the 2,000-image training set
// is the binding constraint, and the 100% train accuracy above says so.
console.log(`  final test accuracy ${(accuracy(finalTest, test.labels) * 100).toFixed(1)}%  ` +
  `(chance 10%, linear-only 88.0%, one hidden layer 90.0%)\n`);
printConfusion(confusion(finalTest, test.labels));

// show the first thing it got wrong — the interesting output of any classifier
const predicted = argmax(finalTest, 1);
for (let i = 0; i < test.count; i++) {
  if (predicted.data[i] !== test.labels[i]) {
    console.log(`\n  first mistake: image ${i} is a ${test.labels[i]}, called a ${predicted.data[i]}`);
    printDigit(test.images, i);
    break;
  }
}
