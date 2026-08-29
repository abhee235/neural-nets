/**
 * DEEP DIVE — A three-hidden-layer dense network on MNIST
 * ════════════════════════════════════════════════════════
 * Part 3 of 6: Neural Net Primitives (extends Chapter 15)
 *
 * WHAT WE'RE BUILDING:  a real classifier — 784 pixels in, ten digits out,
 *                       three hidden layers, 111,146 parameters, trained by
 *                       the same five lines as XOR.
 * WHY IT MATTERS:       every chapter so far trained on a toy you could hold
 *                       in your head. This is the test of whether the LIBRARY
 *                       is real, not whether neural networks are.
 * WHAT THIS UNLOCKS:    → Ch 20 (LayerNorm & Dropout) — the 100%/91% gap you
 *                       are about to produce is exactly what dropout is for.
 *
 * REFERENCE: docs/deep-dives/ch-15-mnist-dense-network.md
 *
 * ──────────────────────────────────────────────────────────────────────────
 * WHY THIS FILE EXISTS
 * ──────────────────────────────────────────────────────────────────────────
 * NOTHING NEW IS IMPORTED. Linear (Ch 13), relu (Ch 11),
 * crossEntropyFromLogits (Ch 12), Adam (Ch 14), argmax (Ch 05). If this file
 * runs, the library is finished enough to be useful.
 *
 * What IS new is everything around the model: reading bytes off disk,
 * scaling them, cutting them into batches, and shuffling those batches. None
 * of that is machine learning, and all of it is where real projects break.
 *
 * ──────────────────────────────────────────────────────────────────────────
 * THE RUNNING EXAMPLE — used in every trace below
 * ──────────────────────────────────────────────────────────────────────────
 * The vendored subset at data/mnist/subset.bin.gz:
 *
 *     2000 training images   200 of each digit
 *      500 test images        50 of each digit
 *      784 pixels each        28 × 28, one byte per pixel, 0 = paper 255 = ink
 *
 * One batch of 64 travels like this — the whole network in four shapes:
 *
 *     [64, 784]  --W₁[128,784]-->  [64, 128]  --relu-->  [64, 128]
 *                --W₂ [64,128]-->  [64,  64]  --relu-->  [64,  64]
 *                --W₃  [32,64]-->  [64,  32]  --relu-->  [64,  32]
 *                --W₄  [10,32]-->  [64,  10]   logits, NO activation
 *
 * Parameters, which are not distributed how you would guess:
 *
 *     layer 1   784→128   100,352 + 128  =  100,480     90.4%
 *     layer 2   128→ 64     8,192 +  64  =    8,256      7.4%
 *     layer 3    64→ 32     2,048 +  32  =    2,080      1.9%
 *     layer 4    32→ 10       320 +  10  =      330      0.3%
 *                                           ─────────
 *                                            111,146
 *
 * WHAT A CORRECT IMPLEMENTATION PRODUCES (machine-verified, 30 epochs,
 * Adam at 1e-3, batch 64 — your numbers will differ by a few tenths because
 * Linear's init is random, but not by more than that):
 *
 *     epoch     loss   train acc   test acc
 *         1   1.6025      77.0%      71.8%
 *         5   0.1787      97.1%      88.2%
 *        15   0.0127     100.0%      90.8%
 *        30   0.0018     100.0%      91.4%
 *
 *     ~83 seconds. 930 optimizer steps.
 *
 * If your test accuracy lands near 91% and your train accuracy reaches 100%,
 * everything below is right. If train accuracy sticks near 10%, something is
 * severed from the graph — check the PITFALLs.
 *
 * ──────────────────────────────────────────────────────────────────────────
 * WHAT IS GIVEN, AND WHY
 * ──────────────────────────────────────────────────────────────────────────
 * makeRandom, printConfusion and printDigit are written for you. They are a
 * linear congruential generator and two functions that turn numbers into
 * characters. Neither teaches anything about neural networks, and debugging
 * an ASCII ramp is not the point of this exercise.
 *
 * Everything else is yours.
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
const BATCH = 64;
const EPOCHS = 30;

/** One half of the dataset, in the four forms the training loop needs. */
interface Split {
  /** [count, 784], pixels already scaled to 0..1 */
  images: Tensor;
  /** [count, 10], one-hot — the shape crossEntropyFromLogits expects */
  oneHot: Tensor;
  /** the plain digit for each row, for accuracy and the confusion matrix */
  labels: Uint8Array;
  count: number;
}

// ═══ GIVEN — plumbing, not learning ═════════════════════════════════════════

/** Seeded uniform generator, so a rerun reproduces the same curve. */
function makeRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

/** counts[actual][predicted], printed as a grid with per-digit recall. */
function printConfusion(counts: number[][]): void {
  console.log("            predicted");
  console.log("        " + Array.from({ length: CLASSES }, (_, d) => String(d).padStart(4)).join(""));
  for (let actual = 0; actual < CLASSES; actual++) {
    const row = counts[actual]!;
    const cells = row.map((n) => (n === 0 ? "   ." : String(n).padStart(4))).join("");
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

// ═══ YOURS — everything below ═══════════════════════════════════════════════

/**
 * Read data/mnist/subset.bin.gz and return both splits.
 *
 * ── THE FILE FORMAT (documented in scripts/make_mnist_subset.py) ───────────
 * The file is gzipped. After `Bun.gunzipSync` you hold a flat Uint8Array:
 *
 *     offset  0    4 bytes    magic "MNSB"        — check it, fail loudly
 *     offset  4    4 bytes    nTrain, uint32 LITTLE-endian   → 2000
 *     offset  8    4 bytes    nTest,  uint32 little-endian   →  500
 *     offset 12    nTrain*784 train images, one byte per pixel
 *     ...          nTrain     train labels, one byte each, 0-9
 *     ...          nTest*784  test images
 *     ...          nTest      test labels
 *
 * Read the two counts with a DataView and `getUint32(offset, true)` — the
 * `true` is little-endian. Then walk a cursor forward through the four
 * blocks in order. Do not hardcode 2000 and 500; read them.
 *
 * ── STEP 1: SCALING — the one line that is load-bearing ────────────────────
 * Pixels are stored 0..255 and must be USED as 0..1:
 *
 *     scaled[i] = raw[i] / 255
 *
 * This is not tidying. Ch 13 derived the `he` scale √(2/inputDim) assuming
 * inputs are roughly unit-sized — the whole argument was about controlling
 * the variance of a sum of 784 terms. Feed raw bytes and every term is up to
 * 255× too large, so the first pre-activations are enormous and every layer
 * after sits in the flat tail. It will not crash. It will just learn badly,
 * for a reason nothing in the output points at.
 *
 * ── STEP 2: ONE-HOT — why targets are [count, 10] and not [count] ──────────
 * A label is the digit 7. The loss wants a row of ten numbers:
 *
 *     label 7  →  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
 *
 * So allocate count*10 zeros and set ONE per row:
 *
 *     oneHot[i * 10 + label[i]] = 1
 *
 * Ch 12 built crossEntropyFromLogits around this convention: the mask must
 * be the same shape as the logits so the true-class logit can be picked out
 * with mul and sum and STAY IN THE GRAPH.
 *
 * ── PITFALL: building the Tensor ───────────────────────────────────────────
 * A Tensor is a plain object — { data, shape, ndim, size }. You can build
 * one directly from a Float64Array, or call createTensor with a normal
 * array. Do NOT convert 1.5 million numbers through a plain array if you can
 * fill a Float64Array in place; the intermediate is pure waste.
 *
 * Returns both splits. Shapes: images [2000, 784], oneHot [2000, 10].
 */
function load(): { train: Split; test: Split } {
  throw new Error("Not implemented — read the chapter doc first");
}

/**
 * Return the numbers 0..count-1 in a random order. Fisher-Yates.
 *
 * ── WHY THIS EXISTS AT ALL ─────────────────────────────────────────────────
 * The vendored file is stratified but ORDERED — all the 0s, then all the 1s.
 * Cut that into batches of 64 without shuffling and almost every batch is a
 * single digit. The gradient of a batch is an AVERAGE (Ch 12's mean over the
 * batch), and an average over one class is not a useful direction: each step
 * drags the weights toward "everything is a 3", then "everything is a 4".
 *
 * Shuffle and every batch is a mixture, so every step is an average over ten
 * digits pulling against each other. That is the direction you want.
 *
 * ── THE ALGORITHM ──────────────────────────────────────────────────────────
 * Walk i from count-1 down to 1; pick j uniformly in [0, i]; swap. Use the
 * `random` you are given, not Math.random, or the run stops being
 * reproducible and you cannot compare two attempts.
 *
 * ── PITFALL: reshuffle EVERY epoch ─────────────────────────────────────────
 * Call this once per epoch, not once for the whole run. A fixed order means
 * the same 64 images travel together 30 times, and the same 16 leftovers are
 * dropped 30 times — see the note in train().
 */
function shuffledIndices(count: number, random: () => number): Uint32Array {
  throw new Error("Not implemented — read the chapter doc first");
}

/**
 * Copy `size` rows out of a [n, width] tensor into a fresh [size, width] one,
 * taking row numbers from `rows[from]` … `rows[from + size - 1]`.
 *
 * ── WHY A GATHER AND NOT A SLICE ───────────────────────────────────────────
 * After shuffling, the rows you want are scattered: batch 1 might be rows
 * 1832, 45, 1201, … A slice takes a contiguous run, which is exactly what
 * shuffling was meant to prevent. So copy row by row.
 *
 * ── THE INDEXING — Ch 01's row-major layout, doing real work ───────────────
 * Row r of the OUTPUT comes from row rows[from + r] of the SOURCE:
 *
 *     src = rows[from + r] * width        start of that row in source.data
 *     dst = r * width                     start of the row you are writing
 *     copy `width` numbers from src to dst
 *
 * Used for both images (width 784) and one-hot targets (width 10), which is
 * why width is read off source.shape[1] rather than passed in.
 *
 * ── PITFALL: the same permutation for both ─────────────────────────────────
 * You will call this twice per batch — once for images, once for targets —
 * and BOTH must use the same `rows` and the same `from`. Different orders
 * and you train image i against label j, which produces a network that
 * learns nothing while looking completely healthy.
 */
function gatherRows(source: Tensor, rows: Uint32Array, from: number, size: number): Tensor {
  throw new Error("Not implemented — read the chapter doc first");
}

/**
 * Build 784 → 128 → 64 → 32 → 10 and return forward + parameters.
 *
 * ── THE SHAPE, AND WHY IT NARROWS ──────────────────────────────────────────
 * Four Linear layers with relu between each PAIR — three relus, not four:
 *
 *     Linear(784, 128)  relu  Linear(128, 64)  relu
 *     Linear(64, 32)    relu  Linear(32, 10)   ← nothing after this one
 *
 * Each layer must describe its input in fewer numbers than it received. 128
 * is not room enough to store a 784-pixel image, so the layer has to keep
 * what separates digits and discard the rest. That is the intuition; be
 * honest that 128/64/32 are conventional, not derived.
 *
 * ── PITFALL: NO SOFTMAX ON THE OUTPUT ──────────────────────────────────────
 * The last layer emits ten RAW LOGITS — unbounded scores, not probabilities.
 * crossEntropyFromLogits applies the softmax internally; that was the entire
 * point of Ch 12's log-sum-exp. Apply it here too and it happens twice.
 * Measured in Ch 15: it does not crash, it reaches 0.4018 where correct
 * usage reaches 0.0008, and looks like a network that needs more epochs.
 *
 * ── PITFALL: relu on the output ────────────────────────────────────────────
 * Equally wrong for a different reason — it would forbid negative logits,
 * and a logit must be free to go negative to say "definitely not this class".
 *
 * ── INITIALISATION ─────────────────────────────────────────────────────────
 * Leave it at the default. Linear's default is `he`, which is what Ch 13
 * derived for relu layers, and it is correct here.
 *
 * ── parameters(): the Ch 13 contract, four layers deep ─────────────────────
 * Return ONE FLAT LIST of all eight tensors — W and b from each layer. The
 * optimizer must not learn that layers exist. `layers.flatMap(l =>
 * l.parameters())` is the whole implementation, and it is what lets Ch 14's
 * Adam scale from XOR to this without a line changing.
 *
 * Verify: 8 tensors, 111,146 numbers.
 */
function makeModel(): {
  forward: (x: TensorValue) => TensorValue;
  parameters: () => TensorValue[];
} {
  throw new Error("Not implemented — read the chapter doc first");
}

/**
 * Fraction correct. The highest logit is the prediction.
 *
 * ── WHY argmax AND NOT softmax ─────────────────────────────────────────────
 * softmax is monotonic — it never reorders. The largest logit is the largest
 * probability, so for a prediction the softmax is wasted work. Use
 * argmax(logits, 1) to collapse [count, 10] into [count], then compare
 * against labels.
 *
 * ── WHAT TO EXPECT ─────────────────────────────────────────────────────────
 * Chance is exactly 10%, because the subset is stratified. Ch 12's point
 * shows up here in full: this number moves in visible steps while the loss
 * moves smoothly, and it stops moving at epoch ~15 while the loss keeps
 * falling by another factor of seven. Train on the loss, report this.
 */
function accuracy(logits: Tensor, labels: Uint8Array): number {
  throw new Error("Not implemented — read the chapter doc first");
}

/**
 * counts[actual][predicted] — a 10×10 grid of where the mistakes go.
 *
 * ── WHY A SINGLE NUMBER IS NOT ENOUGH ──────────────────────────────────────
 * 91% means 45 mistakes out of 500. An accuracy figure cannot tell you
 * whether those are spread evenly or piled onto one digit. The matrix can,
 * and the off-diagonal cells name the pairs the network finds genuinely
 * similar. In the reference run digit 6 was weakest at 86% recall.
 *
 * Start with a 10×10 grid of zeros, take argmax over the logits, and for
 * each test row increment counts[trueLabel][predicted]. Every row should sum
 * to 50, and the whole grid to 500 — a cheap check that you indexed
 * [actual][predicted] and not the transpose.
 */
function confusion(logits: Tensor, labels: Uint8Array): number[][] {
  throw new Error("Not implemented — read the chapter doc first");
}

/**
 * Train for EPOCHS epochs and report progress. Returns the final mean loss.
 *
 * ── THE FIVE LINES — unchanged from XOR, and that is the point ─────────────
 *
 *     optimizer.zeroGrad();                              // 1. forget
 *     const logits = model.forward(new TensorValue(x));  // 2. guess
 *     const loss = crossEntropyFromLogits(logits, y);    // 3. score
 *     loss.backward();                                   // 4. blame
 *     optimizer.step();                                  // 5. move
 *
 * Scaling from four XOR rows to 111,146 parameters changed the data pipeline
 * completely and changed this loop not at all. One backward() still fills
 * all eight parameter tensors, by the hand-off traced in Ch 15: each layer
 * blames its own W and b, then passes what is left to the layer beneath.
 *
 * ── THE THREE NESTED IDEAS ─────────────────────────────────────────────────
 *
 *     one BATCH  = 64 images → one forward, one backward, one step
 *     one EPOCH  = every image seen once = ⌊2000/64⌋ = 31 batches
 *     the RUN    = 30 epochs = 930 optimizer steps
 *
 * Structure: for each epoch, shuffle; then walk `start` from 0 in steps of
 * 64 while `start + 64 <= count`; gather both batches; run the five lines;
 * accumulate the loss so you can report the epoch mean.
 *
 * ── THE LEFTOVER 16 ────────────────────────────────────────────────────────
 * 2000 = 31×64 + 16, and the condition above drops that remainder. This is
 * fine, but only BECAUSE you reshuffle: it is a different 16 every epoch.
 * With a fixed order those same 16 images would never be trained on.
 *
 * ── PITFALL: where zeroGrad goes ───────────────────────────────────────────
 * FIRST, before the forward. Ch 08 showed backward() accumulates with +=.
 * Omit it and gradients from every previous batch pile up and the loss
 * climbs. Put it between backward() and step() and the gradients are wiped
 * before they are used — the loss sits perfectly still and looks like a
 * learning-rate problem. Two different bugs, two different signatures.
 *
 * ── PITFALL: wrap the batch fresh, but not the model ───────────────────────
 * `new TensorValue(x)` per batch is correct — x is new data each time, and a
 * leaf with no history. Do not rebuild the LAYERS inside the loop; their
 * weights must persist across all 930 steps.
 *
 * ── REPORTING ──────────────────────────────────────────────────────────────
 * Every fifth epoch, run the full train and test sets through forward and
 * print loss, train accuracy and test accuracy. Watching those two accuracy
 * numbers separate is the most instructive output in this file.
 */
function train(
  model: { forward: (x: TensorValue) => TensorValue; parameters: () => TensorValue[] },
  optimizer: { step(): void; zeroGrad(): void },
  train_: Split,
  test: Split,
  random: () => number,
): number {
  throw new Error("Not implemented — read the chapter doc first");
}

// ═══ RUN ════════════════════════════════════════════════════════════════════

try {
  const { train: trainSplit, test } = load();
  console.log(`  loaded ${trainSplit.count} training and ${test.count} test images, ${PIXELS} pixels each`);

  const model = makeModel();
  const params = model.parameters();
  console.log(`  ${params.reduce((s, p) => s + p.data.size, 0).toLocaleString()} parameters in ${params.length} tensors`);
  console.log(`  shapes: ${params.map((p) => `[${p.data.shape}]`).join(" ")}\n`);

  const optimizer = new Adam(params, 1e-3);
  const started = performance.now();
  train(model, optimizer, trainSplit, test, makeRandom(7));
  console.log(`\n  trained in ${((performance.now() - started) / 1000).toFixed(1)} s\n`);

  const finalTest = model.forward(new TensorValue(test.images)).data;
  // Baselines measured on this exact subset, five runs each, same 30 epochs:
  //   chance                                  10.0%
  //   linear only 784 -> 10 (7,850 params)    87.8%  median
  //   one hidden  784 -> 128 -> 10            89.8%  median
  //   three hidden (this)                     91.2%  median
  console.log(`  final test accuracy ${(accuracy(finalTest, test.labels) * 100).toFixed(1)}%  ` +
    `(chance 10%, linear-only 87.8%, one hidden layer 89.8%)\n`);
  printConfusion(confusion(finalTest, test.labels));

  // The first mistake is more informative than the score. In the reference
  // run it was a 5 called a 6 — with 99.81% confidence. softmax always names
  // a winner, so a confident output is not a correct one.
  const predicted = argmax(finalTest, 1);
  for (let i = 0; i < test.count; i++) {
    if (predicted.data[i] !== test.labels[i]) {
      console.log(`\n  first mistake: image ${i} is a ${test.labels[i]}, called a ${predicted.data[i]}`);
      printDigit(test.images, i);
      break;
    }
  }
} catch (error) {
  if (error instanceof Error && error.message.includes("Not implemented")) {
    console.log("  pending —", error.message);
    console.log("  fill in the stubs above, in this order: load, shuffledIndices,");
    console.log("  gatherRows, makeModel, accuracy, confusion, train.");
  } else throw error;
}

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO 1: stop training at epoch 15, where train accuracy first hits 100%.
//         Does test accuracy suffer? That is early stopping, and you have now
//         measured the case for it.
//
// TODO 2: delete every relu from makeModel, leaving four Linears stacked.
//         Predict the accuracy BEFORE running. Ch 09 says four stacked linear
//         layers collapse to one — so it should match the 87.8% linear
//         baseline, not beat it.
//
// TODO 3: skip the /255 in load() and train again. The failure is quiet:
//         watch the first epoch's loss and compare it to 1.6025.
//
// TODO 4: shuffle ONCE before epoch 1 instead of every epoch. Accuracy barely
//         moves — but now name the 16 images that are never trained on.
//
// TODO 5: swap Adam for SGD at lr 0.1. XOR was well-conditioned and SGD tied
//         with Adam there. 784 correlated pixel inputs are not. Does Ch 14's
//         argument for per-parameter scaling finally pay off?
