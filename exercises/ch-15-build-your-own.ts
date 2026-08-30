/**
 * EXERCISES — Ch 15: Build One Yourself
 * ══════════════════════════════════════
 * Prereq : Linear (Ch 13), relu (Ch 11), mseLoss (Ch 12), SGD/Adam (Ch 14)
 * Run    : bun run exercises/ch-15-build-your-own.ts
 *
 * `ch-15-training-loop.ts` showed you a finished network. This one is empty.
 *
 * The problem is NOT XOR, deliberately. XOR has four rows you can memorise;
 * this has a thousand points you cannot. If your network learns it, that is
 * because the machinery works — not because you recognised the answer.
 *
 * THE PROBLEM — "is this point inside the circle?"
 *
 *     Points are scattered over the square −1 ≤ x, y ≤ 1.
 *     Label 1 if the point lies inside a circle of radius 0.6, else 0.
 *
 *              ·  ·  ·  ·  ·  ·  ·
 *              ·  ·  ▓  ▓  ▓  ·  ·        ▓ = inside  (label 1)
 *              ·  ▓  ▓  ▓  ▓  ▓  ·        · = outside (label 0)
 *              ·  ▓  ▓  ▓  ▓  ▓  ·
 *              ·  ·  ▓  ▓  ▓  ·  ·
 *              ·  ·  ·  ·  ·  ·  ·
 *
 * WHY THIS PROBLEM: no straight line separates inside from outside — the
 * boundary is curved, so a single `Linear` cannot do it, for the same reason
 * Ch 11 gave for XOR. You need a hidden layer. And because `relu` units are
 * hinges (Ch 11), the network cannot draw a true circle either: it will
 * approximate one with straight segments. With enough units you get a
 * polygon that looks round. You will SEE this in the output — that is the
 * point of the exercise.
 *
 * WHAT YOU WRITE: the four functions marked TODO.
 * WHAT IS GIVEN:  the random number generator and the picture renderer.
 *                 Neither teaches you anything about neural networks.
 */
import { Linear } from "../src/nn/linear.ts";
import { relu } from "../src/nn/activations.ts";
import { mseLoss } from "../src/nn/losses.ts";
import { SGD, SGDMomentum } from "../src/optim/sgd.ts";
import { Adam } from "../src/optim/adam.ts";
import { TensorValue } from "../src/autograd/grad.ts";
import { createTensor, type Tensor } from "../src/tensor/types.ts";

// ─── GIVEN: plumbing, not learning ───────────────────────────────────────────

/**
 * A seeded uniform generator, so every run gives the same dataset and your
 * results are comparable between attempts. Plain LCG — not Box-Muller,
 * because we want points spread evenly over a square, not a bell curve.
 */
function makeRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

/** True label for a point: 1 inside the circle, 0 outside. */
function trueLabel(x: number, y: number): number {
  return x * x + y * y < RADIUS * RADIUS ? 1 : 0;
}

/**
 * Draw what the network currently believes, as a character grid.
 * Runs the model over a lattice of points and prints one character each.
 * `#` where it says inside, `.` where it says outside, `+` where it is
 * unsure (output between 0.35 and 0.65) — the unsure band IS the decision
 * boundary, so you can watch it tighten as training proceeds.
 */
function renderBoundary(predict: (x: number, y: number) => number, size = 25): void {
  for (let row = 0; row < size; row++) {
    const y = 1 - (2 * row) / (size - 1);
    let line = "  ";
    for (let col = 0; col < size; col++) {
      const x = -1 + (2 * col) / (size - 1);
      const p = predict(x, y);
      line += p > 0.65 ? "#" : p < 0.35 ? "." : "+";
    }
    // the true circle, drawn alongside for comparison
    line += "   ";
    for (let col = 0; col < size; col++) {
      const x = -1 + (2 * col) / (size - 1);
      line += trueLabel(x, y) ? "#" : ".";
    }
    console.log(line);
  }
  console.log("   ^ what your network learned          ^ the truth");
}

/** Run a block; report cleanly if something it needs is still a stub. */
function stage(title: string, body: () => void): void {
  console.log(`\n─── ${title} ───`);
  try {
    body();
  } catch (error) {
    if (error instanceof Error && error.message.includes("Not implemented")) {
      console.log("  pending —", error.message);
    } else throw error;
  }
}

const RADIUS = 0.6;
const NUM_POINTS = 1000;

// ─── TODO 1: the dataset ─────────────────────────────────────────────────────
/**
 * Generate `count` random points and their labels.
 *
 * RETURN two tensors:
 *   inputs  — shape [count, 2]   each row is one point, [x, y]
 *   targets — shape [count, 1]   each row is that point's label, 0 or 1
 *
 * HOW: call `random()` for each coordinate. It returns a number in [0, 1),
 * and you need [-1, 1) — so scale and shift it. Then `trueLabel(x, y)`
 * gives you the label.
 *
 * SHAPE WARNING: `createTensor(data, shape)` takes a FLAT array. For
 * [count, 2] the data is laid out row-major: [x0, y0, x1, y1, x2, y2, ...]
 * — not two separate lists. Ch 01's row-major layout, still true here.
 *
 * WHY targets is [count, 1] and not [count]: it has to match the shape your
 * model outputs, or `mseLoss` will broadcast them against each other and
 * silently compute something that is not the loss you meant.
 */
function makeDataset(count: number, random: () => number): { inputs: Tensor; targets: Tensor } {
  throw new Error("Not implemented — read the chapter doc first");
}

// ─── TODO 2: the model ───────────────────────────────────────────────────────
/**
 * Build the network and return an object with `forward` and `parameters`.
 *
 * SHAPE: input is 2 numbers (x, y), output is 1 number (how strongly it
 * believes "inside"). So the first Linear takes 2 in, the last gives 1 out.
 * Everything between is yours — that was this chapter's "how wide?" section.
 *
 * START WITH: `new Linear(2, hidden)` → `relu` → `new Linear(hidden, 1)`,
 * and hidden = 16. Sixteen hinges means a 16-sided polygon at best, which
 * is enough to look convincingly round.
 *
 * `parameters()` must return ONE FLAT LIST from BOTH layers — that is the
 * Ch 13 contract that lets the optimizer stay ignorant of layers. Spread
 * them: [...layer1.parameters(), ...layer2.parameters()]
 *
 * THEN COME BACK and try hidden = 3. Watch the picture become a triangle.
 * That is the clearest demonstration in this course that a relu network is
 * made of straight pieces.
 */
function makeModel(hidden: number): {
  forward: (x: TensorValue) => TensorValue;
  parameters: () => TensorValue[];
} {
  throw new Error("Not implemented — read the chapter doc first");
}

// ─── TODO 3: accuracy ────────────────────────────────────────────────────────
/**
 * What fraction of predictions are on the right side of 0.5?
 *
 * Threshold each prediction, compare to the target, average. Three lines.
 *
 * REMEMBER Ch 12's point, which you will watch happen again here: this
 * number moves in steps while the loss moves smoothly. Train on the loss,
 * report this.
 */
function accuracy(predictions: Tensor, targets: Tensor): number {
  throw new Error("Not implemented — read the chapter doc first");
}

// ─── TODO 4: the training loop ───────────────────────────────────────────────
/**
 * Train the model for `steps` steps and return the final loss.
 *
 * THE FIVE LINES, in this order — the order is the whole lesson:
 *
 *     optimizer.zeroGrad();                        forget last step's blame
 *     const out  = model.forward(inputsAsValue);   guess
 *     const loss = mseLoss(out, targets);          score the guess
 *     loss.backward();                             assign blame
 *     optimizer.step();                            move
 *
 * Wrap `inputs` once in a `new TensorValue(...)` — it is a leaf, it never
 * changes, and rebuilding it every step just makes garbage.
 *
 * Print the loss and accuracy every `reportEvery` steps so you can watch it.
 *
 * IF IT DOES NOT LEARN, the chapter gave you the order to suspect things in:
 * learning rate first, then initialisation, then width — and check
 * `zeroGrad()` is actually there before any of them.
 */
function train(
  model: { forward: (x: TensorValue) => TensorValue; parameters: () => TensorValue[] },
  optimizer: { step(): void; zeroGrad(): void },
  inputs: Tensor,
  targets: Tensor,
  steps: number,
  reportEvery: number,
): number {
  throw new Error("Not implemented — read the chapter doc first");
}

// ─── RUN IT ──────────────────────────────────────────────────────────────────

stage("E1: look at the problem before modelling it", () => {
  const { inputs, targets } = makeDataset(NUM_POINTS, makeRandom(42));
  console.log("  inputs shape ", JSON.stringify(inputs.shape), " targets shape", JSON.stringify(targets.shape));
  let inside = 0;
  for (const v of targets.data) inside += v;
  console.log(`  ${inside} of ${NUM_POINTS} points are inside the circle`);
  console.log("  a model that always guessed 'outside' would already score",
    `${(100 * (1 - inside / NUM_POINTS)).toFixed(1)}%.`);
  console.log("  that is the number to beat — accuracy alone can flatter a useless model.");
});

stage("E2: train it, and watch the boundary appear", () => {
  const { inputs, targets } = makeDataset(NUM_POINTS, makeRandom(42));
  const model = makeModel(16);
  const optimizer = new Adam(model.parameters(), 0.02);

  const final = train(model, optimizer, inputs, targets, 1500, 300);
  console.log(`  final loss ${final.toFixed(6)}\n`);

  renderBoundary((x, y) =>
    model.forward(new TensorValue(createTensor([x, y], [1, 2]))).data.data[0]!);
});

stage("E3: the same network, three hinges wide", () => {
  const { inputs, targets } = makeDataset(NUM_POINTS, makeRandom(42));
  const model = makeModel(3);
  const optimizer = new Adam(model.parameters(), 0.02);
  train(model, optimizer, inputs, targets, 1500, 1500);

  renderBoundary((x, y) =>
    model.forward(new TensorValue(createTensor([x, y], [1, 2]))).data.data[0]!);
  console.log("  three relu units, three straight cuts. it cannot curve — it can only");
  console.log("  fold. every 'curved' boundary you have ever seen was a polygon with");
  console.log("  enough sides that you stopped noticing.");
});

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO 5: swap Adam for SGD at lr 0.1. Does it still find the circle? This
//         problem is less well-conditioned than XOR — a fair test of whether
//         Ch 14's argument for Adam actually pays off somewhere.
//
// TODO 6: delete the `relu` from makeModel, leaving two Linears back to back.
//         Predict what the picture will look like BEFORE running it. Ch 09
//         told you exactly what happens when linear layers stack.
//
// TODO 7: move the circle off-centre — label on (x−0.3)² + (y+0.2)² < 0.36.
//         Which parameters have to change to track it, the weights or the
//         biases? Reason it out from `y = xWᵀ + b`, then check.
//
// TODO 8: replace the circle with a ring: inside if 0.4 < radius < 0.7. Now
//         the network needs TWO boundaries rather than one. Does hidden = 16
//         still cope, and if not, what does the failure look like?
