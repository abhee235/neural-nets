/**
 * EXERCISES — Ch 15: The Training Loop
 * ══════════════════════════════════════
 * Prereq : Linear (Ch 13), relu (Ch 11), mseLoss (Ch 12), SGD/Adam (Ch 14)
 * Run    : bun run exercises/ch-15-training-loop.ts
 *
 * This chapter adds no new machinery — it assembles what you already have.
 * The whole of training is five lines:
 *
 *     optimizer.zeroGrad();
 *     const loss = lossFn(model.forward(x), y);
 *     loss.backward();
 *     optimizer.step();
 *
 * The problem is XOR, which Ch 11 opened Part 3 with and trained by hand
 * because none of these classes existed yet. Now they do.
 */
import { Linear } from "../src/nn/linear.ts";
import { relu } from "../src/nn/activations.ts";
import { mseLoss } from "../src/nn/losses.ts";
import { SGD, SGDMomentum } from "../src/optim/sgd.ts";
import { Adam } from "../src/optim/adam.ts";
import { TensorValue } from "../src/autograd/grad.ts";
import { createTensor } from "../src/tensor/types.ts";

/** Run a block; report cleanly if something it needs is still a stub. */
function stage(title: string, body: () => void): void {
  console.log(`\n─── ${title} ───`);
  try {
    body();
  } catch (error) {
    if (error instanceof Error && error.message.includes("not implemented")) {
      console.log("  pending —", error.message);
    } else throw error;
  }
}

// XOR: the two classes sit on opposite diagonals, so no straight line splits them.
const INPUTS = createTensor([0, 0, 0, 1, 1, 0, 1, 1], [4, 2]);
const TARGETS = createTensor([0, 1, 1, 0], [4, 1]);
const WANT = [0, 1, 1, 0];

/** Two Linear layers with a relu between them — the smallest net that can do XOR. */
function makeModel(hidden = 8) {
  const layer1 = new Linear(2, hidden);
  const layer2 = new Linear(hidden, 1);
  return {
    forward: (x: TensorValue) => layer2.forward(relu(layer1.forward(x))),
    // Ch 13's contract: two layers, one flat list, and the optimizer never
    // needs to know a layer exists.
    parameters: () => [...layer1.parameters(), ...layer2.parameters()],
  };
}

/** Three lines. Threshold at 0.5, compare, average. */
function accuracy(predictions: Float64Array): number {
  let correct = 0;
  for (let i = 0; i < WANT.length; i++) {
    if ((predictions[i]! > 0.5 ? 1 : 0) === WANT[i]) correct++;
  }
  return correct / WANT.length;
}

// ─── E0: a gradient through TWO layers, by hand ──────────────────────────────
// Every trace so far has been through ONE layer. This is the smallest network
// with two: 1 → 1 → 1, weights set by hand so every number is checkable.
//
//   x=1 ──► [W₁=2, b₁=0] ──► relu ──► [W₂=3, b₂=0] ──► y,  target 0
stage("E0: how blame travels through two layers", () => {
  const a = new Linear(1, 1), b = new Linear(1, 1);
  a.weight.data = createTensor([2], [1, 1]); a.bias!.data = createTensor([0], [1]);
  b.weight.data = createTensor([3], [1, 1]); b.bias!.data = createTensor([0], [1]);

  const x = new TensorValue(createTensor([1], [1, 1]));
  const preAct = a.forward(x);
  const hidden = relu(preAct);
  const out = b.forward(hidden);
  const loss = mseLoss(out, createTensor([0], [1, 1]));

  console.log("  forward:  x=1 → layer1 → ", preAct.data.data[0],
    " → relu → ", hidden.data.data[0], " → layer2 → ", out.data.data[0],
    "  loss", loss.data.data[0]);

  loss.backward();

  console.log("  backward, from ONE call:");
  console.log("    dL/dy                  ", out.grad!.data[0], "  = 2(6−0)");
  console.log("    layer2  dL/dW₂ = 12·h  ", b.weight.grad!.data[0], "   dL/db₂", b.bias!.grad!.data[0]);
  console.log("    dL/dh   = 12·W₂        ", hidden.grad!.data[0], "  ← this is what travels DOWN");
  console.log("    relu gate (preAct > 0) ", preAct.grad!.data[0], "  passed through unchanged");
  console.log("    layer1  dL/dW₁ = 36·x  ", a.weight.grad!.data[0], "   dL/db₁", a.bias!.grad!.data[0]);
  console.log("  layer2's x.grad IS layer1's upstream. that is the hand-off.");

  const params = [...a.parameters(), ...b.parameters()];
  console.log("  every parameter now has a gradient:",
    params.map((p) => p.grad!.data[0]).join(", "), " — four tensors, one backward()");
});

// ─── E1: the five lines, once ────────────────────────────────────────────────
stage("E1: one training step, in full", () => {
  const model = makeModel();
  const optimizer = new SGD(model.parameters(), 0.1);

  optimizer.zeroGrad();                                       // 1. forget
  const prediction = model.forward(new TensorValue(INPUTS));  // 2. guess
  const loss = mseLoss(prediction, TARGETS);                  // 3. score
  loss.backward();                                            // 4. blame
  optimizer.step();                                           // 5. move

  console.log("  loss after one step:", loss.data.data[0]!.toFixed(6));
  console.log("  parameters the optimizer walked:", model.parameters().length, "tensors,",
    model.parameters().reduce((s, p) => s + p.data.size, 0), "numbers");
  console.log("  shapes:", model.parameters().map((p) => `[${p.data.shape}]`).join(" "));
});

// ─── E2: the loop — XOR solved with the real classes ─────────────────────────
stage("E2: 600 steps of SGD at lr 0.1", () => {
  const model = makeModel();
  const optimizer = new SGD(model.parameters(), 0.1);
  console.log("  step     loss       accuracy");
  for (let step = 0; step <= 600; step++) {
    optimizer.zeroGrad();
    const out = model.forward(new TensorValue(INPUTS));
    const loss = mseLoss(out, TARGETS);
    loss.backward();
    if ([0, 50, 100, 200, 600].includes(step)) {
      console.log(`  ${String(step).padStart(4)}   ${loss.data.data[0]!.toFixed(6)}      ${(accuracy(out.data.data) * 100).toFixed(0)}%`);
    }
    optimizer.step();
  }
  const final = model.forward(new TensorValue(INPUTS));
  console.log("  predictions:", Array.from(final.data.data).map((v) => v.toFixed(4)).join("  "), "  want 0, 1, 1, 0");
});

// ─── E3: loss is smooth, accuracy is a staircase ─────────────────────────────
// Ch 12's argument, live: this is why you train on one and report the other.
stage("E3: when does accuracy stop changing?", () => {
  const model = makeModel();
  const optimizer = new SGD(model.parameters(), 0.1);
  let previousAcc = -1;
  let firstPerfect = -1, lossThen = 0;
  const changes: string[] = [];
  for (let step = 0; step <= 600; step++) {
    optimizer.zeroGrad();
    const out = model.forward(new TensorValue(INPUTS));
    const loss = mseLoss(out, TARGETS);
    loss.backward();
    const acc = accuracy(out.data.data);
    if (acc !== previousAcc && previousAcc >= 0) {
      changes.push(`step ${step}: ${(previousAcc * 100).toFixed(0)}→${(acc * 100).toFixed(0)}%`);
    }
    if (acc === 1 && firstPerfect < 0) { firstPerfect = step; lossThen = loss.data.data[0]!; }
    previousAcc = acc;
    optimizer.step();
  }
  console.log("  accuracy changed at:", changes.join("   "), " — that is all, out of 600 steps");
  console.log("  loss changed at all 600");
  console.log(`  accuracy first hit 100% at step ${firstPerfect}, loss still ${lossThen.toFixed(4)}`);
  console.log("  training ran on for the rest. the loss had more to say; accuracy did not.");
});

// ─── E4: the learning rate cliff ─────────────────────────────────────────────
// Reduced trial count so this stays quick; the doc's table used 30.
stage("E4: how often does each setup actually solve XOR? (10 inits, 2000 steps)", () => {
  const solve = (make: (p: TensorValue[]) => { step(): void; zeroGrad(): void }) => {
    let solved = 0;
    for (let trial = 0; trial < 10; trial++) {
      const model = makeModel();
      const optimizer = make(model.parameters());
      let loss = 1;
      for (let s = 0; s < 2000; s++) {
        optimizer.zeroGrad();
        const l = mseLoss(model.forward(new TensorValue(INPUTS)), TARGETS);
        l.backward();
        optimizer.step();
        loss = l.data.data[0]!;
      }
      if (loss < 1e-4) solved++;
    }
    return solved;
  };
  const rows: [string, (p: TensorValue[]) => any][] = [
    ["SGD          lr 0.1  ", (p) => new SGD(p, 0.1)],
    ["SGD          lr 0.5  ", (p) => new SGD(p, 0.5)],
    ["SGD          lr 1.0  ", (p) => new SGD(p, 1.0)],
    ["SGDMomentum  lr 0.05 ", (p) => new SGDMomentum(p, 0.05, 0.9)],
    ["Adam         lr 0.01 ", (p) => new Adam(p, 0.01)],
  ];
  for (const [name, make] of rows) {
    const n = solve(make);
    console.log(`  ${name}  ${String(n).padStart(2)}/10  ${"#".repeat(n * 2)}`);
  }
  console.log("  a cliff, not a slope — and plain SGD at a sane rate is hard to beat here.");
});

// ─── E5: what a dead network looks like ──────────────────────────────────────
stage("E5: the signature of failure", () => {
  const model = makeModel();
  const optimizer = new SGD(model.parameters(), 1.0);   // deliberately too high
  for (let s = 0; s < 500; s++) {
    optimizer.zeroGrad();
    mseLoss(model.forward(new TensorValue(INPUTS)), TARGETS).backward();
    optimizer.step();
  }
  const out = model.forward(new TensorValue(INPUTS));
  const loss = mseLoss(out, TARGETS);
  console.log("  lr = 1.0 →  loss", loss.data.data[0]!.toFixed(6),
    "  predictions", Array.from(out.data.data).map((v) => v.toFixed(4)).join("  "));
  const p = Array.from(out.data.data);
  const allSame = Math.max(...p) - Math.min(...p) < 1e-6;
  console.log("  all four predictions identical?", allSame);
  console.log("  THAT is the tell: four different inputs, one output. The hidden");
  console.log("  layer is gone — dead relu units, killed by oversized steps. The loss");
  console.log("  value itself is large and erratic, not a fixed number to look for.");
});

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO 1: delete optimizer.zeroGrad() and watch the loss climb. Then move it to
//         between backward() and step() and watch the loss sit perfectly still.
//         Two different bugs, two different signatures.
//
// TODO 2: shrink the hidden layer. makeModel(2) is the theoretical minimum
//         (Ch 11's E6 proves a 2-unit solution exists). How often does it
//         actually find one?
//
// TODO 3: swap mseLoss for crossEntropyFromLogits. XOR is really a
//         classification problem — does the loss that Ch 12 recommended for
//         classification do better here?
