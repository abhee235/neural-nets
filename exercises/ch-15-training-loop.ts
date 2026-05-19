/**
 * EXERCISES — Ch 15: Training Loop
 * ══════════════════════════════════
 * Prereq : src/nn/linear.ts + activations + losses + optim/adam implemented
 * Run    : bun run exercises/ch-15-training-loop.ts
 *
 * A complete training loop: forward → loss → backward → optimizer step.
 * We train a 2-layer MLP to classify 2D spiral data.
 */
import { Linear }       from "../src/nn/linear.ts";
import { relu }         from "../src/nn/activations.ts";
import { crossEntropyFromLogits } from "../src/nn/losses.ts";
import { Adam }         from "../src/optim/adam.ts";
import { TensorValue }  from "../src/autograd/grad.ts";
import { createTensor } from "../src/tensor/creation.ts";

// ─── E1: Spiral dataset generator ────────────────────────────────────────────
function generateSpiral(n: number, classes: number): { x: number[][], y: number[] } {
  const xs: number[][] = [];
  const ys: number[] = [];
  for (let c = 0; c < classes; c++) {
    for (let i = 0; i < n; i++) {
      const r    = i / n;
      const t    = (i / n) * 4 + c * (4 / classes) + (Math.random() - 0.5) * 0.2;
      xs.push([r * Math.sin(t), r * Math.cos(t)]);
      ys.push(c);
    }
  }
  return { x: xs, y: ys };
}

const dataset = generateSpiral(50, 3);   // 150 samples, 3 classes
console.log("Dataset: 150 samples, 3 classes");

// ─── E2: 2-layer MLP ─────────────────────────────────────────────────────────
const W1 = new Linear(2, 16, true, "he");
const W2 = new Linear(16, 3, true, "xavier");
const params = [...W1.parameters(), ...W2.parameters()];
const opt    = new Adam(params, 0.01);

// ─── E3: Training loop ────────────────────────────────────────────────────────
for (let epoch = 0; epoch < 200; epoch++) {
  const xData = dataset.x.flat();
  const xv    = new TensorValue(createTensor(xData, [150, 2]));

  const h     = relu(W1.forward(xv));
  const logit = W2.forward(h);

  // Compute cross-entropy for each sample
  let totalLoss = new TensorValue(createTensor([0], [1]));
  for (let i = 0; i < 150; i++) {
    const rowLogits = new TensorValue(
      createTensor(Array.from(logit.data.data.slice(i * 3, i * 3 + 3)), [3])
    );
    const ce = crossEntropyFromLogits(rowLogits, dataset.y[i]!);
    totalLoss = totalLoss.add(ce);
  }
  // Average loss
  const avgLoss = totalLoss.mul(new TensorValue(createTensor([1 / 150], [1])));
  avgLoss.backward();
  opt.step();
  opt.zeroGrad();

  if (epoch % 40 === 0) {
    console.log(\`epoch \${epoch}: loss = \${avgLoss.data.data[0]!.toFixed(4)}\`);
  }
}

// ─── E4: Accuracy ────────────────────────────────────────────────────────────
let correct = 0;
for (let i = 0; i < 150; i++) {
  const xv     = new TensorValue(createTensor(dataset.x[i]!, [1, 2]));
  const h      = relu(W1.forward(xv));
  const logits = W2.forward(h);
  const pred   = Array.from(logits.data.data).indexOf(Math.max(...Array.from(logits.data.data)));
  if (pred === dataset.y[i]) correct++;
}
console.log(\`\nFinal accuracy: \${(correct / 150 * 100).toFixed(1)}%  (random=33%, good=>85%)\`);

// ─── STRETCH ─────────────────────────────────────────────────────────────────
// TODO: add a 3rd hidden layer (16→16→16→3).
//       Does accuracy improve? Does training slow down?
