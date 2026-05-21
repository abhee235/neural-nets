# Chapter 15: Training Loop

> **Part 3 of 6 — Neural Net Primitives**
> Source: [`src/nn/linear.ts`](../../src/nn/linear.ts) · [`src/optim/sgd.ts`](../../src/optim/sgd.ts) · [`src/optim/adam.ts`](../../src/optim/adam.ts)
> Tests: [`src/nn/linear.test.ts`](../../src/nn/linear.test.ts)
> Exercise: [`exercises/ch-15-training-loop.ts`](../../exercises/ch-15-training-loop.ts)

---

## Learning Goals

By the end of this chapter you can:

- Build a small MLP from `Linear` + activation + `Linear`.
- Wire forward → loss → zeroGrad → backward → step into one training step.
- Iterate over batches, log loss, and confirm the loss curve goes down.
- Fit a non-trivial dataset (spiral, XOR, or moons) and visualise the decision boundary.
- Detect divergence and recover by lowering the learning rate.

---

## Intuition First

Every neural-network training routine — from a 2-layer MLP to GPT-4 — is the same five-line ritual:

1. `predictions = model.forward(inputs)`
2. `loss = lossFn(predictions, targets)`
3. `optimizer.zeroGrad()`
4. `loss.backward()`
5. `optimizer.step()`

The rest is data plumbing, logging, and validation.

---

## Mental Model

```text
  ┌──────────────────────────────────────────┐
  │ for epoch in 0..E:                      │
  │   for batch in shuffle(dataset):        │
  │     y_hat = model.forward(batch.x)      │
  │     loss  = lossFn(y_hat, batch.y)      │
  │     optimizer.zeroGrad()                │
  │     loss.backward()                     │
  │     optimizer.step()                    │
  │   log(epoch, mean_loss, val_loss)       │
  └──────────────────────────────────────────┘
```

---

## Concepts

### Multi-Layer Perceptron (MLP)

An MLP stacks Linear layers with activation functions in between:

```
Input: [batchSize, inputDim]
   ↓  Linear(inputDim, hiddenDim)
   ↓  ReLU
   ↓  Linear(hiddenDim, hiddenDim)
   ↓  ReLU
   ↓  Linear(hiddenDim, outputDim)
Output: [batchSize, outputDim]   ← logits (before softmax/sigmoid)
```

The depth (number of layers) and width (hiddenDim) are hyperparameters.
More depth → more expressive, harder to train. More width → more capacity.

### The Full Training Loop

```
Initialize model parameters (random weights)
Initialize optimizer

for epoch in range(numEpochs):
  for batch in batches:
    # 1. Forward pass: run input through the model
    predictions = model.forward(batchInputs)

    # 2. Compute loss: how wrong are the predictions?
    loss = crossEntropyFromLogits(predictions, batchLabels)

    # 3. Zero gradients: must clear before backward (or they accumulate)
    optimizer.zeroGrad(model.parameters())

    # 4. Backward pass: compute gradients via autograd
    loss.backward()

    # 5. Update parameters: move in the direction that reduces loss
    optimizer.step(model.parameters())

  # Log progress
  print(`Epoch ${epoch}: loss = ${loss.data.toFixed(4)}`)
```

### Batching

Instead of processing one sample at a time, process many samples simultaneously (a "batch").
Benefits:
1. **Speed:** matrix operations on batches are much faster than loops over single samples
2. **Gradient quality:** the gradient averaged over a batch is less noisy than a single sample
3. **Stability:** smoother loss curve, more stable convergence

Typical batch sizes: 32, 64, 128, 256. For transformers: 8–64 depending on sequence length.

### Train vs Eval Mode

Some components behave differently during training vs inference:
- **Dropout** (Ch 20): randomly zeroes activations during training, does nothing during eval
- **BatchNorm**: uses batch statistics during training, running statistics during eval

Best practice: models have a `mode` flag, and `train()` / `eval()` methods:

```typescript
model.train();   // enable dropout, etc.
// ... training loop ...
model.eval();    // disable dropout for evaluation
```

### Learning Curves

Track and log:
- **Training loss**: loss on the training set. Should decrease.
- **Validation loss**: loss on held-out data. Should also decrease; if it rises, you're overfitting.
- **Training accuracy** (for classification): fraction of correct predictions.

When training loss drops but validation loss rises → **overfitting**: the model memorized
training data and fails to generalize. Solutions: more data, dropout, weight decay.

### The Toy Dataset: Spiral Classification

A classic 2D toy problem: two interleaved spirals that can't be separated linearly.
Any model that achieves > 90% accuracy has truly learned a non-linear boundary.

Data: N points from each class, sampled around a spiral in 2D space:
```
for i in range(N):
  angle = i / N * turns * 2π + classOffset
  radius = i / N
  x = radius * cos(angle) + noise
  y = radius * sin(angle) + noise
  label = classIndex
```

---

## What to Implement

| Symbol | Description |
|--------|-------------|
| `class MLP` | Multi-layer perceptron. Constructor takes `[inputDim, ...hiddenDims, outputDim]`. |
| `MLP.forward(x)` | Stack linear layers with ReLU in between. Last layer has no activation. |
| `MLP.parameters()` | Flatten all layer parameters into a single array. |
| `generateSpiral(n, classes)` | Generate the spiral toy dataset. Returns `{inputs, labels}`. |
| `accuracy(predictions, labels)` | Compute classification accuracy. |
| `train(model, data, opts)` | Run the full training loop; return loss history. |

---

## TypeScript Hints

```typescript
export class MLP {
  layers: Linear[];

  // dims: [inputDim, hidden1, hidden2, ..., outputDim]
  constructor(dims: number[]) {
    this.layers = [];
    for (let i = 0; i < dims.length - 1; i++) {
      this.layers.push(new Linear(dims[i]!, dims[i + 1]!, "he"));
    }
  }

  forward(x: Value): Value {
    let out = x;
    for (let i = 0; i < this.layers.length; i++) {
      out = this.layers[i]!.forward(out);
      // Apply ReLU on all layers except the last (output) layer
      if (i < this.layers.length - 1) {
        out = out.relu();   // (add relu() to Value in Ch 11)
      }
    }
    return out;
  }

  parameters(): Value[] {
    return this.layers.flatMap(layer => layer.parameters());
  }
}

// Training loop
export function train(
  model: MLP,
  inputs: Value,
  labels: Value,
  opts: { epochs: number; lr: number; batchSize: number }
): number[] {
  const optimizer = new Adam(opts.lr);
  const lossHistory: number[] = [];

  for (let epoch = 0; epoch < opts.epochs; epoch++) {
    // Forward
    const logits = model.forward(inputs);
    const loss = crossEntropyFromLogits(logits, labels);

    // Backward
    optimizer.zeroGrad(model.parameters());
    loss.backward();
    optimizer.step(model.parameters());

    lossHistory.push(loss.data as number);
    if (epoch % 100 === 0) {
      console.log(`Epoch ${epoch}: loss = ${(loss.data as number).toFixed(4)}`);
    }
  }

  return lossHistory;
}
```

---

## Common Pitfalls

- Forgetting `zeroGrad()` — gradients accumulate and the model explodes after a few steps.
- Computing loss on the *training* set and never on a held-out set — you cannot detect overfitting.
- Logging the last batch's loss instead of the epoch mean — too noisy to read.
- Shuffling the labels but not the inputs (or vice versa) by accident.
- Letting `dropout`/`train` mode leak into evaluation — turn it off for validation.

---

## How to Verify

Run the tests and the exercise. Both should pass cleanly with no warnings:

```bash
bun run exercises/ch-15-training-loop.ts
```

---

## Self-Check Questions

1. Why must `zeroGrad` come BEFORE `backward()`, not after `step()`?
2. In the spiral dataset, a linear model (single `Linear` layer, no activation) achieves
   ~50% accuracy. Why can't it do better?
3. What is "epoch" vs "step" (or "iteration")? If your dataset has 1000 samples and
   batch size 32, how many steps are in one epoch?
4. Train the MLP twice: once with ReLU, once without any activation (just stacked linears).
   What do you expect to see?
5. What is the difference between training loss at epoch 1 vs epoch 100 for a properly
   training model? What should both values tell you?

---

## End of Part 3

You now have a complete working neural network framework:
- Tensor math (Part 1)
- Automatic differentiation (Part 2)
- Activations, losses, layers, optimizers, and training loop (Part 3)

This is essentially a tiny PyTorch. From here, we build the components specifically needed
for language models and transformers.

---

## Further Reading

- [Karpathy — A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) — the *exact* habits this chapter trains you to develop.
- [Smith — Cyclical Learning Rates](https://arxiv.org/abs/1506.01186) — an early but practical look at LR scheduling.
- [Stanford CS231n — practical training notes](https://cs231n.github.io/neural-networks-3/) — babysitting the learning process; how to read loss curves.
- [PyTorch — training loop tutorial](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) — the same ritual in a production library.

---

## Next Chapter

**[Char Tokenizer](../part-4-tokenizer-and-inputs/ch-16-char-tokenizer.md)** — switch from synthetic data to text, the input format every language model needs.
