# Chapter 15: Training Loop

> **Part 3 of 6 — Neural Net Primitives**
> `src/ch-15-training-loop/`

---

## What You're Building

A complete MLP (Multi-Layer Perceptron) from scratch using the components from Ch 11–13,
plus a full training loop: forward pass → loss → backward → optimizer step. Train it on
a toy dataset (XOR or spiral classification) and watch the loss decrease.

---

## Why This Matters

This chapter closes Part 3. Every component you've built — tensors, autograd, activations,
loss, layers, optimizer — is orchestrated here into a working neural network that actually
learns. This is the full training loop used, at a larger scale, to train GPT.

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

## → Next Part

**Part 4 — Language Model Inputs (Ch 16–21):** tokenize text into numbers, embed tokens
into vectors, add positional information, and normalize with LayerNorm.
