#!/usr/bin/env python3
"""
Scaffold all src/ch-* starter directories for the 30-chapter neural-nets course.
Run with: python3 scripts/scaffold.py
"""

import os, textwrap

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")

# ---------------------------------------------------------------------------
# Chapter data
# ---------------------------------------------------------------------------
# Each entry: (folder, num, title, part, part_name, what, why, unlocks, doc, funcs, tests)
# funcs : list of (name, params, return_type, one-line description)
# tests : list of pending test description strings

CHAPTERS = [
    # ── Part 1: Tensor Library ──────────────────────────────────────────────
    dict(
        folder="ch-01-tensor-type-system",
        num="01", title="Tensor Type System",
        part=1, part_name="Tensor Library",
        what="The core Tensor type: a Float64Array of flat data plus a shape array. Every later chapter depends on this definition.",
        why="Tensors are the universal data type of neural networks. Every weight matrix, activation, and gradient is a Tensor.",
        unlocks="Ch 02 (Tensor Creation) — factories that construct Tensor values.",
        doc="docs/part-1-tensor-library/ch-01-tensor-type-system.md",
        funcs=[
            ("createTensor", "data: number[], shape: number[]", "Tensor",
             "Create a Tensor from a flat data array and a shape. Validates data.length === product(shape)."),
            ("scalar", "value: number", "Tensor",
             "Create a rank-0 scalar Tensor wrapping a single number."),
            ("isTensor", "value: unknown", "value is Tensor",
             "Runtime type guard — returns true when value has the Tensor structure."),
            ("toString", "t: Tensor", "string",
             "Human-readable representation showing shape and data for debugging."),
        ],
        tests=[
            "createTensor stores data flat and records shape correctly",
            "size equals the product of all shape dimensions",
            "ndim equals the number of dimensions in shape",
            "scalar creates a shape-[] tensor with a single element",
            "flatIndex maps multi-dimensional indices to the correct flat offset",
            "createTensor throws when data.length does not match shape product",
        ]
    ),
    dict(
        folder="ch-02-tensor-creation",
        num="02", title="Tensor Creation",
        part=1, part_name="Tensor Library",
        what="Factory functions that produce Tensors filled with specific values: zeros, ones, random, identity, and ranges.",
        why="Every chapter from 03 onward calls these factories. Weight matrices start as randn; masks start as zeros; LayerNorm gamma starts as ones.",
        unlocks="Ch 03 (Elementwise Ops & Broadcasting).",
        doc="docs/part-1-tensor-library/ch-02-tensor-creation.md",
        funcs=[
            ("zeros", "shape: number[]", "Tensor", "Tensor filled with 0.0."),
            ("ones", "shape: number[]", "Tensor", "Tensor filled with 1.0."),
            ("fill", "shape: number[], value: number", "Tensor", "Tensor filled with a constant."),
            ("randn", "shape: number[]", "Tensor", "Tensor of samples from N(0,1) using Box-Muller transform."),
            ("eye", "n: number", "Tensor", "n×n identity matrix."),
            ("arange", "start: number, stop: number, step?: number", "Tensor", "1-D tensor of evenly spaced values."),
            ("linspace", "start: number, stop: number, n: number", "Tensor", "1-D tensor of n evenly-spaced values from start to stop inclusive."),
        ],
        tests=[
            "zeros produces a tensor of all 0.0 with the correct shape",
            "ones produces a tensor of all 1.0 with the correct shape",
            "randn output has shape matching the requested shape",
            "Box-Muller samples have mean ≈ 0 and std ≈ 1 for large n",
            "eye produces 1s on the diagonal and 0s elsewhere",
            "arange(0, 5, 1) produces [0, 1, 2, 3, 4]",
            "linspace(0, 1, 5) produces [0, 0.25, 0.5, 0.75, 1.0]",
        ]
    ),
    dict(
        folder="ch-03-elementwise-ops-broadcasting",
        num="03", title="Elementwise Ops & Broadcasting",
        part=1, part_name="Tensor Library",
        what="Elementwise arithmetic (add, sub, mul, div) and NumPy-style broadcasting so tensors of different shapes can be combined.",
        why="Broadcasting is used everywhere — adding a bias vector to a batch, scaling attention scores, applying a mask.",
        unlocks="Ch 04 (Matrix Ops) — depends on broadcasting for batched ops.",
        doc="docs/part-1-tensor-library/ch-03-elementwise-ops-broadcasting.md",
        funcs=[
            ("broadcastShapes", "a: number[], b: number[]", "number[]",
             "Return the output shape when broadcasting shapes a and b together (NumPy rules)."),
            ("broadcast", "t: Tensor, shape: number[]", "Tensor",
             "Expand t to the target shape by repeating along size-1 dimensions."),
            ("add", "a: Tensor, b: Tensor", "Tensor", "Elementwise addition with broadcasting."),
            ("sub", "a: Tensor, b: Tensor", "Tensor", "Elementwise subtraction with broadcasting."),
            ("mul", "a: Tensor, b: Tensor", "Tensor", "Elementwise multiplication with broadcasting."),
            ("div", "a: Tensor, b: Tensor", "Tensor", "Elementwise division with broadcasting."),
            ("addScalar", "t: Tensor, s: number", "Tensor", "Add scalar s to every element."),
            ("mulScalar", "t: Tensor, s: number", "Tensor", "Multiply every element by scalar s."),
            ("applyFn", "t: Tensor, fn: (x: number) => number", "Tensor",
             "Apply an arbitrary elementwise function — the building block for activations."),
        ],
        tests=[
            "add([1,2], [3,4]) === [4,6] for matching shapes",
            "broadcasting adds [3,1] + [1,4] to produce [3,4]",
            "broadcastShapes([3,1], [1,4]) === [3,4]",
            "broadcastShapes throws when shapes are incompatible",
            "mulScalar doubles every element when s=2",
            "div(a, a) produces all-ones for nonzero a",
        ]
    ),
    dict(
        folder="ch-04-matrix-ops",
        num="04", title="Matrix Operations",
        part=1, part_name="Tensor Library",
        what="Matrix multiply, batched matmul, transpose, reshape, flatten — the structural operations that transformers use constantly.",
        why="matMul is the dominant compute operation in attention and linear layers. Reshape and transpose are used to split/merge attention heads.",
        unlocks="Ch 05 (Reductions) and directly by Ch 22-30 attention and transformer chapters.",
        doc="docs/part-1-tensor-library/ch-04-matrix-ops.md",
        funcs=[
            ("matMul", "a: Tensor, b: Tensor", "Tensor",
             "Matrix multiply. For 2-D: (M,K)×(K,N)→(M,N). For N-D: batched matmul on last two dims."),
            ("transpose", "t: Tensor, axes?: number[]", "Tensor",
             "Permute tensor axes. Default reverses all axes; axes overrides the permutation."),
            ("reshape", "t: Tensor, newShape: number[]", "Tensor",
             "Return a view with a new shape. Total element count must be unchanged."),
            ("flatten", "t: Tensor, startAxis?: number", "Tensor",
             "Collapse axes from startAxis onward into a single dimension."),
            ("squeeze", "t: Tensor, axis: number", "Tensor",
             "Remove a dimension of size 1 at the given axis."),
            ("unsqueeze", "t: Tensor, axis: number", "Tensor",
             "Insert a size-1 dimension at the given axis."),
        ],
        tests=[
            "(M×K) × (K×N) produces shape (M×N)",
            "matMul value check: [[1,2],[3,4]] × [[1],[1]] === [[3],[7]]",
            "transpose of a (3,4) matrix produces shape (4,3)",
            "reshape preserves element count and total size",
            "reshape throws when new shape has incompatible element count",
            "flatten collapses all dimensions into one",
            "batched matMul (B,M,K) × (B,K,N) produces (B,M,N)",
        ]
    ),
    dict(
        folder="ch-05-reductions",
        num="05", title="Reductions & Statistical Ops",
        part=1, part_name="Tensor Library",
        what="Axis-wise reductions: sum, mean, max, min, argmax, and softmax — all with optional keepDims.",
        why="sum and mean are used inside LayerNorm. Softmax produces attention probabilities. argmax is used in generation.",
        unlocks="Ch 06 (Math Primitives) and Ch 20 (LayerNorm) and Ch 22 (Self-Attention).",
        doc="docs/part-1-tensor-library/ch-05-reductions.md",
        funcs=[
            ("sum", "t: Tensor, axis?: number, keepDims?: boolean", "Tensor",
             "Sum along an axis (or all elements if axis omitted). keepDims inserts a size-1 dimension."),
            ("mean", "t: Tensor, axis?: number, keepDims?: boolean", "Tensor",
             "Mean along an axis. Used in LayerNorm to compute per-token mean."),
            ("max", "t: Tensor, axis?: number, keepDims?: boolean", "Tensor",
             "Max value along an axis. Used in numerically-stable softmax."),
            ("min", "t: Tensor, axis?: number, keepDims?: boolean", "Tensor",
             "Min value along an axis."),
            ("argmax", "t: Tensor, axis?: number", "Tensor",
             "Index of the maximum value along an axis. Used during greedy token generation."),
            ("softmax", "t: Tensor, axis?: number", "Tensor",
             "Numerically stable softmax: subtract max before exp. Produces a probability distribution."),
        ],
        tests=[
            "sum over all elements equals the scalar total",
            "mean of [1,2,3,4] === 2.5",
            "sum along axis=1 of a (3,4) tensor produces shape (3,) or (3,1) with keepDims",
            "keepDims=true preserves the reduced axis as size 1",
            "softmax output sums to 1.0 along the target axis",
            "softmax is shift-invariant: softmax(x) === softmax(x + c)",
            "argmax returns the index of the maximum element",
        ]
    ),
    dict(
        folder="ch-06-math-primitives",
        num="06", title="Math Primitives",
        part=1, part_name="Tensor Library",
        what="Elementwise math functions applied to tensors: exp, log, sqrt, pow, abs, clip, tanh, sigmoid.",
        why="exp and log are needed for cross-entropy and softmax. tanh and sigmoid are activation functions. sqrt appears in LayerNorm and attention scaling.",
        unlocks="Ch 07 (Calculus for ML) — numerical gradient checking needs these.",
        doc="docs/part-1-tensor-library/ch-06-math-primitives.md",
        funcs=[
            ("exp", "t: Tensor", "Tensor", "Elementwise e^x."),
            ("log", "t: Tensor", "Tensor", "Elementwise natural log ln(x). Use a small epsilon guard to avoid log(0)."),
            ("sqrt", "t: Tensor", "Tensor", "Elementwise √x."),
            ("pow", "t: Tensor, exponent: number", "Tensor", "Elementwise x^exponent."),
            ("abs", "t: Tensor", "Tensor", "Elementwise |x|."),
            ("clip", "t: Tensor, min: number, max: number", "Tensor",
             "Clamp every element to [min, max]. Used to prevent log(0) and exploding activations."),
            ("tanh", "t: Tensor", "Tensor", "Elementwise tanh(x) = (e^x - e^-x)/(e^x + e^-x)."),
            ("sigmoid", "t: Tensor", "Tensor", "Elementwise σ(x) = 1/(1+e^-x)."),
        ],
        tests=[
            "exp(0) === 1 for every element",
            "log(exp(x)) ≈ x (round-trip identity)",
            "exp(log(x)) ≈ x for positive x",
            "clip constrains values to [min, max]",
            "tanh output is in (-1, 1) for any finite input",
            "sigmoid output is in (0, 1) for any finite input",
            "sqrt(4) === 2 elementwise",
        ]
    ),

    # ── Part 2: Autodiff ────────────────────────────────────────────────────
    dict(
        folder="ch-07-calculus-for-ml",
        num="07", title="Calculus for ML",
        part=2, part_name="Autodiff Engine",
        what="Numerical gradient checking via finite differences — the verification tool used throughout the autodiff chapters.",
        why="Before implementing symbolic gradients, we need a way to check them. This chapter gives us that tool.",
        unlocks="Ch 08a (Autograd Forward) — once you can numerically verify a gradient, you can confidently implement it.",
        doc="docs/part-2-autodiff/ch-07-calculus-for-ml.md",
        funcs=[
            ("numericalGradient",
             "fn: (x: number) => number, x: number, h?: number", "number",
             "Symmetric finite difference: (f(x+h) - f(x-h)) / 2h. Approximates df/dx."),
            ("numericalGradientTensor",
             "fn: (t: Tensor) => number, t: Tensor, h?: number", "Tensor",
             "Compute the numerical gradient of a scalar-valued function w.r.t. each element of t."),
            ("checkGradient",
             "analytical: number, numerical: number, tolerance?: number", "boolean",
             "Return true if analytical and numerical gradients agree within tolerance."),
        ],
        tests=[
            "numericalGradient of x^2 at x=3 ≈ 6",
            "numericalGradient of sin(x) at x=0 ≈ 1",
            "numericalGradient of exp(x) at x=0 ≈ 1",
            "checkGradient passes for gradients within tolerance",
            "checkGradient fails when gradients differ significantly",
        ]
    ),
    dict(
        folder="ch-08a-autograd-forward",
        num="08a", title="Autograd — Forward Pass",
        part=2, part_name="Autodiff Engine",
        what="A scalar-valued computation graph node (Value) that records its operation and inputs so the backward pass can trace the graph.",
        why="The computation graph is the core data structure of automatic differentiation. Every operation wraps its result in a Value, recording how it was computed.",
        unlocks="Ch 08b (Autograd Backward) — backward traversal uses the graph built here.",
        doc="docs/part-2-autodiff/ch-08a-autograd-forward.md",
        funcs=[
            ("Value.constructor", "data: number", "Value",
             "Create a leaf Value node with the given scalar data."),
            ("Value.add", "other: Value", "Value",
             "Return a new Value representing this + other, recording both inputs."),
            ("Value.mul", "other: Value", "Value",
             "Return a new Value representing this * other, recording both inputs."),
            ("Value.pow", "exponent: number", "Value",
             "Return a new Value representing this^exponent."),
            ("Value.exp", "", "Value", "Return a new Value representing e^this."),
            ("Value.log", "", "Value", "Return a new Value representing ln(this)."),
            ("Value.tanh", "", "Value", "Return a new Value representing tanh(this)."),
            ("Value.relu", "", "Value", "Return a new Value representing max(0, this)."),
        ],
        tests=[
            "addition produces the correct scalar value",
            "multiplication produces the correct scalar value",
            "the _inputs array records both operands of a binary op",
            "chained operations build a graph of depth > 1",
            "exp(0).data === 1",
            "tanh(0).data === 0",
        ]
    ),
    dict(
        folder="ch-08b-autograd-backward",
        num="08b", title="Autograd — Backward Pass",
        part=2, part_name="Autodiff Engine",
        what="Topological sort of the computation graph and reverse-mode autodiff: running _backward functions from output to inputs.",
        why="Backpropagation IS the backward pass of a computation graph. Every neural network training step calls backward() once per batch.",
        unlocks="Ch 09 (Gradient Descent) — SGD consumes the gradients produced here.",
        doc="docs/part-2-autodiff/ch-08b-autograd-backward.md",
        funcs=[
            ("topoSort", "root: Value", "Value[]",
             "Return all nodes reachable from root in topological order (inputs before outputs)."),
            ("Value.backward", "", "void",
             "Set this.grad = 1, then run _backward for all nodes in reverse topo order."),
            ("Value.zeroGrad", "", "void",
             "Reset grad to 0 on this node. Call before each new backward pass."),
        ],
        tests=[
            "backward on z = x*y computes dz/dx = y and dz/dy = x",
            "backward on z = x^2 computes dz/dx = 2x",
            "backward through a chain of ops accumulates gradients correctly",
            "zeroGrad resets grad to 0",
            "numerical gradient check passes for add, mul, pow, exp, log, tanh",
            "topoSort visits inputs before outputs",
        ]
    ),
    dict(
        folder="ch-09-gradient-descent",
        num="09", title="Gradient Descent",
        part=2, part_name="Autodiff Engine",
        what="SGD optimizer: update parameters by moving in the direction of the negative gradient.",
        why="Gradient descent is how neural networks learn. Every training step calls step() to move parameters closer to a loss minimum.",
        unlocks="Ch 10 (Tensor Autograd Bridge) — next we extend autograd from scalar to tensor.",
        doc="docs/part-2-autodiff/ch-09-gradient-descent.md",
        funcs=[
            ("SGD.constructor", "learningRate: number", "SGD",
             "Create an SGD optimizer with the given learning rate."),
            ("SGD.step", "params: Value[]", "void",
             "Update each parameter: param.data -= learningRate * param.grad."),
            ("SGD.zeroGrad", "params: Value[]", "void",
             "Reset grad to 0 on all parameters before the next backward pass."),
        ],
        tests=[
            "SGD step moves a parameter in the direction that reduces loss",
            "SGD on L=(w-3)^2 converges w toward 3",
            "zeroGrad resets all parameter gradients to 0",
            "learning rate controls step size: larger lr → larger parameter change",
        ]
    ),
    dict(
        folder="ch-10-tensor-autograd-bridge",
        num="10", title="Tensor Autograd Bridge",
        part=2, part_name="Autodiff Engine",
        what="Extends Value to wrap a full Tensor. Implements tensor-aware backward rules: broadcast, reduction, matmul, reshape, transpose.",
        why="Neural networks operate on tensors, not scalars. This chapter is the bridge that makes the rest of the course possible.",
        unlocks="Ch 11 (Activation Functions) — the first chapter that creates differentiable tensor operations.",
        doc="docs/part-2-autodiff/ch-10-tensor-autograd-bridge.md",
        funcs=[
            ("sumToShape", "grad: Tensor, targetShape: number[]", "Tensor",
             "Reverse broadcasting by summing a gradient over broadcasted dimensions."),
            ("Value.add (tensor)", "other: Value", "Value",
             "Tensor add with broadcast-aware backward: sums grad back to original shapes."),
            ("Value.mul (tensor)", "other: Value", "Value",
             "Tensor multiply with broadcast-aware backward."),
            ("Value.sum", "axis?: number, keepDims?: boolean", "Value",
             "Reduction with backward that broadcasts upstream gradient back to input shape."),
            ("Value.mean", "axis?: number, keepDims?: boolean", "Value",
             "Mean reduction with backward: distributes grad / n to each input element."),
            ("Value.matMul", "other: Value", "Value",
             "Tensor matmul with backward: dA = dZ @ B^T, dB = A^T @ dZ."),
            ("Value.reshape", "newShape: number[]", "Value",
             "Reshape forward with backward that reshapes grad back to original shape."),
            ("Value.transpose", "axes?: number[]", "Value",
             "Transpose forward with backward that applies inverse axis permutation."),
            ("checkTensorGradient",
             "fn: (inputs: Value[]) => Value, inputs: Value[], tolerance?: number", "boolean",
             "Numerical gradient check for tensor-valued operations."),
        ],
        tests=[
            "sumToShape sums over broadcasted axes correctly",
            "add backward matches finite differences for equal shapes",
            "add backward matches finite differences for broadcasted shapes",
            "mean backward gives each element grad/n",
            "matMul backward matches finite differences for both operands",
            "reshape backward reshapes grad to original shape",
            "transpose backward applies the inverse permutation",
            "checkTensorGradient passes for all ops",
        ]
    ),

    # ── Part 3: Neural Net Primitives ───────────────────────────────────────
    dict(
        folder="ch-11-activation-functions",
        num="11", title="Activation Functions",
        part=3, part_name="Neural Net Primitives",
        what="Differentiable nonlinear activation functions built on top of the tensor Value: ReLU, GELU, sigmoid, tanh, softmax.",
        why="Without nonlinearity, stacked linear layers collapse into a single linear map and cannot learn complex functions.",
        unlocks="Ch 12 (Loss Functions) — uses softmax for the cross-entropy loss.",
        doc="docs/part-3-neural-net-primitives/ch-11-activation-functions.md",
        funcs=[
            ("relu", "x: Value", "Value",
             "ReLU(x) = max(0, x). Gradient is 1 for x > 0, else 0."),
            ("gelu", "x: Value", "Value",
             "GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³))). Used inside FFN blocks in GPT."),
            ("sigmoid", "x: Value", "Value",
             "σ(x) = 1/(1+e^-x). Gradient is σ(x)(1-σ(x))."),
            ("softmax", "x: Value, axis?: number", "Value",
             "Numerically stable softmax. Produces attention weights and output probabilities."),
        ],
        tests=[
            "relu(2) === 2 and relu(-2) === 0",
            "relu gradient is 0 for negative inputs and 1 for positive inputs",
            "sigmoid output is in (0,1) for any finite input",
            "sigmoid(0) === 0.5",
            "softmax output sums to 1.0 along the target axis",
            "gelu is smooth near x=0 unlike ReLU",
            "numerical gradient check passes for relu, sigmoid, gelu, softmax",
        ]
    ),
    dict(
        folder="ch-12-loss-functions",
        num="12", title="Loss Functions",
        part=3, part_name="Neural Net Primitives",
        what="MSE loss and cross-entropy loss, with a numerically stable log-sum-exp used inside cross-entropy.",
        why="The loss function is what gradient descent minimizes. Cross-entropy is the training objective for every language model in this course.",
        unlocks="Ch 13 (Linear Layer) — linear layers need a loss to train against.",
        doc="docs/part-3-neural-net-primitives/ch-12-loss-functions.md",
        funcs=[
            ("mseLoss", "predictions: Value, targets: Tensor", "Value",
             "Mean squared error: mean((predictions - targets)^2)."),
            ("logSumExp", "x: Value, axis?: number", "Value",
             "Numerically stable log(sum(exp(x))): subtract max before computing."),
            ("crossEntropyFromLogits", "logits: Value, targets: Tensor", "Value",
             "Stable cross-entropy combining log-sum-exp and the correct logit. Gradient shortcut: softmax(logits) - one_hot(targets)."),
        ],
        tests=[
            "mseLoss of perfect predictions is 0",
            "mseLoss gradient points toward targets",
            "logSumExp is shift-invariant: logSumExp(x) === logSumExp(x+c) + c",
            "crossEntropyFromLogits produces lower loss when the correct class has higher logit",
            "numerical gradient check passes for mseLoss and crossEntropyFromLogits",
        ]
    ),
    dict(
        folder="ch-13-linear-layer",
        num="13", title="Linear Layer",
        part=3, part_name="Neural Net Primitives",
        what="A fully-connected linear layer: y = xW^T + b, with Kaiming and Xavier initialisation options.",
        why="Linear is the most-used module in transformers. Q, K, V projections, FFN layers, and the output head are all Linear layers.",
        unlocks="Ch 14 (Optimizers) — the layer produces parameters that the optimizer will update.",
        doc="docs/part-3-neural-net-primitives/ch-13-linear-layer.md",
        funcs=[
            ("Linear.constructor",
             "inputDim: number, outputDim: number, bias?: boolean, init?: 'he'|'xavier'",
             "Linear",
             "Initialise weight [outputDim, inputDim] and optional bias [outputDim]."),
            ("Linear.forward", "x: Value", "Value",
             "Compute y = x @ W^T + b. Input shape: [*, inputDim]. Output: [*, outputDim]."),
            ("Linear.parameters", "", "Value[]",
             "Return [weight, bias] (or [weight] when bias=false) for the optimizer."),
        ],
        tests=[
            "forward output shape is [..., outputDim]",
            "forward with bias adds a learnable offset",
            "parameters() returns the weight and bias Values",
            "numerical gradient check passes for forward w.r.t. weight and bias",
            "He init variance ≈ 2/inputDim for large layers",
        ]
    ),
    dict(
        folder="ch-14-optimizers",
        num="14", title="Optimizers",
        part=3, part_name="Neural Net Primitives",
        what="SGD with momentum, and Adam — the two optimizers used throughout this course.",
        why="Adam is the default optimizer for transformers. SGD + momentum is the baseline that Adam improves on.",
        unlocks="Ch 15 (Training Loop) — the loop calls optimizer.step() after every backward pass.",
        doc="docs/part-3-neural-net-primitives/ch-14-optimizers.md",
        funcs=[
            ("SGDMomentum.constructor", "learningRate: number, momentum?: number",
             "SGDMomentum", "SGD with momentum buffer. Default momentum=0.9."),
            ("SGDMomentum.step", "params: Value[]", "void",
             "Update velocity buffer and parameters: v = momentum*v - lr*grad; param += v."),
            ("Adam.constructor",
             "learningRate?: number, beta1?: number, beta2?: number, epsilon?: number",
             "Adam",
             "Adam with defaults lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8."),
            ("Adam.step", "params: Value[]", "void",
             "Update first/second moment estimates and apply bias-corrected parameter update."),
            ("Adam.zeroGrad", "params: Value[]", "void",
             "Zero gradients on all parameters."),
        ],
        tests=[
            "Adam step reduces loss on a simple quadratic",
            "SGD with momentum accumulates velocity",
            "Adam bias correction dominates at step 1",
            "zeroGrad resets all gradients",
            "Adam converges faster than vanilla SGD on L=(w-3)^2",
        ]
    ),
    dict(
        folder="ch-15-training-loop",
        num="15", title="Training Loop",
        part=3, part_name="Neural Net Primitives",
        what="A complete MLP (multi-layer perceptron) and full training loop: forward → loss → zero grad → backward → step.",
        why="This is the exact pattern used to train every model in this course, at every scale from toy MLP to full transformer.",
        unlocks="Ch 16 (Char Tokenizer) — begins Part 4, the language-model input stack.",
        doc="docs/part-3-neural-net-primitives/ch-15-training-loop.md",
        funcs=[
            ("MLP.constructor", "dims: number[]", "MLP",
             "Stack Linear layers: dims[i] → dims[i+1] with ReLU between layers."),
            ("MLP.forward", "x: Value", "Value",
             "Forward pass through all layers with ReLU activations (no activation on output)."),
            ("MLP.parameters", "", "Value[]",
             "Flatten all layer parameters into a single array for the optimizer."),
            ("generateSpiral",
             "n: number, classes: number, noise?: number",
             "{ inputs: Tensor, labels: Tensor }",
             "2-D spiral dataset — a classic nonlinear classification benchmark."),
            ("accuracy", "logits: Value, labels: Tensor", "number",
             "Fraction of samples where argmax(logits) === label."),
            ("train",
             "model: MLP, inputs: Value, labels: Value, opts: { epochs: number, lr: number }",
             "number[]",
             "Run the full training loop and return the loss history."),
        ],
        tests=[
            "MLP forward output shape is [batch, outputDim]",
            "MLP achieves near-zero training loss on XOR after enough epochs",
            "loss decreases monotonically on a simple convex objective",
            "train returns a loss history of length opts.epochs",
            "MLP achieves > 90% accuracy on spiral dataset with 3 hidden layers",
        ]
    ),

    # ── Part 4: Tokenizer & Inputs ──────────────────────────────────────────
    dict(
        folder="ch-16-char-tokenizer",
        num="16", title="Character-Level Tokenizer",
        part=4, part_name="Language Model Inputs",
        what="A character-level tokenizer with special tokens (PAD, UNK, BOS, EOS), batch encoding with padding, and an attention mask.",
        why="Text must become integers before a neural network can process it. The tokenizer is the entry point of every language model pipeline.",
        unlocks="Ch 17 (BPE Tokenizer) or skip directly to Ch 18 (Token Embeddings).",
        doc="docs/part-4-tokenizer-and-inputs/ch-16-char-tokenizer.md",
        funcs=[
            ("buildVocab", "text: string", "{ stoi: Map<string, number>, itos: Map<number, string>, vocabSize: number }",
             "Scan text, collect unique characters, assign integer IDs starting after special tokens."),
            ("CharTokenizer.constructor", "text: string", "CharTokenizer",
             "Build vocabulary from text and assign IDs to PAD, UNK, BOS, EOS special tokens."),
            ("CharTokenizer.encode", "text: string, maxLen?: number", "number[]",
             "String → integer IDs. Unknown chars map to UNK. Optionally truncates/pads to maxLen."),
            ("CharTokenizer.decode", "ids: number[]", "string",
             "Integer IDs → string. Special tokens are skipped."),
            ("CharTokenizer.encodeBatch",
             "texts: string[], maxLen: number",
             "{ ids: Tensor, mask: Tensor }",
             "Encode multiple strings with padding to maxLen. Returns ids and binary attention mask."),
        ],
        tests=[
            "encode then decode is a round trip for known characters",
            "unknown characters map to UNK id",
            "encodeBatch pads shorter sequences to maxLen",
            "attention mask is 1 for real tokens and 0 for padding",
            "vocabSize includes all unique chars plus special tokens",
        ]
    ),
    dict(
        folder="ch-17-bpe-tokenizer",
        num="17", title="BPE Tokenizer",
        part=4, part_name="Language Model Inputs",
        what="Byte-Pair Encoding: iteratively merge the most frequent adjacent pair to build a subword vocabulary.",
        why="BPE tokenization handles rare and compound words efficiently. GPT-2 uses BPE with 50k merges.",
        unlocks="Ch 18 (Token Embeddings) — BPE produces integer IDs that feed the embedding table.",
        doc="docs/part-4-tokenizer-and-inputs/ch-17-bpe-tokenizer.md",
        funcs=[
            ("countPairs", "corpus: string[][]", "Map<string, number>",
             "Count adjacent symbol pairs across all tokenized strings in the corpus."),
            ("mergePair", "corpus: string[][], pair: [string, string]", "string[][]",
             "Replace every occurrence of pair with a new merged symbol."),
            ("BPETokenizer.train", "text: string, vocabSize: number", "void",
             "Run BPE training: start with char vocab, merge most-frequent pair repeatedly."),
            ("BPETokenizer.encode", "text: string", "number[]",
             "Apply learned merges to segment text into subword IDs."),
            ("BPETokenizer.decode", "ids: number[]", "string",
             "Convert subword IDs back to text."),
        ],
        tests=[
            "training reduces the number of distinct tokens versus character-level",
            "most frequent pair is merged first",
            "encode then decode is a round trip",
            "merged tokens appear in vocabulary after training",
        ]
    ),
    dict(
        folder="ch-18-token-embeddings",
        num="18", title="Token Embeddings",
        part=4, part_name="Language Model Inputs",
        what="A learnable embedding lookup table that maps integer token IDs to dense dModel-dimensional vectors.",
        why="Embeddings are the first trainable layer every language model. Words that appear in similar contexts will have similar embedding vectors after training.",
        unlocks="Ch 19 (Positional Encoding) — PE is added to the embedding output.",
        doc="docs/part-4-tokenizer-and-inputs/ch-18-token-embeddings.md",
        funcs=[
            ("Embedding.constructor", "vocabSize: number, dModel: number", "Embedding",
             "Initialise a weight table of shape [vocabSize, dModel] with small normal values."),
            ("Embedding.forward", "ids: Tensor", "Value",
             "Lookup rows from the weight table. ids: [batch, seq] → output: [batch, seq, dModel]."),
            ("Embedding.parameters", "", "Value[]",
             "Return [this.weight] — the full embedding table is one parameter."),
        ],
        tests=[
            "forward output shape is [batch, seq, dModel]",
            "each row of the output matches the corresponding row of the weight matrix",
            "gradient flows back only to the rows that were looked up",
            "numerical gradient check passes for embedding lookup",
        ]
    ),
    dict(
        folder="ch-19-positional-encoding",
        num="19", title="Positional Encoding",
        part=4, part_name="Language Model Inputs",
        what="Sinusoidal positional encoding added to token embeddings so the model knows token order.",
        why="Self-attention is permutation-equivariant — without positional information the model cannot distinguish 'dog bites man' from 'man bites dog'.",
        unlocks="Ch 20 (LayerNorm & Dropout) — LayerNorm is applied after PE in transformer blocks.",
        doc="docs/part-4-tokenizer-and-inputs/ch-19-positional-encoding.md",
        funcs=[
            ("PositionalEncoding.constructor", "maxSeqLen: number, dModel: number", "PositionalEncoding",
             "Precompute the PE table: PE[pos, 2i] = sin(pos/10000^(2i/d)), PE[pos, 2i+1] = cos(...)."),
            ("PositionalEncoding.forward", "x: Value", "Value",
             "Add PE[:seqLen, :] to x. x shape: [batch, seqLen, dModel]."),
        ],
        tests=[
            "PE table shape is [maxSeqLen, dModel]",
            "even columns are sin, odd columns are cos",
            "forward output shape matches input shape",
            "PE values are bounded in [-1, 1]",
            "adding PE does not change the shape of the tensor",
            "PE for different positions are different vectors",
        ]
    ),
    dict(
        folder="ch-20-layernorm-dropout",
        num="20", title="LayerNorm & Dropout",
        part=4, part_name="Language Model Inputs",
        what="Layer normalisation (pre-norm variant) and inverted dropout — the two regularisation/stabilisation tools used inside every transformer block.",
        why="LayerNorm makes deep transformer training stable. Dropout prevents overfitting, especially on small datasets.",
        unlocks="Ch 21 (Mask Cookbook) — the last input-stack piece before attention.",
        doc="docs/part-4-tokenizer-and-inputs/ch-20-layernorm-dropout.md",
        funcs=[
            ("LayerNorm.constructor", "dModel: number, eps?: number", "LayerNorm",
             "Initialise gamma (ones) and beta (zeros) of shape [dModel]."),
            ("LayerNorm.forward", "x: Value", "Value",
             "Normalise over the last axis: xHat = (x-mean)/sqrt(var+eps), then y = gamma*xHat + beta."),
            ("LayerNorm.parameters", "", "Value[]",
             "Return [gamma, beta]."),
            ("Dropout.constructor", "rate: number", "Dropout",
             "Initialise with drop probability rate."),
            ("Dropout.forward", "x: Value", "Value",
             "During training: zero random elements, scale survivors by 1/(1-rate). During eval: identity."),
            ("Dropout.train", "", "void", "Enable dropout (training mode)."),
            ("Dropout.eval", "", "void", "Disable dropout (evaluation mode)."),
        ],
        tests=[
            "LayerNorm output has mean ≈ 0 and std ≈ 1 along last axis",
            "LayerNorm preserves shape",
            "gamma and beta are learnable: gradient flows to both",
            "Dropout zeros approximately rate fraction of elements during training",
            "Dropout eval mode returns input unchanged",
            "numerical gradient check passes for LayerNorm forward",
        ]
    ),
    dict(
        folder="ch-21-mask-cookbook",
        num="21", title="Attention Mask Cookbook",
        part=4, part_name="Language Model Inputs",
        what="Utilities to create, convert, expand, and combine attention masks — from tokenizer binary masks to the additive masks consumed by softmax.",
        why="Mask bugs are silent: the model still runs, but attends to padding or future tokens and fails to learn. These utilities make masking explicit and testable.",
        unlocks="Ch 22 (Self-Attention) — attention can now consume masks with clear, tested semantics.",
        doc="docs/part-4-tokenizer-and-inputs/ch-21-mask-cookbook.md",
        funcs=[
            ("binaryMaskFromIds", "ids: Tensor, padId: number", "Tensor",
             "1 for non-pad token IDs, 0 for pad IDs."),
            ("toAdditiveMask", "binaryMask: Tensor", "Tensor",
             "Convert 1/0 binary mask to 0/-1e9 additive mask."),
            ("expandPaddingMask", "mask: Tensor", "Tensor",
             "Reshape [batch, keyLen] to [batch, 1, 1, keyLen] for attention score broadcasting."),
            ("causalMask", "queryLen: number, keyLen?: number", "Tensor",
             "Upper-triangular additive mask: 0 for allowed positions, -1e9 for future positions."),
            ("expandCausalMask", "mask: Tensor", "Tensor",
             "Reshape [queryLen, keyLen] to [1, 1, queryLen, keyLen]."),
            ("combineMasks", "masks: Tensor[]", "Tensor",
             "Sum broadcast-compatible additive masks. Any position blocked by either mask is -1e9."),
            ("makeDecoderSelfAttentionMask",
             "targetIds: Tensor, padId: number", "Tensor",
             "Combine target padding mask and causal mask for decoder self-attention."),
        ],
        tests=[
            "pad tokens produce 0 in binary mask",
            "toAdditiveMask converts 1 to 0 and 0 to -1e9",
            "expandPaddingMask produces shape [batch, 1, 1, keyLen]",
            "causalMask blocks upper triangle (future positions)",
            "combined decoder mask blocks both future tokens and padding",
            "after softmax, masked positions receive probability ≈ 0",
        ]
    ),

    # ── Part 5: Attention ───────────────────────────────────────────────────
    dict(
        folder="ch-22-self-attention",
        num="22", title="Self-Attention",
        part=5, part_name="Attention Mechanism",
        what="Scaled dot-product self-attention with Q, K, V projections and optional causal masking.",
        why="Self-attention is the core mechanism of transformers. It lets every token directly attend to every other token in a single O(n²) operation.",
        unlocks="Ch 23 (Multi-Head Attention) — runs multiple attention heads in parallel.",
        doc="docs/part-5-attention/ch-22-self-attention.md",
        funcs=[
            ("scaledDotProductAttention",
             "Q: Value, K: Value, V: Value, mask?: Tensor", "Value",
             "Compute softmax((Q @ K^T) / sqrt(dHead)) @ V. Shapes: Q/K/V [batch, seq, dHead]."),
            ("SelfAttention.constructor",
             "dModel: number, dHead: number", "SelfAttention",
             "Initialise W_Q, W_K, W_V projection matrices."),
            ("SelfAttention.forward", "x: Value, mask?: Tensor", "Value",
             "Project x to Q, K, V then run scaledDotProductAttention."),
            ("SelfAttention.parameters", "", "Value[]",
             "Return all three projection matrix parameters."),
        ],
        tests=[
            "forward output shape is [batch, seq, dHead]",
            "attention weights sum to 1.0 along the key dimension",
            "causal mask prevents token i from attending to token j > i",
            "scaling by 1/sqrt(dHead) reduces logit magnitude",
            "numerical gradient check passes for scaledDotProductAttention",
        ]
    ),
    dict(
        folder="ch-23-multi-head-attention",
        num="23", title="Multi-Head Attention",
        part=5, part_name="Attention Mechanism",
        what="Multi-head attention: run numHeads attention heads in parallel, split from dModel, then concat and project back.",
        why="Multiple heads let the model attend to different types of relationships simultaneously (syntax, semantics, coreference, etc).",
        unlocks="Ch 24 (Cross-Attention) — cross-attention reuses the multi-head attention pattern.",
        doc="docs/part-5-attention/ch-23-multi-head-attention.md",
        funcs=[
            ("MultiHeadAttention.constructor",
             "dModel: number, numHeads: number", "MultiHeadAttention",
             "Initialise W_Q, W_K, W_V of shape [dModel, dModel] and W_O output projection."),
            ("MultiHeadAttention.splitHeads", "x: Value", "Value",
             "Reshape [batch, seq, dModel] → [batch, numHeads, seq, dHead]."),
            ("MultiHeadAttention.mergeHeads", "x: Value", "Value",
             "Reshape [batch, numHeads, seq, dHead] → [batch, seq, dModel]."),
            ("MultiHeadAttention.forward", "x: Value, mask?: Tensor", "Value",
             "Project to Q/K/V, split heads, attend, merge, project output."),
            ("MultiHeadAttention.parameters", "", "Value[]",
             "Return all four projection matrices."),
        ],
        tests=[
            "forward output shape is [batch, seq, dModel]",
            "splitHeads produces shape [batch, numHeads, seq, dHead]",
            "mergeHeads inverts splitHeads",
            "dModel must be divisible by numHeads",
            "numerical gradient check passes for forward pass",
        ]
    ),
    dict(
        folder="ch-24-cross-attention",
        num="24", title="Cross-Attention",
        part=5, part_name="Attention Mechanism",
        what="Cross-attention: Q from the decoder, K and V from the encoder — the bridge between encoder and decoder in the full transformer.",
        why="Cross-attention is what lets the decoder 'read' the encoder's representation of the source sequence at every decoder step.",
        unlocks="Ch 25 (FFN Block) — with attention done, we build the remaining sublayer of each transformer block.",
        doc="docs/part-5-attention/ch-24-cross-attention.md",
        funcs=[
            ("CrossAttention.constructor",
             "dModel: number, numHeads: number", "CrossAttention",
             "Same projection matrices as MultiHeadAttention; Q comes from decoder, K/V from encoder."),
            ("CrossAttention.forward",
             "query: Value, context: Value, mask?: Tensor", "Value",
             "Compute multi-head attention with Q=query, K=context, V=context."),
            ("CrossAttention.parameters", "", "Value[]",
             "Return all four projection matrices."),
        ],
        tests=[
            "forward output shape matches query shape [batch, tgtLen, dModel]",
            "K and V are derived from context, not query",
            "source mask blocks encoder padding positions in cross-attention",
            "numerical gradient check passes for cross-attention forward",
        ]
    ),

    # ── Part 6: Transformer ─────────────────────────────────────────────────
    dict(
        folder="ch-25-feedforward-block",
        num="25", title="Feed-Forward Block",
        part=6, part_name="Transformer",
        what="The position-wise FFN: two linear layers with GELU activation and a 4× hidden expansion.",
        why="The FFN is the second sublayer in every transformer block and accounts for ~2/3 of all parameters.",
        unlocks="Ch 26 (Encoder Block) — FFN + attention + LayerNorm + residual = one encoder block.",
        doc="docs/part-6-transformer/ch-25-feedforward-block.md",
        funcs=[
            ("FFN.constructor",
             "dModel: number, dFf?: number", "FFN",
             "Two Linear layers: dModel → dFf (default 4×dModel) → dModel."),
            ("FFN.forward", "x: Value", "Value",
             "Apply linear1, GELU, linear2. Shape: [batch, seq, dModel] → same."),
            ("FFN.parameters", "", "Value[]",
             "Return parameters from both linear layers."),
        ],
        tests=[
            "forward output shape is [batch, seq, dModel]",
            "intermediate hidden dim is 4 × dModel by default",
            "GELU activation is applied between the two linear layers",
            "numerical gradient check passes for FFN forward",
        ]
    ),
    dict(
        folder="ch-26-encoder-block",
        num="26", title="Encoder Block",
        part=6, part_name="Transformer",
        what="One complete encoder block: pre-norm → multi-head self-attention → residual → pre-norm → FFN → residual. Stacked into Encoder.",
        why="The encoder block is the fundamental repeating unit of the transformer encoder. Understanding one block means understanding all N.",
        unlocks="Ch 27 (Decoder Block) — decoder block adds cross-attention between self-attention and FFN.",
        doc="docs/part-6-transformer/ch-26-encoder-block.md",
        funcs=[
            ("EncoderBlock.constructor",
             "dModel: number, numHeads: number, dFf: number, dropoutRate?: number",
             "EncoderBlock",
             "Initialise LayerNorm × 2, MultiHeadAttention, FFN, Dropout."),
            ("EncoderBlock.forward", "x: Value, mask?: Tensor", "Value",
             "Pre-norm self-attention sublayer then pre-norm FFN sublayer with residual connections."),
            ("EncoderBlock.parameters", "", "Value[]",
             "All parameters from attention, FFN, and LayerNorms."),
            ("Encoder.constructor",
             "numLayers: number, dModel: number, numHeads: number, dFf: number, dropoutRate?: number",
             "Encoder",
             "Stack numLayers EncoderBlocks."),
            ("Encoder.forward", "x: Value, mask?: Tensor", "Value",
             "Pass x through each EncoderBlock in sequence."),
        ],
        tests=[
            "forward output shape is [batch, seq, dModel]",
            "residual connection is present: output includes the input",
            "pre-norm is applied before attention and before FFN",
            "stacked encoder output shape matches single block output shape",
        ]
    ),
    dict(
        folder="ch-27-decoder-block",
        num="27", title="Decoder Block",
        part=6, part_name="Transformer",
        what="One decoder block: masked self-attention + cross-attention + FFN, each with pre-norm and residual. Stacked into Decoder.",
        why="The decoder generates output tokens autoregressively. Its causal mask ensures position i never sees position j > i during training.",
        unlocks="Ch 28 (Sequence Data & Objectives) — now we need the data pipeline to train the full model.",
        doc="docs/part-6-transformer/ch-27-decoder-block.md",
        funcs=[
            ("DecoderBlock.constructor",
             "dModel: number, numHeads: number, dFf: number, dropoutRate?: number",
             "DecoderBlock",
             "Initialise LayerNorm × 3, MaskedSelfAttention, CrossAttention, FFN, Dropout."),
            ("DecoderBlock.forward",
             "x: Value, encoderOut: Value, srcMask?: Tensor, tgtMask?: Tensor", "Value",
             "Masked self-attention → cross-attention → FFN, all pre-norm with residuals."),
            ("DecoderBlock.parameters", "", "Value[]",
             "All parameters from both attention modules, FFN, and LayerNorms."),
            ("Decoder.constructor",
             "numLayers: number, dModel: number, numHeads: number, dFf: number, dropoutRate?: number",
             "Decoder",
             "Stack numLayers DecoderBlocks."),
            ("Decoder.forward",
             "x: Value, encoderOut: Value, srcMask?: Tensor, tgtMask?: Tensor", "Value",
             "Pass x through each DecoderBlock, passing encoderOut and masks unchanged."),
        ],
        tests=[
            "forward output shape is [batch, tgtLen, dModel]",
            "causal mask prevents attending to future target tokens",
            "cross-attention query comes from decoder, K/V come from encoder",
            "stacked decoder output shape matches single block output",
        ]
    ),
    dict(
        folder="ch-28-sequence-data-objectives",
        num="28", title="Sequence Data & Training Objectives",
        part=6, part_name="Transformer",
        what="Sequence batching, shifted decoder targets, loss masking, perplexity, and an overfit-one-batch debug helper.",
        why="Most transformer bugs are data-pipeline bugs, not model bugs. This chapter makes training objectives explicit and testable before the final assembly.",
        unlocks="Ch 29 (Full Transformer) — the final model has a verified training pipeline.",
        doc="docs/part-6-transformer/ch-28-sequence-data-objectives.md",
        funcs=[
            ("shiftRight", "targetIds: Tensor, bosId: number", "Tensor",
             "Build decoder input by prepending BOS and removing the last token."),
            ("maskedCrossEntropy", "logits: Value, targetIds: Tensor, lossMask: Tensor", "Value",
             "Cross-entropy averaged only over positions where lossMask === 1."),
            ("perplexity", "loss: number", "number",
             "Exponentiated average loss: e^loss."),
            ("makeReversalDataset",
             "count: number, minLen: number, maxLen: number, alphabet?: string",
             "Array<{ source: string, target: string }>",
             "Generate toy source→target pairs where target is the reversed source."),
            ("splitDataset",
             "examples: unknown[], validFraction: number, seed?: number",
             "{ train: unknown[], valid: unknown[] }",
             "Deterministic train/validation split using a seeded shuffle."),
        ],
        tests=[
            "shiftRight prepends BOS and shifts all tokens right by one",
            "maskedCrossEntropy ignores pad positions",
            "perplexity of loss=0 is 1",
            "reversal dataset target equals reversed source string",
            "dataset split is deterministic with the same seed",
            "one-batch overfit reduces loss to near zero",
        ]
    ),
    dict(
        folder="ch-29-full-transformer",
        num="29", title="Full Transformer",
        part=6, part_name="Transformer",
        what="Complete encoder-decoder Transformer assembled from all previous chapters. Train it on character-level sequence reversal.",
        why="This is the original 'Attention Is All You Need' architecture, reproduced from scratch. Every chapter from 01 to 28 feeds into this.",
        unlocks="Ch 30 (Decoder-Only GPT) — the modern decoder-only variant.",
        doc="docs/part-6-transformer/ch-29-full-transformer.md",
        funcs=[
            ("Transformer.constructor", "cfg: TransformerConfig", "Transformer",
             "Initialise source/target embeddings, PE, encoder, decoder, final LayerNorm, output projection."),
            ("Transformer.encode", "src: Tensor, srcMask?: Tensor", "Value",
             "Embed + scale + PE + dropout + encoder stack."),
            ("Transformer.decode",
             "tgt: Tensor, encoderOut: Value, srcMask?: Tensor, tgtMask?: Tensor", "Value",
             "Embed + scale + PE + dropout + decoder stack + final LayerNorm."),
            ("Transformer.forward", "src: Tensor, tgt: Tensor, srcMask?: Tensor", "Value",
             "Full forward pass: encode + decode + output projection → logits."),
            ("Transformer.parameters", "", "Value[]",
             "All parameters from all sub-modules."),
            ("generate",
             "model: Transformer, srcTokens: number[], maxLen: number, bosId: number, eosId: number",
             "number[]",
             "Greedy autoregressive decoding: extend target one token at a time until EOS."),
        ],
        tests=[
            "forward output shape is [batch, tgtLen, vocabSize]",
            "encoder output shape is [batch, srcLen, dModel]",
            "model can overfit a single reversal example",
            "generate produces EOS at the end for a trained model",
            "parameter count matches theoretical calculation for given config",
        ]
    ),
    dict(
        folder="ch-30-decoder-only-gpt",
        num="30", title="Decoder-Only GPT",
        part=6, part_name="Transformer",
        what="A GPT-style decoder-only transformer trained with next-token prediction. No encoder, no cross-attention — just masked self-attention and FFN.",
        why="Decoder-only language models are the dominant architecture behind modern large language models. This chapter completes the course.",
        unlocks="KV cache, LoRA, MoE, quantization — see docs/part-6-transformer/ch-30-decoder-only-gpt.md.",
        doc="docs/part-6-transformer/ch-30-decoder-only-gpt.md",
        funcs=[
            ("DecoderOnlyBlock.constructor",
             "dModel: number, numHeads: number, dFf: number, dropoutRate?: number",
             "DecoderOnlyBlock",
             "Pre-norm masked self-attention + FFN sublayers. No cross-attention."),
            ("DecoderOnlyBlock.forward", "x: Value, causalMask: Tensor", "Value",
             "Self-attention then FFN, both pre-norm with residual connections."),
            ("GPT.constructor", "cfg: GPTConfig", "GPT",
             "Token embedding, positional encoding, N decoder-only blocks, final LayerNorm, output projection."),
            ("GPT.forward", "inputIds: Tensor", "Value",
             "Embed, add PE, run blocks with causal mask, project to logits. Shape: [batch, seq, vocabSize]."),
            ("GPT.parameters", "", "Value[]",
             "All parameters from all sub-modules."),
            ("makeLanguageModelingBatch",
             "tokenIds: number[], blockSize: number, batchSize: number",
             "{ inputs: Tensor, targets: Tensor }",
             "Sample random windows of length blockSize; target is input shifted right by 1."),
            ("generateText",
             "model: GPT, tokenizer: { encode: (s: string) => number[], decode: (ids: number[]) => string }, prompt: string, maxNewTokens: number, temperature?: number",
             "string",
             "Autoregressive token generation with optional temperature sampling."),
        ],
        tests=[
            "forward output shape is [batch, seq, vocabSize]",
            "causal mask ensures position i cannot attend to position j > i",
            "language modeling batch shifts targets by one token",
            "model can overfit a tiny repeated string to near-zero loss",
            "greedy generation is deterministic for the same input",
            "temperature < 1 makes distribution sharper, temperature > 1 makes it flatter",
        ]
    ),
]

# ---------------------------------------------------------------------------
# Template helpers
# ---------------------------------------------------------------------------

def make_index_ts(ch: dict) -> str:
    imports_hint = ""
    if ch["part"] > 1:
        prev_part = ch["part"]
        prev_num_hint = "previous chapter"
        imports_hint = f"// import {{ ... }} from \"../ch-{str(int(ch['num'].replace('a','').replace('b','')) - 1).zfill(2)}-*/index.ts\";"

    func_stubs = []
    for name, params, ret, desc in ch["funcs"]:
        clean_name = name.split(".")[0] if "." in name else name
        func_stubs.append(f"""/**
 * {desc}
 */
export function {clean_name}({params}): {ret} {{
  throw new Error("Not implemented — read {ch['doc']} first");
}}""")

    funcs_text = "\n\n".join(func_stubs)
    num_str = f"Chapter {ch['num'].upper()}: {ch['title']}"
    eq_line = "═" * max(len(num_str), 40)

    return f"""/**
 * {num_str}
 * {eq_line}
 * Part {ch['part']} of 6: {ch['part_name']}
 *
 * WHAT WE'RE BUILDING:
 *   {ch['what']}
 *
 * WHY IT MATTERS:
 *   {ch['why']}
 *
 * WHAT THIS UNLOCKS:
 *   → {ch['unlocks']}
 *
 * REFERENCE: {ch['doc']}
 */

// ─── Imports from previous chapters ─────────────────────────────────────────
{imports_hint}

// ─── Types ───────────────────────────────────────────────────────────────────

// ─── Implementation ──────────────────────────────────────────────────────────

{funcs_text}

// ─── Demo (run with: bun run src/{ch['folder']}/index.ts) ────────────────────
// Uncomment to try:
// console.log("Chapter {ch['num']} demo:");
"""


def make_index_test(ch: dict) -> str:
    groups: dict[str, list[str]] = {}
    for name, _, _, _ in ch["funcs"]:
        top = name.split(".")[0]
        groups.setdefault(top, [])

    for test in ch["tests"]:
        first_func = list(groups.keys())[0] if groups else "general"
        placed = False
        for key in groups:
            if key.lower() in test.lower():
                groups[key].append(test)
                placed = True
                break
        if not placed:
            groups[first_func].append(test)

    describe_blocks = []
    for func_name, tests in groups.items():
        todo_lines = "\n".join(f'  it.todo("{t}");' for t in tests)
        describe_blocks.append(f"""describe("{func_name}", () => {{
{todo_lines}
}});""")

    blocks_text = "\n\n".join(describe_blocks)

    return f"""/**
 * Tests for Chapter {ch['num'].upper()}: {ch['title']}
 *
 * Each test verifies a specific mathematical property.
 * Read the test names — they teach you what the functions should do.
 *
 * Run: bun test src/{ch['folder']}
 */
import {{ describe, it, expect }} from "bun:test";
// Uncomment as you implement:
// import {{ {", ".join(f[0].split(".")[0] for f in ch["funcs"])} }} from "./index.ts";

const EPSILON = 1e-6;
// Helper: check two floats are close
const close = (a: number, b: number) => Math.abs(a - b) < EPSILON;

{blocks_text}
"""


# ---------------------------------------------------------------------------
# Write files
# ---------------------------------------------------------------------------

created = 0
for ch in CHAPTERS:
    folder = os.path.join(BASE, ch["folder"])
    os.makedirs(folder, exist_ok=True)

    ts_path = os.path.join(folder, "index.ts")
    test_path = os.path.join(folder, "index.test.ts")

    if not os.path.exists(ts_path):
        with open(ts_path, "w") as f:
            f.write(make_index_ts(ch))
        created += 1

    if not os.path.exists(test_path):
        with open(test_path, "w") as f:
            f.write(make_index_test(ch))
        created += 1

print(f"Created {created} files across {len(CHAPTERS)} chapters in {BASE}/")
print("Run: bun test  (all tests should show as todo/pending, none failing)")
