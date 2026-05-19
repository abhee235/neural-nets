/**
 * Tests for autograd/grad.ts
 * Chapter 10 — Tensor Autograd Bridge
 *
 * Run: bun test src/autograd/grad.test.ts
 */
import { describe, it, expect } from "bun:test";
// import { TensorValue, sumToShape, checkTensorGradient } from "./grad.ts";
// import { zeros, ones, createTensor } from "../tensor/index.ts";

describe("sumToShape", () => {
  it.todo("sums over broadcast axes to recover the original shape");
  it.todo("is a no-op when grad already has the target shape");
  it.todo("[3,4] summed to [1,4] reduces axis 0");
});

describe("TensorValue — forward", () => {
  it.todo("add output shape matches the broadcast shape");
  it.todo("matMul output shape is (M,N) for (M,K) × (K,N)");
  it.todo("sum reduces to a scalar when no axis is given");
});

describe("TensorValue — backward", () => {
  it.todo("add backward: each input receives the upstream gradient");
  it.todo("add backward with broadcast: grad is summed to original shape");
  it.todo("mean backward distributes grad/n to each element");
  it.todo("matMul backward: dA = dZ @ Bᵀ, dB = Aᵀ @ dZ");
  it.todo("reshape backward: grad has the original pre-reshape shape");
  it.todo("transpose backward applies the inverse permutation");
});

describe("checkTensorGradient", () => {
  it.todo("passes for add");
  it.todo("passes for matMul");
  it.todo("passes for mean");
});
