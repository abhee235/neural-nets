/**
 * neural-nets — A transformer library built from scratch in TypeScript.
 * ══════════════════════════════════════════════════════════════════════
 * Zero npm dependencies.  Every function you import was written by you.
 *
 * Import by module (recommended for large files):
 *
 *   import { Tensor, zeros, matMul }     from "./tensor/index.ts";
 *   import { TensorValue }               from "./autograd/index.ts";
 *   import { Linear, MultiHeadAttention, GPT } from "./nn/index.ts";
 *   import { Adam }                      from "./optim/index.ts";
 *   import { CharTokenizer, causalMask } from "./tokenizer/index.ts";
 *
 * Or import everything from here:
 *
 *   import { zeros, Linear, Adam, GPT, CharTokenizer } from "./index.ts";
 *
 * Implementation order follows the 30-chapter curriculum in docs/.
 */
export * from "./tensor/index.ts";
export * from "./autograd/index.ts";
export * from "./nn/index.ts";
export * from "./optim/index.ts";
export * from "./tokenizer/index.ts";
export * from "./utils/numerical.ts";
export * from "./utils/data.ts";
