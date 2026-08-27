/**
 * optim/index.ts — public API for the optimizer module.
 *
 * import { SGD, SGDMomentum, Adam, AdamW } from "../optim/index.ts";
 *
 * All three operate on TensorValue, which is what Ch 13's
 * Linear.parameters() hands back.
 *
 * Ch 09's scalar versions live in sgd-scalar.ts as SGDScalar and
 * SGDMomentumScalar. They are deliberately NOT re-exported here: they are
 * reference material for reading beside the tensor rebuild, not part of the
 * library's surface. Import that file directly if you want them.
 */
export * from "./sgd.ts";
export * from "./adam.ts";
