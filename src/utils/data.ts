/**
 * utils/data.ts
 * ══════════════════════════════════════════════════════════
 * Dataset helpers: sequence batching, shifted targets, loss masking,
 * perplexity, and dataset splitting.
 *
 * Chapter: 28 — Sequence Data & Training Objectives
 * Doc:     docs/part-6-transformer/ch-28-sequence-data-objectives.md
 */
import type { Tensor } from "../tensor/index.ts";
import type { TensorValue } from "../autograd/grad.ts";

/**
 * Build decoder input by prepending BOS and dropping the last token.
 *   targets     = [w1, w2, w3, EOS]
 *   shiftRight  = [BOS, w1, w2, w3]
 */
export function shiftRight(targetIds: Tensor, bosId: number): Tensor {
  throw new Error("shiftRight not implemented");
}

/**
 * Cross-entropy averaged only over real (non-pad) token positions.
 * lossMask: 1 for real tokens, 0 for padding.
 */
export function maskedCrossEntropy(
  logits: TensorValue,
  targetIds: Tensor,
  lossMask: Tensor
): TensorValue {
  throw new Error("maskedCrossEntropy not implemented");
}

/**
 * Exponentiated average loss:  perplexity = e^loss
 * Perfect model → perplexity 1.  Random over V tokens → perplexity V.
 */
export function perplexity(loss: number): number {
  throw new Error("perplexity not implemented");
}

/**
 * Toy reversal dataset: target is the reversed source string.
 * Great for sanity-checking an encoder-decoder transformer.
 */
export function makeReversalDataset(
  count: number,
  minLen: number,
  maxLen: number,
  alphabet?: string
): Array<{ source: string; target: string }> {
  throw new Error("makeReversalDataset not implemented");
}

/**
 * Deterministic train/validation split using a seeded shuffle.
 */
export function splitDataset<T>(
  examples: T[],
  validFraction: number,
  seed?: number
): { train: T[]; valid: T[] } {
  throw new Error("splitDataset not implemented");
}
