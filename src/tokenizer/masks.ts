/**
 * tokenizer/masks.ts
 * ══════════════════════════════════════════════════════════
 * Attention mask utilities.
 *
 * Key insight: tokenizers produce BINARY masks (1=real, 0=pad), but
 * attention softmax needs ADDITIVE masks (0=attend, -1e9=block).
 * This module bridges the two representations and provides causal masks.
 *
 * Chapter: 21 — Attention Mask Cookbook
 * Doc:     docs/part-4-tokenizer-and-inputs/ch-21-mask-cookbook.md
 */
import type { Tensor } from "../tensor/index.ts";

/** Large negative value that drives a softmax probability to ≈ 0. */
export const MASK_VALUE = -1e9;

/**
 * Binary mask from token IDs: 1 for non-pad, 0 for pad.
 * Input shape: [batch, seq].  Output shape: [batch, seq].
 */
export function binaryMaskFromIds(ids: Tensor, padId: number): Tensor {
  throw new Error("binaryMaskFromIds not implemented");
}

/**
 * Convert 1/0 binary mask → 0/−1e9 additive mask.
 *   1 (real)    → 0      (logit unchanged)
 *   0 (padding) → −1e9   (logit → −∞ before softmax → prob ≈ 0)
 */
export function toAdditiveMask(binaryMask: Tensor): Tensor {
  throw new Error("toAdditiveMask not implemented");
}

/**
 * Reshape [batch, keyLen] → [batch, 1, 1, keyLen] so it broadcasts over
 * attention scores of shape [batch, heads, queryLen, keyLen].
 */
export function expandPaddingMask(mask: Tensor): Tensor {
  throw new Error("expandPaddingMask not implemented");
}

/**
 * Upper-triangular causal mask: position i cannot attend to j > i.
 *   shape:  [queryLen, keyLen]
 *   values: 0 for allowed positions, −1e9 for future positions.
 */
export function causalMask(queryLen: number, keyLen?: number): Tensor {
  throw new Error("causalMask not implemented");
}

/**
 * Expand causal mask [queryLen, keyLen] → [1, 1, queryLen, keyLen].
 */
export function expandCausalMask(mask: Tensor): Tensor {
  throw new Error("expandCausalMask not implemented");
}

/**
 * Sum a list of additive masks.  Any position blocked by any one mask
 * stays at −1e9 after softmax.
 */
export function combineMasks(masks: Tensor[]): Tensor {
  throw new Error("combineMasks not implemented");
}

/**
 * Build the combined decoder self-attention mask:
 *   target padding mask + causal mask
 * Shape: [batch, 1, tgtLen, tgtLen]
 */
export function makeDecoderSelfAttentionMask(
  targetIds: Tensor,
  padId: number
): Tensor {
  throw new Error("makeDecoderSelfAttentionMask not implemented");
}
