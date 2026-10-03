import { multiplyMatrices, transposeMatrix } from './matrix-multiply.js';

/**
 * Compute raw query-key scores as QKᵀ without changing the input precision.
 *
 * @param {{ matrix: number[][], shape: [number, number] }} Q
 * @param {{ matrix: number[][], shape: [number, number] }} K
 * @returns {{ KT: { matrix: number[][], shape: [number, number] }, S: { matrix: number[][], shape: [number, number] } }}
 */
export function computeAttentionScores(Q, K) {
  const KT = transposeMatrix(K);
  const S = multiplyMatrices(Q, KT);
  return { KT, S };
}
