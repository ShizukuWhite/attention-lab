import { multiplyMatrices } from './matrix-multiply.js';

function frozenMatrix(rows) {
  const matrix = Object.freeze(rows.map((row) => Object.freeze([...row])));
  const shape = Object.freeze([matrix.length, matrix[0]?.length ?? 2]);
  return Object.freeze({ matrix, shape });
}

export const PROJECTION_WEIGHTS = Object.freeze({
  Wq: frozenMatrix([[1, 0], [0, 1], [1, -1]]),
  Wk: frozenMatrix([[1, 1], [1, -1], [-1, 0]]),
  Wv: frozenMatrix([[1, -1], [1, 1], [0, 1]]),
});

/**
 * Apply the three fixed teaching projections to an embedding matrix X.
 *
 * @param {{ matrix: number[][], shape: [number, number] }} x
 * @returns {{ Q: { matrix: number[][], shape: [number, number] }, K: { matrix: number[][], shape: [number, number] }, V: { matrix: number[][], shape: [number, number] } }}
 */
export function projectQKV(x) {
  return {
    Q: multiplyMatrices(x, PROJECTION_WEIGHTS.Wq),
    K: multiplyMatrices(x, PROJECTION_WEIGHTS.Wk),
    V: multiplyMatrices(x, PROJECTION_WEIGHTS.Wv),
  };
}
