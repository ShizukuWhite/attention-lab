import { multiplyMatrices, validateMatrix } from './matrix-multiply.js';

function validateDK(dK) {
  if (typeof dK !== 'number') {
    throw new TypeError('dK must be a positive safe integer.');
  }
  if (!Number.isSafeInteger(dK) || dK <= 0) {
    throw new RangeError('dK must be a positive safe integer.');
  }
}

/**
 * Divide every raw score by sqrt(dK), preserving the input precision.
 * @param {{ matrix: number[][], shape: [number, number] }} scores
 * @param {number} dK
 * @returns {{ matrix: number[][], shape: [number, number] }}
 */
export function scaleScores(scores, dK) {
  validateDK(dK);
  const { matrix, rows, columns } = validateMatrix(scores, 'Scores');
  return {
    matrix: matrix.map((row) => row.map((score) => score / Math.sqrt(dK))),
    shape: [rows, columns],
  };
}

/**
 * Apply a numerically stable softmax independently to each Query row.
 * @param {{ matrix: number[][], shape: [number, number] }} scaled
 * @returns {{ weights: { matrix: number[][], shape: [number, number] }, rowSteps: Array<{ row: number, max: number, denominator: number, weightSum: number, steps: Array<{ index: number, scaled: number, shifted: number, exponential: number, weight: number }> }> }}
 */
export function softmaxRows(scaled) {
  const { matrix, rows, columns } = validateMatrix(scaled, 'Scaled scores');
  if (rows > 0 && columns === 0) {
    throw new RangeError('Softmax requires at least one Key position per Query row.');
  }

  const weights = [];
  const rowSteps = [];
  for (let rowIndex = 0; rowIndex < rows; rowIndex += 1) {
    const scaledRow = matrix[rowIndex];
    let max = scaledRow[0];
    for (let columnIndex = 1; columnIndex < columns; columnIndex += 1) {
      if (scaledRow[columnIndex] > max) max = scaledRow[columnIndex];
    }
    const shifted = scaledRow.map((value) => value - max);
    const exponentials = shifted.map((value) => Math.exp(value));
    const denominator = exponentials.reduce((sum, value) => sum + value, 0);
    const weightRow = exponentials.map((value) => value / denominator);
    const steps = scaledRow.map((value, index) => ({
      index,
      scaled: value,
      shifted: shifted[index],
      exponential: exponentials[index],
      weight: weightRow[index],
    }));
    weights.push(weightRow);
    rowSteps.push({
      row: rowIndex,
      max,
      denominator,
      weightSum: weightRow.reduce((sum, value) => sum + value, 0),
      steps,
    });
  }

  return { weights: { matrix: weights, shape: [rows, columns] }, rowSteps };
}

/**
 * Compute scaled dot-product attention from a previously calculated QKᵀ score matrix.
 * @param {{ matrix: number[][], shape: [number, number] }} scores
 * @param {{ matrix: number[][], shape: [number, number] }} values
 * @param {number} dK
 * @returns {{ dK: number, scale: number, scaled: object, weights: object, output: object, rowSteps: Array<object> }}
 */
export function computeScaledAttention(scores, values, dK) {
  validateDK(dK);
  const scoreInfo = validateMatrix(scores, 'Scores');
  const valueInfo = validateMatrix(values, 'Values');
  if (scoreInfo.columns !== valueInfo.rows) {
    throw new RangeError('Score key count must match the number of value rows.');
  }

  const scaled = scaleScores(scores, dK);
  const { weights, rowSteps } = softmaxRows(scaled);
  const output = multiplyMatrices(weights, values);
  return {
    dK,
    scale: Math.sqrt(dK),
    scaled,
    weights,
    output,
    rowSteps,
  };
}
