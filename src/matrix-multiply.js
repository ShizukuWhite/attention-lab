import { dotProductSteps } from './dot-product.js';

function validateMatrix(value, label) {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError(`${label} must be a matrix object.`);
  }
  if (!Array.isArray(value.matrix)) {
    throw new TypeError(`${label}.matrix must be an array.`);
  }

  const { matrix, shape } = value;
  if (!Array.isArray(shape) || shape.length !== 2) {
    throw new RangeError(`${label}.shape must contain two dimensions.`);
  }
  const [rows, columns] = shape;
  if (!Number.isSafeInteger(rows) || rows < 0 || !Number.isSafeInteger(columns) || columns < 0) {
    throw new RangeError(`${label}.shape dimensions must be non-negative safe integers.`);
  }
  if (matrix.length !== rows) {
    throw new RangeError(`${label}.matrix row count does not match its shape.`);
  }

  for (let rowIndex = 0; rowIndex < rows; rowIndex += 1) {
    if (!Object.hasOwn(matrix, rowIndex)) {
      throw new TypeError(`${label}.matrix must not contain missing rows.`);
    }
    const row = matrix[rowIndex];
    if (!Array.isArray(row)) {
      throw new TypeError(`${label}.matrix rows must be arrays.`);
    }
    if (row.length !== columns) {
      throw new RangeError(`${label}.matrix row width does not match its shape.`);
    }
    for (let columnIndex = 0; columnIndex < columns; columnIndex += 1) {
      if (!Object.hasOwn(row, columnIndex)) {
        throw new TypeError(`${label}.matrix rows must not contain missing components.`);
      }
      if (typeof row[columnIndex] !== 'number' || !Number.isFinite(row[columnIndex])) {
        throw new TypeError(`${label}.matrix components must be finite numbers.`);
      }
    }
  }

  return { matrix, rows, columns };
}

function validatePair(leftValue, rightValue) {
  const left = validateMatrix(leftValue, 'Left input');
  const right = validateMatrix(rightValue, 'Right input');
  if (left.columns !== right.rows) {
    throw new RangeError('Matrix inner dimensions must match.');
  }
  return { left, right, dimension: left.columns };
}

/**
 * Transpose a dense matrix while retaining its declared dimensions, including
 * dimensions that cannot be inferred from an empty outer array.
 *
 * @param {{ matrix: number[][], shape: [number, number] }} value
 * @returns {{ matrix: number[][], shape: [number, number] }}
 */
export function transposeMatrix(value) {
  const { matrix: input, rows, columns } = validateMatrix(value, 'Input');
  const matrix = Array.from({ length: columns }, () => []);

  for (let rowIndex = 0; rowIndex < rows; rowIndex += 1) {
    for (let columnIndex = 0; columnIndex < columns; columnIndex += 1) {
      matrix[columnIndex].push(input[rowIndex][columnIndex]);
    }
  }

  return { matrix, shape: [columns, rows] };
}

function readColumn(matrix, dimension, column) {
  const values = [];
  for (let index = 0; index < dimension; index += 1) {
    values.push(matrix[index][column]);
  }
  return values;
}

function calculateCell(left, right, dimension, row, column) {
  const leftRow = [...left.matrix[row]];
  const rightColumn = readColumn(right.matrix, dimension, column);
  const calculation = dimension === 0
    ? { steps: [], result: 0 }
    : dotProductSteps(leftRow, rightColumn);

  return {
    row,
    column,
    leftRow,
    rightColumn,
    dimension,
    leftShape: [dimension],
    rightShape: [dimension],
    outputShape: [],
    steps: calculation.steps,
    result: calculation.result,
  };
}

/**
 * Multiply two dense matrices with explicit shapes.
 *
 * @param {{ matrix: number[][], shape: [number, number] }} left
 * @param {{ matrix: number[][], shape: [number, number] }} right
 * @returns {{ matrix: number[][], shape: [number, number] }}
 */
export function multiplyMatrices(leftValue, rightValue) {
  const { left, right, dimension } = validatePair(leftValue, rightValue);
  const matrix = [];

  for (let row = 0; row < left.rows; row += 1) {
    const outputRow = [];
    for (let column = 0; column < right.columns; column += 1) {
      outputRow.push(calculateCell(left, right, dimension, row, column).result);
    }
    matrix.push(outputRow);
  }

  return {
    matrix,
    shape: [left.rows, right.columns],
  };
}

/**
 * Trace one output cell as the dot product of one input row and one input column.
 *
 * @param {{ matrix: number[][], shape: [number, number] }} left
 * @param {{ matrix: number[][], shape: [number, number] }} right
 * @param {number} row
 * @param {number} column
 * @returns {{ row: number, column: number, leftRow: number[], rightColumn: number[], dimension: number, leftShape: [number], rightShape: [number], outputShape: [], steps: Array<{ index: number, left: number, right: number, product: number, runningSum: number }>, result: number }}
 */
export function matrixCellSteps(leftValue, rightValue, row, column) {
  const { left, right, dimension } = validatePair(leftValue, rightValue);
  if (!Number.isSafeInteger(row) || row < 0 || row >= left.rows
    || !Number.isSafeInteger(column) || column < 0 || column >= right.columns) {
    throw new RangeError('Matrix cell indices must identify an output cell.');
  }

  return calculateCell(left, right, dimension, row, column);
}
