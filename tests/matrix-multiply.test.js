import test from 'node:test';
import assert from 'node:assert/strict';
import { matrixCellSteps, multiplyMatrices } from '../src/matrix-multiply.js';

const left = {
  matrix: [[1, 2, 3], [-1, 0, 2]],
  shape: [2, 3],
};
const right = {
  matrix: [[1, 0], [0, -1], [2, 1]],
  shape: [3, 2],
};

test('multiplies rectangular matrices and preserves the output shape', () => {
  assert.deepEqual(multiplyMatrices(left, right), {
    matrix: [[7, 1], [3, 2]],
    shape: [2, 2],
  });
});

test('shows one output cell as a row-by-column dot product with running sums', () => {
  assert.deepEqual(matrixCellSteps(left, right, 0, 1), {
    row: 0,
    column: 1,
    leftRow: [1, 2, 3],
    rightColumn: [0, -1, 1],
    dimension: 3,
    leftShape: [3],
    rightShape: [3],
    outputShape: [],
    steps: [
      { index: 0, left: 1, right: 0, product: 0, runningSum: 0 },
      { index: 1, left: 2, right: -1, product: -2, runningSum: -2 },
      { index: 2, left: 3, right: 1, product: 3, runningSum: 1 },
    ],
    result: 1,
  });
});

test('every output cell agrees with its independently traced dot product', () => {
  const product = multiplyMatrices(left, right);

  for (let row = 0; row < product.shape[0]; row += 1) {
    for (let column = 0; column < product.shape[1]; column += 1) {
      assert.equal(product.matrix[row][column], matrixCellSteps(left, right, row, column).result);
    }
  }
});

test('retains declared columns for an empty row dimension', () => {
  assert.deepEqual(multiplyMatrices(
    { matrix: [], shape: [0, 3] },
    { matrix: [[1, 0], [0, 1], [1, 1]], shape: [3, 2] },
  ), { matrix: [], shape: [0, 2] });
});

test('defines a zero-width inner product as zero', () => {
  const zeroInnerLeft = { matrix: [[], []], shape: [2, 0] };
  const zeroInnerRight = { matrix: [], shape: [0, 2] };

  assert.deepEqual(multiplyMatrices(zeroInnerLeft, zeroInnerRight), {
    matrix: [[0, 0], [0, 0]],
    shape: [2, 2],
  });
  assert.deepEqual(matrixCellSteps(zeroInnerLeft, zeroInnerRight, 1, 0), {
    row: 1,
    column: 0,
    leftRow: [],
    rightColumn: [],
    dimension: 0,
    leftShape: [0],
    rightShape: [0],
    outputShape: [],
    steps: [],
    result: 0,
  });
});

test('supports zero output columns without guessing the inner shape', () => {
  assert.deepEqual(multiplyMatrices(
    { matrix: [[1], [2]], shape: [2, 1] },
    { matrix: [[]], shape: [1, 0] },
  ), { matrix: [[], []], shape: [2, 0] });
});

test('validates matrices even when the multiplication has no output cells', () => {
  assert.throws(() => multiplyMatrices(
    { matrix: [], shape: [0, 3] },
    { matrix: [[1, 2]], shape: [1, 2] },
  ), RangeError);
});

test('rejects malformed and sparse matrices with TypeError', () => {
  const sparseOuter = [];
  sparseOuter.length = 1;
  const sparseRow = [];
  sparseRow.length = 1;

  for (const matrix of [null, [], { matrix: null, shape: [0, 0] }, { matrix: sparseOuter, shape: [1, 0] }, { matrix: [sparseRow], shape: [1, 1] }]) {
    assert.throws(() => multiplyMatrices(matrix, { matrix: [], shape: [0, 0] }), TypeError);
  }
  assert.throws(() => multiplyMatrices({ matrix: [['1']], shape: [1, 1] }, { matrix: [['1']], shape: [1, 1] }), TypeError);
  assert.throws(() => multiplyMatrices({ matrix: [[NaN]], shape: [1, 1] }, { matrix: [[1]], shape: [1, 1] }), TypeError);
  assert.throws(() => multiplyMatrices({ matrix: [[Infinity]], shape: [1, 1] }, { matrix: [[1]], shape: [1, 1] }), TypeError);
});

test('rejects invalid or inconsistent shapes with RangeError', () => {
  for (const shape of [[-1, 2], [1.5, 2], [Number.MAX_SAFE_INTEGER + 1, 0], [1], null]) {
    assert.throws(() => multiplyMatrices(
      { matrix: [[1]], shape },
      { matrix: [[1]], shape: [1, 1] },
    ), RangeError);
  }
  assert.throws(() => multiplyMatrices({ matrix: [[1]], shape: [2, 1] }, { matrix: [[1]], shape: [1, 1] }), RangeError);
  assert.throws(() => multiplyMatrices({ matrix: [[1, 2]], shape: [1, 1] }, { matrix: [[1]], shape: [1, 1] }), RangeError);
  assert.throws(() => multiplyMatrices(left, { matrix: [[1], [2]], shape: [2, 1] }), RangeError);
});

test('rejects invalid cell indices with RangeError', () => {
  for (const [row, column] of [[-1, 0], [2, 0], [0, -1], [0, 2], [0.5, 0], [0, NaN]]) {
    assert.throws(() => matrixCellSteps(left, right, row, column), RangeError);
  }
  assert.throws(() => matrixCellSteps({ matrix: [], shape: [0, 3] }, right, 0, 0), RangeError);
});

test('rejects product and accumulation overflow with RangeError', () => {
  assert.throws(() => multiplyMatrices(
    { matrix: [[Number.MAX_VALUE]], shape: [1, 1] },
    { matrix: [[2]], shape: [1, 1] },
  ), RangeError);
  assert.throws(() => multiplyMatrices(
    { matrix: [[Number.MAX_VALUE, Number.MAX_VALUE]], shape: [1, 2] },
    { matrix: [[1], [1]], shape: [2, 1] },
  ), RangeError);
});

test('does not mutate inputs, preserves precision, and returns detached data', () => {
  const mutableLeft = { matrix: [[0.123456789]], shape: [1, 1] };
  const mutableRight = { matrix: [[1]], shape: [1, 1] };
  const first = multiplyMatrices(mutableLeft, mutableRight);
  const trace = matrixCellSteps(mutableLeft, mutableRight, 0, 0);

  assert.equal(first.matrix[0][0], 0.123456789);
  assert.deepEqual(mutableLeft, { matrix: [[0.123456789]], shape: [1, 1] });
  first.matrix[0][0] = 99;
  first.shape[0] = 99;
  trace.leftRow[0] = 99;
  trace.rightColumn[0] = 99;
  trace.steps[0].product = 99;
  trace.leftShape[0] = 99;

  assert.deepEqual(multiplyMatrices(mutableLeft, mutableRight), {
    matrix: [[0.123456789]],
    shape: [1, 1],
  });
  assert.equal(matrixCellSteps(mutableLeft, mutableRight, 0, 0).result, 0.123456789);
});
