import assert from 'node:assert/strict';
import test from 'node:test';
import { matrixCellSteps } from '../src/matrix-multiply.js';
import {
  computeScaledAttention,
  scaleScores,
  softmaxRows,
} from '../src/scaled-attention.js';

const rawScores = {
  matrix: [
    [-0.06, 0.51, 0.12, 0.21],
    [1.01, 0.79, 0.98, 1.34],
    [0.28, 0.62, 0.40, 0.58],
    [0.76, 1.04, 0.88, 1.24],
  ],
  shape: [4, 4],
};
const values = {
  matrix: [[0.9, 0.6], [1.1, -0.1], [1, 0.4], [1.6, 0.7]],
  shape: [4, 2],
};
const expectedScaled = [
  [-0.04242640687119285, 0.36062445840513924, 0.08485281374238570, 0.14849242404917498],
  [0.71417784899841300, 0.55861435713737254, 0.69296464556281657, 0.94752308678997368],
  [0.19798989873223331, 0.43840620433565947, 0.28284271247461901, 0.41012193308819756],
  [0.53740115370177612, 0.73539105243400943, 0.62225396744416182, 0.87681240867131893],
];
const expectedWeights = [
  [0.20650582980243074, 0.30901181387364382, 0.23453571149219812, 0.24994664483172732],
  [0.24403574677836998, 0.20887818306538620, 0.23891348867127880, 0.30817258148496501],
  [0.21754990613755347, 0.27667529083174919, 0.23681543712736677, 0.26895936590333057],
  [0.21224088284062193, 0.25871104129152213, 0.23103626353403380, 0.29801181233382215],
];
const expectedOutput = [
  [1.16021858530615770, 0.36177925247318243],
  [1.18138779251968063, 0.43681983226847040],
  [1.16728815801141791, 0.38586014558263527],
  [1.18345410324538331, 0.40249619962250997],
];

function assertMatrixNear(actual, expected, tolerance = 1e-12) {
  assert.deepEqual(actual.shape, [expected.length, expected[0]?.length ?? actual.shape[1]]);
  assert.equal(actual.matrix.length, expected.length);
  for (let row = 0; row < expected.length; row += 1) {
    assert.equal(actual.matrix[row].length, expected[row].length);
    for (let column = 0; column < expected[row].length; column += 1) {
      assert.ok(
        Math.abs(actual.matrix[row][column] - expected[row][column]) < tolerance,
        `matrix[${row}, ${column}]: got ${actual.matrix[row][column]}, expected ${expected[row][column]}`,
      );
    }
  }
}

function assertNear(actual, expected, tolerance = 1e-12) {
  assert.ok(Math.abs(actual - expected) < tolerance, `got ${actual}, expected ${expected}`);
}

function deepFreeze(value) {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    Object.freeze(value);
    for (const child of Object.values(value)) deepFreeze(child);
  }
  return value;
}

test('default I love my gecko matrices match the independent scaled-attention oracle', () => {
  const result = computeScaledAttention(rawScores, values, 2);
  assert.equal(result.dK, 2);
  assert.equal(result.scale, Math.sqrt(2));
  assertMatrixNear(result.scaled, expectedScaled);
  assertMatrixNear(result.weights, expectedWeights);
  assertMatrixNear(result.output, expectedOutput);
  assert.deepEqual(result.scaled.shape, [4, 4]);
  assert.deepEqual(result.weights.shape, [4, 4]);
  assert.deepEqual(result.output.shape, [4, 2]);
  for (const row of result.weights.matrix) assertNear(row.reduce((sum, weight) => sum + weight, 0), 1);

  const row = result.rowSteps[0];
  assert.equal(row.row, 0);
  assertNear(row.max, 0.36062445840513924);
  assertNear(row.denominator, 3.2361222293232583);
  assertNear(row.weightSum, 1);
  const expectedExponentials = [0.66827810630849152, 1, 0.75898622953004870, 0.80885789348471807];
  assert.deepEqual(row.steps.map((step) => step.index), [0, 1, 2, 3]);
  row.steps.forEach((step, index) => {
    assertNear(step.scaled, expectedScaled[0][index]);
    assertNear(step.shifted, expectedScaled[0][index] - row.max);
    assertNear(step.exponential, expectedExponentials[index]);
    assertNear(step.weight, result.weights.matrix[0][index]);
  });

  const outputTrace = matrixCellSteps(result.weights, values, 0, 1);
  assert.equal(outputTrace.steps.length, 4);
  assertNear(outputTrace.result, result.output.matrix[0][1]);
});

test('softmax normalizes each Query row and multiplies by V in the documented direction', () => {
  const scores = {
    matrix: [[0, 0, 0], [0, Math.log(2), Math.log(3)]],
    shape: [2, 3],
  };
  const v = { matrix: [[1, 10], [2, 20], [4, 40]], shape: [3, 2] };
  const result = computeScaledAttention(scores, v, 1);
  assertMatrixNear(result.weights, [[1 / 3, 1 / 3, 1 / 3], [1 / 6, 1 / 3, 1 / 2]]);
  assertMatrixNear(result.output, [[7 / 3, 70 / 3], [17 / 6, 85 / 3]]);
  assert.deepEqual(result.scaled.shape, [2, 3]);
  assert.deepEqual(result.output.shape, [2, 2]);
});

test('row softmax stays finite for large values and is invariant to adding a row constant', () => {
  const large = softmaxRows({ matrix: [[1000, 1001], [0, 1]], shape: [2, 2] });
  const expected = [0.2689414213699951, 0.7310585786300049];
  large.weights.matrix.forEach((row) => {
    row.forEach((weight, index) => assertNear(weight, expected[index]));
    assert.ok(row.every(Number.isFinite));
  });
  const shifted = softmaxRows({ matrix: [[0, 1], [0, 1]], shape: [2, 2] });
  shifted.weights.matrix[0].forEach((weight, index) => assertNear(weight, large.weights.matrix[0][index]));
  assert.deepEqual(large.rowSteps[0].steps.map((step) => step.shifted), [-1, 0]);
  assert.equal(large.rowSteps[0].max, 1001);
});

test('scaling and the output trace retain full precision before display rounding', () => {
  const scores = { matrix: [[0.123456789123, 0.987654321987]], shape: [1, 2] };
  const v = { matrix: [[3.1415926535], [-2.7182818284]], shape: [2, 1] };
  const scaled = scaleScores(scores, 3);
  scores.matrix[0].forEach((value, index) => {
    const exact = value / Math.sqrt(3);
    assert.equal(scaled.matrix[0][index], exact);
    assert.notEqual(scaled.matrix[0][index], Number(exact.toFixed(6)));
  });
  const result = computeScaledAttention(scores, v, 3);
  const trace = matrixCellSteps(result.weights, v, 0, 0);
  assertNear(trace.result, result.output.matrix[0][0]);
  assert.equal(trace.steps[0].left, result.weights.matrix[0][0]);
  assert.equal(trace.steps[1].left, result.weights.matrix[0][1]);
});

test('empty self-attention preserves explicit shapes and rejects invalid dK', () => {
  const scores = { matrix: [], shape: [0, 0] };
  const v = { matrix: [], shape: [0, 2] };
  const result = computeScaledAttention(scores, v, 2);
  assert.deepEqual(result.scaled, { matrix: [], shape: [0, 0] });
  assert.deepEqual(result.weights, { matrix: [], shape: [0, 0] });
  assert.deepEqual(result.output, { matrix: [], shape: [0, 2] });
  assert.deepEqual(result.rowSteps, []);
  assert.throws(() => computeScaledAttention(scores, v, 0), RangeError);
  assert.throws(() => softmaxRows({ matrix: [[]], shape: [1, 0] }), RangeError);
});

test('single and repeated tokens retain their positions and expected value combinations', () => {
  const single = computeScaledAttention(
    { matrix: [[0.64]], shape: [1, 1] },
    { matrix: [[0.9, 0.6]], shape: [1, 2] },
    2,
  );
  assert.deepEqual(single.weights.matrix, [[1]]);
  assertMatrixNear(single.output, [[0.9, 0.6]]);

  const repeated = computeScaledAttention(
    { matrix: [[1.01, 1.01], [1.01, 1.01]], shape: [2, 2] },
    { matrix: [[1.1, -0.1], [1.1, -0.1]], shape: [2, 2] },
    2,
  );
  assertMatrixNear(repeated.weights, [[0.5, 0.5], [0.5, 0.5]]);
  assertMatrixNear(repeated.output, [[1.1, -0.1], [1.1, -0.1]]);
  assert.deepEqual(repeated.rowSteps.map((row) => row.steps.map((step) => step.index)), [[0, 1], [0, 1]]);
});

test('unknown Query uses uniform weights, unknown Key can receive weight, and unknown V contributes zero', () => {
  const result = computeScaledAttention(
    { matrix: [[-0.06, 0], [0, 0]], shape: [2, 2] },
    { matrix: [[0.9, 0.6], [0, 0]], shape: [2, 2] },
    2,
  );
  const knownWeight = 1 / (1 + Math.exp(0.06 / Math.sqrt(2)));
  assertNear(result.weights.matrix[0][0], knownWeight);
  assertNear(result.weights.matrix[0][1], 1 - knownWeight);
  assert.ok(result.weights.matrix[0][1] > 0);
  assertNear(result.output.matrix[0][0], knownWeight * 0.9);
  assertNear(result.output.matrix[1][0], 0.45);
  assertNear(result.output.matrix[1][1], 0.3);
  assert.equal(result.rowSteps[0].steps[1].weight * 0, 0);

  const allUnknown = computeScaledAttention(
    { matrix: [[0, 0], [0, 0]], shape: [2, 2] },
    { matrix: [[0, 0], [0, 0]], shape: [2, 2] },
    2,
  );
  assertMatrixNear(allUnknown.weights, [[0.5, 0.5], [0.5, 0.5]]);
  assertMatrixNear(allUnknown.output, [[0, 0], [0, 0]]);
});

test('all input matrices and dK are validated, including otherwise empty inputs', () => {
  const v = { matrix: [[1]], shape: [1, 1] };
  const s = { matrix: [[1]], shape: [1, 1] };
  assert.throws(() => computeScaledAttention({ matrix: [[1, 2]], shape: [1, 2] }, v, 1), RangeError);
  for (const badDK of [0, -1, 1.5, Number.MAX_SAFE_INTEGER + 1]) {
    assert.throws(() => scaleScores({ matrix: [], shape: [0, 0] }, badDK), RangeError);
  }
  assert.throws(() => scaleScores(s, '2'), TypeError);
  assert.throws(() => scaleScores({ matrix: [[1], [2, 3]], shape: [2, 1] }, 1), RangeError);
  assert.throws(() => scaleScores({ matrix: [[1]], shape: [2, 1] }, 1), RangeError);
  assert.throws(() => scaleScores({ matrix: [[NaN]], shape: [1, 1] }, 1), TypeError);
  assert.throws(() => softmaxRows({ matrix: [[Infinity]], shape: [1, 1] }), TypeError);
  assert.throws(() => computeScaledAttention(s, { matrix: [[NaN]], shape: [1, 1] }, 1), TypeError);
});

test('computation clones results and never mutates inputs or cross-links result matrices', () => {
  const frozenScores = deepFreeze(structuredClone(rawScores));
  const frozenValues = deepFreeze(structuredClone(values));
  const beforeScores = structuredClone(frozenScores);
  const beforeValues = structuredClone(frozenValues);
  const result = computeScaledAttention(frozenScores, frozenValues, 2);
  assert.deepEqual(frozenScores, beforeScores);
  assert.deepEqual(frozenValues, beforeValues);

  result.scaled.matrix[0][0] = 123;
  result.weights.matrix[0][0] = 456;
  result.output.matrix[0][0] = 789;
  result.rowSteps[0].steps[0].weight = 999;
  assert.equal(frozenScores.matrix[0][0], -0.06);
  assert.equal(frozenValues.matrix[0][0], 0.9);
  assert.notEqual(result.scaled.matrix[0][0], result.weights.matrix[0][0]);
  assert.notEqual(result.weights.matrix[0][0], result.output.matrix[0][0]);
});
