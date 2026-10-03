import test from 'node:test';
import assert from 'node:assert/strict';
import { analyzeText } from '../src/tokenizer.js';
import { embedTokens } from '../src/embedding.js';
import { matrixCellSteps } from '../src/matrix-multiply.js';
import { projectQKV } from '../src/projection.js';
import { computeAttentionScores } from '../src/attention-score.js';

function closeTo(actual, expected, tolerance = 1e-12) {
  assert.ok(Math.abs(actual - expected) < tolerance, `${actual} should be within ${tolerance} of ${expected}`);
}

function assertMatrixClose(actual, expected) {
  assert.deepEqual(actual.shape, [expected.length, expected[0]?.length ?? 0]);
  assert.equal(actual.matrix.length, expected.length);
  for (let row = 0; row < expected.length; row += 1) {
    assert.equal(actual.matrix[row].length, expected[row].length);
    for (let column = 0; column < expected[row].length; column += 1) {
      closeTo(actual.matrix[row][column], expected[row][column]);
    }
  }
}

function analyze(text) {
  const embedding = embedTokens(analyzeText(text).tokens);
  const projections = projectQKV({ matrix: embedding.matrix, shape: embedding.shape });
  return { embedding, projections, scores: computeAttentionScores(projections.Q, projections.K) };
}

test('raw scores preserve query-key direction', () => {
  const { KT, S } = computeAttentionScores(
    { matrix: [[1, 2], [3, 4]], shape: [2, 2] },
    { matrix: [[5, 6], [7, 8]], shape: [2, 2] },
  );

  assert.deepEqual(KT, { matrix: [[5, 7], [6, 8]], shape: [2, 2] });
  assert.deepEqual(S, { matrix: [[17, 23], [39, 53]], shape: [2, 2] });
  assert.equal(S.matrix[0][1], 23);
  assert.equal(S.matrix[1][0], 39);
});

test('raw score shape preserves different query and key position counts', () => {
  const { KT, S } = computeAttentionScores(
    { matrix: [[1, 2], [3, 4]], shape: [2, 2] },
    { matrix: [[5, 6], [7, 8], [9, 10]], shape: [3, 2] },
  );

  assert.deepEqual(KT, { matrix: [[5, 7, 9], [6, 8, 10]], shape: [2, 3] });
  assert.deepEqual(S, { matrix: [[17, 23, 29], [39, 53, 67]], shape: [2, 3] });
});

test('default projected scores match every hand-calculated cell and trace', () => {
  const { projections, scores } = analyze('I love my gecko');
  assertMatrixClose(scores.KT, [[0.8, 0.7, 0.8, 1.1], [-0.5, 0.5, -0.2, -0.2]]);
  assert.deepEqual(scores.S.shape, [4, 4]);

  const expected = [
    [-0.06, 0.51, 0.12, 0.21],
    [1.01, 0.79, 0.98, 1.34],
    [0.28, 0.62, 0.4, 0.58],
    [0.76, 1.04, 0.88, 1.24],
  ];
  for (let row = 0; row < expected.length; row += 1) {
    for (let column = 0; column < expected[row].length; column += 1) {
      closeTo(scores.S.matrix[row][column], expected[row][column]);
    }
  }

  const trace = matrixCellSteps(projections.Q, scores.KT, 0, 3);
  const expectedLeft = [0.3, 0.6];
  const expectedRight = [1.1, -0.2];
  const expectedSteps = [
    { index: 0, left: 0.3, right: 1.1, product: 0.33, runningSum: 0.33 },
    { index: 1, left: 0.6, right: -0.2, product: -0.12, runningSum: 0.21 },
  ];
  for (let index = 0; index < 2; index += 1) {
    closeTo(trace.leftRow[index], expectedLeft[index]);
    closeTo(trace.rightColumn[index], expectedRight[index]);
    assert.equal(trace.steps[index].index, expectedSteps[index].index);
    for (const key of ['left', 'right', 'product', 'runningSum']) {
      closeTo(trace.steps[index][key], expectedSteps[index][key]);
    }
  }
  closeTo(trace.result, 0.21);
  assert.deepEqual(trace.leftShape, [2]);
  assert.deepEqual(trace.rightShape, [2]);
  assert.deepEqual(trace.outputShape, []);
});

test('scores preserve empty single repeat and unknown-token positions', () => {
  const empty = analyze('');
  assert.deepEqual(empty.projections.Q.shape, [0, 2]);
  assert.deepEqual(empty.projections.K.shape, [0, 2]);
  assert.deepEqual(empty.scores.KT, { matrix: [[], []], shape: [2, 0] });
  assert.deepEqual(empty.scores.S, { matrix: [], shape: [0, 0] });

  const single = analyze('I').scores.S;
  assert.deepEqual(single.shape, [1, 1]);
  closeTo(single.matrix[0][0], -0.06);

  const repeated = analyze('love love').scores.S;
  assert.deepEqual(repeated.shape, [2, 2]);
  assertMatrixClose(repeated, [[0.79, 0.79], [0.79, 0.79]]);

  const unknown = analyze('I mystery').scores.S;
  assertMatrixClose(unknown, [[-0.06, 0], [0, 0]]);
});

test('score composition rejects mismatched feature dimensions and preserves inputs', () => {
  for (const [Q, K] of [
    [
      { matrix: [[1, 2]], shape: [1, 2] },
      { matrix: [[1, 2, 3]], shape: [1, 3] },
    ],
    [
      { matrix: [], shape: [0, 2] },
      { matrix: [], shape: [0, 3] },
    ],
  ]) {
    assert.throws(() => computeAttentionScores(Q, K), RangeError);
  }

  const Q = { matrix: [[0.123456789, 0.25]], shape: [1, 2] };
  const K = { matrix: [[0.5, -0.75]], shape: [1, 2] };
  const originalQ = structuredClone(Q);
  const originalK = structuredClone(K);
  const { KT, S } = computeAttentionScores(Q, K);

  assert.equal(S.matrix[0][0], 0.123456789 * 0.5 + 0.25 * -0.75);
  assert.deepEqual(Q, originalQ);
  assert.deepEqual(K, originalK);
  assert.notStrictEqual(KT.matrix, K.matrix);
  assert.notStrictEqual(S.matrix, Q.matrix);
  S.matrix[0][0] = 42;
  KT.matrix[0][0] = 42;
  assert.deepEqual(Q, originalQ);
  assert.deepEqual(K, originalK);
});
