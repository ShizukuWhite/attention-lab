import test from 'node:test';
import assert from 'node:assert/strict';
import { analyzeText } from '../src/tokenizer.js';
import { embedTokens } from '../src/embedding.js';
import { PROJECTION_WEIGHTS, projectQKV } from '../src/projection.js';

const nearMatrix = (actual, expected) => {
  assert.deepEqual(actual.shape, expected.shape);
  assert.equal(actual.matrix.length, expected.matrix.length);
  for (let row = 0; row < expected.matrix.length; row += 1) {
    assert.equal(actual.matrix[row].length, expected.matrix[row].length);
    for (let column = 0; column < expected.matrix[row].length; column += 1) {
      assert.ok(
        Math.abs(actual.matrix[row][column] - expected.matrix[row][column]) < 1e-12,
        `${actual.matrix[row][column]} is not near ${expected.matrix[row][column]} at [${row}, ${column}]`,
      );
    }
  }
};

const inputFor = (text) => {
  const embedding = embedTokens(analyzeText(text).tokens);
  return { matrix: embedding.matrix, shape: embedding.shape, rows: embedding.rows };
};

test('projects the default sentence into the fixed Q, K, and V matrices', () => {
  const result = projectQKV(inputFor('I love my gecko'));

  nearMatrix(result.Q, {
    matrix: [[0.3, 0.6], [1.2, -0.1], [0.6, 0.4], [1.2, 0.4]],
    shape: [4, 2],
  });
  nearMatrix(result.K, {
    matrix: [[0.8, -0.5], [0.7, 0.5], [0.8, -0.2], [1.1, -0.2]],
    shape: [4, 2],
  });
  nearMatrix(result.V, {
    matrix: [[0.9, 0.6], [1.1, -0.1], [1, 0.4], [1.6, 0.7]],
    shape: [4, 2],
  });
});

test('uses distinct fixed matrices with explicit [3, 2] shapes', () => {
  assert.deepEqual(PROJECTION_WEIGHTS.Wq, {
    matrix: [[1, 0], [0, 1], [1, -1]],
    shape: [3, 2],
  });
  assert.deepEqual(PROJECTION_WEIGHTS.Wk, {
    matrix: [[1, 1], [1, -1], [-1, 0]],
    shape: [3, 2],
  });
  assert.deepEqual(PROJECTION_WEIGHTS.Wv, {
    matrix: [[1, -1], [1, 1], [0, 1]],
    shape: [3, 2],
  });
});

test('preserves token positions for repeated tokens and keeps unknown rows at zero', () => {
  const embedding = embedTokens(analyzeText('love love mystery').tokens);
  const result = projectQKV({ matrix: embedding.matrix, shape: embedding.shape });

  for (const key of ['Q', 'K', 'V']) {
    assert.deepEqual(result[key].shape, [3, 2]);
    assert.deepEqual(result[key].matrix[0], result[key].matrix[1]);
    assert.notStrictEqual(result[key].matrix[0], result[key].matrix[1]);
    assert.deepEqual(result[key].matrix[2], [0, 0]);
  }
  assert.deepEqual(embedding.rows.map(({ index, source }) => [index, source]), [
    [0, 'manual'],
    [1, 'manual'],
    [2, 'unknown'],
  ]);
});

test('supports one token and an empty sequence while preserving output shapes', () => {
  const one = projectQKV(inputFor('I'));
  assert.deepEqual(one.Q.shape, [1, 2]);
  assert.deepEqual(one.K.shape, [1, 2]);
  assert.deepEqual(one.V.shape, [1, 2]);

  const empty = projectQKV({ matrix: [], shape: [0, 3] });
  assert.deepEqual(empty.Q, { matrix: [], shape: [0, 2] });
  assert.deepEqual(empty.K, { matrix: [], shape: [0, 2] });
  assert.deepEqual(empty.V, { matrix: [], shape: [0, 2] });
});

test('rejects input matrices whose declared embedding width is not three', () => {
  for (const input of [
    { matrix: [[1, 2]], shape: [1, 2] },
    { matrix: [], shape: [0, 2] },
  ]) {
    assert.throws(() => projectQKV(input), RangeError);
  }
});

test('deep-freezes weights and does not share input or output arrays', () => {
  for (const weight of Object.values(PROJECTION_WEIGHTS)) {
    assert.ok(Object.isFrozen(weight));
    assert.ok(Object.isFrozen(weight.shape));
    assert.ok(Object.isFrozen(weight.matrix));
    for (const row of weight.matrix) assert.ok(Object.isFrozen(row));
  }

  const input = inputFor('I love');
  const originalInput = structuredClone(input);
  const first = projectQKV(input);
  first.Q.matrix[0][0] = 99;
  first.Q.shape[0] = 99;
  first.K.matrix[0][0] = 88;
  first.V.matrix[0][0] = 77;

  assert.deepEqual(input, originalInput);
  nearMatrix(projectQKV(input).Q, {
    matrix: [[0.3, 0.6], [1.2, -0.1]],
    shape: [2, 2],
  });
});
