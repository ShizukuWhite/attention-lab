import test from 'node:test';
import assert from 'node:assert/strict';
import { analyzeText } from '../src/tokenizer.js';
import { D_MODEL, embedTokens } from '../src/embedding.js';

test('maps the default sentence to the four fixed three-dimensional teaching vectors', () => {
  const tokens = analyzeText('I love my gecko').tokens;
  const result = embedTokens(tokens);

  assert.equal(D_MODEL, 3);
  assert.equal(result.dimension, 3);
  assert.deepEqual(result.vectorShape, [3]);
  assert.deepEqual(result.shape, [4, 3]);
  assert.deepEqual(result.rows, [
    { text: 'I', index: 0, vector: [0.2, 0.7, 0.1], source: 'manual' },
    { text: 'love', index: 1, vector: [0.8, 0.3, 0.4], source: 'manual' },
    { text: 'my', index: 2, vector: [0.4, 0.6, 0.2], source: 'manual' },
    { text: 'gecko', index: 3, vector: [0.7, 0.9, 0.5], source: 'manual' },
  ]);
  assert.deepEqual(result.matrix, result.rows.map((row) => row.vector));
});

test('keeps repeated tokens at independent positions with independent vector arrays', () => {
  const result = embedTokens(analyzeText('love love').tokens);

  assert.deepEqual(result.rows.map((row) => row.index), [0, 1]);
  assert.deepEqual(result.rows.map((row) => row.vector), [
    [0.8, 0.3, 0.4],
    [0.8, 0.3, 0.4],
  ]);
  assert.notStrictEqual(result.rows[0].vector, result.rows[1].vector);
  assert.notStrictEqual(result.matrix[0], result.matrix[1]);
});

test('uses a zero-vector placeholder for unknown text without normalizing tokens', () => {
  const result = embedTokens(analyzeText('i love! 我喜欢壁虎 constructor __proto__').tokens);

  assert.deepEqual(result.rows.map(({ text, source, vector }) => ({ text, source, vector })), [
    { text: 'i', source: 'unknown', vector: [0, 0, 0] },
    { text: 'love!', source: 'unknown', vector: [0, 0, 0] },
    { text: '我喜欢壁虎', source: 'unknown', vector: [0, 0, 0] },
    { text: 'constructor', source: 'unknown', vector: [0, 0, 0] },
    { text: '__proto__', source: 'unknown', vector: [0, 0, 0] },
  ]);
});

test('returns an empty matrix while preserving the two-dimensional empty shape', () => {
  const result = embedTokens([]);

  assert.equal(result.dimension, 3);
  assert.deepEqual(result.vectorShape, [3]);
  assert.deepEqual(result.rows, []);
  assert.deepEqual(result.matrix, []);
  assert.deepEqual(result.shape, [0, 3]);
});

test('returns detached arrays without mutating input tokens, duplicate rows, or later results', () => {
  const tokens = analyzeText('love love gecko').tokens;
  const originalTokens = structuredClone(tokens);
  const firstResult = embedTokens(tokens);

  firstResult.rows[0].vector[0] = 99;
  firstResult.matrix[1][1] = 88;
  firstResult.shape[0] = 77;
  firstResult.vectorShape[0] = 66;

  assert.deepEqual(tokens, originalTokens);
  assert.deepEqual(firstResult.rows[1].vector, [0.8, 0.3, 0.4]);

  const nextResult = embedTokens(tokens);
  assert.deepEqual(nextResult.rows.map((row) => row.vector), [
    [0.8, 0.3, 0.4],
    [0.8, 0.3, 0.4],
    [0.7, 0.9, 0.5],
  ]);
  assert.deepEqual(nextResult.matrix, nextResult.rows.map((row) => row.vector));
  assert.deepEqual(nextResult.shape, [3, 3]);
  assert.deepEqual(nextResult.vectorShape, [3]);
});
