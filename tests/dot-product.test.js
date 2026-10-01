import test from 'node:test';
import assert from 'node:assert/strict';
import { analyzeText } from '../src/tokenizer.js';
import { embedTokens } from '../src/embedding.js';
import { dotProductSteps } from '../src/dot-product.js';

const near = (actual, expected) => {
  assert.ok(Math.abs(actual - expected) < 1e-12, `${actual} is not near ${expected}`);
};

function vectorsFor(text, leftIndex, rightIndex) {
  const rows = embedTokens(analyzeText(text).tokens).rows;
  return [rows[leftIndex].vector, rows[rightIndex].vector];
}

test('shows each I/gecko product and running sum with vector and scalar shapes', () => {
  const [left, right] = vectorsFor('I love my gecko', 0, 3);
  const result = dotProductSteps(left, right);

  assert.equal(result.dimension, 3);
  assert.deepEqual(result.leftShape, [3]);
  assert.deepEqual(result.rightShape, [3]);
  assert.deepEqual(result.outputShape, []);
  assert.deepEqual(result.steps.map(({ index }) => index), [0, 1, 2]);
  result.steps.forEach((step, index) => {
    assert.equal(step.left, left[index]);
    assert.equal(step.right, right[index]);
  });
  result.steps.forEach((step, index) => near(step.product, [0.14, 0.63, 0.05][index]));
  result.steps.forEach((step, index) => near(step.runningSum, [0.14, 0.77, 0.82][index]));
  near(result.result, 0.82);
  assert.equal(result.result, result.steps.at(-1).runningSum);
});

test('supports dot products above one without normalizing them', () => {
  const [love, gecko] = vectorsFor('love gecko', 0, 1);
  near(dotProductSteps(love, gecko).result, 1.03);
});

test('computes a vector self dot product', () => {
  const [love] = vectorsFor('love', 0, 0);
  near(dotProductSteps(love, love).result, 0.89);
});

test('returns zero when either input is a zero vector', () => {
  near(dotProductSteps([0, 0, 0], [0.7, 0.9, 0.5]).result, 0);
});

test('supports arbitrary dimensions and preserves signed contributions', () => {
  const result = dotProductSteps([2, -3], [4, 5]);

  assert.equal(result.dimension, 2);
  assert.deepEqual(result.leftShape, [2]);
  assert.deepEqual(result.rightShape, [2]);
  assert.deepEqual(result.outputShape, []);
  assert.deepEqual(result.steps, [
    { index: 0, left: 2, right: 4, product: 8, runningSum: 8 },
    { index: 1, left: -3, right: 5, product: -15, runningSum: -7 },
  ]);
  assert.equal(result.result, -7);
});

test('does not mutate inputs or let result changes affect later calls', () => {
  const left = [0.2, 0.7, 0.1];
  const right = [0.7, 0.9, 0.5];
  const originalLeft = [...left];
  const originalRight = [...right];
  const first = dotProductSteps(left, right);

  first.steps[0].left = 99;
  first.steps[0].product = 99;
  first.steps[0].runningSum = 99;
  first.leftShape[0] = 99;
  first.rightShape[0] = 99;
  first.outputShape.push(99);
  first.steps.pop();

  assert.deepEqual(left, originalLeft);
  assert.deepEqual(right, originalRight);
  const second = dotProductSteps(left, right);
  assert.equal(second.result, 0.8200000000000001);
  assert.deepEqual(second.leftShape, [3]);
  assert.deepEqual(second.rightShape, [3]);
  assert.deepEqual(second.outputShape, []);
  assert.equal(second.steps.length, 3);
});

test('keeps full floating-point precision in products and results', () => {
  const result = dotProductSteps([0.123456789], [1]);

  assert.equal(result.steps[0].product, 0.123456789);
  assert.equal(result.steps[0].runningSum, 0.123456789);
  assert.equal(result.result, 0.123456789);
});

test('rejects non-array and sparse inputs with TypeError', () => {
  assert.throws(() => dotProductSteps(null, [1]), TypeError);
  assert.throws(() => dotProductSteps([1], {}), TypeError);
  assert.throws(() => dotProductSteps([, 1], [1, 1]), TypeError);
});

test('rejects non-finite and non-number components with TypeError', () => {
  for (const component of ['1', NaN, Infinity, -Infinity]) {
    assert.throws(() => dotProductSteps([component], [1]), TypeError);
  }
});

test('rejects empty or mismatched vector dimensions with RangeError', () => {
  assert.throws(() => dotProductSteps([], []), RangeError);
  assert.throws(() => dotProductSteps([1], [1, 2]), RangeError);
});

test('rejects product or running-sum overflow with RangeError', () => {
  assert.throws(() => dotProductSteps([Number.MAX_VALUE], [2]), RangeError);
  assert.throws(() => dotProductSteps([Number.MAX_VALUE, Number.MAX_VALUE], [1, 1]), RangeError);
});
