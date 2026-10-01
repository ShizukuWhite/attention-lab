import test from 'node:test';
import assert from 'node:assert/strict';
import { analyzeText } from '../src/tokenizer.js';

test('analyzes the default sentence into indexed tokens and sequence shape', () => {
  const result = analyzeText('I love my gecko');

  assert.deepEqual(result.tokens.map((token) => token.text), ['I', 'love', 'my', 'gecko']);
  assert.deepEqual(result.tokens.map((token) => token.index), [0, 1, 2, 3]);
  assert.deepEqual(result.tokens.map((token) => token.codePointLength), [1, 4, 2, 5]);
  assert.equal(result.text, 'I love my gecko');
  assert.equal(result.sequenceLength, 4);
  assert.deepEqual(result.shape, [4]);
});

test('uses spaces, tabs, and newlines as separators and ignores surrounding whitespace', () => {
  const result = analyzeText(' \tI\n love  ');

  assert.deepEqual(result.tokens.map((token) => token.text), ['I', 'love']);
  assert.equal(result.sequenceLength, 2);
  assert.deepEqual(result.shape, [2]);
});

test('returns an empty sequence for empty and whitespace-only input', () => {
  for (const text of ['', ' \t\n  ']) {
    const result = analyzeText(text);

    assert.equal(result.text, text);
    assert.deepEqual(result.tokens, []);
    assert.equal(result.sequenceLength, 0);
    assert.deepEqual(result.shape, [0]);
  }
});

test('keeps repeated words at separate sequence positions', () => {
  const result = analyzeText('go go');

  assert.deepEqual(result.tokens.map((token) => token.text), ['go', 'go']);
  assert.deepEqual(result.tokens.map((token) => token.index), [0, 1]);
});

test('keeps punctuation attached to its whitespace-delimited token', () => {
  const result = analyzeText('hello, world!');

  assert.deepEqual(result.tokens.map((token) => token.text), ['hello,', 'world!']);
});

test('does not split text without whitespace, including Chinese text', () => {
  assert.deepEqual(analyzeText('我喜欢壁虎').tokens.map((token) => token.text), ['我喜欢壁虎']);
  assert.deepEqual(analyzeText('我 喜欢 壁虎').tokens.map((token) => token.text), ['我', '喜欢', '壁虎']);
});

test('counts Unicode code points rather than UTF-16 code units', () => {
  assert.equal(analyzeText('😀').tokens[0].codePointLength, 1);
  assert.equal(analyzeText('e\u0301').tokens[0].codePointLength, 2);
});
