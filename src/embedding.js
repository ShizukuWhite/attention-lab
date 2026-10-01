export const D_MODEL = 3;

const manualEmbeddings = new Map([
  ['I', [0.2, 0.7, 0.1]],
  ['love', [0.8, 0.3, 0.4]],
  ['my', [0.4, 0.6, 0.2]],
  ['gecko', [0.7, 0.9, 0.5]],
]);
const unknownVector = [0, 0, 0];

/**
 * Look up one manually assigned teaching vector for each token.
 * Unknown text gets a zero-vector placeholder; no model inference occurs.
 *
 * @param {Array<{ text: string, index: number, codePointLength: number }>} tokens
 * @returns {{ dimension: number, vectorShape: [number], rows: Array<{ text: string, index: number, vector: number[], source: 'manual' | 'unknown' }>, matrix: number[][], shape: [number, number] }}
 */
export function embedTokens(tokens) {
  const rows = tokens.map((token) => {
    const isManual = manualEmbeddings.has(token.text);
    const vector = isManual ? manualEmbeddings.get(token.text) : unknownVector;

    return {
      text: token.text,
      index: token.index,
      vector: [...vector],
      source: isManual ? 'manual' : 'unknown',
    };
  });

  return {
    dimension: D_MODEL,
    vectorShape: [D_MODEL],
    rows,
    matrix: rows.map((row) => [...row.vector]),
    shape: [rows.length, D_MODEL],
  };
}
