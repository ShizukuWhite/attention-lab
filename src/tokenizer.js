/**
 * Split text on whitespace for the Phase 1 teaching playground.
 * This intentionally does not model a production language-model tokenizer.
 *
 * @param {string} text
 * @returns {{ text: string, tokens: Array<{ text: string, index: number, codePointLength: number }>, sequenceLength: number, shape: [number] }}
 */
export function analyzeText(text) {
  const tokens = [...text.matchAll(/\S+/gu)].map(([tokenText], index) => ({
    text: tokenText,
    index,
    codePointLength: [...tokenText].length,
  }));
  const sequenceLength = tokens.length;

  return {
    text,
    tokens,
    sequenceLength,
    shape: [sequenceLength],
  };
}
