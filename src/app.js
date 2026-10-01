import { analyzeText } from './tokenizer.js';
import { embedTokens } from './embedding.js';

const input = document.querySelector('#token-input');
const analyzeButton = document.querySelector('#analyze-button');
const analysisStatus = document.querySelector('#analysis-status');
const staleNote = document.querySelector('#stale-note');
const sequenceLength = document.querySelector('#sequence-length');
const sequenceShape = document.querySelector('#sequence-shape');
const tokenList = document.querySelector('#token-list');
const detailEmpty = document.querySelector('#detail-empty');
const detailContent = document.querySelector('#detail-content');
const detailToken = document.querySelector('#detail-token');
const detailIndex = document.querySelector('#detail-index');
const detailOrdinal = document.querySelector('#detail-ordinal');
const detailCodePoints = document.querySelector('#detail-code-points');
const detailVector = document.querySelector('#detail-vector');
const detailVectorShape = document.querySelector('#detail-vector-shape');
const detailVectorSource = document.querySelector('#detail-vector-source');
const embeddingShape = document.querySelector('#embedding-shape');
const embeddingDimension = document.querySelector('#embedding-dimension');
const embeddingMatrixBody = document.querySelector('#embedding-matrix-body');

let currentAnalysis;
let currentEmbedding;
let selectedIndex = null;
let lastStaleState = false;

function setAnalysisStatus(message) {
  analysisStatus.textContent = message;
}

function updateStaleState() {
  const isStale = input.value !== currentAnalysis.text;
  staleNote.hidden = !isStale;

  if (isStale !== lastStaleState) {
    setAnalysisStatus(
      isStale
        ? '输入已修改，请点击 Analyze 更新下方 Tokens 与 Embedding。'
        : '当前输入与最近一次分析一致。',
    );
    lastStaleState = isStale;
  }
}

function renderDetails() {
  const selectedToken = currentAnalysis.tokens.find((token) => token.index === selectedIndex);
  const selectedEmbedding = currentEmbedding.rows.find((row) => row.index === selectedIndex);

  if (!selectedToken || !selectedEmbedding) {
    detailContent.hidden = true;
    detailEmpty.hidden = false;
    detailEmpty.textContent = currentAnalysis.tokens.length === 0
      ? '当前序列为空：N = 0，Token Shape 为 [0]，Embedding Shape 为 [0, 3]，没有可选 Token。'
      : '点击一个 Token，这里会显示它的位置和对应的 Embedding 向量。';
    return;
  }

  detailEmpty.hidden = true;
  detailContent.hidden = false;
  detailToken.textContent = selectedToken.text;
  detailIndex.textContent = String(selectedToken.index);
  detailOrdinal.textContent = String(selectedToken.index + 1);
  detailCodePoints.textContent = String(selectedToken.codePointLength);
  detailVector.textContent = `[${selectedEmbedding.vector.join(', ')}]`;
  detailVectorShape.textContent = `[${currentEmbedding.vectorShape.join(', ')}]`;
  detailVectorSource.textContent = selectedEmbedding.source === 'manual'
    ? '人工设定'
    : '未知 Token：零向量占位';
}

function renderEmbeddingMatrix() {
  embeddingShape.textContent = `[${currentEmbedding.shape.join(', ')}]`;
  embeddingDimension.textContent = String(currentEmbedding.dimension);
  const fragment = document.createDocumentFragment();

  if (currentEmbedding.rows.length === 0) {
    const row = document.createElement('tr');
    const cell = document.createElement('td');
    cell.className = 'matrix-empty';
    cell.colSpan = 6;
    cell.textContent = '当前没有矩阵行。空序列的 Embedding Shape 仍为 [0, 3]。';
    row.append(cell);
    embeddingMatrixBody.replaceChildren(row);
    return;
  }

  for (const embeddingRow of currentEmbedding.rows) {
    const row = document.createElement('tr');
    const isSelected = embeddingRow.index === selectedIndex;
    if (isSelected) row.classList.add('matrix-row--selected');

    const position = document.createElement('td');
    position.className = 'matrix-index';
    position.textContent = String(embeddingRow.index);

    const token = document.createElement('td');
    token.className = 'matrix-token';
    const tokenText = document.createElement('span');
    tokenText.className = 'matrix-token-text';
    tokenText.textContent = embeddingRow.text;
    token.append(tokenText);
    if (isSelected) {
      const selectedMarker = document.createElement('span');
      selectedMarker.className = 'matrix-selected-marker';
      selectedMarker.textContent = '当前选中';
      token.append(selectedMarker);
    }

    row.append(position, token);
    for (const component of embeddingRow.vector) {
      const value = document.createElement('td');
      value.className = 'matrix-number';
      value.textContent = String(component);
      row.append(value);
    }

    const source = document.createElement('td');
    source.className = embeddingRow.source === 'manual'
      ? 'matrix-source'
      : 'matrix-source matrix-source--unknown';
    source.textContent = embeddingRow.source === 'manual'
      ? '人工设定'
      : '未知 Token：零向量占位';
    row.append(source);
    fragment.append(row);
  }

  embeddingMatrixBody.replaceChildren(fragment);
}

function renderTokens() {
  const fragment = document.createDocumentFragment();

  if (currentAnalysis.tokens.length === 0) {
    const emptyItem = document.createElement('li');
    emptyItem.className = 'token-empty';
    emptyItem.textContent = '还没有 Token。输入文本后点击 Analyze。';
    fragment.append(emptyItem);
    tokenList.replaceChildren(fragment);
    renderDetails();
    return;
  }

  for (const token of currentAnalysis.tokens) {
    const item = document.createElement('li');
    item.className = 'token-item';

    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'token-button';
    button.dataset.tokenIndex = String(token.index);
    button.setAttribute('aria-pressed', String(token.index === selectedIndex));
    button.setAttribute('aria-label', `位置 ${token.index}，Token ${token.text}`);
    if (token.index === selectedIndex) button.classList.add('token-button--selected');

    const value = document.createElement('span');
    value.className = 'token-value';
    value.textContent = token.text;

    const position = document.createElement('span');
    position.className = 'token-position';
    position.textContent = `index ${token.index}`;

    const state = document.createElement('span');
    state.className = 'token-state';
    state.textContent = token.index === selectedIndex ? '当前选中' : '查看详情';

    button.append(value, position, state);
    item.append(button);
    fragment.append(item);
  }

  tokenList.replaceChildren(fragment);
  renderDetails();
}

tokenList.addEventListener('click', (event) => {
  const target = event.target;
  const button = target instanceof Element ? target.closest('button[data-token-index]') : null;
  if (!button) return;

  selectedIndex = Number(button.dataset.tokenIndex);
  for (const tokenButton of tokenList.querySelectorAll('.token-button')) {
    const isSelected = Number(tokenButton.dataset.tokenIndex) === selectedIndex;
    tokenButton.setAttribute('aria-pressed', String(isSelected));
    tokenButton.classList.toggle('token-button--selected', isSelected);
    tokenButton.querySelector('.token-state').textContent = isSelected ? '当前选中' : '查看详情';
  }
  renderDetails();
  renderEmbeddingMatrix();
  updateStaleState();
});

function renderAnalysis() {
  sequenceLength.textContent = String(currentAnalysis.sequenceLength);
  sequenceShape.textContent = `[${currentAnalysis.shape.join(', ')}]`;
  renderTokens();
  renderEmbeddingMatrix();
  updateStaleState();
}

function analyzeCurrentInput(statusMessage) {
  currentAnalysis = analyzeText(input.value);
  currentEmbedding = embedTokens(currentAnalysis.tokens);
  selectedIndex = currentAnalysis.tokens.length > 0 ? 0 : null;
  lastStaleState = false;
  renderAnalysis();
  setAnalysisStatus(`${statusMessage}：当前有 ${currentAnalysis.sequenceLength} 个 Token。`);
}

analyzeButton.addEventListener('click', () => {
  analyzeCurrentInput('Analyze 完成');
});

input.addEventListener('input', updateStaleState);

for (const exampleButton of document.querySelectorAll('[data-example]')) {
  exampleButton.addEventListener('click', () => {
    input.value = exampleButton.dataset.example;
    analyzeCurrentInput('示例已分析');
  });
}

analyzeCurrentInput('已分析默认示例');
