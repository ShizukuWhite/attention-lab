import { analyzeText } from './tokenizer.js';
import { embedTokens } from './embedding.js';
import { dotProductSteps } from './dot-product.js';
import { createProjectionView } from './projection-view.js';

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
const dotLeftSelect = document.querySelector('#dot-left-select');
const dotRightSelect = document.querySelector('#dot-right-select');
const pairLeftToken = document.querySelector('#pair-left-token');
const pairRightToken = document.querySelector('#pair-right-token');
const pairLeftVector = document.querySelector('#pair-left-vector');
const pairRightVector = document.querySelector('#pair-right-vector');
const pairLeftShape = document.querySelector('#pair-left-shape');
const pairRightShape = document.querySelector('#pair-right-shape');
const pairLeftSource = document.querySelector('#pair-left-source');
const pairRightSource = document.querySelector('#pair-right-source');
const dotProductEmpty = document.querySelector('#dot-product-empty');
const singleTokenNote = document.querySelector('#single-token-note');
const dotProductResults = document.querySelector('#dot-product-results');
const dotProductStepsBody = document.querySelector('#dot-product-steps');
const dotProductEquation = document.querySelector('#dot-product-equation');
const dotProductValue = document.querySelector('#dot-product-value');
const dotOutputShape = document.querySelector('#dot-output-shape');
const dotResultShape = document.querySelector('#dot-result-shape');
const dotUnknownNote = document.querySelector('#dot-unknown-note');
const projectionView = createProjectionView(document.querySelector('#projection-panel'));

let currentAnalysis;
let currentEmbedding;
let selectedIndex = null;
let pairLeftIndex = null;
let pairRightIndex = null;
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
        ? '输入已修改；请点击 Analyze 同步 Tokens、Embedding、点积与 Q/K/V。'
        : 'Tokens、Embedding、点积与 Q/K/V 都来自当前输入的最近一次 Analyze。',
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

function updateEmbeddingMatrixSelection() {
  const renderedRows = embeddingMatrixBody.querySelectorAll('tr');
  currentEmbedding.rows.forEach((embeddingRow, rowIndex) => {
    const row = renderedRows[rowIndex];
    if (!row) return;

    const isSelected = embeddingRow.index === selectedIndex;
    row.classList.toggle('matrix-row--selected', isSelected);
    const tokenCell = row.querySelector('.matrix-token');
    let marker = tokenCell.querySelector('.matrix-selected-marker');

    if (isSelected && !marker) {
      marker = document.createElement('span');
      marker.className = 'matrix-selected-marker';
      marker.textContent = '当前选中';
      tokenCell.append(marker);
    } else if (!isSelected && marker) {
      marker.remove();
    }
  });
}

function formatDotNumber(value) {
  return String(Number(value.toFixed(6)));
}

function formatVector(vector) {
  return `[${vector.map(formatDotNumber).join(', ')}]`;
}

function embeddingSourceLabel(source) {
  return source === 'manual' ? '人工设定' : '未知 Token：零向量占位';
}

function populateDotProductSelectors() {
  const rows = currentEmbedding.rows;
  const leftOptions = document.createDocumentFragment();
  const rightOptions = document.createDocumentFragment();

  for (const row of rows) {
    const makeOption = () => {
      const option = document.createElement('option');
      option.value = String(row.index);
      option.textContent = `位置 ${row.index} · ${row.text}`;
      return option;
    };
    leftOptions.append(makeOption());
    rightOptions.append(makeOption());
  }

  dotLeftSelect.replaceChildren(leftOptions);
  dotRightSelect.replaceChildren(rightOptions);
  dotLeftSelect.disabled = rows.length === 0;
  dotRightSelect.disabled = rows.length === 0;
  dotLeftSelect.value = pairLeftIndex === null ? '' : String(pairLeftIndex);
  dotRightSelect.value = pairRightIndex === null ? '' : String(pairRightIndex);
}

function renderDotProduct() {
  const rows = currentEmbedding.rows;
  const leftRow = rows.find((row) => row.index === pairLeftIndex);
  const rightRow = rows.find((row) => row.index === pairRightIndex);
  const hasRows = rows.length > 0;

  dotProductEmpty.hidden = hasRows;
  dotProductResults.hidden = !hasRows;
  singleTokenNote.hidden = rows.length !== 1;
  dotLeftSelect.disabled = !hasRows;
  dotRightSelect.disabled = !hasRows;
  dotOutputShape.textContent = '[]';
  dotResultShape.textContent = '[]';

  if (!hasRows || !leftRow || !rightRow) return;

  dotLeftSelect.value = String(pairLeftIndex);
  dotRightSelect.value = String(pairRightIndex);
  pairLeftToken.textContent = `位置 ${leftRow.index} · ${leftRow.text}`;
  pairRightToken.textContent = `位置 ${rightRow.index} · ${rightRow.text}`;
  pairLeftVector.textContent = formatVector(leftRow.vector);
  pairRightVector.textContent = formatVector(rightRow.vector);
  pairLeftShape.textContent = `[${leftRow.vector.length}]`;
  pairRightShape.textContent = `[${rightRow.vector.length}]`;
  pairLeftSource.textContent = embeddingSourceLabel(leftRow.source);
  pairRightSource.textContent = embeddingSourceLabel(rightRow.source);

  const calculation = dotProductSteps(leftRow.vector, rightRow.vector);
  const fragment = document.createDocumentFragment();
  for (const step of calculation.steps) {
    const row = document.createElement('tr');
    for (const [className, value] of [
      ['dot-index', String(step.index)],
      ['dot-number', formatDotNumber(step.left)],
      ['dot-number', formatDotNumber(step.right)],
      ['dot-number', formatDotNumber(step.product)],
      ['dot-number', formatDotNumber(step.runningSum)],
    ]) {
      const cell = document.createElement('td');
      cell.className = className;
      cell.textContent = value;
      row.append(cell);
    }
    fragment.append(row);
  }
  dotProductStepsBody.replaceChildren(fragment);

  const terms = calculation.steps.map((step) => (
    `${formatDotNumber(step.left)} × ${formatDotNumber(step.right)}`
  ));
  dotProductEquation.textContent = `${terms.join(' + ')} = ${formatDotNumber(calculation.result)}`;
  dotProductValue.textContent = formatDotNumber(calculation.result);
  dotOutputShape.textContent = `[${calculation.outputShape.join(', ')}]`;
  dotResultShape.textContent = `[${calculation.outputShape.join(', ')}]`;

  const includesUnknown = leftRow.source === 'unknown' || rightRow.source === 'unknown';
  dotUnknownNote.hidden = !includesUnknown;
  if (includesUnknown) {
    dotUnknownNote.textContent = '零结果来自未知 Token 的零向量占位；这不代表两个词不相关或词义相同。';
  }
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
  updateEmbeddingMatrixSelection();
  updateStaleState();
});

function updateDotProductSelection() {
  pairLeftIndex = dotLeftSelect.value === '' ? null : Number(dotLeftSelect.value);
  pairRightIndex = dotRightSelect.value === '' ? null : Number(dotRightSelect.value);
  renderDotProduct();
  updateStaleState();
}

dotLeftSelect.addEventListener('change', updateDotProductSelection);
dotRightSelect.addEventListener('change', updateDotProductSelection);

function renderAnalysis() {
  sequenceLength.textContent = String(currentAnalysis.sequenceLength);
  sequenceShape.textContent = `[${currentAnalysis.shape.join(', ')}]`;
  populateDotProductSelectors();
  renderTokens();
  renderEmbeddingMatrix();
  renderDotProduct();
  projectionView.render(currentEmbedding);
  updateStaleState();
}

function analyzeCurrentInput(statusMessage) {
  currentAnalysis = analyzeText(input.value);
  currentEmbedding = embedTokens(currentAnalysis.tokens);
  selectedIndex = currentAnalysis.tokens.length > 0 ? 0 : null;
  pairLeftIndex = currentEmbedding.rows.length > 0 ? currentEmbedding.rows[0].index : null;
  pairRightIndex = currentEmbedding.rows.length > 1
    ? currentEmbedding.rows.at(-1).index
    : pairLeftIndex;
  lastStaleState = false;
  renderAnalysis();
  setAnalysisStatus(`${statusMessage}：当前有 ${currentAnalysis.sequenceLength} 个 Token；Tokens、Embedding、点积与 Q/K/V 已同步。`);
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
