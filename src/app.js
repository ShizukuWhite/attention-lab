import { analyzeText } from './tokenizer.js';

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

let currentAnalysis;
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
        ? '输入已修改，请点击 Analyze 更新下方结果。'
        : '当前输入与最近一次分析一致。',
    );
    lastStaleState = isStale;
  }
}

function renderDetails() {
  const selectedToken = currentAnalysis.tokens.find((token) => token.index === selectedIndex);

  if (!selectedToken) {
    detailContent.hidden = true;
    detailEmpty.hidden = false;
    detailEmpty.textContent = currentAnalysis.tokens.length === 0
      ? '当前序列为空：N = 0，Shape 为 [0]，没有可选 Token。'
      : '点击一个 Token，这里会显示它在序列中的位置。';
    return;
  }

  detailEmpty.hidden = true;
  detailContent.hidden = false;
  detailToken.textContent = selectedToken.text;
  detailIndex.textContent = String(selectedToken.index);
  detailOrdinal.textContent = String(selectedToken.index + 1);
  detailCodePoints.textContent = String(selectedToken.codePointLength);
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
  const button = event.target.closest('button[data-token-index]');
  if (!button) return;

  selectedIndex = Number(button.dataset.tokenIndex);
  for (const tokenButton of tokenList.querySelectorAll('.token-button')) {
    const isSelected = Number(tokenButton.dataset.tokenIndex) === selectedIndex;
    tokenButton.setAttribute('aria-pressed', String(isSelected));
    tokenButton.classList.toggle('token-button--selected', isSelected);
    tokenButton.querySelector('.token-state').textContent = isSelected ? '当前选中' : '查看详情';
  }
  renderDetails();
  updateStaleState();
});

function renderAnalysis() {
  sequenceLength.textContent = String(currentAnalysis.sequenceLength);
  sequenceShape.textContent = `[${currentAnalysis.shape.join(', ')}]`;
  renderTokens();
  updateStaleState();
}

function analyzeCurrentInput(statusMessage) {
  currentAnalysis = analyzeText(input.value);
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
