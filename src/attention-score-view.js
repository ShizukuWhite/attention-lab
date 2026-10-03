import { matrixCellSteps } from './matrix-multiply.js';

function formatNumber(value) {
  return String(Number(value.toFixed(6)));
}

function formatFactor(value) {
  const formatted = formatNumber(value);
  return value < 0 ? `(${formatted})` : formatted;
}

function formatShape(shape) {
  return `[${shape.join(', ')}]`;
}

function sourceLabel(source) {
  return source === 'manual' ? '人工设定' : '未知 Token · 零向量占位';
}

/**
 * Create the view for the raw QKᵀ matrix and one selected score trace.
 *
 * @param {Element} root
 * @returns {{ render(embedding: object, projections: object, scores: object): void }}
 */
export function createAttentionScoreView(root) {
  const document = root.ownerDocument;
  const query = (selector) => root.querySelector(selector);
  const ktHead = query('#attention-score-kt-head');
  const ktBody = query('#attention-score-kt-body');
  const scoreHead = query('#attention-score-s-head');
  const scoreBody = query('#attention-score-s-body');

  let currentEmbedding = null;
  let currentProjections = null;
  let currentScores = null;
  let selection = null;

  function makeElement(tagName, className, text) {
    const element = document.createElement(tagName);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
  }

  function renderFlow() {
    query('#attention-score-flow').textContent = `Q ${formatShape(currentProjections.Q.shape)} × Kᵀ ${formatShape(currentScores.KT.shape)} → S ${formatShape(currentScores.S.shape)}`;
    query('#attention-score-q-shape').textContent = formatShape(currentProjections.Q.shape);
    query('#attention-score-k-shape').textContent = formatShape(currentProjections.K.shape);
    query('#attention-score-v-shape').textContent = formatShape(currentProjections.V.shape);
    query('#attention-score-s-shape-summary').textContent = formatShape(currentScores.S.shape);
  }

  function renderTransposedKeys() {
    query('#attention-score-kt-shape').textContent = formatShape(currentScores.KT.shape);
    const header = document.createDocumentFragment();
    const componentHeading = makeElement('th', 'attention-score-row-axis', '分量 t');
    componentHeading.scope = 'col';
    header.append(componentHeading);

    for (const token of currentEmbedding.rows) {
      const cell = makeElement('th', 'attention-score-token-axis', `Key 位置 ${token.index} · ${token.text}`);
      cell.scope = 'col';
      cell.dataset.attentionScoreKtColumn = String(token.index);
      header.append(cell);
    }
    ktHead.replaceChildren(header);

    const body = document.createDocumentFragment();
    for (let component = 0; component < currentScores.KT.shape[0]; component += 1) {
      const row = makeElement('tr');
      const axis = makeElement('th', 'attention-score-row-axis', String(component));
      axis.scope = 'row';
      row.append(axis);
      for (let keyPosition = 0; keyPosition < currentScores.KT.shape[1]; keyPosition += 1) {
        const cell = makeElement('td', 'attention-score-number attention-score-kt-cell', formatNumber(currentScores.KT.matrix[component][keyPosition]));
        cell.dataset.attentionScoreKeyColumn = String(keyPosition);
        row.append(cell);
      }
      body.append(row);
    }
    ktBody.replaceChildren(body);

    const empty = query('#attention-score-kt-empty');
    empty.hidden = currentEmbedding.rows.length !== 0;
    if (currentEmbedding.rows.length === 0) {
      empty.textContent = '空序列仍保留 Kᵀ Shape [2, 0]：表中有分量 t=0、t=1 两行，但没有 Key 位置列。';
    }
  }

  function makeScoreButton(rowIndex, columnIndex) {
    const token = currentEmbedding.rows[columnIndex];
    const value = currentScores.S.matrix[rowIndex][columnIndex];
    const cell = makeElement('td', 'attention-score-value-cell');
    const button = makeElement('button', 'attention-score-cell-button');
    const number = makeElement('span', 'attention-score-cell-number', formatNumber(value));
    const marker = makeElement('span', 'attention-score-cell-marker', '当前选中');
    marker.hidden = true;
    button.type = 'button';
    button.dataset.attentionScoreCell = '';
    button.dataset.row = String(rowIndex);
    button.dataset.column = String(columnIndex);
    button.setAttribute('aria-pressed', 'false');
    button.setAttribute(
      'aria-label',
      `S[${rowIndex}, ${columnIndex}]，Query 位置 ${currentEmbedding.rows[rowIndex].index} · ${currentEmbedding.rows[rowIndex].text}，Key 位置 ${token.index} · ${token.text}，原始分数 ${formatNumber(value)}`,
    );
    button.append(number, marker);
    cell.append(button);
    return cell;
  }

  function renderScoreMatrix() {
    query('#attention-score-s-shape').textContent = formatShape(currentScores.S.shape);
    const header = document.createDocumentFragment();
    const corner = makeElement('th', 'attention-score-corner-axis', 'Query 位置 i ↓ / Key 位置 j →');
    corner.scope = 'col';
    header.append(corner);

    for (const token of currentEmbedding.rows) {
      const cell = makeElement('th', 'attention-score-token-axis', `Key 位置 ${token.index} · ${token.text}`);
      cell.scope = 'col';
      cell.dataset.attentionScoreSColumn = String(token.index);
      header.append(cell);
    }
    scoreHead.replaceChildren(header);

    const body = document.createDocumentFragment();
    for (let rowIndex = 0; rowIndex < currentScores.S.shape[0]; rowIndex += 1) {
      const row = makeElement('tr');
      const token = currentEmbedding.rows[rowIndex];
      const axis = makeElement('th', 'attention-score-token-axis attention-score-query-axis', `Query 位置 ${token.index} · ${token.text}`);
      axis.scope = 'row';
      axis.dataset.attentionScoreSRow = String(token.index);
      row.append(axis);

      for (let columnIndex = 0; columnIndex < currentScores.S.shape[1]; columnIndex += 1) {
        row.append(makeScoreButton(rowIndex, columnIndex));
      }
      body.append(row);
    }
    scoreBody.replaceChildren(body);
  }

  function updateScoreButtons() {
    for (const button of root.querySelectorAll('button[data-attention-score-cell]')) {
      const isSelected = selection !== null
        && Number(button.dataset.row) === selection.row
        && Number(button.dataset.column) === selection.column;
      button.setAttribute('aria-pressed', String(isSelected));
      button.classList.toggle('attention-score-cell-button--selected', isSelected);
      button.querySelector('.attention-score-cell-marker').hidden = !isSelected;
    }
  }

  function updateAxisHighlights() {
    for (const axis of root.querySelectorAll('[data-attention-score-kt-column], [data-attention-score-key-column], [data-attention-score-s-column], [data-attention-score-s-row]')) {
      const isColumnAxis = axis.hasAttribute('data-attention-score-kt-column')
        || axis.hasAttribute('data-attention-score-key-column')
        || axis.hasAttribute('data-attention-score-s-column');
      const axisPosition = Number(
        axis.dataset.attentionScoreKtColumn
          ?? axis.dataset.attentionScoreKeyColumn
          ?? axis.dataset.attentionScoreSColumn
          ?? axis.dataset.attentionScoreSRow,
      );
      const selectedPosition = selection === null
        ? null
        : isColumnAxis ? selection.column : selection.row;
      axis.classList.toggle('attention-score-axis--selected', selectedPosition === axisPosition);
    }
  }

  function renderTrace() {
    const trace = query('#attention-score-trace');
    const empty = query('#attention-score-empty-note');
    const selectedStatus = query('#attention-score-selected');
    const unknownNote = query('#attention-score-selected-unknown-note');
    const unknownRows = currentEmbedding.rows.filter((row) => row.source === 'unknown');
    query('#attention-score-unknown-note').hidden = unknownRows.length === 0;
    query('#attention-score-unknown-note').textContent = '未知 Token 的 X 行使用 [0, 0, 0] 占位，因此对应 Q/K 行为 [0, 0]，S 中对应的分数行与列为 0。这来自教学占位规则，不是模型判断或词义结论。';

    if (!selection || !currentEmbedding || !currentProjections || !currentScores) {
      trace.hidden = true;
      selectedStatus.textContent = '';
      selectedStatus.hidden = true;
      empty.hidden = currentEmbedding?.rows.length !== 0;
      unknownNote.hidden = true;
      return;
    }

    const queryToken = currentEmbedding.rows[selection.row];
    const keyToken = currentEmbedding.rows[selection.column];
    const calculation = matrixCellSteps(
      currentProjections.Q,
      currentScores.KT,
      selection.row,
      selection.column,
    );

    trace.hidden = false;
    empty.hidden = true;
    query('#attention-score-vector-equation').textContent = `S[${selection.row}, ${selection.column}] = q_${selection.row} · k_${selection.column}`;
    selectedStatus.hidden = false;
    selectedStatus.textContent = `当前选择 S[${selection.row}, ${selection.column}]：Query 位置 ${queryToken.index} · ${queryToken.text} 对 Key 位置 ${keyToken.index} · ${keyToken.text}，raw score = ${formatNumber(calculation.result)}。`;
    query('#attention-score-trace-cell').textContent = `S[${selection.row}, ${selection.column}]`;
    query('#attention-score-trace-route').textContent = `Q 第 ${selection.row} 行 × Kᵀ 第 ${selection.column} 列（对应 K 第 ${selection.column} 行）`;
    query('#attention-score-query-token').textContent = `位置 ${queryToken.index} · ${queryToken.text}`;
    query('#attention-score-query-vector').textContent = `[${calculation.leftRow.map(formatNumber).join(', ')}]`;
    query('#attention-score-query-shape').textContent = formatShape(calculation.leftShape);
    query('#attention-score-query-source').textContent = sourceLabel(queryToken.source);
    query('#attention-score-key-token').textContent = `位置 ${keyToken.index} · ${keyToken.text}`;
    query('#attention-score-key-vector').textContent = `[${calculation.rightColumn.map(formatNumber).join(', ')}]`;
    query('#attention-score-key-shape').textContent = formatShape(calculation.rightShape);
    query('#attention-score-key-source').textContent = sourceLabel(keyToken.source);
    query('#attention-score-k-transpose-note').textContent = `K[${selection.column}, t] = Kᵀ[t, ${selection.column}]：K 第 ${selection.column} 行就是 Kᵀ 第 ${selection.column} 列。`;
    query('#attention-score-result').textContent = formatNumber(calculation.result);
    query('#attention-score-result-shape').textContent = formatShape(calculation.outputShape);

    const stepsBody = query('#attention-score-steps');
    const steps = document.createDocumentFragment();
    for (const step of calculation.steps) {
      const row = makeElement('tr');
      for (const value of [
        String(step.index),
        formatNumber(step.left),
        formatNumber(step.right),
        formatNumber(step.product),
        formatNumber(step.runningSum),
      ]) {
        row.append(makeElement('td', 'dot-number', value));
      }
      steps.append(row);
    }
    stepsBody.replaceChildren(steps);

    const terms = calculation.steps.map((step) => (
      `${formatFactor(step.left)} × ${formatFactor(step.right)}`
    ));
    query('#attention-score-equation').textContent = `${terms.join(' + ')} = ${formatNumber(calculation.result)}`;

    const unknownDetails = [];
    if (queryToken.source === 'unknown') {
      unknownDetails.push(`Query 位置 ${queryToken.index} 的 Token「${queryToken.text}」是未知 Token，因此 Q 行为 [0, 0]，这一行的分数为 0。`);
    }
    if (keyToken.source === 'unknown') {
      unknownDetails.push(`Key 位置 ${keyToken.index} 的 Token「${keyToken.text}」是未知 Token，因此 K 行为 [0, 0]，这一列的分数为 0。`);
    }
    unknownNote.hidden = unknownDetails.length === 0;
    unknownNote.textContent = unknownDetails.join(' ');
  }

  function updateSelection() {
    updateScoreButtons();
    updateAxisHighlights();
    renderTrace();
  }

  function render(embedding, projections, scores) {
    currentEmbedding = embedding;
    currentProjections = projections;
    currentScores = scores;
    selection = embedding.rows.length > 0
      ? { row: 0, column: embedding.rows.length - 1 }
      : null;

    renderFlow();
    renderTransposedKeys();
    renderScoreMatrix();
    updateSelection();
  }

  root.addEventListener('click', (event) => {
    const target = event.target;
    const button = target instanceof Element ? target.closest('button[data-attention-score-cell]') : null;
    if (!button || !root.contains(button)) return;

    selection = {
      row: Number(button.dataset.row),
      column: Number(button.dataset.column),
    };
    updateSelection();
  });

  return { render };
}
