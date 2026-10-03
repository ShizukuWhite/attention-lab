import { matrixCellSteps } from './matrix-multiply.js';

function formatNumber(value) {
  return String(Number(value.toFixed(6)));
}

function formatShape(shape) {
  return `[${shape.join(', ')}]`;
}

function sourceLabel(source) {
  return source === 'manual' ? '人工设定' : '未知 Token · 零向量占位';
}

/**
 * Create the Phase 6 view. It renders only the supplied Analyze snapshot;
 * selection changes update trace and highlighting without recomputing scores.
 * @param {Element} root
 * @returns {{ render(embedding: object, projections: object, scores: object, attention: object): void }}
 */
export function createScaledAttentionView(root) {
  const document = root.ownerDocument;
  const query = (selector) => root.querySelector(selector);
  const scaledHead = query('#scaled-attention-scaled-head');
  const scaledBody = query('#scaled-attention-scaled-body');
  const weightsHead = query('#scaled-attention-weights-head');
  const weightsBody = query('#scaled-attention-weights-body');
  const outputHead = query('#scaled-attention-output-head');
  const outputBody = query('#scaled-attention-output-body');
  const valueColumnHead = query('#scaled-attention-v-column-head');
  const valueColumnBody = query('#scaled-attention-v-column-body');

  let currentEmbedding = null;
  let currentProjections = null;
  let currentScores = null;
  let currentAttention = null;
  let selection = null;

  function makeElement(tagName, className, text) {
    const element = document.createElement(tagName);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
  }

  function renderFlow() {
    const queryShape = currentProjections.Q.shape;
    const keyTransposeShape = currentScores.KT.shape;
    const scoreShape = currentScores.S.shape;
    const scaledShape = currentAttention.scaled.shape;
    const weightShape = currentAttention.weights.shape;
    const valueShape = currentProjections.V.shape;
    const outputShape = currentAttention.output.shape;
    const dK = queryShape[1];
    query('#scaled-attention-flow').textContent = `Q ${formatShape(queryShape)} × Kᵀ ${formatShape(keyTransposeShape)} → S ${formatShape(scoreShape)} → scaled ${formatShape(scaledShape)} → A ${formatShape(weightShape)} × V ${formatShape(valueShape)} → O ${formatShape(outputShape)}`;
    query('#scaled-attention-d-k').textContent = `d_k = Q.shape[1] = ${dK}`;
    query('#scaled-attention-scale').textContent = `√d_k = √${dK} ≈ ${formatNumber(currentAttention.scale)}`;
    query('#scaled-attention-scaled-shape').textContent = formatShape(scaledShape);
    query('#scaled-attention-weights-shape').textContent = formatShape(weightShape);
    query('#scaled-attention-output-shape').textContent = formatShape(outputShape);
    query('#scaled-attention-output-shape-summary').textContent = formatShape(outputShape);
  }

  function makeTokenAxis(token, className, positionLabel) {
    return makeElement('th', className, `${positionLabel} ${token.index} · ${token.text}`);
  }

  function renderInputMatrix(head, body, value, valueClass, interactiveRows) {
    const header = document.createDocumentFragment();
    const corner = makeElement('th', 'scaled-attention-corner-axis', 'Query 位置 i ↓ / Key 位置 j →');
    corner.scope = 'col';
    header.append(corner);
    for (const token of currentEmbedding.rows) {
      const axis = makeTokenAxis(token, 'scaled-attention-token-axis', 'Key 位置');
      axis.scope = 'col';
      header.append(axis);
    }
    head.replaceChildren(header);

    const rows = document.createDocumentFragment();
    for (let rowIndex = 0; rowIndex < value.shape[0]; rowIndex += 1) {
      const row = makeElement('tr');
      row.dataset.phase6Row = String(rowIndex);
      const token = currentEmbedding.rows[rowIndex];
      const axis = makeElement('th', 'scaled-attention-query-axis');
      axis.scope = 'row';
      if (interactiveRows) {
        const button = makeElement('button', 'scaled-attention-query-button');
        button.type = 'button';
        button.dataset.phase6QueryRow = String(rowIndex);
        button.setAttribute('aria-pressed', 'false');
        button.setAttribute('aria-label', `选择 Query 行 ${rowIndex}，位置 ${token.index}，Token ${token.text}`);
        const label = makeElement('span', '', `Query 位置 ${token.index} · ${token.text}`);
        const marker = makeElement('span', 'scaled-attention-query-marker', '当前 Query');
        marker.hidden = true;
        button.append(label, marker);
        axis.append(button);
      } else {
        axis.textContent = `Query 位置 ${token.index} · ${token.text}`;
      }
      row.append(axis);

      for (let columnIndex = 0; columnIndex < value.shape[1]; columnIndex += 1) {
        const cell = makeElement('td', valueClass, formatNumber(value.matrix[rowIndex][columnIndex]));
        cell.dataset.phase6KeyColumn = String(columnIndex);
        row.append(cell);
      }
      rows.append(row);
    }
    body.replaceChildren(rows);
  }

  function makeOutputButton(rowIndex, columnIndex) {
    const token = currentEmbedding.rows[rowIndex];
    const value = currentAttention.output.matrix[rowIndex][columnIndex];
    const cell = makeElement('td', 'scaled-attention-output-cell');
    const button = makeElement('button', 'scaled-attention-output-button');
    const number = makeElement('span', '', formatNumber(value));
    const marker = makeElement('span', 'scaled-attention-output-marker', '当前选择');
    marker.hidden = true;
    button.type = 'button';
    button.dataset.phase6OutputCell = '';
    button.dataset.row = String(rowIndex);
    button.dataset.column = String(columnIndex);
    button.setAttribute('aria-pressed', 'false');
    button.setAttribute(
      'aria-label',
      `O[${rowIndex}, ${columnIndex}]，Query 位置 ${token.index} · ${token.text}，输出分量 ${columnIndex}，数值 ${formatNumber(value)}`,
    );
    button.append(number, marker);
    cell.append(button);
    return cell;
  }

  function renderOutputMatrix() {
    const header = document.createDocumentFragment();
    const corner = makeElement('th', 'scaled-attention-corner-axis', 'Query 位置 i ↓ / 输出分量 c →');
    corner.scope = 'col';
    header.append(corner);
    for (let columnIndex = 0; columnIndex < currentAttention.output.shape[1]; columnIndex += 1) {
      const axis = makeElement('th', 'scaled-attention-component-axis', `输出分量 ${columnIndex}`);
      axis.scope = 'col';
      axis.dataset.phase6OutputColumn = String(columnIndex);
      header.append(axis);
    }
    outputHead.replaceChildren(header);

    const rows = document.createDocumentFragment();
    for (let rowIndex = 0; rowIndex < currentAttention.output.shape[0]; rowIndex += 1) {
      const row = makeElement('tr');
      row.dataset.phase6Row = String(rowIndex);
      const token = currentEmbedding.rows[rowIndex];
      const axis = makeElement('th', 'scaled-attention-query-axis', `Query 位置 ${token.index} · ${token.text}`);
      axis.scope = 'row';
      row.append(axis);
      for (let columnIndex = 0; columnIndex < currentAttention.output.shape[1]; columnIndex += 1) {
        row.append(makeOutputButton(rowIndex, columnIndex));
      }
      rows.append(row);
    }
    outputBody.replaceChildren(rows);
  }

  function renderValueColumn() {
    const header = document.createDocumentFragment();
    const positionAxis = makeElement('th', 'scaled-attention-corner-axis', 'Key 位置 j');
    positionAxis.scope = 'col';
    const valueAxis = makeElement('th', 'scaled-attention-component-axis', `V[j, ${selection.column}]`);
    valueAxis.scope = 'col';
    header.append(positionAxis, valueAxis);
    valueColumnHead.replaceChildren(header);
    query('#scaled-attention-v-column-shape').textContent = formatShape([currentProjections.V.shape[0]]);
    query('#scaled-attention-v-column-label').textContent = `当前 V 输出分量 c = ${selection.column}，来源 V = XWv`;

    const rows = document.createDocumentFragment();
    for (let rowIndex = 0; rowIndex < currentProjections.V.shape[0]; rowIndex += 1) {
      const row = makeElement('tr');
      const token = currentEmbedding.rows[rowIndex];
      const tokenAxis = makeElement('th', 'scaled-attention-token-axis', `位置 ${token.index} · ${token.text}`);
      tokenAxis.scope = 'row';
      const value = makeElement('td', 'scaled-attention-number', formatNumber(currentProjections.V.matrix[rowIndex][selection.column]));
      value.dataset.phase6KeyColumn = String(rowIndex);
      row.append(tokenAxis, value);
      rows.append(row);
    }
    valueColumnBody.replaceChildren(rows);
  }

  function updateHighlights() {
    for (const row of root.querySelectorAll('[data-phase6-row]')) {
      row.classList.toggle('scaled-attention-row--selected', Number(row.dataset.phase6Row) === selection.row);
    }
    for (const button of root.querySelectorAll('[data-phase6-query-row]')) {
      const isSelected = Number(button.dataset.phase6QueryRow) === selection.row;
      button.setAttribute('aria-pressed', String(isSelected));
      button.classList.toggle('scaled-attention-query-button--selected', isSelected);
      button.querySelector('.scaled-attention-query-marker').hidden = !isSelected;
    }
    for (const button of root.querySelectorAll('[data-phase6-output-cell]')) {
      const isSelected = Number(button.dataset.row) === selection.row
        && Number(button.dataset.column) === selection.column;
      button.setAttribute('aria-pressed', String(isSelected));
      button.classList.toggle('scaled-attention-output-button--selected', isSelected);
      button.querySelector('.scaled-attention-output-marker').hidden = !isSelected;
    }
    for (const axis of root.querySelectorAll('[data-phase6-output-column]')) {
      axis.classList.toggle('scaled-attention-axis--selected', Number(axis.dataset.phase6OutputColumn) === selection.column);
    }
  }

  function renderRowTrace() {
    const rowInfo = currentAttention.rowSteps[selection.row];
    const queryToken = currentEmbedding.rows[selection.row];
    query('#scaled-attention-row-title').textContent = `Query 行 ${selection.row} · 位置 ${queryToken.index} · ${queryToken.text}`;
    query('#scaled-attention-row-max').textContent = formatNumber(rowInfo.max);
    query('#scaled-attention-row-denominator').textContent = formatNumber(rowInfo.denominator);
    query('#scaled-attention-row-weight-sum').textContent = `${formatNumber(rowInfo.weightSum)} ≈ 1`;
    query('#scaled-attention-row-weight-list').textContent = `[${rowInfo.steps.map((step) => formatNumber(step.weight)).join(', ')}]`;

    const rows = document.createDocumentFragment();
    for (const step of rowInfo.steps) {
      const token = currentEmbedding.rows[step.index];
      const row = makeElement('tr');
      const source = makeElement('td', 'scaled-attention-source-cell', `Key 位置 ${token.index} · ${token.text} · ${sourceLabel(token.source)}`);
      row.append(source);
      for (const value of [
        currentScores.S.matrix[selection.row][step.index],
        currentScores.S.matrix[selection.row][step.index] / currentAttention.scale,
        step.scaled,
        step.shifted,
        step.exponential,
        step.exponential / rowInfo.denominator,
        step.weight,
      ]) {
        row.append(makeElement('td', 'scaled-attention-number', formatNumber(value)));
      }
      rows.append(row);
    }
    query('#scaled-attention-row-steps').replaceChildren(rows);
  }

  function renderOutputTrace() {
    const rowIndex = selection.row;
    const columnIndex = selection.column;
    const queryToken = currentEmbedding.rows[rowIndex];
    const calculation = matrixCellSteps(currentAttention.weights, currentProjections.V, rowIndex, columnIndex);
    query('#scaled-attention-output-trace-cell').textContent = `O[${rowIndex}, ${columnIndex}]`;
    query('#scaled-attention-output-trace-route').textContent = `A 第 ${rowIndex} 行 ${formatShape(calculation.leftShape)} × V 第 ${columnIndex} 列 ${formatShape(calculation.rightShape)} → O[${rowIndex}, ${columnIndex}] ${formatShape(calculation.outputShape)}`;
    query('#scaled-attention-output-trace-query').textContent = `Query 位置 ${queryToken.index} · ${queryToken.text}`;
    query('#scaled-attention-output-trace-value').textContent = formatNumber(calculation.result);

    const rows = document.createDocumentFragment();
    for (const step of calculation.steps) {
      const token = currentEmbedding.rows[step.index];
      const row = makeElement('tr');
      row.append(
        makeElement('td', 'scaled-attention-number', `位置 ${token.index} · ${token.text}`),
        makeElement('td', 'scaled-attention-source-cell', sourceLabel(token.source)),
        makeElement('td', 'scaled-attention-number', formatNumber(step.left)),
        makeElement('td', 'scaled-attention-number', formatNumber(step.right)),
        makeElement('td', 'scaled-attention-number', formatNumber(step.product)),
        makeElement('td', 'scaled-attention-number', formatNumber(step.runningSum)),
      );
      rows.append(row);
    }
    query('#scaled-attention-output-steps').replaceChildren(rows);

    const terms = calculation.steps.map((step) => `${formatNumber(step.left)} × ${formatNumber(step.right)}`);
    query('#scaled-attention-output-equation').textContent = `${terms.join(' + ')} = ${formatNumber(calculation.result)}`;

    const unknownDetails = [];
    if (queryToken.source === 'unknown') {
      unknownDetails.push(`Query 位置 ${queryToken.index} 的「${queryToken.text}」使用零向量占位，所以这一行 raw score 为 0，权重均匀；加权组合其他 V 后，输出通常仍非零。`);
    }
    const unknownValuePositions = calculation.steps
      .filter((step) => currentEmbedding.rows[step.index].source === 'unknown');
    if (unknownValuePositions.length > 0) {
      unknownDetails.push('未知 Key 的零分数仍可能得到非零权重；对应的 V 同样是零向量，因此该位置的乘积贡献为 0。');
    }
    const unknownNote = query('#scaled-attention-unknown-note');
    unknownNote.textContent = unknownDetails.join(' ');
    unknownNote.hidden = unknownDetails.length === 0;
  }

  function clearTrace() {
    for (const selector of [
      '#scaled-attention-row-title',
      '#scaled-attention-row-max',
      '#scaled-attention-row-denominator',
      '#scaled-attention-row-weight-sum',
      '#scaled-attention-row-weight-list',
      '#scaled-attention-output-trace-cell',
      '#scaled-attention-output-trace-route',
      '#scaled-attention-output-trace-query',
      '#scaled-attention-output-trace-value',
      '#scaled-attention-output-equation',
      '#scaled-attention-v-column-label',
    ]) {
      query(selector).textContent = '';
    }
    query('#scaled-attention-row-steps').replaceChildren();
    query('#scaled-attention-output-steps').replaceChildren();
    query('#scaled-attention-v-column-head').replaceChildren();
    query('#scaled-attention-v-column-body').replaceChildren();
    query('#scaled-attention-v-column-shape').textContent = formatShape([0]);
    query('#scaled-attention-unknown-note').textContent = '';
    query('#scaled-attention-unknown-note').hidden = true;
    query('#scaled-attention-empty-note').hidden = false;
    query('#scaled-attention-selected').textContent = '';
    query('#scaled-attention-selected').hidden = true;
    query('#scaled-attention-trace').hidden = true;
  }

  function updateSelection() {
    if (!selection) {
      clearTrace();
      return;
    }
    updateHighlights();
    renderValueColumn();
    renderRowTrace();
    renderOutputTrace();
    const token = currentEmbedding.rows[selection.row];
    const selected = query('#scaled-attention-selected');
    selected.textContent = `本阶段当前选择 Query 位置 ${token.index} · ${token.text}，输出分量 ${selection.column}；追踪 A 的整行和 O[${selection.row}, ${selection.column}]。`;
    selected.hidden = false;
    query('#scaled-attention-empty-note').hidden = true;
    query('#scaled-attention-trace').hidden = false;
  }

  function render(embedding, projections, scores, attention) {
    currentEmbedding = embedding;
    currentProjections = projections;
    currentScores = scores;
    currentAttention = attention;
    selection = embedding.rows.length > 0 ? { row: 0, column: 0 } : null;

    renderFlow();
    renderInputMatrix(scaledHead, scaledBody, attention.scaled, 'scaled-attention-number', false);
    renderInputMatrix(weightsHead, weightsBody, attention.weights, 'scaled-attention-number', true);
    renderOutputMatrix();
    query('#scaled-attention-empty-note').textContent = '当前序列为空：scaled Shape [0, 0]、A Shape [0, 0]、O Shape [0, 2]。没有 Query 行或输出格可选择。';
    updateSelection();
  }

  root.addEventListener('click', (event) => {
    const target = event.target;
    const queryButton = target instanceof Element ? target.closest('button[data-phase6-query-row]') : null;
    const outputButton = target instanceof Element ? target.closest('button[data-phase6-output-cell]') : null;
    if (queryButton && root.contains(queryButton)) {
      selection = { row: Number(queryButton.dataset.phase6QueryRow), column: selection.column };
      updateSelection();
      return;
    }
    if (outputButton && root.contains(outputButton)) {
      selection = { row: Number(outputButton.dataset.row), column: Number(outputButton.dataset.column) };
      updateSelection();
    }
  });

  return { render };
}
