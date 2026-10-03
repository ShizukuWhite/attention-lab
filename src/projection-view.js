import { matrixCellSteps } from './matrix-multiply.js';
import { PROJECTION_WEIGHTS } from './projection.js';

const OUTPUTS = ['Q', 'K', 'V'];
const WEIGHT_FOR_OUTPUT = { Q: 'Wq', K: 'Wk', V: 'Wv' };

function formatNumber(value) {
  return String(Number(value.toFixed(6)));
}

function formatFactor(value) {
  const formatted = formatNumber(value);
  return value < 0 ? `(${formatted})` : formatted;
}

function sourceLabel(source) {
  return source === 'manual' ? '人工设定' : '未知 · 零向量占位';
}

/**
 * Create the DOM view for Q/K/V projections.
 *
 * @param {Element} root
 * @returns {{ render(embedding: { matrix: number[][], shape: [number, number], rows: Array<{ text: string, index: number, source: 'manual' | 'unknown' }> }, projections: { Q: { matrix: number[][], shape: [number, number] }, K: { matrix: number[][], shape: [number, number] }, V: { matrix: number[][], shape: [number, number] } }): void }}
 */
export function createProjectionView(root) {
  const document = root.ownerDocument;
  const query = (selector) => root.querySelector(selector);
  const inputMatrixBody = root.closest('main')?.querySelector('#embedding-matrix-body')
    ?? document.querySelector('#embedding-matrix-body');
  const weightBodies = {
    Wq: query('#projection-wq-body'),
    Wk: query('#projection-wk-body'),
    Wv: query('#projection-wv-body'),
  };
  const outputBodies = {
    Q: query('#projection-q-body'),
    K: query('#projection-k-body'),
    V: query('#projection-v-body'),
  };
  const outputShapeLabels = {
    Q: query('#projection-q-shape'),
    K: query('#projection-k-shape'),
    V: query('#projection-v-shape'),
  };

  let currentEmbedding = null;
  let projected = null;
  let selection = null;

  function makeElement(tagName, className, text) {
    const element = document.createElement(tagName);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
  }

  function renderWeights() {
    for (const [key, weight] of Object.entries(PROJECTION_WEIGHTS)) {
      const body = weightBodies[key];
      const fragment = document.createDocumentFragment();

      for (let rowIndex = 0; rowIndex < weight.shape[0]; rowIndex += 1) {
        const row = makeElement('tr');
        const index = makeElement('th', 'projection-weight-index', String(rowIndex));
        index.scope = 'row';
        row.append(index);

        for (let columnIndex = 0; columnIndex < weight.shape[1]; columnIndex += 1) {
          const cell = makeElement('td', 'projection-weight-cell', formatNumber(weight.matrix[rowIndex][columnIndex]));
          cell.dataset.projectionWeight = key;
          cell.dataset.weightColumn = String(columnIndex);
          row.append(cell);
        }

        fragment.append(row);
      }

      body.replaceChildren(fragment);
    }
  }

  function makeOutputCell(output, rowIndex, columnIndex, value, token) {
    const cell = makeElement('td', 'projection-output-cell');
    const button = makeElement('button', 'projection-cell-button');
    const number = makeElement('span', 'projection-cell-number', formatNumber(value));
    const marker = makeElement('span', 'projection-cell-marker', '当前选中');
    marker.hidden = true;
    button.type = 'button';
    button.dataset.projectionCell = '';
    button.dataset.projection = output;
    button.dataset.row = String(rowIndex);
    button.dataset.column = String(columnIndex);
    button.setAttribute('aria-pressed', 'false');
    button.setAttribute(
      'aria-label',
      `${output}[${rowIndex}, ${columnIndex}]，位置 ${token.index}，Token ${token.text}，数值 ${formatNumber(value)}`,
    );
    button.append(number, marker);
    cell.append(button);
    return cell;
  }

  function renderOutput(output) {
    const body = outputBodies[output];
    const fragment = document.createDocumentFragment();
    const matrix = projected[output];
    outputShapeLabels[output].textContent = `[${matrix.shape.join(', ')}]`;

    for (let rowIndex = 0; rowIndex < matrix.shape[0]; rowIndex += 1) {
      const token = currentEmbedding.rows[rowIndex];
      const row = makeElement('tr');

      const position = makeElement('td', 'matrix-index', String(token.index));
      const tokenCell = makeElement('td', 'matrix-token');
      const tokenText = makeElement('span', 'matrix-token-text', token.text);
      tokenCell.append(tokenText);
      row.append(position, tokenCell);

      for (let columnIndex = 0; columnIndex < matrix.shape[1]; columnIndex += 1) {
        row.append(makeOutputCell(output, rowIndex, columnIndex, matrix.matrix[rowIndex][columnIndex], token));
      }

      const source = makeElement(
        'td',
        token.source === 'manual' ? 'matrix-source' : 'matrix-source matrix-source--unknown',
        sourceLabel(token.source),
      );
      row.append(source);
      fragment.append(row);
    }

    if (matrix.shape[0] === 0) {
      const row = makeElement('tr');
      const cell = makeElement('td', 'matrix-empty', '当前没有矩阵行；空序列的输出 Shape 仍保留为 [0, 2]。');
      cell.colSpan = 5;
      row.append(cell);
      fragment.append(row);
    }

    body.replaceChildren(fragment);
  }

  function renderOutputs() {
    for (const output of OUTPUTS) renderOutput(output);
  }

  function clearHighlights() {
    for (const row of inputMatrixBody.querySelectorAll('.projection-input-row--trace')) {
      row.classList.remove('projection-input-row--trace');
    }
    for (const cell of root.querySelectorAll('.projection-weight-cell--trace')) {
      cell.classList.remove('projection-weight-cell--trace');
    }
  }

  function updateOutputButtons() {
    for (const button of root.querySelectorAll('button[data-projection-cell]')) {
      const isSelected = selection !== null
        && button.dataset.projection === selection.output
        && Number(button.dataset.row) === selection.row
        && Number(button.dataset.column) === selection.column;
      button.setAttribute('aria-pressed', String(isSelected));
      button.classList.toggle('projection-cell-button--selected', isSelected);
      button.querySelector('.projection-cell-marker').hidden = !isSelected;
    }
  }

  function renderTrace() {
    const trace = query('#projection-trace');
    const emptyNote = query('#projection-empty-note');
    const unknownNote = query('#projection-unknown-note');

    if (!selection || !currentEmbedding || !projected) {
      trace.hidden = true;
      emptyNote.hidden = !currentEmbedding || currentEmbedding.rows.length !== 0;
      clearHighlights();
      return;
    }

    const output = selection.output;
    const weightKey = WEIGHT_FOR_OUTPUT[output];
    const weight = PROJECTION_WEIGHTS[weightKey];
    const token = currentEmbedding.rows[selection.row];
    const calculation = matrixCellSteps(
      { matrix: currentEmbedding.matrix, shape: currentEmbedding.shape },
      weight,
      selection.row,
      selection.column,
    );

    trace.hidden = false;
    emptyNote.hidden = true;
    unknownNote.hidden = !currentEmbedding.rows.some((row) => row.source === 'unknown');
    query('#projection-x-shape').textContent = `[${currentEmbedding.shape.join(', ')}]`;
    query('#projection-output-shape').textContent = `[${projected.Q.shape.join(', ')}]`;
    query('#projection-output-shape-summary').textContent = `[${projected.Q.shape.join(', ')}]`;
    query('#projection-trace-cell').textContent = `${output}[${selection.row}, ${selection.column}]`;
    query('#projection-trace-token').textContent = `位置 ${token.index} · ${token.text}`;
    query('#projection-trace-route').textContent = `X 第 ${selection.row} 行 × ${weightKey} 第 ${selection.column} 列`;
    query('#projection-input-row').textContent = `[${calculation.leftRow.map(formatNumber).join(', ')}]`;
    query('#projection-input-row-shape').textContent = `[${calculation.leftShape.join(', ')}]`;
    query('#projection-input-source').textContent = sourceLabel(token.source);
    query('#projection-weight-column').textContent = `[${calculation.rightColumn.map(formatNumber).join(', ')}]`;
    query('#projection-weight-column-shape').textContent = `[${calculation.rightShape.join(', ')}]`;
    query('#projection-result').textContent = formatNumber(calculation.result);
    query('#projection-result-shape').textContent = `[${calculation.outputShape.join(', ')}]`;

    const stepsBody = query('#projection-steps');
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
    query('#projection-equation').textContent = `${output}[${selection.row}, ${selection.column}] = ${terms.join(' + ')} = ${formatNumber(calculation.result)}`;

    clearHighlights();
    const inputRows = inputMatrixBody.querySelectorAll('tr');
    inputRows[selection.row]?.classList.add('projection-input-row--trace');
    for (const cell of root.querySelectorAll(`.projection-weight-cell[data-projection-weight="${weightKey}"]`)) {
      if (Number(cell.dataset.weightColumn) === selection.column) {
        cell.classList.add('projection-weight-cell--trace');
      }
    }
  }

  function updateSelection() {
    updateOutputButtons();
    renderTrace();
  }

  function render(embedding, projections) {
    currentEmbedding = embedding;
    projected = projections;
    selection = embedding.rows.length > 0
      ? { output: 'Q', row: 0, column: 0 }
      : null;

    query('#projection-x-shape').textContent = `[${embedding.shape.join(', ')}]`;
    query('#projection-output-shape').textContent = `[${projected.Q.shape.join(', ')}]`;
    query('#projection-output-shape-summary').textContent = `[${projected.Q.shape.join(', ')}]`;
    renderOutputs();
    updateSelection();
    query('#projection-unknown-note').hidden = !embedding.rows.some((row) => row.source === 'unknown');
    if (embedding.rows.some((row) => row.source === 'unknown')) {
      query('#projection-unknown-note').textContent = '未知 Token 的 X 行使用 [0, 0, 0] 零向量占位，因此对应的 Q、K、V 行也是 [0, 0]。这些零值来自教学占位规则，不是模型判断或词义结论。';
    }
    if (embedding.rows.length === 0) {
      query('#projection-empty-note').textContent = '当前序列为空：X 的 Shape 为 [0, 3]，Q、K、V 的 Shape 都为 [0, 2]。权重矩阵仍保留 [3, 2]，但没有可选择的输出格。';
    }
  }

  renderWeights();
  root.addEventListener('click', (event) => {
    const target = event.target;
    const button = target instanceof Element ? target.closest('button[data-projection-cell]') : null;
    if (!button || !root.contains(button)) return;

    selection = {
      output: button.dataset.projection,
      row: Number(button.dataset.row),
      column: Number(button.dataset.column),
    };
    updateSelection();
  });

  return { render };
}
