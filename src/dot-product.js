/**
 * Multiply corresponding vector components and accumulate their sum.
 *
 * @param {number[]} left
 * @param {number[]} right
 * @returns {{ dimension: number, leftShape: [number], rightShape: [number], outputShape: [], steps: Array<{ index: number, left: number, right: number, product: number, runningSum: number }>, result: number }}
 */
export function dotProductSteps(left, right) {
  if (!Array.isArray(left) || !Array.isArray(right)) {
    throw new TypeError('Dot product inputs must be arrays.');
  }

  if (left.length === 0 || right.length === 0) {
    throw new RangeError('Dot product vectors must not be empty.');
  }

  if (left.length !== right.length) {
    throw new RangeError('Dot product vectors must have the same dimension.');
  }

  for (let index = 0; index < left.length; index += 1) {
    if (!Object.hasOwn(left, index) || !Object.hasOwn(right, index)) {
      throw new TypeError('Dot product vectors must not contain missing components.');
    }
    if (typeof left[index] !== 'number' || !Number.isFinite(left[index])) {
      throw new TypeError('Dot product components must be finite numbers.');
    }
    if (typeof right[index] !== 'number' || !Number.isFinite(right[index])) {
      throw new TypeError('Dot product components must be finite numbers.');
    }
  }

  const steps = [];
  let runningSum = 0;

  for (let index = 0; index < left.length; index += 1) {
    const product = left[index] * right[index];
    if (!Number.isFinite(product)) {
      throw new RangeError('A dot product component overflowed.');
    }

    runningSum += product;
    if (!Number.isFinite(runningSum)) {
      throw new RangeError('The dot product sum overflowed.');
    }

    steps.push({
      index,
      left: left[index],
      right: right[index],
      product,
      runningSum,
    });
  }

  return {
    dimension: left.length,
    leftShape: [left.length],
    rightShape: [right.length],
    outputShape: [],
    steps,
    result: runningSum,
  };
}
