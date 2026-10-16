import assert from 'node:assert/strict';

export function wasmConcurrency(value = process.env.PAPERLEET_WASM_CONCURRENCY ?? '1') {
  assert(/^[12]$/.test(value), 'PAPERLEET_WASM_CONCURRENCY must be 1 or 2');
  return Number(value);
}

/** Each case still runs in its own process/VM; only independent cases overlap. */
export async function runWasmCases(cases, concurrency, run, settled = () => {}) {
  assert(Number.isInteger(concurrency) && concurrency >= 1 && concurrency <= 2, 'Use one or two WASM processes');
  assert(cases.length && cases.every(id => typeof id === 'string' && id.length) && new Set(cases).size === cases.length,
    'WASM cases must be nonempty, distinct module IDs');
  const results = new Array(cases.length);
  let cursor = 0;
  async function worker() {
    while (cursor < cases.length) {
      const index = cursor++, id = cases[index];
      let result;
      try {
        result = await run(id);
        assert(result?.id === id && typeof result.passed === 'boolean', 'WASM child returned a mismatched or incomplete result');
      } catch (error) {
        result = {id, passed:false, error:String(error?.stderr || error?.message || error).slice(-12000)};
      }
      results[index] = result;
      settled(result);
    }
  }
  await Promise.all(Array.from({length:Math.min(concurrency, cases.length)}, worker));
  assert.deepEqual(results.map(result => result.id), cases, 'Every WASM case must appear exactly once in catalog order');
  return results;
}
