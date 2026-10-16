import assert from 'node:assert/strict';
import {runWasmCases, wasmConcurrency} from './wasm-cases.mjs';

assert.equal(wasmConcurrency('1'), 1);
assert.equal(wasmConcurrency('2'), 2);
for (const value of ['', '0', '3', '1.5', '-1', 'Infinity', ' 2', '2x']) {
  assert.throws(() => wasmConcurrency(value), /must be 1 or 2/);
}
for (const [ids, count] of [[[], 1], [['a', 'a'], 2], [['a'], 0], [['a'], 3], [['a'], 1.5]]) {
  await assert.rejects(() => runWasmCases(ids, count, () => { throw Error('Must not start invalid work'); }));
}

// Hold actual case completions, rather than timing sleeps, to prove admission.
const pending = new Map(), started = [], completed = [];
let active = 0, peak = 0;
const work = runWasmCases(['a','b','c','d','e'], 2, id => {
  started.push(id); active++; peak = Math.max(peak, active);
  return new Promise((resolve, reject) => pending.set(id, (failed = false) => {
    active--;
    if (failed) reject(Error('fixture child failure'));
    else resolve({id, passed:true});
  }));
}, result => completed.push(result.id));
assert.deepEqual(started, ['a','b']);
pending.get('b')(); await Promise.resolve(); await Promise.resolve();
assert.deepEqual(started, ['a','b','c']);
pending.get('c')(true); await Promise.resolve(); await Promise.resolve();
assert.deepEqual(started, ['a','b','c','d'], 'A failed child must not hide remaining modules');
pending.get('a')(); await Promise.resolve(); await Promise.resolve();
assert.deepEqual(started, ['a','b','c','d','e']);
pending.get('e')(); pending.get('d')();
const results = await work;
assert.equal(peak, 2);
assert.equal(active, 0);
assert.deepEqual(results.map(row => row.id), ['a','b','c','d','e']);
assert.deepEqual(completed, ['b','c','a','e','d']);
assert.deepEqual(results.map(row => row.passed), [true,true,false,true,true]);
assert.match(results[2].error, /fixture child failure/);
assert.equal(new Set(started).size, 5);

const malformed = await runWasmCases(['wrong','partial','valid'], 1, async id =>
  id === 'wrong' ? {id:'different',passed:true} : id === 'partial' ? {id} : {id,passed:true});
assert.deepEqual(malformed.map(row => row.passed), [false,false,true]);
assert.equal((await runWasmCases(['single'], 2, async id => ({id,passed:true}))).length, 1);
console.log('PASS WASM case scheduling: bounded processes, ordered exact-once coverage, failure continuation, malformed child rejection and strict concurrency configuration');
