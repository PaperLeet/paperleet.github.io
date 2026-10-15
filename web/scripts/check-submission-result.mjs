import assert from 'node:assert/strict';
import React, { act } from 'react';
import { createRoot } from 'react-dom/client';
import { renderToStaticMarkup } from 'react-dom/server';
import { JSDOM } from 'jsdom';
import { createServer } from 'vite';

const server = await createServer({ server: { middlewareMode: true }, appType: 'custom', plugins: [{
  name: 'submission-result-test-account', enforce: 'pre',
  resolveId(source, importer) { if (source === './social' && importer?.endsWith('/SubmissionResultPage.tsx')) return '\0submission-result-test-account'; },
  load(id) { if (id === '\0submission-result-test-account') return 'export const useAccount = () => globalThis.__submissionResultTestAccount;'; },
}] });
try {
  const { parseSubmissionResult, loadSubmissionResult, submissionFailurePreview } = await server.ssrLoadModule('/src/submissionResultData.ts');
  const id = 'a'.repeat(32);
  const report = { problem: 'rl/dpo', source_sha256: 'b'.repeat(64), summary: { T0: [1, 1] }, tests: [{ id: 'test_loss', layer: 'T0', passed: true }] };
  const run = { id, problem: 'rl/dpo', created_at: '2026-10-15T12:00:00+00:00', accepted: true, source_sha256: 'b'.repeat(64), code: 'private submitted code', report };
  const payload = { user_id: 'alice', run };
  const { submissionDestination } = await server.ssrLoadModule('/src/submissionNavigation.ts');
  for (const completed of [report,
    { ...report, summary: { T0: [0, 1] }, tests: [{ ...report.tests[0], passed: false }] },
    { ...report, error: 'Judge interrupted' },
  ]) {
    assert.deepEqual(submissionDestination({ ...completed, run_id: id }, 'user:alice', { id: report.problem, number: 3 }),
      { href: `#/submission/${id}` },
      'Accepted, failed, and incomplete attempts expose an explicit saved-result link without automatic navigation');
  }
  assert.equal(submissionDestination({ ...report, run_id: '../another-account' }, 'user:alice', { id: report.problem, number: 3 }), undefined);
  assert.equal(parseSubmissionResult(payload, 'alice', id).code, run.code);
  assert.throws(() => parseSubmissionResult(payload, 'bob', id), /owner/);
  assert.throws(() => parseSubmissionResult(payload, 'alice', 'c'.repeat(32)), /URL/);
  assert.throws(() => parseSubmissionResult({ ...payload, run: { ...run, source_sha256: 'c'.repeat(64) } }, 'alice', id), /source/);
  assert.throws(() => parseSubmissionResult({ ...payload, run: { ...run, report: { ...report, error: 'incomplete' } } }, 'alice', id), /accepted/);
  const partial = parseSubmissionResult({ ...payload, run: { ...run, accepted: false } }, 'alice', id);
  assert.equal(partial.accepted, false, 'All tests in a partial attempt cannot become an accepted full submission');

  const states = [];
  const reader = loadSubmissionResult('alice', id, async (path) => {
    assert.equal(path, `/api/practice/runs/${id}`);
    return Response.json(payload);
  }, state => states.push(state));
  await reader.done;
  assert.deepEqual(states.map(state => state.status), ['loading', 'ready']);

  for (const [status, expected] of [[401, 'auth'], [403, 'auth'], [404, 'missing'], [503, 'error']]) {
    const events = [];
    const attempt = loadSubmissionResult('alice', id, async () => new Response('', { status }), state => events.push(state));
    await attempt.done;
    assert.equal(events.at(-1).status, expected);
    assert.equal(events.some(state => state.run), false, 'An unavailable response must not reveal a previous record');
  }
  const identityStates = [];
  await loadSubmissionResult('bob', id, async () => Response.json(payload), state => identityStates.push(state)).done;
  assert.equal(identityStates.at(-1).status, 'identity');

  let reply, signal;
  const obsolete = [];
  const oldRequest = loadSubmissionResult('alice', id, (_, currentSignal) => {
    signal = currentSignal;
    return new Promise(resolve => { reply = resolve; });
  }, state => obsolete.push(state));
  oldRequest.dispose();
  assert.equal(signal.aborted, true);
  reply(Response.json(payload));
  await oldRequest.done;
  assert.deepEqual(obsolete.map(state => state.status), ['loading'], 'Late results from another account or URL must never publish');

  let calls = 0;
  const invalid = [];
  await loadSubmissionResult('alice', '../runs', async () => { calls++; return Response.json(payload); }, state => invalid.push(state)).done;
  assert.equal(calls, 0);
  assert.equal(invalid.at(-1).status, 'missing');
  for (const stalledBody of [false, true]) {
    const originalSet = globalThis.setTimeout, originalClear = globalThis.clearTimeout;
    let fireDeadline, deadlineCleared = false, timeoutSignal;
    const timeoutStates = [];
    try {
      globalThis.setTimeout = (callback, delay) => {
        assert.equal(delay, 12_000, 'A stalled result has a bounded 12-second deadline');
        fireDeadline = callback;
        return 491;
      };
      globalThis.clearTimeout = timer => { if (timer === 491) deadlineCleared = true; };
      const stalled = loadSubmissionResult('alice', id, (_, signal) => {
        timeoutSignal = signal;
        return stalledBody ? Promise.resolve({ ok: true, status: 200, json: () => new Promise(() => {}) }) : new Promise(() => {});
      }, state => timeoutStates.push(state));
      await Promise.resolve();
      await Promise.resolve();
      fireDeadline();
      await stalled.done;
      assert.equal(timeoutSignal.aborted, true);
      assert.equal(deadlineCleared, true);
      assert.deepEqual(timeoutStates.map(state => state.status), ['loading', 'error'],
        'A stalled connection or response body must show retry instead of a permanent spinner');
    } finally {
      globalThis.setTimeout = originalSet;
      globalThis.clearTimeout = originalClear;
    }
  }
  assert.equal(submissionFailurePreview('a\nb\nc\nd\ne\nf\ng\nh'), 'b\nc\nd\ne\nf\ng\nh');
  assert.equal(submissionFailurePreview('x'.repeat(1300)).length, 1201);
  const { default: ReportView } = await server.ssrLoadModule('/src/ReportView.tsx');
  const historical = renderToStaticMarkup(React.createElement(ReportView, {
    report, problem: { id: 'rl/dpo', tests: { T0: 99 } }, sourceState: 'current', source: run.code, accepted: true,
  }));
  assert(historical.includes('Accepted') && !historical.includes('Selected tests passed'),
    'An accepted historical attempt keeps its recorded verdict when the catalog gains tests');
  const partialReport = renderToStaticMarkup(React.createElement(ReportView, {
    report, problem: { id: 'rl/dpo', tests: { T0: 1 } }, sourceState: 'current', source: run.code, accepted: false,
  }));
  assert(partialReport.includes('Selected tests passed') && !partialReport.includes('Accepted'),
    'A historical partial run cannot become accepted when the current test requirements change');

  const dom = new JSDOM('<div id="app"></div>', { url: 'https://paperleet.test' });
  const originals = new Map();
  const requests = [];
  const globals = {
    window: dom.window, document: dom.window.document, navigator: dom.window.navigator,
    localStorage: dom.window.localStorage, IS_REACT_ACT_ENVIRONMENT: true,
    __submissionResultTestAccount: { user: { id: 'alice' }, loading: false, refresh: async () => {} },
    fetch: (path, options) => new Promise(resolve => requests.push({ path, options, resolve })),
  };
  let root;
  try {
    for (const [key, value] of Object.entries(globals)) {
      originals.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
      Object.defineProperty(globalThis, key, { configurable: true, value, writable: true });
    }
    const { default: Page } = await server.ssrLoadModule('/src/SubmissionResultPage.tsx');
    const problems = [{ id: 'rl/dpo', number: 3, title: 'DPO', module: { optional: false }, tests: { T0: 1 } },
      { id: 'rl/grpo', number: 4, title: 'GRPO', module: { optional: false } }];
    root = createRoot(document.getElementById('app'));
    const draw = (runId = id) => React.createElement(Page, { runId, problems });
    await act(async () => root.render(draw()));
    assert(document.body.textContent.includes('Loading your result'));
    assert.equal(requests.length, 1);
    await act(async () => requests[0].resolve(Response.json(payload)));
    assert(document.body.textContent.includes('Accepted'));
    assert.equal(document.activeElement?.id, 'submission-result-title', 'A completed result is announced via the focused heading');
    assert(document.body.textContent.includes('Saved to your account'));
    assert(!document.body.textContent.includes(run.code), 'The result landing page does not expose submitted source');
    assert.equal([...document.querySelectorAll('a')].find(link => link.textContent === 'Next problem')?.getAttribute('href'), '#/4');

    const secondId = 'c'.repeat(32);
    await act(async () => root.render(draw(secondId)));
    assert(!document.body.textContent.includes('Accepted'), 'A new result URL must hide the previous attempt while loading');
    globalThis.__submissionResultTestAccount = { ...globals.__submissionResultTestAccount, user: { id: 'bob' } };
    await act(async () => root.render(draw(secondId)));
    assert(requests[1].options.signal.aborted, 'An account change aborts the old result request');
    await act(async () => requests[1].resolve(Response.json({ ...payload, run: { ...run, id: secondId } })));
    assert(!document.body.textContent.includes('Accepted'), 'A late old-account response cannot render a private result');
    await act(async () => requests[2].resolve(Response.json({ ...payload, run: { ...run, id: secondId } })));
    assert(document.body.textContent.includes('Your account has changed'));

    globalThis.__submissionResultTestAccount = { ...globals.__submissionResultTestAccount, user: null };
    await act(async () => root.render(draw()));
    assert(document.body.textContent.includes('Sign in to view your result'));
    assert.equal(requests.length, 3, 'A signed-out result page must not request private history');
    globalThis.__submissionResultTestAccount = globals.__submissionResultTestAccount;
    await act(async () => root.render(draw()));
    await act(async () => requests[3].resolve(new Response('', { status: 503 })));
    assert(document.body.textContent.includes('Your result could not load'));
    await act(async () => [...document.querySelectorAll('button')].find(button => button.textContent === 'Retry').click());
    assert.equal(requests.length, 5);
    await act(async () => requests[4].resolve(Response.json({ ...payload, run: { ...run, accepted: false } })));
    assert(document.body.textContent.includes('Not accepted') && document.body.textContent.includes('did not cover every required test'),
      'A partial passing attempt is explained without implying acceptance or failed assertions');

    // Keeping Submit inside the workspace must not remove the deliberate path
    // from a history entry to its immutable saved result.
    const { default: History } = await server.ssrLoadModule('/src/SolutionHistory.tsx');
    dom.window.history.replaceState(null, '', '#/3?from=collections/llm');
    const practiceRoute = dom.window.location.href;
    const historyRequest = requests.length;
    await act(async () => root.render(React.createElement(History, { owner: 'user:alice', problem: problems[0], revision: 0 })));
    assert.equal(requests.length, historyRequest + 1);
    assert(requests[historyRequest].path.startsWith('/api/practice/runs?'));
    await act(async () => requests[historyRequest].resolve(Response.json({ user_id: 'alice', runs: [run] })));
    assert.equal(dom.window.location.href, practiceRoute, 'Loading completed submissions keeps the current practice route');
    const savedResultLink = document.querySelector('.solution-history-source-heading a');
    assert.equal(savedResultLink?.textContent, 'View result');
    assert.equal(savedResultLink?.getAttribute('href'), `#/submission/${id}`);
    const navigated = new Promise(resolve => dom.window.addEventListener('hashchange', resolve, { once: true }));
    await act(async () => { savedResultLink.click(); await navigated; });
    assert.equal(dom.window.location.hash, `#/submission/${id}`, 'Only explicit history activation opens the saved result page');
    const explicitRequest = requests.length;
    await act(async () => root.render(draw()));
    await act(async () => requests[explicitRequest].resolve(Response.json(payload)));
    assert(document.body.textContent.includes('Accepted'), 'Existing direct result links still render their saved verdict');
    assert.equal([...document.querySelectorAll('a')].find(link => link.textContent === 'Back to code')?.getAttribute('href'), '#/3');
    assert.equal(requests.every(request => !request.options.method || request.options.method === 'GET'), true, 'Viewing and retrying results must never submit code');
  } finally {
    if (root) await act(async () => root.unmount());
    for (const [key, descriptor] of originals) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else delete globalThis[key];
    }
    dom.window.close();
  }
  console.log('PASS explicit submission links without redirects, history navigation, immutable result, authoritative verdict, private account/URL isolation, cancellation, retry, focused result and partial-suite UI');
} finally {
  await server.close();
}
