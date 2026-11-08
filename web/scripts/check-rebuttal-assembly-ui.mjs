import assert from 'node:assert/strict';
import React, { act } from 'react';
import { JSDOM } from 'jsdom';
import { createServer } from 'vite';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';

const hostedSupported = (await readFile('src/RebuttalStudio.tsx', 'utf8')).includes('HostedStudioSession');
const dom = new JSDOM('<!doctype html><div id="root"></div>', { url: 'https://paperleet.test/rebuttal/?paper=paper_a', pretendToBeVisual: true });
const context = { locale: 'en', account: { user: { id: 'account_a' }, loading: false, capabilities: { rebuttal_assembly_drafts: true, rebuttal_reviewer_versions: true } }, request: null };
const globals = { window: dom.window, document: dom.window.document, navigator: dom.window.navigator, location: dom.window.location, history: dom.window.history,
  localStorage: dom.window.localStorage, sessionStorage: dom.window.sessionStorage, Element: dom.window.Element, HTMLElement: dom.window.HTMLElement,
  HTMLInputElement: dom.window.HTMLInputElement, HTMLTextAreaElement: dom.window.HTMLTextAreaElement, HTMLButtonElement: dom.window.HTMLButtonElement,
  IS_REACT_ACT_ENVIRONMENT: true, __assemblyTest: context,
  fetch: async path => { assert(String(path).endsWith('/api/rebuttal/status')); return new Response(JSON.stringify({ user_id: context.account.user?.id, configured: false, model: null, provider: null, source: null, available: true })); } };
const prior = new Map(Object.keys(globals).map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
for (const [key, value] of Object.entries(globals)) Object.defineProperty(globalThis, key, { value, configurable: true, writable: true });
let copied = '', downloaded;
Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText: async text => { copied = text; } } });
const createURL = URL.createObjectURL, revokeURL = URL.revokeObjectURL;
URL.createObjectURL = blob => { downloaded = blob; return 'blob:test-assembly'; }; URL.revokeObjectURL = () => {};
window.HTMLAnchorElement.prototype.click = function () {};
const mocks = {
  './social': 'export const useAccount=()=>globalThis.__assemblyTest.account;export const socialApi=(...args)=>globalThis.__assemblyTest.request(...args);export const errorMessage=error=>error.message;',
  './i18n': 'export const useI18n=()=>({locale:globalThis.__assemblyTest.locale,t:(zh,en)=>globalThis.__assemblyTest.locale==="zh"?zh:en});',
  './apiClient': 'export const apiFetch=(...args)=>fetch(...args);',
};
const cacheDir = await mkdtemp(`${tmpdir()}/paperleet-assembly-ui-`);
const server = await createServer({ configFile: false, cacheDir, esbuild: { jsx: 'automatic' }, server: { middlewareMode: true, watch: null, hmr: false, ws: false }, appType: 'custom', optimizeDeps: { noDiscovery: true, include: [] }, plugins: [{ name: 'assembly-test-boundaries', enforce: 'pre', resolveId(source) { if (mocks[source]) return '\0assembly:' + source; }, load(id) { return id.startsWith('\0assembly:') ? mocks[id.slice(10)] : undefined; } }] });
let root;
try {
  const { createRoot } = await import('react-dom/client');
  const { default: Studio } = await server.ssrLoadModule('/src/RebuttalStudio.tsx');
  const { newProject, WORKSPACE_KEY, countWords } = await server.ssrLoadModule('/src/rebuttalModel.ts');
  const { toOpenReviewMarkdown } = await server.ssrLoadModule('/src/rebuttalFormatting.ts');
  const project = { ...newProject(), id: 'paper_a', title: 'Assembly paper', opening: 'Original shared opening', reviews: [{ id: 'review_a', name: 'Reviewer A', text: 'Review' }, { id: 'all', name: 'Reviewer with all ID', text: 'Review' }] };
  localStorage.setItem(WORKSPACE_KEY, JSON.stringify({ version: 1, activeProjectId: project.id, projects: [project] }));
  const documents = new Map(), requests = [], shared = [];
  context.request = async (path, body) => {
    const scope = decodeURIComponent(path.split('/').at(-1)), user = context.account.user.id, key = `${user}:${scope}`;
    assert(path.includes('/papers/paper_a/drafts/'));
    const current = documents.get(key) || { user_id: user, paper_id: project.id, scope, source: null, revision: 0, updated_at: null };
    if (!body) return current;
    assert.equal(body.expected_user_id, user); assert.equal(body.expected_revision, current.revision);
    requests.push({ path, body: structuredClone(body) });
    const result = { ...current, source: body.source, revision: current.revision + 1, updated_at: '2030-01-01T00:00:00Z' }; documents.set(key, result); return result;
  };
  let bridge;
  const session = { project, role: 'owner', readOnly: false, saveStatus: 'saved', onChange: value => shared.push(value), onBack() {}, onShare() {}, onRefresh() {}, onPersonalDraft: value => { bridge = value; } };
  root = createRoot(document.querySelector('#root'));
  const render = async (key, overrides = {}, hosted = hostedSupported) => act(async () => root.render(React.createElement(React.StrictMode, null, React.createElement(Studio, { key, ...(hosted ? { hosted: { ...session, ...overrides } } : {}) }))));
  const button = text => [...document.querySelectorAll('button')].find(node => node.textContent.trim() === text || node.getAttribute('aria-label') === text);
  const click = async text => { const node = button(text); assert(node, `Missing ${text}`); assert.equal(node.disabled, false, `${text} is disabled`); await act(async () => node.click()); };
  const source = () => document.querySelector('textarea[aria-label="OpenReview Markdown"]');
  const change = async (node, value) => act(async () => { const prototype = node instanceof window.HTMLTextAreaElement ? window.HTMLTextAreaElement.prototype : window.HTMLSelectElement.prototype; Object.getOwnPropertyDescriptor(prototype, 'value').set.call(node, value); node.dispatchEvent(new window.Event(node.tagName === 'SELECT' ? 'change' : 'input', { bubbles: true })); });
  const flush = async () => { if (hostedSupported) await act(async () => { assert(bridge); assert.equal(await bridge.flush(), true); }); };
  await render('first'); await click('Assemble');
  assert(source()); assert.equal(source().readOnly, false, 'StrictMode replay finishes loading and enables the personal editor');
  assert.equal(source().value, 'Original shared opening'); assert.equal(requests.length, 0, 'Auto assembly performs no personal save');
  await change(document.querySelector('.rst-export aside textarea'), 'Updated shared opening'); assert.equal(source().value, 'Updated shared opening');
  const sharedBeforeEdit = shared.length, personal = 'My private answer with $\\alpha$ and `literal \\ code`.';
  await change(source(), personal); await flush(); assert.equal(shared.length, sharedBeforeEdit, 'Autosave never modifies the shared paper');
  assert.equal(document.querySelector('.rst-export-meter strong').firstChild.textContent, String(countWords(personal)));
  await click('Preview'); assert.match(document.querySelector('.rbe-rendered').textContent, /My private answer/); await click('OpenReview source');
  await click('Copy for OpenReview'); assert.equal(copied, toOpenReviewMarkdown(personal));
  await click('.md'); assert.equal(await downloaded.text(), toOpenReviewMarkdown(personal));
  await click(hostedSupported ? 'Save shared version' : 'Save current version');
  const version = hostedSupported ? shared.at(-1).snapshots.at(-1) : JSON.parse(localStorage.getItem(WORKSPACE_KEY)).projects[0].snapshots.at(-1);
  assert.equal(version.markdown, toOpenReviewMarkdown(personal));
  await change(document.querySelector('.rst-export aside textarea'), 'Newer shared content'); assert.equal(source().value, personal, 'Later shared edits cannot overwrite personal edits');
  await change(document.querySelector('.rst-export aside select'), 'reviewer-all'); assert.equal(source().value, 'Newer shared content');
  await change(source(), 'Reviewer-only rewrite'); await flush();
  await change(document.querySelector('.rst-export aside select'), 'all'); assert.equal(source().value, personal);
  let confirmations = 0; window.confirm = () => { confirmations++; return false; };
  await click('Reassemble'); assert.equal(source().value, personal);
  window.confirm = () => { confirmations++; return true; }; await click('Reassemble'); await flush(); assert.equal(source().value, 'Newer shared content'); assert.equal(confirmations, 2);
  await change(source(), ''); await flush(); await render('reload'); await click('Assemble'); assert.equal(source().value, '', 'Intentional blank survives remount');

  if (hostedSupported) {
    const normalDraftRequest = context.request;
    // Before the first personal read succeeds, generated text must not masquerade as the user's draft.
    context.account.user = { id: 'slow_reader' };
    let resolveFirstRead;
    const firstRead = new Promise(resolve => { resolveFirstRead = resolve; });
    context.request = (path, body) => body ? normalDraftRequest(path, body) : firstRead;
    await render('slow-first-read'); await click('Assemble');
    assert.equal(source().value, ''); assert.equal(source().readOnly, true);
    assert.equal(button('Copy for OpenReview').disabled, true); assert.equal(button('.md').disabled, true);
    assert.equal(document.querySelector('.rst-export-meter strong').firstChild.textContent, '0');
    await act(async () => resolveFirstRead({ user_id: 'slow_reader', paper_id: project.id, scope: 'all', source: 'Previously saved personal answer', revision: 1, updated_at: '2030-01-01T00:00:00Z' }));
    assert.equal(source().value, 'Previously saved personal answer'); assert.equal(button('Copy for OpenReview').disabled, false); assert.equal(button('.md').disabled, false);

    context.account.user = { id: 'failed_reader' }; context.request = async () => { throw Error('Initial read failed'); };
    await render('failed-first-read'); await click('Assemble');
    assert.equal(source().value, ''); assert.equal(source().readOnly, true);
    assert.equal(button('Copy for OpenReview').disabled, true); assert.equal(button('.md').disabled, true);
    assert.match(document.querySelector('.rst-assembly-error').textContent, /Could not load your personal draft/);

    // After a successful read, a failed save still allows recovery of the actual edited text.
    context.account.user = { id: 'failed_writer' }; context.request = normalDraftRequest;
    await render('failed-save'); await click('Assemble'); await change(source(), 'Unsaved personal recovery answer');
    context.request = async (path, body) => { if (body) throw Error('Save unavailable'); return normalDraftRequest(path, body); };
    await act(async () => { assert.equal(await bridge.flush(), false); });
    assert.equal(source().value, 'Unsaved personal recovery answer'); assert.equal(button('Copy for OpenReview').disabled, false); assert.equal(button('.md').disabled, false);
    await click('Copy for OpenReview'); assert.equal(copied, 'Unsaved personal recovery answer'); await click('.md'); assert.equal(await downloaded.text(), copied);
    context.request = normalDraftRequest;

    // Completing the final concern must open the canonical reviewer draft scope, not its raw ID.
    context.account.user = { id: 'completion_owner' };
    const scopeReads = [];
    context.request = (path, body) => { if (!body) scopeReads.push(path.split('/').at(-1)); return normalDraftRequest(path, body); };
    const concern = { id: 'concern_a', reviewerId: 'review_a', title: 'Concern A', quote: 'Clarify this point', category: 'clarification', priority: 'medium', status: 'drafting', suggestion: '', outline: '', response: 'Completed reviewer answer' };
    await render('complete-reviewer', { project: { ...project, issues: [concern] } }); await click('Responses'); await click('Complete & continue');
    assert.equal(document.querySelector('.rst-export aside select').value, 'reviewer-review_a');
    assert.equal(scopeReads.at(-1), 'reviewer-review_a'); assert.match(source().value, /Completed reviewer answer/); assert.equal(source().readOnly, false);
    context.request = normalDraftRequest;

    const beforeViewerShared = shared.length; context.account.user = { id: 'viewer_b' };
    await render('viewer_b', { role: 'viewer', readOnly: true }); await click('Assemble');
    assert.equal(source().readOnly, false); assert.equal(button('Save shared version').disabled, true); assert.equal(document.querySelector('.rst-export aside textarea').disabled, true);
    await change(source(), 'Viewer private source'); await flush(); assert.equal(shared.length, beforeViewerShared); assert.equal(documents.get('viewer_b:all').source, 'Viewer private source');
    assert.equal(documents.get('account_a:all').source, '');
    context.account.capabilities = {}; await render('old-api', { role: 'viewer', readOnly: true }); await click('Assemble'); assert.equal(source().readOnly, true); assert.match(document.body.textContent, /Personal drafts are updating/);
    assert.equal(source().value, 'Original shared opening'); assert.equal(button('Copy for OpenReview').disabled, false); assert.equal(button('.md').disabled, false, 'Old backends retain their generated export without personal drafts');
    context.account.capabilities = { rebuttal_assembly_drafts: true, rebuttal_reviewer_versions: true };

    // The actual hosted wrapper must settle both queues before following the API settings link.
    const { default: HostedHub } = await server.ssrLoadModule('/src/HostedRebuttal.tsx');
    const personalRequest = context.request, events = [];
    const owner = { id: 'navigation_owner', username: 'navigation_owner', display_name: 'Navigation owner' };
    context.account.user = owner;
    let serverProject = structuredClone(project), serverRevision = 1, sharedConflict = false, personalFailure = false;
    context.request = async (path, body) => {
      if (path.includes('/drafts/')) {
        if (body) { events.push('personal-save'); if (personalFailure) throw Error('Offline personal save'); }
        return personalRequest(path, body);
      }
      assert.equal(path, 'rebuttal/papers/paper_a');
      if (body) {
        events.push('shared-save');
        if (sharedConflict) throw Object.assign(Error('revision_conflict'), { status: 409, code: 'revision_conflict' });
        assert.equal(body.expected_revision, serverRevision); serverProject = structuredClone(body.project); serverRevision++;
      } else events.push('shared-read');
      return { project: serverProject, revision: serverRevision, role: 'owner', owner, members: [{ user: owner, role: 'owner' }] };
    };
    await act(async () => root.render(React.createElement(HostedHub, { route: 'rebuttal/paper_a' })));
    for (let tick = 0; !button('Assemble') && tick < 30; tick++) await act(async () => { await new Promise(resolve => setTimeout(resolve, 0)); });
    const setApi = async () => {
      const link = document.querySelector('.personal-ai-connection a');
      assert.equal(link?.textContent.trim(), 'Set API'); assert.equal(link.getAttribute('href'), '#/account/ai');
      await act(async () => link.dispatchEvent(new window.MouseEvent('click', { bubbles: true, cancelable: true, button: 0 })));
      for (let tick = 0; tick < 4; tick++) await act(async () => new Promise(resolve => setTimeout(resolve, 0)));
    };
    assert.equal(button('Refresh'), undefined);
    await click('Assemble'); await change(document.querySelector('.rst-export aside textarea'), 'Shared navigation edit'); await change(source(), 'Personal navigation edit');
    const start = events.length; await setApi();
    assert.deepEqual(events.slice(start), ['shared-save', 'personal-save'], 'Opening API settings saves both queues without reloading the shared document');
    assert.equal(location.hash, '#/account/ai');
    assert.equal(serverProject.opening, 'Shared navigation edit');
    assert.equal(documents.get('navigation_owner:all').source, 'Personal navigation edit');
    history.replaceState(null, '', '#/rebuttal/paper_a');
    sharedConflict = true; personalFailure = true;
    await change(document.querySelector('.rst-export aside textarea'), 'Conflicting shared edit'); await change(source(), 'Personal recovery through reload'); await setApi();
    assert.equal(location.hash, '#/rebuttal/paper_a', 'Failed saves prevent navigation away from the current paper');
    assert(button('Load server version')); const beforeReload = events.filter(item => item === 'shared-read').length;
    await click('Load server version'); assert.equal(events.filter(item => item === 'shared-read').length, beforeReload, 'Failed personal save prevents the shared reload from retiring its queue');
    assert.equal(source().value, 'Personal recovery through reload');
    personalFailure = false; await click('Load server version'); await click('Assemble'); assert.equal(source().value, 'Personal recovery through reload');
    context.request = personalRequest; context.locale = 'zh';
    await render('local', {}, false); await click('拼接回复'); await change(source(), '本地个人回复');
    assert.match(document.body.textContent, /个人草稿已保存到本机/);
    await render('local-reload', {}, false); await click('拼接回复'); assert.equal(source().value, '本地个人回复');
    const localDraftKey = Object.keys(localStorage).find(key => key.startsWith('paperleet.rebuttal.assembly.local.') && key.endsWith('.all.v1'));
    assert(localDraftKey); localStorage.setItem(localDraftKey, '{corrupt personal draft');
    await render('local-corrupt', {}, false); await click('拼接回复');
    assert.equal(source().value, ''); assert.equal(source().readOnly, true); assert.equal(button('复制到 OpenReview').disabled, true); assert.equal(button('.md').disabled, true);
    assert.equal(localStorage.getItem(localDraftKey), '{corrupt personal draft', 'Unreadable personal storage is never overwritten by automatic assembly');
    const getItem = window.Storage.prototype.getItem;
    window.Storage.prototype.getItem = function (key) { if (key.startsWith('paperleet.rebuttal.assembly.local.')) throw Error('Storage unavailable'); return getItem.call(this, key); };
    try {
      await render('local-unreadable', {}, false); await click('拼接回复');
      assert.equal(source().value, ''); assert.equal(source().readOnly, true); assert.equal(button('复制到 OpenReview').disabled, true); assert.equal(button('.md').disabled, true);
    } finally { window.Storage.prototype.getItem = getItem; }
  }
  console.log('PASS assembly UI: actual StrictMode editor, private autosave, follow-until-edited generation, reviewer-all isolation, deliberate blank/reload, explicit reset, toolbar preview/copy/download/counts, initial-read export guards with save-failure recovery, deliberate shared snapshots, viewer isolation, coordinated shared/personal saves before API navigation and conflict-reload recovery, and bilingual local persistence');
} finally {
  if (root) await act(async () => root.unmount()); await server.close(); await rm(cacheDir, { recursive: true, force: true });
  URL.createObjectURL = createURL; URL.revokeObjectURL = revokeURL; dom.window.close();
  for (const [key, value] of prior) value ? Object.defineProperty(globalThis, key, value) : delete globalThis[key];
}
await import('./check-rebuttal-versions-ui.mjs');
