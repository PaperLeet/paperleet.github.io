import assert from 'node:assert/strict';
import React, { act } from 'react';
import { JSDOM } from 'jsdom';
import { createServer } from 'vite';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';

const hostedSupported = (await readFile('src/RebuttalStudio.tsx', 'utf8')).includes('HostedStudioSession');
const dom = new JSDOM('<!doctype html><div id="root"></div>', { url: 'https://paperleet.test/rebuttal/?paper=paper_a', pretendToBeVisual: true });
const context = { locale: 'en', account: { user: { id: 'account_a' }, loading: false, capabilities: { rebuttal_assembly_drafts: true } }, request: null };
const globals = { window: dom.window, document: dom.window.document, navigator: dom.window.navigator, location: dom.window.location, history: dom.window.history,
  localStorage: dom.window.localStorage, sessionStorage: dom.window.sessionStorage, Element: dom.window.Element, HTMLElement: dom.window.HTMLElement,
  HTMLInputElement: dom.window.HTMLInputElement, HTMLTextAreaElement: dom.window.HTMLTextAreaElement, HTMLButtonElement: dom.window.HTMLButtonElement,
  IS_REACT_ACT_ENVIRONMENT: true, __assemblyTest: context,
  fetch: async path => { assert(String(path).endsWith('/api/rebuttal/status')); return new Response(JSON.stringify({ configured: false, model: null })); } };
const prior = new Map(Object.keys(globals).map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
for (const [key, value] of Object.entries(globals)) Object.defineProperty(globalThis, key, { value, configurable: true, writable: true });
let copied = '', downloaded;
Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText: async text => { copied = text; } } });
const createURL = URL.createObjectURL, revokeURL = URL.revokeObjectURL;
URL.createObjectURL = blob => { downloaded = blob; return 'blob:test-assembly'; }; URL.revokeObjectURL = () => {};
window.HTMLAnchorElement.prototype.click = function () {};
const mocks = {
  './social': 'export const useAccount=()=>globalThis.__assemblyTest.account;export const socialApi=(...args)=>globalThis.__assemblyTest.request(...args);',
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
    const beforeViewerShared = shared.length; context.account.user = { id: 'viewer_b' };
    await render('viewer_b', { role: 'viewer', readOnly: true }); await click('Assemble');
    assert.equal(source().readOnly, false); assert.equal(button('Save shared version').disabled, true); assert.equal(document.querySelector('.rst-export aside textarea').disabled, true);
    await change(source(), 'Viewer private source'); await flush(); assert.equal(shared.length, beforeViewerShared); assert.equal(documents.get('viewer_b:all').source, 'Viewer private source');
    assert.equal(documents.get('account_a:all').source, '');
    context.account.capabilities = {}; await render('old-api', { role: 'viewer', readOnly: true }); await click('Assemble'); assert.equal(source().readOnly, true); assert.match(document.body.textContent, /Personal drafts are updating/);
    context.account.capabilities = { rebuttal_assembly_drafts: true }; context.locale = 'zh';
    await render('local', {}, false); await click('拼接回复'); await change(source(), '本地个人回复');
    assert.match(document.body.textContent, /个人草稿已保存到本机/);
    await render('local-reload', {}, false); await click('拼接回复'); assert.equal(source().value, '本地个人回复');
  }
  console.log('PASS assembly UI: actual StrictMode editor, private autosave, follow-until-edited generation, reviewer-all isolation, deliberate blank/reload, explicit reset, toolbar preview/copy/download/counts, deliberate shared snapshots, viewer isolation and bilingual local persistence');
} finally {
  if (root) await act(async () => root.unmount()); await server.close(); await rm(cacheDir, { recursive: true, force: true });
  URL.createObjectURL = createURL; URL.revokeObjectURL = revokeURL; dom.window.close();
  for (const [key, value] of prior) value ? Object.defineProperty(globalThis, key, value) : delete globalThis[key];
}
