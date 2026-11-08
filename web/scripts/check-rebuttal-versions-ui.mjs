import assert from 'node:assert/strict';
import React, { act } from 'react';
import { JSDOM } from 'jsdom';
import { createServer } from 'vite';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';

const dom = new JSDOM('<!doctype html><div id="root"></div>', { url: 'https://paperleet.test/#/rebuttal/versions_paper', pretendToBeVisual: true });
const context = { locale: 'en', account: { user: { id: 'versions_owner' }, loading: false,
  capabilities: { rebuttal_assembly_drafts: true, rebuttal_reviewer_versions: true } }, request: null };
const globals = { window: dom.window, document: dom.window.document, navigator: dom.window.navigator,
  location: dom.window.location, history: dom.window.history, localStorage: dom.window.localStorage, sessionStorage: dom.window.sessionStorage,
  Element: dom.window.Element, HTMLElement: dom.window.HTMLElement, HTMLInputElement: dom.window.HTMLInputElement,
  HTMLTextAreaElement: dom.window.HTMLTextAreaElement, HTMLButtonElement: dom.window.HTMLButtonElement,
  IS_REACT_ACT_ENVIRONMENT: true, __versionsTest: context,
  fetch: async path => { assert(String(path).endsWith('/api/rebuttal/status')); return new Response(JSON.stringify({ configured: false, model: null })); } };
const prior = new Map(Object.keys(globals).map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
for (const [key, value] of Object.entries(globals)) Object.defineProperty(globalThis, key, { value, configurable: true, writable: true });
let downloaded;
const createURL = URL.createObjectURL, revokeURL = URL.revokeObjectURL;
URL.createObjectURL = blob => { downloaded = blob; return 'blob:version-download'; }; URL.revokeObjectURL = () => {};
window.HTMLAnchorElement.prototype.click = function () {};
const mocks = {
  './social': 'export const useAccount=()=>globalThis.__versionsTest.account;export const socialApi=(...args)=>globalThis.__versionsTest.request(...args);',
  './i18n': 'export const useI18n=()=>({locale:globalThis.__versionsTest.locale,t:(zh,en)=>globalThis.__versionsTest.locale==="zh"?zh:en});',
  './apiClient': 'export const apiFetch=(...args)=>fetch(...args);',
};
const cacheDir = await mkdtemp(`${tmpdir()}/paperleet-versions-ui-`);
const server = await createServer({ configFile: false, cacheDir, esbuild: { jsx: 'automatic' }, server: { middlewareMode: true, watch: null, hmr: false, ws: false }, appType: 'custom', optimizeDeps: { noDiscovery: true, include: [] }, plugins: [{ name: 'version-test-boundaries', enforce: 'pre', resolveId(source) { if (mocks[source]) return '\0versions:' + source; }, load(id) { return id.startsWith('\0versions:') ? mocks[id.slice(10)] : undefined; } }] });
let root;
try {
  const { createRoot } = await import('react-dom/client');
  const { default: Studio } = await server.ssrLoadModule('/src/RebuttalStudio.tsx');
  const { newProject, validateWorkspace } = await server.ssrLoadModule('/src/rebuttalModel.ts');
  const { toOpenReviewMarkdown } = await server.ssrLoadModule('/src/rebuttalFormatting.ts');
  const stamp = '2030-01-01T00:00:00Z';
  const legacy = { id: 'legacy_v1', label: 'Earlier full response', markdown: 'LEGACY original complete response', createdAt: stamp };
  const orphan = { id: 'removed_v1', label: 'Custom archived version', markdown: 'REMOVED reviewer historical response', createdAt: stamp, scope: 'reviewer-removed', reviewerName: 'Former reviewer' };
  const project = { ...newProject(), id: 'versions_paper', title: 'Scoped version history', opening: 'Generated opening',
    reviews: [{ id: 'r1', name: 'Reviewer One', text: 'Review one' }, { id: 'r2', name: 'Reviewer Two', text: 'Review two' }], snapshots: [legacy, orphan] };
  const documents = new Map(['all', 'reviewer-r1', 'reviewer-r2'].map(scope => [scope, { user_id: context.account.user.id, paper_id: project.id, scope,
    source: scope === 'all' ? 'ALL personal complete response' : scope === 'reviewer-r1' ? 'R1 personal first response' : 'R2 private loaded response', revision: 1, updated_at: stamp }]));
  let resolveR2;
  const delayedR2 = new Promise(resolve => { resolveR2 = resolve; });
  let delayR2 = true;
  const reads = [], shared = [];
  context.request = async (path, body) => {
    assert(path.startsWith('rebuttal/papers/versions_paper/drafts/'));
    const scope = decodeURIComponent(path.split('/').at(-1));
    assert.notEqual(scope, 'legacy', 'Legacy versions never become a private draft API scope');
    if (scope === 'reviewer-removed') throw Object.assign(Error('Removed reviewer'), { status: 404 });
    const current = documents.get(scope); assert(current, `Unexpected scope ${scope}`);
    if (!body) { reads.push(scope); return scope === 'reviewer-r2' && delayR2 ? delayedR2 : current; }
    assert.equal(body.expected_user_id, context.account.user.id); assert.equal(body.expected_revision, current.revision);
    const saved = { ...current, source: body.source, revision: current.revision + 1 }; documents.set(scope, saved); return saved;
  };
  let bridge;
  const session = { project, role: 'owner', readOnly: false, saveStatus: 'saved', onChange: value => shared.push(value),
    onBack() {}, onShare() {}, onRefresh() {}, onPersonalDraft: value => { bridge = value; } };
  root = createRoot(document.querySelector('#root'));
  const render = async (key, overrides = {}) => act(async () => root.render(React.createElement(React.StrictMode, null, React.createElement(Studio, { key, hosted: { ...session, ...overrides } }))));
  const button = text => [...document.querySelectorAll('button')].find(node => node.textContent.trim() === text || node.getAttribute('aria-label') === text);
  const click = async text => { const node = button(text); assert(node, `Missing button ${text}`); assert.equal(node.disabled, false, `${text} disabled`); await act(async () => node.click()); };
  const change = async (node, value) => { assert(node); await act(async () => { const prototype = node instanceof window.HTMLTextAreaElement ? window.HTMLTextAreaElement.prototype : window.HTMLSelectElement.prototype; Object.getOwnPropertyDescriptor(prototype, 'value').set.call(node, value); node.dispatchEvent(new window.Event(node.tagName === 'SELECT' ? 'change' : 'input', { bubbles: true })); }); };
  const scopeSelect = () => document.querySelector('select[aria-label="Version scope"],select[aria-label="版本范围"]');
  const preview = () => document.querySelector('.rst-version-content')?.textContent || '';
  const labels = () => [...document.querySelectorAll('.rst-version-list button strong')].map(node => node.textContent);
  const lastVersion = () => shared.at(-1).snapshots.at(-1);

  await render('versions');
  const navigation = document.querySelector('.rst-paper-navigation'), heading = document.querySelector('.rst-paper-heading');
  assert(navigation?.contains(button('My papers')) && navigation.contains(button('Refresh')));
  assert(heading?.contains(document.querySelector('.rst-paper-identity h1')) && heading.contains(document.querySelector('.rst-save')));
  assert(navigation.compareDocumentPosition(heading) & window.Node.DOCUMENT_POSITION_FOLLOWING, 'Navigation precedes the paper identity row');
  assert(!heading.contains(button('My papers')) && !heading.contains(button('Refresh')));

  await click('Versions');
  assert.equal(scopeSelect().value, 'reviewer-r1', 'Entering versions from overview starts with the first reviewer');
  const options = [...scopeSelect().options].map(option => [option.value, option.textContent]);
  assert.deepEqual(options.map(option => option[0]), ['reviewer-r1', 'reviewer-r2', 'reviewer-removed', 'all', 'legacy']);
  assert.match(options.find(option => option[0] === 'reviewer-removed')[1], /Former reviewer.*removed/i);
  assert.deepEqual(labels(), []); assert(!preview().includes('LEGACY') && !preview().includes('REMOVED'));
  await click('Save shared version');
  assert.equal(lastVersion().scope, 'reviewer-r1'); assert.equal(lastVersion().reviewerName, 'Reviewer One'); assert.equal(lastVersion().label, 'Version 1');
  assert.equal(lastVersion().markdown, 'R1 personal first response'); assert.deepEqual(labels(), ['V1']); assert.match(preview(), /R1 personal first response/);

  await change(scopeSelect(), 'reviewer-r2');
  assert.equal(button('Save shared version').disabled, true, 'A reviewer cannot save another scope while its private draft is loading');
  assert(!preview().includes('R1 personal first response')); assert.deepEqual(labels(), []);
  const before = shared.length; await act(async () => button('Save shared version').click()); assert.equal(shared.length, before);
  delayR2 = false; await act(async () => resolveR2(documents.get('reviewer-r2')));
  await click('Save shared version');
  assert.equal(lastVersion().scope, 'reviewer-r2'); assert.equal(lastVersion().reviewerName, 'Reviewer Two'); assert.equal(lastVersion().label, 'Version 1');
  assert.equal(lastVersion().markdown, 'R2 private loaded response'); assert.deepEqual(labels(), ['V1']); assert.match(preview(), /R2 private loaded response/);

  await change(scopeSelect(), 'reviewer-r1');
  assert.match(preview(), /R1 personal first response/); assert(!preview().includes('R2 private loaded response'));
  await click('Assemble');
  assert.equal(document.querySelector('.rst-export aside select').value, 'reviewer-r1', 'History and assembly retain the selected reviewer');
  const secondR1 = 'R1 personal second response with $\\alpha$';
  await change(document.querySelector('textarea[aria-label="OpenReview Markdown"]'), secondR1);
  await act(async () => { assert.equal(await bridge.flush(), true); });
  await click('Save shared version');
  assert.equal(lastVersion().scope, 'reviewer-r1'); assert.equal(lastVersion().label, 'Version 2'); assert.equal(lastVersion().markdown, toOpenReviewMarkdown(secondR1));
  await click('Versions');
  assert.equal(scopeSelect().value, 'reviewer-r1'); assert.deepEqual(labels(), ['V2', 'V1']); assert.match(preview(), /R1 personal second response/);

  await change(scopeSelect(), 'all'); await click('Save shared version');
  assert.equal(lastVersion().scope, 'all'); assert.equal(lastVersion().label, 'Version 1'); assert.equal(lastVersion().markdown, 'ALL personal complete response');
  assert.equal(lastVersion().reviewerName, undefined); assert.deepEqual(labels(), ['V1']); assert.match(preview(), /ALL personal complete response/);
  await click('Assemble'); assert.equal(document.querySelector('.rst-export aside select').value, 'all'); await click('Versions'); assert.equal(scopeSelect().value, 'all');

  await change(scopeSelect(), 'legacy');
  assert.equal(button('Save shared version').disabled, true); assert.deepEqual(labels(), ['Earlier full response']); assert.match(preview(), /LEGACY original complete response/);
  await click('Markdown'); assert.equal(await downloaded.text(), legacy.markdown);
  await change(scopeSelect(), 'reviewer-removed');
  assert.equal(button('Save shared version').disabled, true); assert.deepEqual(labels(), ['Custom archived version']); assert.match(preview(), /REMOVED reviewer historical response/);
  await click('Markdown'); assert.equal(await downloaded.text(), orphan.markdown);
  assert.deepEqual(shared.at(-1).snapshots.filter(item => item.id === legacy.id || item.id === orphan.id), [legacy, orphan]);
  const roundTrip = validateWorkspace({ version: 1, projects: [shared.at(-1)], activeProjectId: project.id }).projects[0];
  assert.deepEqual(roundTrip.snapshots, shared.at(-1).snapshots, 'Saved scope metadata and legacy snapshots survive reload validation');

  // A new UI on the old backend may browse old snapshots, but cannot submit unknown scope fields.
  context.account.capabilities = { rebuttal_assembly_drafts: true };
  await render('old-version-api', { project: shared.at(-1) }); await click('Versions');
  assert.equal(button('Save shared version').disabled, true); assert.match(document.body.textContent, /Versions are updating/);
  assert.match(preview(), /R1 personal second response/);
  await change(scopeSelect(), 'legacy'); assert.match(preview(), /LEGACY original complete response/);
  await click('Markdown'); assert.equal(await downloaded.text(), legacy.markdown);
  assert(reads.includes('reviewer-r1') && reads.includes('reviewer-r2') && reads.includes('all'));
  console.log('PASS reviewer versions UI: scoped private-source loading, independent numbering, latest selection, All/Legacy/orphan history, immutable downloads, metadata round trip, old-server write gate and two-row paper header structure');
} finally {
  if (root) await act(async () => root.unmount()); await server.close(); await rm(cacheDir, { recursive: true, force: true });
  URL.createObjectURL = createURL; URL.revokeObjectURL = revokeURL; dom.window.close();
  for (const [key, value] of prior) value ? Object.defineProperty(globalThis, key, value) : delete globalThis[key];
}
