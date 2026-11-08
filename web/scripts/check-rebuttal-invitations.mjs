import assert from 'node:assert/strict';
import React, { act } from 'react';
import { JSDOM } from 'jsdom';
import { createServer } from 'vite';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';

const dom = new JSDOM('<!doctype html><div id="root"></div>', { url: 'https://paperleet.test/#/rebuttal', pretendToBeVisual: true });
const context = { account: {}, locale: 'en', href: '#/rebuttal', request: async () => { throw Error('Unexpected request'); }, events: [] };
const globals = { window: dom.window, document: dom.window.document, HTMLElement: dom.window.HTMLElement, Element: dom.window.Element,
  navigator: dom.window.navigator, location: dom.window.location, sessionStorage: dom.window.sessionStorage, IS_REACT_ACT_ENVIRONMENT: true, __rebuttalInvitationTest: context };
const previous = new Map(Object.keys(globals).map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
for (const [key, value] of Object.entries(globals)) Object.defineProperty(globalThis, key, { value, configurable: true, writable: true });
const mocks = {
  './social': 'export const useAccount=()=>globalThis.__rebuttalInvitationTest.account;export const socialApi=(...args)=>globalThis.__rebuttalInvitationTest.request(...args);export const errorMessage=e=>e.message;',
  './inbox': 'export const inboxChanged=owner=>globalThis.__rebuttalInvitationTest.events.push(owner);',
  './i18n': 'export const useI18n=()=>({locale:globalThis.__rebuttalInvitationTest.locale,t:(zh,en)=>globalThis.__rebuttalInvitationTest.locale==="zh"?zh:en});',
  './appLocation': 'export const appLocation={get hash(){return globalThis.__rebuttalInvitationTest.href},set hash(value){globalThis.__rebuttalInvitationTest.href=value}};export const publicHref=href=>href;',
  './accountNavigation': 'export const signInHref=href=>"#/account?return_to="+encodeURIComponent(href);',
  './AppLink': 'import React from "react";export default function Link(props){return React.createElement("a",props)}',
  './deferredFeature': 'import React from "react";export const deferredFeature=()=>({hosted})=>React.createElement("button",{onClick:hosted.onShare},"Manage members");',
};
const cacheDir = await mkdtemp(`${tmpdir()}/paperleet-hub-invitations-`);
const server = await createServer({ configFile: false, cacheDir, esbuild: { jsx: 'automatic' }, server: { middlewareMode: true, watch: null, hmr: false, ws: false }, appType: 'custom', optimizeDeps: { noDiscovery: true, include: [] },
  plugins: [{ name: 'hub-invitation-boundaries', enforce: 'pre', resolveId(source) { if (mocks[source]) return '\0hub:' + source; }, load(id) { return id.startsWith('\0hub:') ? mocks[id.slice(5)] : undefined; } }],
});
const deferred = () => { let resolve, reject; const promise = new Promise((yes, no) => { resolve = yes; reject = no; }); return { promise, resolve, reject }; };
let root;
try {
  const { createRoot } = await import('react-dom/client');
  const { default: Hub } = await server.ssrLoadModule('/src/HostedRebuttal.tsx');
  const { newProject } = await server.ssrLoadModule('/src/rebuttalModel.ts');
  root = createRoot(document.querySelector('#root'));
  const author = { id: 'author_a', username: 'author_a', display_name: 'Author A' }, recipient = { id: 'author_b', username: 'author_b', display_name: 'Author B' };
  const project = { ...newProject(), id: 'paper_a', title: 'Shared paper' };
  const invite = { id: 'invite_a', paper_id: project.id, title: project.title, venue: 'ICLR', owner: author, role: 'editor', created_at: '2030-01-01T00:00:00Z' };
  let invitations = [invite], posts = [], memberInvitations = [], revision = 1, requested = [];
  const detail = () => ({ project, revision, role: 'owner', owner: author, members: [{ user: author, role: 'owner' }, { user: recipient, role: 'viewer' }], invitations: memberInvitations });
  const defaultRequest = async (path, body) => {
    requested.push(path);
    if (body) {
      posts.push({ path, body: structuredClone(body) });
      if (path.endsWith('/members')) { revision++; memberInvitations = body.action === 'remove' ? [] : [{ id: 'pending_c', user: { id: 'author_c', username: body.username, display_name: 'Author C' }, role: body.role, created_at: invite.created_at }]; return detail(); }
      const result = { id: path.split('/').at(-1), paper_id: project.id, status: body.action === 'accept' ? 'accepted' : 'declined' };
      invitations = invitations.filter(item => item.id !== result.id); return result;
    }
    if (path === 'rebuttal/papers') return { papers: [] };
    if (path === 'rebuttal/invitations') return { user_id: context.account.user.id, invitations };
    if (path === 'rebuttal/papers/paper_a') return detail();
    throw Error('Unexpected request: ' + path);
  };
  context.request = defaultRequest;
  const render = async (route = 'rebuttal/invitations') => act(async () => { root.render(React.createElement(Hub, { route })); });
  const button = text => [...document.querySelectorAll('button')].find(node => node.textContent === text || node.getAttribute('aria-label') === text);
  const click = async text => { const node = button(text); assert(node, `Missing button ${text}`); assert.equal(node.disabled, false, `${text} should be enabled`); await act(async () => node.click()); };
  const signIn = user => { context.account = { user, loading: false, capabilities: { inbox_rebuttal_invitations: true } }; };

  // A notification's invitation landing survives the sign-in link.
  context.account = { user: null, loading: false };
  await render();
  assert.equal(new URL(document.querySelector('a').href).hash, '#/account?return_to=%23%2Frebuttal%2Finvitations');
  context.account = { user: author, loading: false, capabilities: {} };
  await render();
  assert.match(document.body.textContent, /Invitations are updating/);
  assert(!requested.includes('rebuttal/invitations'), 'Old APIs receive no invitation endpoint request');
  await render('rebuttal/paper_a'); await click('Manage members');
  assert(!button('Send invitation'), 'Old immediate-grant APIs cannot be called by the new invitation form');
  assert(document.querySelector('select[aria-label="author_b permission"]'), 'Existing member role edits remain available');

  // Pending invitations stay outside Shared with me and decline never navigates into a paper.
  signIn(recipient); context.href = '#/rebuttal/invitations'; await render();
  assert.match(document.body.textContent, /Shared paper/); assert.match(document.body.textContent, /Author A \(@author_a\)/);
  await click('Shared with me'); assert.match(document.body.textContent, /No shared papers/); assert(!document.querySelector('.hreb-paper-open'));
  await click('Invitations (1)'); await click('Decline');
  assert.match(document.body.textContent, /No pending invitations/); assert.equal(context.href, '#/rebuttal/invitations');
  assert.equal(posts.at(-1).body.expected_user_id, recipient.id); assert.deepEqual(context.events, [recipient.id]);

  // The actual controls keep a lost accept request retryable and disable competing decisions.
  invitations = [invite]; await click('Refresh invitations');
  const uncertain = deferred(); context.request = (path, body) => {
    if (!body) return defaultRequest(path, body);
    posts.push({ path, body: structuredClone(body) }); return uncertain.promise;
  };
  await click('Accept'); assert.equal(button('Decline').disabled, true); assert.equal(context.href, '#/rebuttal/invitations');
  await act(async () => uncertain.reject(Error('Connection lost')));
  assert(button('Retry original operation')); assert.equal(button('Decline').disabled, true);
  const original = posts.at(-1); context.request = defaultRequest;
  await click('Retry original operation'); assert.deepEqual(posts.at(-1), original);
  assert.equal(context.href, '#/rebuttal/paper_a'); assert.equal(context.events.length, 2);

  // Late success from an unmounted account does not navigate the new account.
  invitations = [invite]; await click('Refresh invitations');
  const oldResponse = deferred(); context.request = (path, body) => body ? oldResponse.promise : defaultRequest(path, body);
  context.href = '#/rebuttal/invitations'; await click('Accept');
  signIn({ id: 'author_c', username: 'author_c', display_name: 'Author C' }); invitations = []; await render();
  await act(async () => oldResponse.resolve({ id: invite.id, paper_id: project.id, status: 'accepted' }));
  assert.equal(context.href, '#/rebuttal/invitations'); assert.equal(context.events.length, 2);

  // Owner invitations appear separately from active members and cancel through the permission API.
  signIn(author); context.request = defaultRequest; context.locale = 'zh'; await render('rebuttal/paper_a'); await click('Manage members');
  const input = document.querySelector('.hreb-invite input');
  await act(async () => { Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set.call(input, 'author_c'); input.dispatchEvent(new window.Event('input', { bubbles: true })); });
  await click('发送邀请');
  assert.match(document.body.textContent, /邀请已发送 · 等待接受/); assert.match(document.body.textContent, /待接受的邀请/);
  assert.equal(document.querySelectorAll('.hreb-pending-badge').length, 1);
  await click('取消邀请'); assert.equal(document.querySelectorAll('.hreb-pending-badge').length, 0);
  assert.equal(posts.at(-1).body.action, 'remove'); assert.equal(posts.at(-1).body.user_id, 'author_c');
  assert.equal(posts.at(-1).body.expected_user_id, author.id);
  console.log('Rebuttal invitation UI: bilingual pending members, cancellation, capability rollout, sign-in landing, no pre-accept access, exact retries and account-switch navigation passed.');
} finally {
  if (root) await act(async () => root.unmount());
  await server.close(); await rm(cacheDir, { recursive: true, force: true }); dom.window.close();
  for (const [key, value] of previous) value ? Object.defineProperty(globalThis, key, value) : delete globalThis[key];
}
