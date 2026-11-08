import assert from 'node:assert/strict';
import React, { act } from 'react';
import { JSDOM } from 'jsdom';
import { createServer } from 'vite';

const dom = new JSDOM('<!doctype html><div id="root"></div>', { url: 'https://paperleet.test/#/account', pretendToBeVisual: true });
const globals = { window: dom.window, document: dom.window.document, navigator: dom.window.navigator, HTMLElement: dom.window.HTMLElement, location: dom.window.location, localStorage: dom.window.localStorage, IS_REACT_ACT_ENVIRONMENT: true, __accountEntryFixture: undefined };
const previous = new Map(Object.keys(globals).map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
for (const [key, value] of Object.entries(globals)) Object.defineProperty(globalThis, key, { value, writable: true, configurable: true });
localStorage.setItem('paperleet.locale', 'en');
const scrolls = [];
dom.window.HTMLElement.prototype.scrollIntoView = function () { scrolls.push(this); };
const requests = [], account = { user: null, loading: false, capabilities: { auth_mode: 'oauth', oauth_providers: [{ id: 'google', available: true }] } };
globalThis.__accountEntryFixture = { account, request: async (url, body) => { requests.push({ url, body }); throw Error('Fixture stops before external OAuth navigation'); } };
const server = await createServer({ server: { middlewareMode: true, hmr: false, ws: false }, appType: 'custom', optimizeDeps: { noDiscovery: true, include: [] }, plugins: [{
  name: 'account-entry-fixture', enforce: 'pre',
  resolveId(source) { if (source === './social') return '\0account-entry-fixture'; },
  load(id) { if (id === '\0account-entry-fixture') return `import React from 'react';
    export const useAccount=()=>globalThis.__accountEntryFixture.account;
    export const socialApi=(...args)=>globalThis.__accountEntryFixture.request(...args);
    export const errorMessage=e=>e.message;export class SocialError extends Error{};
    export const Avatar=({user})=>React.createElement('span',null,user.display_name);
    export const profileHref=u=>'#/u/'+u.username;export const PersonLink=Avatar;
    export const DateLabel=({value})=>React.createElement('time',null,value);
    export const SocialNotice=({error})=>error?React.createElement('p',{role:'alert'},error):null;
    export const TestAccountBadge=()=>null;`; },
}] });
let root;
try {
  const { createRoot } = await import('react-dom/client');
  const { default: Account } = await server.ssrLoadModule('/src/Account.tsx');
  const { LocaleProvider } = await server.ssrLoadModule('/src/i18n.tsx');
  const { accountEntryHref, isAccountRoute } = await server.ssrLoadModule('/src/accountEntry.ts');
  const { signInHref, accountReturnHref } = await server.ssrLoadModule('/src/accountNavigation.ts');
  const { decodeProblemHash } = await server.ssrLoadModule('/src/reports.ts');
  const { emptyInterviewFilters, interviewHref } = await server.ssrLoadModule('/src/interviewNavigation.ts');
  const interview = interviewHref({ ...emptyInterviewFilters(), company: 'nvidia', q: '视觉 & scale 😀 #/%? +=', page: 3, sort: 'helpful' });
  const valid = ['#/rebuttal/paper_ABC-42', '#/u/researcher?tab=solutions&page=2', interview];
  const invalid = ['https://evil.test/', '//evil.test/', 'javascript:alert(1)', '#/%ZZ', '#/#/rebuttal', '#/account/continue/community', 'rebuttal/paper?private=1', '#/community%252Fproblem%252Farchitecture%252Fattention'];
  root = createRoot(document.querySelector('#root'));
  for (const source of [...valid, ...invalid]) {
    const entry = accountEntryHref(source), route = decodeProblemHash(entry);
    const expected = valid.includes(source) ? accountReturnHref(decodeProblemHash(signInHref(source))) : undefined;
    await act(async () => {
      location.hash = entry;
      root.render(React.createElement(LocaleProvider, null, React.createElement(Account, { key: source, route, returnTo: 'https://evil.test/' })));
    });
    assert.equal(document.querySelector('.account-page > .social-back').getAttribute('href'), expected || '#/community');
    const before = location.href, count = requests.length;
    await act(async () => document.querySelector('.organization-login-button').click());
    assert.equal(requests.length, count + 1);
    assert.equal(requests.at(-1).url, 'account/oauth/google/start');
    assert.equal(requests.at(-1).body.return_to, expected || '#/', 'Only the existing allowlist output can reach OAuth return_to');
    assert.equal(location.href, before, 'Invalid continuations cannot navigate before validation');
  }
  // A direct settings deep link must survive OAuth without allowing arbitrary account routes.
  assert(isAccountRoute('account/ai'));
  for (const route of ['account/api', 'account/ai/other', 'account/ai?api_key=private']) {
    assert.equal(isAccountRoute(route), false);
    assert.equal(accountReturnHref('account/continue/' + route), undefined);
  }
  assert.equal(accountReturnHref('account/continue/account/ai'), '#/account/ai');
  assert.equal(accountReturnHref('account/error/expired/account/ai'), '#/account/ai');
  await act(async () => {
    location.hash = '#/account/ai';
    root.render(React.createElement(LocaleProvider, null, React.createElement(Account, { key: 'api-entry', route: 'account/ai', returnTo: 'https://evil.test/' })));
  });
  await act(async () => document.querySelector('.organization-login-button').click());
  assert.equal(requests.at(-1).url, 'account/oauth/google/start');
  assert.equal(requests.at(-1).body.return_to, '#/account/ai');

  // Exercise the real deferred settings form, including its focus after loading.
  account.user = { id: 'api-owner', username: 'api_owner', display_name: 'API owner', is_test_user: true, show_country: false, show_progress: false };
  account.capabilities = {};
  const settingsReads = [];
  globalThis.__accountEntryFixture.request = async (path, body) => {
    settingsReads.push({ path, body });
    assert.equal(path, 'account/ai-settings'); assert.equal(body, undefined);
    return { user_id: account.user.id, available: true, active_provider: null, revision: 0,
      providers: ['gemini', 'openai', 'anthropic', 'kimi', 'qwen', 'deepseek'].map(id => ({ id, configured: false, model: null, updated_at: null, region: ['kimi', 'qwen'].includes(id) ? 'international' : null })) };
  };
  await act(async () => root.render(React.createElement(LocaleProvider, null, React.createElement(Account, { key: 'api-signed-in', route: 'account/ai' }))));
  for (let tick = 0; !document.querySelector('.ai-settings input[aria-label="API key"]') && tick < 60; tick++) {
    await act(async () => new Promise(resolve => setTimeout(resolve, 10)));
  }
  const apiSection = document.querySelector('.ai-settings'), apiInput = apiSection?.querySelector('input[aria-label="API key"]');
  assert(apiInput, 'The direct API route loads actual settings even when capability metadata is absent');
  assert.equal(apiInput.type, 'password'); assert.equal(apiInput.disabled, false); assert.equal(apiInput.value, '');
  assert.equal(document.activeElement, apiSection, 'The deferred settings section receives focus');
  assert(scrolls.includes(apiSection), 'The API section is scrolled into view after it mounts');
  assert.equal(settingsReads.length, 1); assert.equal(location.hash, '#/account/ai');
  console.log('PASS account entry boundary: validated return links/OAuth destinations, exact API settings continuation, deferred form/focus with missing capability metadata, and rejected malformed/external/recursive continuations');
} finally {
  if (root) await act(async () => root.unmount());
  await server.close(); dom.window.close();
  for (const [key, descriptor] of previous) { if (descriptor) Object.defineProperty(globalThis, key, descriptor); else delete globalThis[key]; }
}
