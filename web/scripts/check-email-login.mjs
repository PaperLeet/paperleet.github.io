import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';
import { createServer } from 'vite';

const dom = new JSDOM('<div id="root"></div>', { url: 'https://paperleet.test/#/account' });
const globals = { window: dom.window, document: dom.window.document, location: dom.window.location,
  localStorage: dom.window.localStorage, IS_REACT_ACT_ENVIRONMENT: true, __emailLoginFixture: undefined };
const originals = new Map(Object.keys(globals).map(name => [name, Object.getOwnPropertyDescriptor(globalThis, name)]));
for (const [name, value] of Object.entries(globals)) Object.defineProperty(globalThis, name, { configurable: true, writable: true, value });
dom.window.HTMLDialogElement.prototype.showModal = function () { this.open = true; };
dom.window.HTMLDialogElement.prototype.close = function () { this.open = false; };
const React = await import('react'), { act } = React;
const { createRoot } = await import('react-dom/client');
const server = await createServer({ cacheDir: 'node_modules/.vite-email-login-check', server: { middlewareMode: true, hmr: false, ws: false }, appType: 'custom',
  optimizeDeps: { noDiscovery: true, include: [] }, plugins: [{
    name: 'email-login-fixture', enforce: 'pre',
    resolveId(source, importer) { if (source === './social' && importer?.endsWith('/Account.tsx')) return '\0email-login-fixture'; },
    load(id) { if (id === '\0email-login-fixture') return `
      import React from 'react';
      export class SocialError extends Error { constructor(message,status,code){super(message);this.status=status;this.code=code;} }
      export const errorMessage=e=>e.message;
      export const useAccount=()=>globalThis.__emailLoginFixture;
      export const socialApi=(...args)=>globalThis.__emailLoginFixture.api(...args);
      export const SocialNotice=({error,children})=>React.createElement('p',{role:error?'alert':'status'},error||children);
      export const Avatar=()=>null; export const TestAccountBadge=()=>null;
      export const profileHref=user=>'#/u/'+user.username;
    `; },
  }] });
let root;
try {
  const { SignIn, OrganizationSignIn } = await server.ssrLoadModule('/src/Account.tsx');
  const { LocaleProvider } = await server.ssrLoadModule('/src/i18n.tsx');
  const { canPublishCommunity } = await server.ssrLoadModule('/src/communityPublishing.ts');
  const { SocialError } = await server.ssrLoadModule('\0email-login-fixture');
  const requests = [], signedIn = [];
  let response, failure, pending;
  const capabilities = { auth_available: true, auth_mode: 'oauth', email_login_policy: 'any_email', email_login_available: true,
    affiliation_verification_available: true, institutions: [], work_email_registration: true };
  globalThis.__emailLoginFixture = { capabilities, setUser(user) { signedIn.push(user); },
    async api(path, payload) { requests.push({ path, payload }); if (pending) return pending; if (failure) throw failure; return response; } };
  const container = document.querySelector('#root');
  const form = () => container.querySelector('.login-card form');
  const email = () => form().querySelector('input[type=email]');
  const stop = async () => { if (root) await act(async () => root.unmount()); root = undefined; };
  async function render(options = {}) {
    const { locale = 'en', organization = false, enabled = true, emailAvailable = true } = options;
    await stop(); localStorage.setItem('paperleet.locale', locale); capabilities.email_login_policy = Object.hasOwn(options, 'policy') ? options.policy : 'any_email';
    response = { sent: true, challenge: 'login-challenge', retry_after: 60 }; failure = pending = undefined;
    root = createRoot(container);
    await act(async () => root.render(React.createElement(LocaleProvider, null, organization
      ? React.createElement(OrganizationSignIn, { emailLoginAvailable: emailAvailable, providers: [{ id: 'google', available: true }] })
      : React.createElement(SignIn, { enabled, workEmailRegistration: true }))));
  }
  async function change(node, value) {
    const props = node[Object.keys(node).find(key => key.startsWith('__reactProps$'))];
    await act(async () => props.onChange({ target: { value } }));
  }
  const submit = () => act(async () => form().dispatchEvent(new dom.window.Event('submit', { bubbles: true, cancelable: true })));
  const click = node => act(async () => node.click());
  for (const locale of ['en', 'zh']) {
    await render({ locale, organization: true });
    assert.equal(container.querySelector('.email-login-alternative>h2').textContent, locale === 'en' ? 'Sign in with an email code' : '使用邮箱验证码登录');
    assert.equal(email().placeholder, 'you@example.com');
    assert(!container.textContent.includes('Which accounts work?') && !container.querySelector('.email-login-alternative summary'));
    assert(container.querySelector('.email-login-compact') && email(), 'Email sign-in is immediately visible beside provider methods');
    assert(!container.textContent.includes(locale === 'en' ? 'cannot use email codes' : '个人邮箱不支持验证码'));
    assert(!container.querySelector('.institution-request'), 'Domain review must not block ordinary sign-in');
  }
  for (const address of ['person@gmail.com', 'person@qq.com', 'person@outlook.com', 'person@unreviewed.example']) {
    await render(); await change(email(), address); const before = requests.length; await submit();
    assert.deepEqual(requests[before], { path: 'account/send-code', payload: { email: address, prefer_email: false } });
    assert.equal(requests.length, before + 1); assert(!container.querySelector('dialog').open);
    assert(email().disabled); assert([...form().querySelectorAll('button')].some(button => button.textContent.includes('Resend in') && button.disabled));
    await change(form().querySelector('input[autocomplete="one-time-code"]'), '123456');
    const user = { id: 'personal', email: address, email_verified: true, can_publish_community: false, affiliation: null, status: 'active' };
    response = { user }; await submit();
    assert.deepEqual(requests.at(-1), { path: 'account/verify', payload: { email: address, code: '123456', challenge: 'login-challenge' } });
    assert.equal(signedIn.at(-1), user); assert.equal(canPublishCommunity(user, capabilities), false);
    assert.equal(canPublishCommunity({ ...user, can_publish_community: undefined }, capabilities), false, 'Verifying a login mailbox cannot grant affiliation publishing permission');
  }
  // An absent, old or unrecognized capability must not promise unsupported login.
  for (const policy of [undefined, 'education_and_approved_domains', 'unknown']) {
    await render({ policy }); await change(email(), 'person@gmail.com'); const before = requests.length; await submit();
    assert.equal(requests.length, before); assert(container.querySelector('[role=alert]').textContent.includes('reviewed school/company'));
    assert.equal(email().placeholder, 'you@university.edu.cn');
    await change(email(), 'student@school.edu.cn'); await submit(); assert.equal(requests.length, before + 1);
  }
  for (const address of ['bad@@gmail.com', 'person@', 'person @gmail.com']) {
    await render(); await change(email(), address); const before = requests.length; await submit();
    assert.equal(requests.length, before); assert(container.querySelector('[role=alert]').textContent.includes('valid email'));
  }
  await render(); await change(email(), 'student@school.edu'); let before = requests.length; await submit();
  assert.equal(requests.length, before); assert(container.querySelector('dialog').open);
  await click([...container.querySelectorAll('dialog button')].find(button => button.textContent === 'Continue with email code'));
  assert.deepEqual(requests.at(-1), { path: 'account/send-code', payload: { email: 'student@school.edu', prefer_email: true } });
  await render({ enabled: false }); await change(email(), 'person@gmail.com'); before = requests.length; await submit(); assert.equal(requests.length, before);
  await render({ organization: true, emailAvailable: false }); assert(!container.querySelector('.email-login-alternative'));
  await render(); await change(email(), 'person@gmail.com'); failure = new SocialError('Unavailable', 503); await submit();
  assert.equal(email().value, 'person@gmail.com'); assert(!email().disabled); assert(container.querySelector('[role=alert]').textContent.includes('temporarily unavailable'));
  failure = undefined; await submit(); assert(form().querySelector('input[autocomplete="one-time-code"]'));
  await render(); await change(email(), 'person@gmail.com'); before = requests.length;
  let release; pending = new Promise(resolve => { release = resolve; }); await submit(); await submit(); assert.equal(requests.length, before + 1, 'Busy submits cannot send duplicate codes');
  await act(async () => release(response));
  assert(requests.every(request => !request.path.startsWith('account/affiliation/')), 'Sign-in and affiliation verification use separate flows');
  console.log('PASS capability-gated personal email login: bilingual copy, four ordinary providers/domains, exact send/verify challenge, no publishing grant, legacy fallback, .edu OAuth choice, cooldown, disabled/unavailable and duplicate-send recovery');
} finally {
  if (root) await act(async () => root.unmount());
  await server.close(); dom.window.close();
  for (const [name, descriptor] of originals) { if (descriptor) Object.defineProperty(globalThis, name, descriptor); else delete globalThis[name]; }
}
