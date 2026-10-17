import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';
import { createServer } from 'vite';
const dom = new JSDOM('<div id="root"></div>', { url: 'https://paperleet.test/#/account' });
const names = ['window', 'document', 'location', 'localStorage', 'IS_REACT_ACT_ENVIRONMENT', '__affiliationFixture'];
const originals = new Map(names.map(name => [name, Object.getOwnPropertyDescriptor(globalThis, name)]));
Object.assign(globalThis, { window: dom.window, document: dom.window.document, location: dom.window.location,
  localStorage: dom.window.localStorage, IS_REACT_ACT_ENVIRONMENT: true });
localStorage.setItem('paperleet.locale', 'zh');
const React = await import('react');
const { act } = React;
const { createRoot } = await import('react-dom/client');
const server = await createServer({ server: { middlewareMode: true, hmr: false }, appType: 'custom',
  optimizeDeps: { noDiscovery: true, include: [] }, plugins: [{
    name: 'affiliation-api-fixture', enforce: 'pre',
    resolveId(source, importer) { if (source === './social' && importer?.endsWith('/AffiliationVerification.tsx')) return '\0affiliation-api-fixture'; },
    load(id) { if (id === '\0affiliation-api-fixture') return `
      import React from 'react';
      export class SocialError extends Error { constructor(message,code){super(message);this.code=code;} }
      export const errorMessage = e => e.message;
      export const useAccount = () => globalThis.__affiliationFixture;
      export const socialApi = (...args) => globalThis.__affiliationFixture.api(...args);
      export const SocialNotice = ({error}) => error ? React.createElement('p',{role:'alert'},error) : null;
    `; },
  }] });
let root;
try {
  const { LocaleProvider } = await server.ssrLoadModule('/src/i18n.tsx');
  const { default: Verification } = await server.ssrLoadModule('/src/AffiliationVerification.tsx');
  const { SocialError } = await server.ssrLoadModule('\0affiliation-api-fixture');
  const requests = [];
  let user = { id: 'personal', email: 'person@gmail.com', affiliation_verified: false };
  let result = { sent: true, challenge: 'bound-challenge', retry_after: 60 }, failure;
  globalThis.__affiliationFixture = {
    capabilities: { affiliation_verification_available: true },
    setUser(next) { user = next; render(); },
    async api(path, payload) { requests.push({ path, payload }); if (failure) throw failure; return result; },
  };
  const container = document.querySelector('#root');
  root = createRoot(container);
  function render() { root.render(React.createElement(LocaleProvider, null, React.createElement(Verification, { user, key: user.id }))); }
  const text = () => container.textContent;
  async function input(selector, value) {
    const element = container.querySelector(selector);
    const props = element[Object.keys(element).find(key => key.startsWith('__reactProps$'))];
    await act(async () => props.onChange({ target: { value } }));
  }
  async function submit() { await act(async () => container.querySelector('form').dispatchEvent(new dom.window.Event('submit', { bubbles: true, cancelable: true }))); }
  await act(async () => render());
  assert.match(text(), /未认证也可登录、练习、发帖和回复/);
  await input('input[type="email"]', 'Student@school.edu'); await submit();
  assert.deepEqual(requests[0], { path: 'account/affiliation/send-code', payload: { email: 'student@school.edu', expected_user_id: 'personal' } });
  assert.match(text(), /10 分钟/);
  assert(container.querySelector('input[type="email"]').disabled);
  assert([...container.querySelectorAll('button')].some(button => button.textContent.includes('秒后可重发') && button.disabled));
  await input('input[inputmode="numeric"]', '123456');
  result = { user: { ...user, affiliation_verified: true, affiliation_email: 'student@school.edu',
    affiliation: { kind: 'email_domain', domain: 'school.edu', name: 'School' } } };
  await submit();
  assert.deepEqual(requests[1], { path: 'account/affiliation/verify', payload: { email: 'student@school.edu', expected_user_id: 'personal', code: '123456', challenge: 'bound-challenge' } });
  assert.equal(user.email, 'person@gmail.com');
  assert.match(text(), /无需重复验证/);
  assert(!container.querySelector('form'));
  user = { ...user, id: 'another', affiliation_verified: false, affiliation: null };
  failure = new SocialError('Capacity full', 'email_quota_exhausted');
  await act(async () => render());
  await input('input[type="email"]', 'other@school.edu'); await submit();
  assert.match(text(), /仍可继续使用账号/);
  globalThis.__affiliationFixture.capabilities.affiliation_verification_available = false;
  await act(async () => render());
  assert(!container.querySelector('form'));
  assert.match(text(), /暂不可用/);
  console.log('PASS optional affiliation send/verify, bound account payload, cooldown, existing proof, quota and old-server compatibility');
} finally {
  if (root) await act(async () => root.unmount());
  await server.close(); dom.window.close();
  for (const [name, descriptor] of originals) { if (descriptor) Object.defineProperty(globalThis, name, descriptor); else delete globalThis[name]; }
}
