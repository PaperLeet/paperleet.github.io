import assert from 'node:assert/strict';
import React, { act } from 'react';
import { createRoot } from 'react-dom/client';
import { createServer } from 'vite';

// Minimal DOM host for React's real form handler and state updates, without a
// browser, account service, or additional test dependency.
class TestNode {
  constructor(type, name, document, namespace = 'http://www.w3.org/1999/xhtml') {
    Object.assign(this, { nodeType: type, nodeName: name, tagName: name, ownerDocument: document,
      namespaceURI: namespace, parentNode: null, childNodes: [], attributes: {}, listeners: new Map(), text: '', style: {} });
  }
  appendChild(node) {
    node.parentNode?.removeChild(node);
    this.childNodes.push(node); node.parentNode = this; return node;
  }
  insertBefore(node, reference) {
    node.parentNode?.removeChild(node);
    this.childNodes.splice(this.childNodes.indexOf(reference), 0, node); node.parentNode = this; return node;
  }
  removeChild(node) {
    this.childNodes.splice(this.childNodes.indexOf(node), 1); node.parentNode = null; return node;
  }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  getAttribute(name) { return this.attributes[name] ?? null; }
  get type() { return this.attributes.type ?? ''; }
  set type(value) { this.attributes.type = value; }
  removeAttribute(name) { delete this.attributes[name]; }
  addEventListener(name, listener) {
    this.listeners.set(name, [...(this.listeners.get(name) || []), listener]);
  }
  removeEventListener(name, listener) {
    this.listeners.set(name, (this.listeners.get(name) || []).filter(value => value !== listener));
  }
  get firstChild() { return this.childNodes[0] ?? null; }
  get textContent() { return this.text + this.childNodes.map(node => node.textContent).join(''); }
  set textContent(value) {
    this.text = String(value);
    this.childNodes.forEach(node => { node.parentNode = null; }); this.childNodes = [];
  }
  get nodeValue() { return this.text; }
  set nodeValue(value) { this.text = String(value); }
}
function findNode(node, matches) {
  if (matches(node)) return node;
  for (const child of node.childNodes) {
    const found = findNode(child, matches);
    if (found) return found;
  }
}

const server = await createServer({
  server: { middlewareMode: true }, appType: 'custom', optimizeDeps: { noDiscovery: true, entries: [] },
  define: { 'import.meta.env.VITE_PUBLIC_SITE': '"0"', 'import.meta.env.VITE_APP_EDITION': '"hosted"', 'import.meta.env.VITE_API_ORIGIN': '""' },
});
const names = ['window', 'document', 'location', 'localStorage', 'fetch', 'createImageBitmap', 'IS_REACT_ACT_ENVIRONMENT'];
const originals = new Map(names.map(name => [name, Object.getOwnPropertyDescriptor(globalThis, name)]));
let root;
try {
  const { LocaleProvider } = await server.ssrLoadModule('/src/i18n.tsx');
  const { ProfileSettings } = await server.ssrLoadModule('/src/Account.tsx');
  const { AccountProvider, useAccount } = await server.ssrLoadModule('/src/social.tsx');
  const document = new TestNode(9, '#document', null);
  document.createElement = name => new TestNode(1, name.toUpperCase(), document);
  document.createElementNS = (namespace, name) => new TestNode(1, name, document, namespace);
  document.createTextNode = text => Object.assign(new TestNode(3, '#text', document), { textContent: text });
  document.documentElement = document.createElement('html');
  document.body = document.createElement('body');
  document.activeElement = document.body;
  const location = { hash: '#/account' };
  const window = { document, location, HTMLIFrameElement: class {}, addEventListener() {}, removeEventListener() {} };
  document.defaultView = window;
  const user = { id: 'fixture-user', username: 'original_name', display_name: 'Fixture User', avatar_color: '#4f7c73',
    bio: 'Existing biography', email: 'fixture@example.test', role: 'member', institution: null, country: '',
    show_institution: false, show_country: false, show_progress: true, created_at: '2026-10-15T00:00:00Z',
    avatar_url: `/api/avatars/${'a'.repeat(32)}/${'b'.repeat(64)}.jpg` };
  const requests = [];
  let completeSave;
  const fetch = async (url, options) => {
    if (url === '/api/account/me') return Response.json({ user, capabilities: {} });
    if (url === '/api/health') return Response.json({ mode: 'hosted', auth_required: true, judge: { ready: false } });
    assert.equal(url, '/api/account/profile', 'Only the mocked profile API may receive saves');
    assert.equal(options.method, 'POST');
    requests.push(JSON.parse(options.body));
    return new Promise(resolve => { completeSave = resolve; });
  };
  Object.assign(globalThis, { window, document, location, fetch, localStorage: { getItem: key => key === 'paperleet.locale' ? 'zh' : null, setItem() {} }, IS_REACT_ACT_ENVIRONMENT: true });

  let account;
  function ObserveAccount() { account = useAccount(); return null; }
  const container = document.createElement('div');
  root = createRoot(container);
  await act(async () => root.render(React.createElement(LocaleProvider, null, React.createElement(AccountProvider, null,
    React.createElement(ProfileSettings, { user }), React.createElement(ObserveAccount)))));
  let form = findNode(container, node => node.tagName === 'FORM');
  assert(form);
  const uploadInput = () => findNode(form, node => node.tagName === 'INPUT' && node.attributes.type === 'file');
  const button = label => findNode(form, node => node.tagName === 'BUTTON' && node.textContent === label);
  assert('disabled' in uploadInput().attributes, 'Old servers must disable new avatar uploads');
  assert('disabled' in button('移除图片').attributes, 'Old servers must not pretend to support avatar deletion');
  assert.match(container.textContent, /头像上传需等待服务更新/);
  const failedImage = findNode(form, node => node.tagName === 'IMG');
  assert(failedImage);
  await act(async () => {
    const event = { type: 'error', target: failedImage, bubbles: false, stopPropagation() {} };
    for (const listener of failedImage.listeners.get('error') || []) listener(event);
  });
  assert(!findNode(form, node => node.tagName === 'IMG'), 'An unavailable avatar image falls back to the letter avatar');
  const saveButton = () => findNode(form, node => node.tagName === 'BUTTON' && node.attributes.type === 'submit');
  async function submit() {
    const event = { type: 'submit', target: form, bubbles: true, defaultPrevented: false,
      preventDefault() { this.defaultPrevented = true; }, stopPropagation() {} };
    await act(async () => {
      for (const listener of container.listeners.get('submit') || []) listener(event);
    });
    assert(event.defaultPrevented, 'Saving stays within the hash-routed app');
  }

  await submit();
  assert.equal(requests.length, 1);
  assert.equal(requests[0].expected_user_id, user.id, 'Profile saves must name the expected authenticated account');
  assert(!('avatar_image' in requests[0]), 'Unchanged or unsupported avatars must not be submitted');
  assert.equal(requests[0].show_institution, true, 'Old servers must also restore public institution visibility when settings are saved');
  assert.equal(location.hash, '#/account', 'Do not leave settings while the save is pending');
  assert('disabled' in saveButton().attributes);
  await act(async () => completeSave(Response.json({ error: 'This username is already taken' }, { status: 400 })));
  assert.equal(location.hash, '#/account', 'A rejected save must keep the user on the settings page');
  assert.match(container.textContent, /This username is already taken/);
  assert.equal(account.user.username, 'original_name');
  assert(!('disabled' in saveButton().attributes), 'The failed form remains available for retry');

  await submit();
  assert.equal(requests.length, 2);
  assert.equal(requests[1].bio, user.bio, 'Retry keeps the unsaved form values');
  const savedUser = { ...user, username: 'server_saved_name', display_name: 'Saved User' };
  await act(async () => completeSave(Response.json({ user: savedUser })));
  assert.equal(account.user.username, savedUser.username, 'The account context receives the saved profile');
  assert.equal(location.hash, '#/u/server_saved_name', 'Successful saves navigate using the server-returned username');
  assert(!container.textContent.includes('This username is already taken'));

  const policy = { username: { interval_seconds: 2592000, next_change_at: null },
    display_name: { interval_seconds: 604800, next_change_at: null },
    avatar: { daily_limit: 3, remaining: 2, reset_at: '2030-10-16T00:00:00Z' } };
  const supported = { ...savedUser, profile_edit_policy: policy,
    affiliation: { kind: 'email_domain', domain: 'example.test', name: 'example.test' } };
  const preparedImage = 'data:image/jpeg;base64,cHJlcGFyZWQ=';
  globalThis.createImageBitmap = async () => ({ width: 800, height: 400, close() {} });
  const createElement = document.createElement;
  document.createElement = name => {
    if (name === 'canvas') return { getContext: () => ({ fillRect() {}, drawImage() {} }), toDataURL: () => preparedImage };
    return createElement(name);
  };
  async function showSettings(next, key) {
    location.hash = '#/account';
    await act(async () => root.render(React.createElement(LocaleProvider, null, React.createElement(AccountProvider, null,
      React.createElement(ProfileSettings, { user: next, key }), React.createElement(ObserveAccount)))));
    form = findNode(container, node => node.tagName === 'FORM');
  }
  async function dispatch(type, target) {
    const event = { type, target, bubbles: true, defaultPrevented: false,
      preventDefault() { this.defaultPrevented = true; }, stopPropagation() {} };
    await act(async () => { for (const listener of container.listeners.get(type) || []) listener(event); });
  }
  await showSettings(supported, 'supported');
  assert(!('disabled' in uploadInput().attributes), 'A supported server enables avatar selection');
  assert.match(container.textContent, /每天最多修改 3 次.*剩余 2 次/);
  assert.match(container.textContent, /固定公开，不能隐藏/);
  const selected = uploadInput();
  selected.files = [{ type: 'image/png', size: 300 }];
  selected.value = 'fixture.png';
  await dispatch('change', selected);
  assert.equal(requests.length, 2, 'File selection and preview must not upload before Save');
  assert(findNode(form, node => node.tagName === 'IMG' && node.attributes.src === preparedImage), 'The prepared avatar is previewed before saving');
  await submit();
  assert.equal(requests[2].avatar_image, preparedImage);
  assert.equal(requests[2].expected_user_id, supported.id);
  const limitedPolicy = { ...policy, avatar: { ...policy.avatar, remaining: 0 } };
  await act(async () => completeSave(Response.json({ error: 'Profile limit', code: 'profile_edit_limit', field: 'avatar',
    next_change_at: '2030-10-16T00:00:00Z', profile_edit_policy: limitedPolicy }, { status: 429 })));
  assert.equal(location.hash, '#/account');
  assert.match(container.textContent, /头像暂时不能再次修改/);
  assert.match(container.textContent, /今天剩余 0 次/);
  assert(findNode(form, node => node.tagName === 'IMG' && node.attributes.src === preparedImage), 'A rejected save retains the selected image for retry');
  await submit();
  assert.equal(requests[3].avatar_image, preparedImage, 'Retry submits the same unsaved image');
  await act(async () => completeSave(Response.json({ user: supported })));
  assert.equal(location.hash, '#/u/server_saved_name');

  await showSettings(supported, 'remove');
  await dispatch('click', button('移除图片'));
  assert.equal(requests.length, 4, 'Removing an image is also staged until Save');
  assert(!findNode(form, node => node.tagName === 'IMG'), 'Removal previews the original letter/color avatar');
  await submit();
  assert.equal(requests[4].avatar_image, null, 'Only an explicit removal sends null');
  await act(async () => completeSave(Response.json({ user: { ...supported, avatar_url: null } })));
  assert.equal(location.hash, '#/u/server_saved_name');

  await showSettings(supported, 'old-account-save');
  await submit();
  const otherUser = { ...supported, id: 'other-account', username: 'another_person', display_name: 'Other Person' };
  await showSettings(otherUser, 'new-account');
  await act(async () => account.setUser(otherUser));
  await act(async () => completeSave(Response.json({ user: supported })));
  assert.equal(account.user.id, otherUser.id, 'A late save must not restore an account after the settings owner changes');
  assert.equal(location.hash, '#/account', 'A late old-account response must not navigate the new account away');
  console.log('PASS profile save: old-server compatibility, private actor identity, staged avatar upload/removal, 429 policy/preview preservation and successful profile navigation');
} finally {
  if (root) await act(async () => root.unmount());
  for (const name of names) {
    const descriptor = originals.get(name);
    if (descriptor) Object.defineProperty(globalThis, name, descriptor);
    else delete globalThis[name];
  }
  await server.close();
}
