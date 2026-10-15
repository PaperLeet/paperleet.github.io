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
const names = ['window', 'document', 'location', 'localStorage', 'fetch', 'IS_REACT_ACT_ENVIRONMENT'];
const originals = new Map(names.map(name => [name, Object.getOwnPropertyDescriptor(globalThis, name)]));
let root;
try {
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
    show_institution: false, show_country: false, show_progress: true, created_at: '2026-10-15T00:00:00Z' };
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
  Object.assign(globalThis, { window, document, location, fetch, localStorage: { setItem() {} }, IS_REACT_ACT_ENVIRONMENT: true });

  let account;
  function ObserveAccount() { account = useAccount(); return null; }
  const container = document.createElement('div');
  root = createRoot(container);
  await act(async () => root.render(React.createElement(AccountProvider, null,
    React.createElement(ProfileSettings, { user }), React.createElement(ObserveAccount))));
  const form = findNode(container, node => node.tagName === 'FORM');
  assert(form);
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
  console.log('PASS profile save: pending and failed saves remain editable, successful saves update the account and open the saved username profile');
} finally {
  if (root) await act(async () => root.unmount());
  for (const name of names) {
    const descriptor = originals.get(name);
    if (descriptor) Object.defineProperty(globalThis, name, descriptor);
    else delete globalThis[name];
  }
  await server.close();
}
