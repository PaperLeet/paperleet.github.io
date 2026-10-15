import assert from 'node:assert/strict';
import React, { act } from 'react';
import { createRoot } from 'react-dom/client';
import { createServer } from 'vite';

// A small host fixture for actual React client rendering. Server rendering does
// not catch descendant errors, so renderToString cannot verify this boundary.
class TestNode {
  constructor(type, name, document) {
    this.nodeType = type;
    this.nodeName = this.tagName = name;
    this.ownerDocument = document;
    this.parentNode = null;
    this.childNodes = [];
    this.style = {};
    this.attributes = {};
    this.namespaceURI = 'http://www.w3.org/1999/xhtml';
    this.text = '';
  }
  appendChild(node) {
    node.parentNode?.removeChild(node);
    this.childNodes.push(node);
    node.parentNode = this;
    return node;
  }
  insertBefore(node, reference) {
    node.parentNode?.removeChild(node);
    this.childNodes.splice(this.childNodes.indexOf(reference), 0, node);
    node.parentNode = this;
    return node;
  }
  removeChild(node) {
    this.childNodes.splice(this.childNodes.indexOf(node), 1);
    node.parentNode = null;
    return node;
  }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  removeAttribute(name) { delete this.attributes[name]; }
  addEventListener() {}
  removeEventListener() {}
  get firstChild() { return this.childNodes[0] ?? null; }
  get textContent() { return this.text + this.childNodes.map(node => node.textContent).join(''); }
  set textContent(value) {
    this.text = String(value);
    this.childNodes.forEach(node => { node.parentNode = null; });
    this.childNodes = [];
  }
}

function findElement(element, type) {
  if (!React.isValidElement(element)) return;
  if (element.type === type) return element;
  for (const child of React.Children.toArray(element.props.children)) {
    const match = findElement(child, type);
    if (match) return match;
  }
}

const server = await createServer({
  server: { middlewareMode: true }, appType: 'custom',
  optimizeDeps: { noDiscovery: true, entries: [] },
});
const names = ['window', 'document', 'localStorage', 'sessionStorage', 'IS_REACT_ACT_ENVIRONMENT'];
const originals = new Map(names.map(name => [name, Object.getOwnPropertyDescriptor(globalThis, name)]));
let root;
try {
  const { AppErrorBoundary } = await server.ssrLoadModule('/src/AppErrorBoundary.tsx');
  const document = new TestNode(9, '#document', null);
  document.createElement = name => new TestNode(1, name.toUpperCase(), document);
  document.createTextNode = text => Object.assign(new TestNode(3, '#text', document), { textContent: text });
  document.documentElement = document.createElement('html');
  document.body = document.createElement('body');
  document.activeElement = document.body;
  const navigations = [];
  const location = { href: 'https://paperleet.test/?view=practice#/architecture/attention', replace: url => navigations.push(url) };
  const storage = new Proxy({}, { get() { throw Error('Recovery must not read or clear browser storage'); } });
  const window = { document, HTMLIFrameElement: class {}, location, localStorage: storage, sessionStorage: storage };
  document.defaultView = window;
  Object.assign(globalThis, { window, document, localStorage: storage, sessionStorage: storage, IS_REACT_ACT_ENVIRONMENT: true });

  const container = document.createElement('div');
  const boundary = React.createRef();
  const caught = [];
  root = createRoot(container, { onCaughtError: error => caught.push(error), onUncaughtError: error => { throw error; } });
  const render = child => act(async () => root.render(React.createElement(AppErrorBoundary, { ref: boundary }, child)));
  await render(React.createElement('p', null, 'Working page'));
  assert.equal(container.textContent, 'Working page');
  assert.deepEqual(navigations, [], 'Healthy rendering never triggers recovery');

  const privateText = 'private-account@example.test secret draft source';
  const failure = new Error(privateText);
  function BrokenPage() { throw failure; }
  await render(React.createElement(BrokenPage));
  assert.deepEqual(caught, [failure], 'A real descendant render error must be caught by React');
  assert.match(container.textContent, /页面暂时无法显示/);
  assert.match(container.textContent, /This page could not be displayed/);
  assert(!container.textContent.includes(privateText), 'The fallback must not display error details or private data');
  assert.equal(container.firstChild.attributes['aria-labelledby'], 'app-recovery-title');
  assert.equal(container.firstChild.firstChild.attributes.role, 'alert');
  assert.deepEqual(navigations, [], 'Catching an error must not reload automatically');

  await render(React.createElement('p', null, 'Changed page'));
  assert.match(container.textContent, /页面暂时无法显示/, 'A normal rerender must not retry a crashed subtree');
  assert.deepEqual(navigations, []);

  // Verify the public button callback separately; the host fixture intentionally
  // does not imitate a browser event system or access React's private DOM keys.
  const button = findElement(boundary.current.render(), 'button');
  assert(button);
  assert.equal(button.props.type, 'button');
  await act(async () => button.props.onClick());
  assert.equal(navigations.length, 1, 'The recovery action performs one explicit reload');
  const destination = new URL(navigations[0]);
  assert.equal(destination.origin, 'https://paperleet.test');
  assert.equal(destination.hash, '#/architecture/attention', 'Recovery preserves the current page');
  assert.equal(destination.searchParams.get('view'), 'practice');
  assert(destination.searchParams.has('_paperleet_reload'), 'Recovery requests a fresh entry document');
  await act(async () => root.unmount());
  root = undefined;
  assert.equal(container.textContent, '');
  console.log('PASS root error boundary: actual client render recovery, bilingual fallback, private error suppression, explicit reload, and untouched storage');
} finally {
  if (root) await act(async () => root.unmount());
  for (const name of names) {
    const descriptor = originals.get(name);
    if (descriptor) Object.defineProperty(globalThis, name, descriptor);
    else delete globalThis[name];
  }
  await server.close();
}
