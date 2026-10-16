import assert from 'node:assert/strict';
import fs from 'node:fs';
import React, { act, useState } from 'react';
import { JSDOM, VirtualConsole } from 'jsdom';
import { createServer } from 'vite';

const errors = [];
const virtualConsole = new VirtualConsole();
virtualConsole.on('jsdomError', error => errors.push(error));
const dom = new JSDOM('<div id="root"></div>', { url: 'http://paperleet.test/', virtualConsole });
const globals = { window: dom.window, document: dom.window.document, Element: dom.window.Element, HTMLElement: dom.window.HTMLElement, localStorage: dom.window.localStorage, location: dom.window.location, scrollY: 0, innerHeight: 900, scrollTo() {}, IS_REACT_ACT_ENVIRONMENT: true, ResizeObserver: class { observe() {} disconnect() {} }, requestAnimationFrame: callback => setTimeout(callback, 0), cancelAnimationFrame: clearTimeout };
const previous = new Map(Object.keys(globals).map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
for (const [key, value] of Object.entries(globals)) Object.defineProperty(globalThis, key, { value, writable: true, configurable: true });
localStorage.setItem('paperleet.locale','zh');
dom.window.matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {} });
dom.window.HTMLElement.prototype.scrollIntoView = function () {};
const server = await createServer({ server: { middlewareMode: true }, appType: 'custom', optimizeDeps: { noDiscovery: true, include: [] } });
let root;
try {
  const { createRoot } = await import('react-dom/client');
  const { PaperLibrary } = await server.ssrLoadModule('/src/PaperViews.tsx');
  const { navigateTabs } = await server.ssrLoadModule('/src/tabNavigation.ts');
  const { LocaleProvider } = await server.ssrLoadModule('/src/i18n.tsx');
  const catalog = JSON.parse(fs.readFileSync('public/data/catalog.json', 'utf8'));
  const h = React.createElement;
  let route = '';
  function Library({ initialRoute = '' }) {
    const [view, setView] = useState(initialRoute);
    return h(PaperLibrary, { ...catalog, saved: {}, route: view, onFilters: href => { route = href; setView(href); } });
  }
  const find = selector => document.querySelector(selector);
  const click = async element => { assert(element); await act(async () => element.click()); };
  const mount = async () => {
    root = createRoot(find('#root'));
    await act(async () => root.render(h(LocaleProvider, null, h(Library))));
  };
  await mount();
  await click(find('[aria-label="紧凑列表"]'));
  assert.equal(find('.library-main').dataset.density, 'compact');
  assert.equal(localStorage.getItem('paperleet.library.density.v1'), 'compact');
  assert.equal(route, '', 'Density must not change search/filter navigation');
  await act(async () => root.unmount());
  await mount();
  assert.equal(find('.library-main').dataset.density, 'compact', 'Density survives a remount');
  await click(find('.difficulty-summary.E'));
  assert(route.includes('difficulty=E'));
  assert.equal(find('.difficulty-summary.E').getAttribute('aria-pressed'), 'true');
  assert.equal(find('[aria-label="模块难度筛选"]').value, 'E', 'Sidebar and toolbar use the same URL state');
  await click(find('.difficulty-summary.E'));
  assert.equal(find('[aria-label="模块难度筛选"]').value, 'all');
  assert(!find('.library-help-card'), 'The generic practice guide card is removed');
  assert(!document.body.textContent.includes('查看使用指南'), 'Practice recommendations have no generic guide action');
  const setItem = dom.window.Storage.prototype.setItem;
  dom.window.Storage.prototype.setItem = () => { throw new Error('Storage blocked'); };
  await click(find('[aria-label="舒适列表"]'));
  assert.equal(find('.library-main').dataset.density, 'comfortable', 'Blocked preference storage must not break the list');
  dom.window.Storage.prototype.setItem = setItem;
  const legacyTopic = '#/?tag=Top-k&sort=title';
  await act(async () => root.render(h(LocaleProvider, null, h(Library, { key: 'legacy-topic', initialRoute: legacyTopic }))));
  const topic = find('[aria-label="主题筛选"]');
  assert.equal([...topic.options].filter(option => option.value.toLowerCase() === 'top-k').length, 1);
  assert.equal(topic.value.toLowerCase(), 'top-k', 'A differently cased existing URL still selects its visible topic');
  const paperLinks = [...document.querySelectorAll('.library-main a[href^="#/papers/"]')];
  assert.deepEqual(paperLinks.map(link => link.getAttribute('href').split('?')[0]).sort(), ['#/papers/llm-training-generation', '#/papers/sdpo']);
  assert(paperLinks.every(link => decodeURIComponent(link.getAttribute('href')).includes('tag=Top-k')), 'Reading keeps the original link spelling');
  topic.focus();
  await act(async () => { topic.value = 'Attention'; topic.dispatchEvent(new dom.window.Event('change', { bubbles: true })); });
  assert(route.includes('tag=Attention') && route.includes('sort=title'));
  assert.equal(document.activeElement, topic, 'Choosing a new topic keeps the same native control and keyboard focus');
  await act(async () => { topic.value = 'all'; topic.dispatchEvent(new dom.window.Event('change', { bubbles: true })); });
  assert.equal(route, '#/?sort=title', 'Clearing a topic preserves the learner-selected paper order');
  await act(async () => root.render(h(PaperLibrary, { ...catalog, saved: {}, progressReady: false })));
  assert(find('.library-progress-ring b').textContent.startsWith('—'), 'Unresolved progress must not render a fabricated zero');

  function Tabs() {
    const [selected, setSelected] = useState(0);
    return h('div', { role: 'tablist', onKeyDown: navigateTabs }, ['Read', 'Tests', 'Code'].map((name, index) => h('button', { key: name, role: 'tab', 'aria-selected': selected === index, tabIndex: selected === index ? 0 : -1, onClick: () => setSelected(index) }, name)));
  }
  await act(async () => root.render(h(Tabs)));
  const tabs = [...document.querySelectorAll('[role="tab"]')];
  const key = async (element, value, extra = {}) => act(async () => element.dispatchEvent(new dom.window.KeyboardEvent('keydown', { key: value, bubbles: true, cancelable: true, ...extra })));
  tabs[0].focus();
  await key(tabs[0], 'ArrowLeft');
  assert.equal(document.activeElement, tabs[2]);
  assert.equal(tabs[2].getAttribute('aria-selected'), 'true');
  await key(tabs[2], 'ArrowRight');
  assert.equal(document.activeElement, tabs[0]);
  await key(tabs[0], 'End');
  assert.equal(document.activeElement, tabs[2]);
  await key(tabs[2], 'Home');
  assert.equal(document.activeElement, tabs[0]);
  await key(tabs[0], 'ArrowRight', { metaKey: true });
  assert.equal(document.activeElement, tabs[0], 'Browser shortcuts stay untouched');
  assert.equal(tabs.filter(tab => tab.tabIndex === 0).length, 1);
  assert.deepEqual(errors, [], 'No uncaught DOM errors are allowed');
  console.log('PASS catalog workbench: density persistence/storage failure, shared URL difficulty filters, native case-alias topic selection and focus with exact reading returns and retained sort, removed generic guide, keyboard tab activation/wrapping/Home/End and modifier isolation');
} finally {
  if (root) await act(async () => root.unmount());
  await server.close(); dom.window.close();
  for (const [key, descriptor] of previous) if (descriptor) Object.defineProperty(globalThis, key, descriptor); else delete globalThis[key];
}
