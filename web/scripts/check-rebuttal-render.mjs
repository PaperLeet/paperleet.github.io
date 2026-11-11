import assert from 'node:assert/strict';
import { readFile, mkdtemp, rm } from 'node:fs/promises';
import { resolve } from 'node:path';
import { tmpdir } from 'node:os';
import React, { act } from 'react';
import { JSDOM, ResourceLoader, VirtualConsole } from 'jsdom';
import { createServer } from 'vite';

const requests = [];
let failFirstBundle = true;
class LocalMathJaxResources extends ResourceLoader {
  fetch(url) {
    const parsed = new URL(url);
    assert.equal(parsed.origin, 'https://paperleet.test', 'Rendering must not download third-party resources');
    assert(parsed.pathname.startsWith('/vendor/mathjax-3.2.2/es5/'), `Unexpected resource ${url}`);
    requests.push(parsed.pathname);
    if (failFirstBundle && parsed.pathname.endsWith('/tex-chtml-full.js')) {
      failFirstBundle = false;
      const failure = Promise.reject(new Error('Intentional offline load for retry coverage'));
      failure.abort = () => {};
      return failure;
    }
    const pending = readFile(resolve('public', '.' + parsed.pathname));
    pending.abort = () => {};
    return pending;
  }
}
const virtualConsole = new VirtualConsole();
const browserErrors = [];
virtualConsole.on('jsdomError', error => browserErrors.push(error));
const dom = new JSDOM('<!doctype html><html><head></head><body><div id="root"></div></body></html>', {
  url: 'https://paperleet.test/rebuttal/', pretendToBeVisual: true, runScripts: 'dangerously',
  resources: new LocalMathJaxResources(), virtualConsole,
});
const globals = { window: dom.window, document: dom.window.document, navigator: dom.window.navigator,
  HTMLElement: dom.window.HTMLElement, Element: dom.window.Element, IS_REACT_ACT_ENVIRONMENT: true };
const previous = new Map(Object.keys(globals).map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
for (const [key, value] of Object.entries(globals)) Object.defineProperty(globalThis, key, { value, configurable: true, writable: true });
const cacheDir = await mkdtemp(`${tmpdir()}/paperleet-rebuttal-render-`);
const server = await createServer({ configFile: false, cacheDir, esbuild: { jsx: 'automatic' },
  server: { middlewareMode: true, watch: null, hmr: false, ws: false }, appType: 'custom', optimizeDeps: { noDiscovery: true, include: [] } });
let root;
try {
  const { default: Preview } = await server.ssrLoadModule('/src/RebuttalPreview.tsx');
  const { renderOpenReviewMarkdown } = await server.ssrLoadModule('/src/openReviewMarkdown.ts');
  const { formatRebuttal, openReviewMath, toOpenReviewMarkdown, REBUTTAL_COLORS } = await server.ssrLoadModule('/src/rebuttalFormatting.ts');
  const { createRoot } = await import('react-dom/client');
  root = createRoot(document.querySelector('#root'));
  const until = async (predicate, description) => {
    const deadline = Date.now() + 15000;
    while (!predicate()) {
      if (Date.now() > deadline) throw new Error(`Timed out: ${description}; ${browserErrors.map(error => error.message).join('; ')}`);
      await act(async () => { await new Promise(resolve => setTimeout(resolve, 10)); });
    }
  };
  const render = async (...sources) => {
    await act(async () => root.render(React.createElement(React.Fragment, null,
      ...sources.map((text, index) => React.createElement(Preview, { key: index, text })))));
  };
  const ready = async () => until(() => [...document.querySelectorAll('.rbe-preview')].every(node => node.dataset.mathStatus === 'ready'), 'MathJax preview readiness');
  const previews = () => [...document.querySelectorAll('.rbe-preview')];
  const mathItems = node => window.MathJax.startup.document.getMathItemsWithin([node]);
  const assertValidMath = (node, count) => {
    assert.equal(node.querySelectorAll('mjx-container').length, count, 'Every intended expression is typeset once');
    assert.equal(mathItems(node).length, count, 'MathJax tracks only the current expressions');
    assert(!node.querySelector('mjx-merror, merror, [data-mjx-error]'), `Valid TeX must not contain a MathJax error: ${mathItems(node).map(item => item.math).join(' / ')}; ${node.querySelector('mjx-merror, merror, [data-mjx-error]')?.outerHTML}`);
  };

  // A failed local asset load exposes Retry and can recover without losing source.
  await render('$offline$');
  await until(() => document.querySelector('.rbe-preview')?.dataset.mathStatus === 'error', 'load error is shown');
  assert.equal(document.querySelector('.rbe-message button')?.textContent, 'Retry');
  assert(!document.querySelector('script[data-rebuttal-mathjax]'), 'Failed script is removed before retry');
  assert.equal(browserErrors.length, 1, 'Only the intentional initial asset failure is reported');
  browserErrors.length = 0;
  await act(async () => document.querySelector('.rbe-message button').click());
  // Updates during script/extension loading must not mutate the latest source.
  await render(String.raw`Stale $\\text{old}$`);
  await render(String.raw`Current $\\text{new}$`, '$k=7$');
  await ready();
  assertValidMath(previews()[0], 1); assertValidMath(previews()[1], 1);
  assert.equal(mathItems(previews()[0])[0].math, String.raw`\text{new}`);
  assert(!document.querySelector('#root').textContent.includes('Stale'));
  assert.equal(requests.filter(path => path.endsWith('/tex-chtml-full.js')).length, 2, 'Multiple previews share one successful MathJax load after the intentional failure');
  assert(requests.some(path => path.endsWith('/ui/safe.js')), 'The real MathJax safe extension is loaded');

  const commands = [String.raw`p_{\text{dec}}`, String.raw`2\times 3`, String.raw`\ell`, String.raw`\rho`,
    String.raw`\max(x,y)`, String.raw`\beta`, String.raw`x\to y`, String.raw`\text{public checkpoint}`, String.raw`p_{\text{dec}}>0`];
  const doubled = commands.map(formula => formula.replaceAll('\\', '\\\\'));
  const sample = [...doubled, ...commands, 'k=7', 'K=23'].map(formula => `$${formula}$`).join('\n\n');
  await render(sample); await ready();
  let preview = previews()[0]; assertValidMath(preview, 20);
  const items = mathItems(preview);
  for (let index = 0; index < commands.length; index++) {
    assert.equal(items[index].math, commands[index], 'marked consumes one slash from doubled commands');
    assert.equal(items[index + commands.length].math, commands[index], 'marked preserves a single slash before a letter');
    assert.equal(items[index].root.toString(), items[index + commands.length].root.toString(), 'Single/double source commands have identical rendered math trees');
  }
  assert.equal(toOpenReviewMarkdown(sample), sample, 'Export retains the exact source that was previewed');

  const mixed = String.raw`$p_{\\text{dec}} + \rho$`;
  await render(mixed); await ready();
  assertValidMath(previews()[0], 1);
  assert.equal(mathItems(previews()[0])[0].math, String.raw`p_{\text{dec}} + \rho`, 'Mixed slash conventions are handled by marked, with no all-or-nothing decoding guess');

  const matrix = String.raw`\begin{matrix}a&b\\c&d\end{matrix}`;
  await render(`$$\n${openReviewMath(matrix)}\n$$`, String.raw`$\\{x\\}$`, String.raw`$a\\,b\\|c$`); await ready();
  for (const node of previews()) assertValidMath(node, 1);
  assert.equal(mathItems(previews()[0])[0].math.trim(), matrix);
  assert.equal(previews()[0].querySelectorAll('mjx-mtr').length, 2, 'Escaped matrix row break retains two rows after Markdown');
  assert.equal(mathItems(previews()[1])[0].math, String.raw`\{x\}`);
  assert.equal(mathItems(previews()[2])[0].math, String.raw`a\,b\|c`);
  await render(String.raw`$\{x\}$`); await ready();
  assert.equal(mathItems(previews()[0])[0].math, '{x}', 'Unescaped punctuation follows marked and is not silently repaired');

  await render(String.raw`\\(x+1\\)`); await ready();
  assertValidMath(previews()[0], 1); assert.equal(mathItems(previews()[0])[0].math, 'x+1', 'Escaped parenthesis delimiters survive Markdown and render inline');

  const table = '| Method | Value |\n| --- | --- |\n| Decoder | ' + String.raw`$p_{\\text{dec}}$` + ' |\n| Baseline | $k=7$ |';
  await render(table); await ready(); assertValidMath(previews()[0], 2);
  assert.equal(previews()[0].querySelectorAll('td mjx-container').length, 2);
  const literalCode = '`$\\\\text{code}$`\n\n```tex\n$\\\\rho$\n```';
  await render(literalCode); await ready(); assertValidMath(previews()[0], 0);
  assert.equal(previews()[0].querySelector('code').textContent, String.raw`$\\text{code}$`);
  assert.equal(previews()[0].querySelector('pre code').textContent.trim(), String.raw`$\\rho$`);
  assert(renderOpenReviewMarkdown('$a _b$ and $c_ d$').includes('<em>'), 'Markdown emphasis is allowed to cross formula delimiters, exactly as in OpenReview');
  const malformed = '**80-ep RAEv2 DiT-XL reproduced)*';
  const fixed = '**80-ep RAEv2 DiT-XL** ($k=7$: public checkpoint; $k=23$: reproduced)';
  await render(malformed, fixed); await ready();
  assert(!previews()[0].querySelector('strong')); assert(previews()[0].querySelector('em'));
  assert(previews()[0].textContent.startsWith('*'), 'A missing bold delimiter remains visible as a source error');
  assert.equal(previews()[1].querySelector('strong').textContent, '80-ep RAEv2 DiT-XL'); assertValidMath(previews()[1], 2);

  const styleSources = REBUTTAL_COLORS.map(color => formatRebuttal('Key result', 'color', 0, 10, color.value).value);
  styleSources.push(...['small', 'large', 'boxed'].map(format => formatRebuttal('Key result', format, 0, 10).value));
  const special = String.raw`Cost $5, 10% A_B {x} C:\tmp & # ^ ~`;
  styleSources.push(formatRebuttal(special, 'color', 0, special.length).value);
  await render(...styleSources); await ready();
  for (const node of previews()) assertValidMath(node, 1);
  for (let index = 0; index < REBUTTAL_COLORS.length; index++) {
    const swatch = document.createElement('span'); swatch.style.color = REBUTTAL_COLORS[index].value;
    assert([...previews()[index].querySelectorAll('[style]')].some(node => node.style.color === swatch.style.color), 'Toolbar color is rendered by MathJax');
    assert.equal(previews()[index].querySelector('[mathcolor]')?.getAttribute('mathcolor'), REBUTTAL_COLORS[index].value);
  }
  assert.equal(previews()[REBUTTAL_COLORS.length].querySelector('[mathsize]')?.getAttribute('mathsize'), '0.85em', 'Small text uses the actual MathJax size');
  assert.equal(previews()[REBUTTAL_COLORS.length + 1].querySelector('[mathsize]')?.getAttribute('mathsize'), '1.2em');
  assert(previews()[REBUTTAL_COLORS.length + 2].querySelector('mjx-box'), 'Toolbar boxed text uses a real math box');
  const specialMath = previews().at(-1).querySelector('mjx-assistive-mml math');
  assert(specialMath, 'Special toolbar text exposes the actual rendered MathML');
  assert.equal(specialMath.textContent.replace(/\s+/g, ' ').trim(), special,
    'Toolbar rendering preserves literal dollars, underscores, braces, backslashes and punctuation exactly');
  assert(!specialMath.querySelector('mspace[linebreak]'), 'A literal backslash in styled prose never becomes a math line break');


  const unsafe = '<script>window.pwned=1</script>\n\n<img src="https://probe.invalid/x" onerror="window.pwned=2">\n\n[bad](javascript:alert(1))\n\n' + String.raw`$\href{javascript:alert(1)}{x}$`;
  await render(unsafe); await ready();
  assert.equal(window.pwned, undefined); assert(!previews()[0].querySelector('script,img,[onerror]'));
  assert(![...previews()[0].querySelectorAll('[href]')].some(node => /^javascript:/i.test(node.getAttribute('href'))));
  assert(previews()[0].textContent.includes('<script>window.pwned=1</script>'), 'Raw HTML remains visible text');
  await render('![remote](https://probe.invalid/image.png)\n\n- [x] done\n- [ ] todo'); await ready();
  assert(!previews()[0].querySelector('img,input')); assert(previews()[0].textContent.includes('[x] done'));

  // Suspend a real typesetting request, change source, then release it: stale
  // work can complete only inside its detached subtree and is removed from MathJax.
  const originalTypeset = window.MathJax.typesetPromise;
  let release, started = false;
  window.MathJax.typesetPromise = async elements => {
    started = true;
    await new Promise(resolve => { release = resolve; });
    window.MathJax.typesetPromise = originalTypeset;
    return originalTypeset(elements);
  };
  await render('$x=1$'); await until(() => started, 'delayed MathJax starts');
  await render('$y=2$');
  await act(async () => release()); await ready();
  assertValidMath(previews()[0], 1); assert.equal(mathItems(previews()[0])[0].math, 'y=2');
  assert.equal([...window.MathJax.startup.document.math].length, 1, 'Detached math from previous edits is cleared');
  await act(async () => root.unmount()); root = undefined;
  await until(() => [...window.MathJax.startup.document.math].length === 0, 'MathJax cleanup after unmount');
  assert.equal(browserErrors.length, 0, browserErrors.map(error => error.message).join('\n'));
  console.log('PASS OpenReview render: actual marked 15 → DOMPurify → MathJax 3.2.2; 20 formulas, mixed slashes, tables, matrix rows, punctuation, literal code/HTML, malformed bold, toolbar styles, safe URLs, shared loading, rapid edits and unmount cleanup');
} finally {
  if (root) await act(async () => root.unmount());
  await server.close(); dom.window.close(); await rm(cacheDir, { recursive: true, force: true });
  for (const [key, descriptor] of previous) { if (descriptor) Object.defineProperty(globalThis, key, descriptor); else delete globalThis[key]; }
}
