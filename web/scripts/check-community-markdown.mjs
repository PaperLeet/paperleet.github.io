import assert from 'node:assert/strict';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { createServer } from 'vite';

const server=await createServer({server:{middlewareMode:true},appType:'custom'});
const originalStorage=globalThis.localStorage;
try {
  const {default:CommunityMarkdown,CommunityFormatHelp}=await server.ssrLoadModule('/src/CommunityMarkdown.tsx');
  const {LocaleProvider}=await server.ssrLoadModule('/src/i18n.tsx');
  globalThis.localStorage={getItem:()=>"zh"};
  const render=body=>renderToStaticMarkup(React.createElement(LocaleProvider,null,React.createElement(CommunityMarkdown,{body})));
  const math=render('Scale $\\alpha/r$.\n\n$$\n\\Delta W = \\frac{\\alpha}{r}BA\n$$');
  assert(math.includes('katex-inline')||math.includes('class="katex"'));
  assert(math.includes('katex-display')&&math.includes('<math')&&math.includes('annotation encoding="application/x-tex"'));
  assert(math.includes('mfrac'),'Fractions must actually render, not merely retain literal dollar delimiters');
  const literal=render('Keep `$x^2$` literal.\n\n```text\n$x^2$\n```');
  assert(!literal.includes('class="katex"')&&literal.includes('$x^2$'));
  assert(render('Cost $5').includes('Cost $5'));
  const code='# 中文 🧪\n\ndef forward(x):\n    return "<script>"\n';
  for(const lang of ['python','c++','']) {
    const result=render('```'+lang+'\n'+code+'```');
    assert(result.includes('community-code-block')&&result.includes('<pre>')&&result.includes('复制代码'));
    assert(result.includes('&lt;script&gt;')&&!result.includes('<script>'));
    assert(result.includes('# 中文 🧪\n\ndef forward(x):\n    return &quot;&lt;script&gt;&quot;'),'Code keeps whitespace and literal content before highlighting loads');
  }
  const inline=render('Use `x.detach()` here.');
  assert(inline.includes('<code>x.detach()</code>')&&!inline.includes('community-code-header'),'Inline code must not become a block or a copy toolbar');
  const unclosed=render('Before $unclosed\n\nAfter text.');
  assert(unclosed.includes('$unclosed')&&unclosed.includes('After text.'));
  const invalid=render('Before $\\frac{1}{$ after.');
  assert(invalid.includes('katex-error')&&invalid.includes('Before')&&invalid.includes('after.'));
  const malicious=render('![pixel](https://tracker.invalid/pixel)\n\n<script>evil()</script>\n\n[unsafe](javascript:evil())\n\n$\\includegraphics{https://tracker.invalid/math.png}$\n\n$\\href{javascript:evil()}{click}$\n\n$\\htmlStyle{background:url(https://tracker.invalid/style)}{x}$');
  assert(!/<img\b|<script\b|href="javascript:|src="https:\/\/tracker|style="[^"]*tracker/i.test(malicious),'Public Markdown/math must not enable raw HTML, script links or external image/style fetches');
  assert(malicious.includes('[pixel]'),'Blocked images retain their alt text');
  const link=render('[Paper](https://arxiv.org/abs/2106.09685)');
  assert(link.includes('target="_blank"')&&link.includes('rel="noopener noreferrer nofollow"'));
  const loop=render('$\\def\\x{\\x}\\x$\n\nStill readable.');
  assert(loop.includes('katex-error')&&loop.includes('Still readable.'),'Macro recursion is bounded and leaves the surrounding thread usable');
  render('$\\gdef\\privateMacro{SHOULD_NOT_LEAK}\\privateMacro$');
  const isolated=render('$\\privateMacro$');
  assert(!isolated.includes('SHOULD_NOT_LEAK')&&isolated.includes('\\privateMacro'),'Macros must not leak across posts, previews or accounts');
  const table=render('| Term | Value |\n| --- | --- |\n| rank | 2 |\n\n- [x] done');
  assert(table.includes('<table>')&&table.includes('type="checkbox"'),'Existing GFM tables and task lists remain available');
  for(const en of [false,true]) {
    globalThis.localStorage={getItem:()=>en?'en':'zh'};
    const result=renderToStaticMarkup(React.createElement(LocaleProvider,null,React.createElement(CommunityMarkdown,{body:'```python\nreturn x\n```'}),React.createElement(CommunityFormatHelp)));
    assert(result.includes(en?'Copy python code':'复制 python 代码')&&result.includes(en?'Math and code formatting':'公式与代码写法'));
    assert(result.includes('$\\alpha/r$')&&result.includes('```python'));
  }
  console.log('PASS community Markdown: inline/display math, literal code, copy controls, bilingual help, readable invalid formulas, bounded/isolated macros, GFM and unchanged public-content protections');
} finally {
  if(originalStorage===undefined)delete globalThis.localStorage;else globalThis.localStorage=originalStorage;
  await server.close();
}
