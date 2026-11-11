import assert from "node:assert/strict";
import React, { act } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { JSDOM } from "jsdom";
import { createServer } from "vite";
import { marked } from "marked";

const dom = new JSDOM('<!doctype html><div id="root"></div>', { url: "https://paperleet.test", pretendToBeVisual: true });
const globals = { window: dom.window, document: dom.window.document, navigator: dom.window.navigator, HTMLElement: dom.window.HTMLElement, HTMLButtonElement: dom.window.HTMLButtonElement, IS_REACT_ACT_ENVIRONMENT: true };
const previous = new Map(Object.keys(globals).map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
for (const [key, value] of Object.entries(globals)) Object.defineProperty(globalThis, key, { value, configurable: true, writable: true });
let copied;
Object.defineProperty(navigator, "clipboard", { value: { writeText: async value => { copied = value; } }, configurable: true });
const server = await createServer({ server: { middlewareMode: true, ws: false }, optimizeDeps: { noDiscovery: true, include: [] }, appType: "custom" });
let root;
try {
  const { formatRebuttal: format, openReviewMath, toOpenReviewMarkdown, rebuttalShortcut, REBUTTAL_COLORS } = await server.ssrLoadModule("/src/rebuttalFormatting.ts");
  const { default: Preview } = await server.ssrLoadModule("/src/RebuttalPreview.tsx");
  const { default: Editor } = await server.ssrLoadModule("/src/RebuttalEditor.tsx");
  const renderPreview = text => new JSDOM(renderToStaticMarkup(React.createElement(Preview, { text }))).window.document;
  const edit = (text, kind) => format(text, kind, 0, text.length);
  const markdownText = source => new JSDOM(marked.parseInline(source, { gfm: true, breaks: false })).window.document.body.textContent;
  for (const [kind, source, expected] of [
    ["bold", "answer", "**answer**"], ["italic", "answer", "*answer*"],
    ["quote", "First\n\nSecond", "> First\n>\n> Second"], ["quote", "> First\n>\n> Second", "First\n\nSecond"],
    ["h1", "## First\nSecond", "# First\n# Second"], ["h2", "First\nSecond", "## First\n## Second"],
    ["ordered", "- First\n- Second", "1. First\n2. Second"], ["list", "1. First\n2. Second", "- First\n- Second"],
  ]) assert.equal(edit(source, kind).value, expected, kind + " applies to complete selected lines");
  assert.equal(format("😀", "bold", 1, 1).value, "**😀**", "Formatting never splits a surrogate pair");
  assert.equal(format("a\nb\nc", "quote", 0, 4).value, "> a\n> b\n\nc", "A selection ending at the next line does not quote that line");
  assert.equal(format("Before\r\nFirst\r\nAfter", "quote", 10, 12).value, "Before\r\n\r\n> First\r\n\r\nAfter");
  const hr = format("Reply", "hr", 0, 0);
  assert.equal(hr.value, "---\n\nReply"); assert.equal(hr.start, 5); assert.equal(hr.end, 5);
  assert.throws(() => format("x", "bold", NaN, 1));
  assert.throws(() => edit("x".repeat(200000), "bold"));
  assert.throws(() => edit("two\nlines", "color"));
  assert.throws(() => edit("x".repeat(201), "boxed"));
  assert.throws(() => format("x", "color", 0, 1, "red;position:fixed"));
  const colored = edit("Key result", "color");
  assert.equal(colored.value, '$' + String.raw`{\color{#3e6775}\text{Key result}}$`, "Generated alphabetic TeX commands use single backslashes");
  const large = edit(colored.value, "large"), boxed = edit(large.value, "boxed");
  assert(large.value.includes(String.raw`\large`) && large.value.includes(String.raw`\color`), "Changing size keeps the existing color");
  assert.equal(edit(boxed.value, "boxed").value, large.value, "Box styling toggles without nesting math delimiters");
  assert.equal(edit(edit("Key result", "small").value, "normal").value, "Key result");
  for (const palette of REBUTTAL_COLORS) {
    const source = format("Key result", "color", 0, 10, palette.value).value;
    assert.equal(source, "$" + String.raw`{\color{` + palette.value + String.raw`}\text{Key result}}$`);
    assert.equal(markdownText(source), source, "Markdown leaves alphabetic commands intact");
  }
  for (const [kind, expected] of [
    ["small", '$' + String.raw`{\small\text{Key result}}$`],
    ["large", '$' + String.raw`{\large\text{Key result}}$`],
    ["boxed", '$' + String.raw`{\boxed{\text{Key result}}}$`],
  ]) assert.equal(edit("Key result", kind).value, expected);
  const oldColored = '$' + String.raw`{\\color{#3e6775}\\text{Key result}}$`;
  assert.equal(edit(oldColored, "large").value, large.value, "Existing doubled toolbar commands are read through marked's escapes");
  const mixedColored = '$' + String.raw`{\\color{#3e6775}\text{Key result}}$`;
  assert.equal(edit(mixedColored, "large").value, large.value, "Mixed single and doubled commands do not rely on a whole-formula guess");
  const special = String.raw`Cost $5, 10% A_B {x} C:\tmp & # ^ ~`;
  for (const text of [special, "**bold** &amp; [link](url) a<b>c", "_first_ and _second_", "a|b"]) {
    assert.equal(edit(edit(text, "small").value, "normal").value, text, "Toolbar styles preserve selected prose and Markdown punctuation when toggled off");
  }
  const legacySpecial = "$" + String.raw`{\\small\\text{Cost \\textdollar{}5, 10\\% A\\_B \\{x\\} C:\\textbackslash{}tmp \\& \\# \\textasciicircum{} \\textasciitilde{}}}$`;
  assert.equal(edit(legacySpecial, "normal").value, special, "Old toolbar textcomp escapes remain editable after switching math engines");
  const specialTex = markdownText(edit(special, "color").value);
  assert(specialTex.includes(String.raw`\$5`) && specialTex.includes(String.raw`C:\\tmp`));
  assert(specialTex.includes(String.raw`\unicode{94}`) && specialTex.includes(String.raw`\unicode{126}`), "Caret and tilde use commands supported inside MathJax text mode");
  assert(!/\\text(?:dollar|backslash|asciicircum|asciitilde)/.test(specialTex), "Generated prose does not use math-only textcomp macros inside text mode");
  const blockDoc = renderPreview([edit("One\nTwo", "quote").value, hr.value, edit("Method", "table").value, edit("Title", "h1").value].join("\n\n"));
  assert.equal(blockDoc.querySelectorAll("blockquote").length, 1); assert.equal(blockDoc.querySelectorAll("hr").length, 1);
  assert.equal(blockDoc.querySelectorAll("th").length, 2); assert.equal(blockDoc.querySelector("h1").textContent, "Title");
  const code = edit("a ``` b\nreturn `x`", "code").value;
  assert(code.startsWith("````\n")); assert.equal(renderPreview(code).querySelector("pre code").textContent.trim(), "a ``` b\nreturn `x`");
  assert.equal(renderPreview(edit("`x`", "code").value).querySelector("code").textContent, "`x`");
  const link = edit("[source]", "link"); assert.equal(link.value.slice(link.start, link.end), "https://");

  const formulas = [
    String.raw`\frac{a}{b}`, String.raw`{\color{#3e6775}\text{Response 1}}`,
    String.raw`\begin{aligned}a&=b\\c&=d\end{aligned}`,
    String.raw`\boxed{\text{Key result}}`, String.raw`\large\text{Label}`, String.raw`\pi\cdot\sin(x)`, String.raw`\{x\}`,
    String.raw`a\,b\;c\!d\|e`, String.raw`\R`, String.raw`a\\b`,
    String.raw`a_b + c_d`, String.raw`\text{a\_b\_c **literal** &amp; <tag>}`, String.raw`\text{Cost \$5}`,
  ];
  for (const formula of formulas) {
    const encoded = openReviewMath(formula);
    assert.equal(markdownText(encoded), formula, "Newly inserted raw TeX survives a real marked pass, including punctuation and row breaks");
    assert.equal(edit(formula, "math").value, `$${encoded}$`, "The Math action treats its selection as raw TeX without guessing an escape convention");
    for (const source of [`$${formula}$`, `$$\n${formula}\n$$`, `$$\n${encoded}\n$$`]) {
      assert.equal(toOpenReviewMarkdown(source), source, "Existing source is unchanged at submission, regardless of its backslash spelling");
    }
  }
  assert.equal(openReviewMath(String.raw`\text{dec}\rho`), String.raw`\text{dec}\rho`);
  assert.equal(openReviewMath(String.raw`\{x\}\,y`), String.raw`\\{x\\}\\,y`, "Control-symbol backslashes are escaped for Markdown");
  assert.equal(openReviewMath(String.raw`a\\b`), String.raw`a\\\\b`, "A raw matrix row break becomes four source backslashes");
  const rawBlock = "\\begin{aligned}\na&=b\\\\\nc&=d\n\\end{aligned}";
  assert.equal(edit(rawBlock, "math").value, `$$\n${openReviewMath(rawBlock)}\n$$`, "Multiline raw TeX uses display delimiters and the same Markdown encoder");
  const literal = String.raw`C:\research\notes`;
  const literalDoc = renderPreview('`$\\\\frac{a}{b}$`\n\n```tex\n$\\\\color{red}x$\n```\n\n' + literal);
  assert(!literalDoc.querySelector("mjx-container")); assert(literalDoc.body.textContent.includes(literal));
  const existingSource = String.raw`Review \\ source. Inline $\frac{a}{b}$.

> $$
> \begin{aligned}a&=b\\c&=d\end{aligned}
> $$
`;
  const literalSource = '`$\\frac{a}{b}$`\n\n```tex\n$\\frac{a}{b}$\n```\n\n' + literal;
  const crlf = '$$\r\n\\begin{aligned}a&=b\\\\c&=d\\end{aligned}\r\n$$';
  for (const source of [existingSource, literalSource, crlf, colored.value, String.raw`$a\\b$`, String.raw`$\\a{x}$`, String.raw`$a\\ b$`, String.raw`$p_{\\text{dec}} + \rho$`, "**80-ep reproduced)*"]) {
    assert.equal(toOpenReviewMarkdown(source), source, "Clipboard, exports and versions preserve all source bytes, including mixed escapes and malformed Markdown");
  }

  const unsafe = '<script>window.pwned=1</script><img src="https://probe.invalid/x" onerror="window.pwned=2"><u>literal</u>\n\n[bad](javascript:alert(1))';
  const unsafeDoc = renderPreview(unsafe);
  assert(!unsafeDoc.querySelector("script,img,[onerror]"));
  assert(![...unsafeDoc.querySelectorAll("a")].some(node => /^javascript:/i.test(node.getAttribute("href") || "")));
  assert(unsafeDoc.body.textContent.includes("<u>literal</u>"), "Raw HTML is displayed literally rather than executed");
  assert(!renderPreview("![image](https://probe.invalid/x)").querySelector("img"), "Preview cannot load arbitrary image resources");

  const keyboard = { key: "b", code: "KeyB", ctrlKey: true, metaKey: false, shiftKey: false, altKey: false };
  assert.equal(rebuttalShortcut(keyboard), "bold");
  assert.equal(rebuttalShortcut({ ...keyboard, shiftKey: true }), "boxed");
  for (const binding of [{ key: "(", code: "Digit9" }, { key: "9", code: "" }, { key: "(", code: "" }]) {
    for (const mod of [{ ctrlKey: true, metaKey: false }, { ctrlKey: false, metaKey: true }]) {
      assert.equal(rebuttalShortcut({ ...keyboard, ...binding, ...mod, shiftKey: true }), "quote");
    }
  }
  assert.equal(rebuttalShortcut({ ...keyboard, key: "Q", code: "KeyQ", metaKey: true, shiftKey: true }), undefined, "The macOS logout shortcut must never be a quote alias");
  assert.equal(rebuttalShortcut({ ...keyboard, isComposing: true }), undefined);
  assert.equal(rebuttalShortcut({ ...keyboard, key: "+", code: "Equal" }), undefined, "Browser zoom is untouched without Shift");
  assert.equal(rebuttalShortcut({ ...keyboard, key: "+", code: "Equal", shiftKey: true }), "large");

  const { createRoot } = await import("react-dom/client");
  let setText;
  function Harness({ readOnly = false, locale = "en" }) {
    const [value, update] = React.useState("First line\nSecond line"); setText = update;
    return React.createElement(Editor, { value, onChange: update, readOnly, locale, actions: React.createElement("button", { type: "button" }, "AI action"), footer: "Footer" });
  }
  root = createRoot(document.querySelector("#root"));
  const render = async props => act(async () => root.render(React.createElement(Harness, props)));
  const named = name => [...document.querySelectorAll("button")].find(node => (node.getAttribute("aria-label") || node.textContent.trim()) === name);
  const click = async name => act(async () => { const button = named(name); assert(button, `Missing ${name}`); button.click(); });
  const key = async (element, options) => act(async () => element.dispatchEvent(new dom.window.KeyboardEvent("keydown", { bubbles: true, cancelable: true, ...options })));
  await render({});
  assert.equal(named("Quote").title, "Quote (Ctrl/⌘+Shift+9)", "The visible shortcut matches the safe binding");
  const field = document.querySelector("textarea"); field.focus(); field.setSelectionRange(0, 5, "backward"); field.scrollTop = 42; field.scrollLeft = 7;
  await click("Bold"); assert.equal(field.value, "**First** line\nSecond line"); assert.equal(document.activeElement, field);
  assert.deepEqual([field.selectionStart, field.selectionEnd, field.selectionDirection, field.scrollTop, field.scrollLeft], [2, 7, "backward", 42, 7]);
  await click("Undo"); assert.equal(field.value, "First line\nSecond line"); assert.equal(field.selectionDirection, "backward");
  await key(field, { key: "z", metaKey: true, shiftKey: true }); assert(field.value.startsWith("**First**"));
  await click("Preview"); assert.equal(document.querySelector("textarea"), field); assert(field.parentElement.hidden);
  assert(document.querySelector(".rbe-rendered strong")); await click("Write"); assert.equal(document.activeElement, field); assert.equal(field.selectionStart, 2); assert.equal(field.scrollTop, 42);
  await key(field, { key: "z", ctrlKey: true }); assert.equal(field.value, "First line\nSecond line");
  field.setSelectionRange(0, field.value.length); await key(field, { key: "(", code: "Digit9", ctrlKey: true, shiftKey: true }); assert.equal(field.value, "> First line\n> Second line");
  await key(field, { key: "z", ctrlKey: true });
  field.setSelectionRange(0, 5); await key(field, { key: "C", metaKey: true, shiftKey: true }); assert(field.value.includes(String.raw`\color`));
  await click("Preview"); await click("Copy"); assert.equal(copied, field.value, "Copy retains the exact OpenReview source, not preview-normalized TeX");
  await click("Write"); const original = field.value;
  await key(field, { key: "b", ctrlKey: true, isComposing: true }); assert.equal(field.value, original);
  named("Bold").focus(); await key(named("Bold"), { key: "ArrowRight" }); assert.equal(document.activeElement, named("Italic"));
  await act(async () => setText("External replacement")); assert(named("Undo").disabled && named("Redo").disabled, "Replacing the document invalidates old undo history");
  field.focus(); field.setSelectionRange(0, 8); await click("Bold");
  await act(async () => {
    Object.getOwnPropertyDescriptor(dom.window.HTMLTextAreaElement.prototype, "value").set.call(field, field.value + " later");
    field.dispatchEvent(new dom.window.Event("input", { bubbles: true }));
  });
  assert(field.value.endsWith(" later")); await key(field, { key: "z", ctrlKey: true }); assert.equal(field.value, "**External** replacement");
  await key(field, { key: "z", ctrlKey: true }); assert.equal(field.value, "External replacement", "Typing and formatting share an ordered undo history");
  await render({ readOnly: true }); assert(field.readOnly); assert(!document.querySelector(".rbe-toolbar")); assert(!named("AI action"));
  await key(field, { key: "b", ctrlKey: true }); assert.equal(field.value, "External replacement");
  await click("Preview"); await click("Copy"); assert.equal(copied, "External replacement");
  await click("Markdown"); assert.equal(document.querySelector("textarea"), field); assert(field.readOnly);
  await act(async () => setText(String.raw`$\frac{a}{b}$`)); await click("Copy"); assert.equal(copied, String.raw`$\frac{a}{b}$`, "Viewer copy preserves exactly the source shown in its preview");
  assert.equal(field.value, String.raw`$\frac{a}{b}$`);
  await render({ locale: "zh" }); assert(named("粗体") && named("分隔线") && named("文字方框"));
  console.log("PASS rebuttal editor: byte-preserving OpenReview export/copy, single-slash toolbar commands, real marked escape round trips, legacy style toggles, Markdown previews, literal HTML, whole-line formatting, Unicode/fences, keyboard shortcuts, selection/scroll preservation, unified undo/redo, external-document reset and read-only editing gates");
} finally {
  if (root) await act(async () => root.unmount());
  await server.close(); dom.window.close();
  for (const [key, descriptor] of previous) { if (descriptor) Object.defineProperty(globalThis, key, descriptor); else delete globalThis[key]; }
}
