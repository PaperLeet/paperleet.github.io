import assert from "node:assert/strict";
import React, { act } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { JSDOM } from "jsdom";
import { createServer } from "vite";

const dom = new JSDOM('<!doctype html><div id="root"></div>', { url: "https://paperleet.test", pretendToBeVisual: true });
const globals = { window: dom.window, document: dom.window.document, navigator: dom.window.navigator, HTMLElement: dom.window.HTMLElement, HTMLButtonElement: dom.window.HTMLButtonElement, IS_REACT_ACT_ENVIRONMENT: true };
const previous = new Map(Object.keys(globals).map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
for (const [key, value] of Object.entries(globals)) Object.defineProperty(globalThis, key, { value, configurable: true, writable: true });
let copied;
Object.defineProperty(navigator, "clipboard", { value: { writeText: async value => { copied = value; } }, configurable: true });
const server = await createServer({ server: { middlewareMode: true, ws: false }, optimizeDeps: { noDiscovery: true, include: [] }, appType: "custom" });
let root;
try {
  const { formatRebuttal: format, openReviewMath, previewRebuttalMath, toOpenReviewMarkdown, rebuttalShortcut, remarkRebuttalMath, REBUTTAL_COLORS } = await server.ssrLoadModule("/src/rebuttalFormatting.ts");
  const { default: Preview } = await server.ssrLoadModule("/src/RebuttalPreview.tsx");
  const { default: Editor } = await server.ssrLoadModule("/src/RebuttalEditor.tsx");
  const renderPreview = text => new JSDOM(renderToStaticMarkup(React.createElement(Preview, { text }))).window.document;
  const edit = (text, kind) => format(text, kind, 0, text.length);
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
  assert(colored.value.includes(String.raw`\\color{#3e6775}`));
  const large = edit(colored.value, "large"), boxed = edit(large.value, "boxed");
  assert(large.value.includes(String.raw`\\large`) && large.value.includes(String.raw`\\color`), "Changing size keeps the existing color");
  assert.equal(edit(boxed.value, "boxed").value, large.value, "Box styling toggles without nesting math delimiters");
  assert.equal(edit(edit("Key result", "small").value, "normal").value, "Key result");
  for (const palette of REBUTTAL_COLORS) {
    const source = format("Key result", "color", 0, 10, palette.value).value;
    const doc = renderPreview(source);
    assert(!doc.querySelector(".katex-error"));
    assert([...doc.querySelectorAll("[style]")].some(node => node.getAttribute("style").includes(palette.value)), "The selected color renders in the actual preview");
  }
  for (const kind of ["small", "large", "boxed"]) {
    const doc = renderPreview(edit("Key result", kind).value);
    assert(doc.querySelector("math")); assert(!doc.querySelector(".katex-error"));
    assert(doc.querySelector("mtext")?.textContent.replace(/\s/g, " ").includes("Key result"));
    if (kind === "boxed") assert(doc.querySelector("menclose") || doc.querySelector(".fbox"), "A box is rendered, not a fake HTML style");
    else assert.equal(doc.querySelector("mstyle[mathsize]").getAttribute("mathsize"), kind === "small" ? "0.85em" : "1.2em", "Preview sizes match MathJax rather than KaTeX's different small size");
  }
  const special = String.raw`Cost $5, 10% A_B {x} C:\tmp & # ^ ~`;
  const specialDoc = renderPreview(edit(special, "color").value);
  assert(!specialDoc.querySelector(".katex-error"), "TeX-special punctuation in selected prose remains valid");
  assert(specialDoc.querySelector("mtext")?.textContent.includes("Cost"));
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
    String.raw`a\,b\;c\!d\|e`, String.raw`\R`,
  ];
  for (const formula of formulas) {
    assert.equal(previewRebuttalMath(formula), formula, "Legacy single-slash TeX remains untouched");
    assert.equal(previewRebuttalMath(openReviewMath(formula)), formula, "OpenReview Markdown escapes decode only inside a formula");
    assert.equal(toOpenReviewMarkdown(`$$\n${formula}\n$$`), `$$\n${openReviewMath(formula)}\n$$`, "Legacy math is escaped at submission without rewriting delimiters or whitespace");
    assert.equal(toOpenReviewMarkdown(`$$\n${openReviewMath(formula)}\n$$`), `$$\n${openReviewMath(formula)}\n$$`, "Already escaped submission source is byte-for-byte stable");
    for (const source of [formula, openReviewMath(formula)]) {
      const doc = renderPreview(`$$\n${source}\n$$`);
      assert(doc.querySelector("math") && !doc.querySelector(".katex-error"));
      assert(!doc.querySelector('mspace[linebreak="newline"]') || formula.includes("aligned"), "Escaped commands cannot become spurious line breaks");
      if (formula.includes("aligned")) assert.equal(doc.querySelectorAll("mtable mtr").length, 2);
    }
  }
  assert.equal(previewRebuttalMath(String.raw`a\\b`), String.raw`a\\b`, "A legacy line break without a command is not mistaken for Markdown escaping");
  assert.equal(previewRebuttalMath(String.raw`\\Huge\\text{Title}`), String.raw`\Large\text{Title}`, "MathJax safe size ceilings are respected");
  const literal = String.raw`C:\research\notes`;
  const tree = { type: "root", children: [{ type: "code", value: String.raw`$\\color{red}x$` }, { type: "text", value: literal }] };
  const unchanged = structuredClone(tree); remarkRebuttalMath()(tree); assert.deepEqual(tree, unchanged);
  const literalDoc = renderPreview('`$\\\\frac{a}{b}$`\n\n```tex\n$\\\\color{red}x$\n```\n\n' + literal);
  assert(!literalDoc.querySelector("math")); assert(literalDoc.body.textContent.includes(literal));
  const legacySource = String.raw`Review \\ source. Inline $\frac{a}{b}$.

> $$
> \begin{aligned}a&=b\\c&=d\end{aligned}
> $$
`;
  const submission = toOpenReviewMarkdown(legacySource);
  assert(submission.includes(String.raw`Review \\ source. Inline $\\frac{a}{b}$`));
  assert(submission.includes(String.raw`> \\begin{aligned}a&=b\\\\c&=d\\end{aligned}`));
  assert.equal(toOpenReviewMarkdown(submission), submission, "Submission normalization is idempotent across quoted math and prose");
  const literalSource = '`$\\frac{a}{b}$`\n\n```tex\n$\\frac{a}{b}$\n```\n\n' + literal;
  assert.equal(toOpenReviewMarkdown(literalSource), literalSource, "Code and ordinary backslashes are never export-normalized");
  const crlf = '$$\r\n\\begin{aligned}a&=b\\\\c&=d\\end{aligned}\r\n$$';
  assert.equal(toOpenReviewMarkdown(crlf), '$$\r\n\\\\begin{aligned}a&=b\\\\\\\\c&=d\\\\end{aligned}\r\n$$');
  assert.equal(toOpenReviewMarkdown(colored.value), colored.value, "Toolbar-generated source needs no second escaping pass");
  for (const source of [String.raw`$a\\b$`, String.raw`$\\a{x}$`, String.raw`$a\\ b$`]) {
    assert.equal(toOpenReviewMarkdown(source), source, "Even-only backslashes are preserved: ambiguous raw line breaks must not cause double escaping");
  }

  const unsafe = '<script>window.pwned=1</script><img src="https://probe.invalid/x" onerror="window.pwned=2"><u>literal</u>\n\n[bad](javascript:alert(1))\n\n' + String.raw`$\htmlClass{injected}{x}$ $\href{javascript:alert(1)}{x}$ $\includegraphics{https://probe.invalid/x}$`;
  const unsafeDoc = renderPreview(unsafe);
  assert(!unsafeDoc.querySelector("script,img,.injected,[onerror]"));
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
  field.setSelectionRange(0, 5); await key(field, { key: "C", metaKey: true, shiftKey: true }); assert(field.value.includes(String.raw`\\color`));
  await click("Preview"); assert(!document.querySelector(".rbe-rendered .katex-error")); await click("Copy"); assert.equal(copied, field.value, "Copy retains the exact OpenReview source, not preview-normalized TeX");
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
  await act(async () => setText(String.raw`$\frac{a}{b}$`)); await click("Copy"); assert.equal(copied, String.raw`$\\frac{a}{b}$`, "Viewer copy also produces valid submission source without mutating the paper");
  assert.equal(field.value, String.raw`$\frac{a}{b}$`);
  await render({ locale: "zh" }); assert(named("粗体") && named("分隔线") && named("文字方框"));
  console.log("PASS rebuttal editor: idempotent AST-only OpenReview export, escaped/legacy math, real color/size/box/table/quote/rule previews, literal HTML and untrusted-math safety, submission-source copy, whole-line formatting, Unicode/fences, keyboard shortcuts, selection/scroll preservation, unified undo/redo, external-document reset and read-only editing gates");
} finally {
  if (root) await act(async () => root.unmount());
  await server.close(); dom.window.close();
  for (const [key, descriptor] of previous) { if (descriptor) Object.defineProperty(globalThis, key, descriptor); else delete globalThis[key]; }
}
