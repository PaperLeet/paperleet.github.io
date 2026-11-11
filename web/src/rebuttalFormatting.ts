import { Lexer } from "marked";

export type RebuttalFormat = "bold" | "italic" | "h1" | "h2" | "color" | "small" | "normal" | "large" | "quote" | "hr" | "boxed" | "code" | "table" | "list" | "ordered" | "link" | "math";
export type RebuttalEdit = { value: string; start: number; end: number };
export const REBUTTAL_COLORS = [
  { value: "#3e6775", en: "Blue", zh: "蓝色" },
  { value: "#8c1b13", en: "Red", zh: "红色" },
  { value: "#235789", en: "Deep blue", zh: "深蓝" },
  { value: "#6f428f", en: "Purple", zh: "紫色" },
  { value: "#333333", en: "Black", zh: "黑色" },
] as const;
const MAX_LENGTH = 200_000;
const slash = "\\";

/** Encode newly inserted raw TeX for Markdown. Alphabetic commands need one slash;
 * control symbols and row breaks need escaping before marked consumes the source.
 * Existing Markdown is never passed through this encoder when copying or exporting.
 */
export function openReviewMath(tex: string): string {
  let source = "";
  for (let index = 0; index < tex.length; index++) {
    const character = tex[index], next = tex[index + 1] || "";
    if (character === slash) {
      if (next === slash) { source += slash.repeat(4); index++; }
      else source += /[a-zA-Z]/.test(next) ? slash : slash.repeat(2);
    } else {
      // Prevent Markdown emphasis, HTML/entities, links and table separators in generated math.
      source += /[*_`~<>&[\]|]/.test(character) ? slash + character : character;
    }
  }
  return source;
}

/** Preview, clipboard, download and saved versions all use the same source bytes. */
export function toOpenReviewMarkdown(source: string): string { return source; }

/** Read only text and escape tokens from an existing toolbar expression. The real
 * marked lexer handles both older doubled commands and new single-slash commands;
 * formatted Markdown is not guessed to be raw TeX or silently repaired.
 */
function readMarkdownText(source: string): string | undefined {
  const tokens = Lexer.lexInline(source, { gfm: true, breaks: false });
  if (tokens.some(token => token.type !== "text" && token.type !== "escape")) return;
  return tokens.map(token => token.type === "escape" ? token.raw.slice(1) : token.raw).join("");
}

const textEscapes: Record<string, string> = {
  "\\": "\\\\", "{": "\\{", "}": "\\}", "$": "\\$", "%": "\\%", "#": "\\#", "&": "\\&", "_": "\\_", "^": "\\unicode{94}", "~": "\\unicode{126}",
};
// MathJax's text parser accepts quoted symbols and unicode, whereas the
// textcomp macros used by the older KaTeX preview are math-mode commands.
const textUnescapes: Record<string, string> = {
  ...Object.fromEntries(Object.entries(textEscapes).map(([text, tex]) => [tex, text])),
  "\\textbackslash{}": "\\", "\\textdollar{}": "$", "\\textasciicircum{}": "^", "\\textasciitilde{}": "~",
};
const textEscapePattern = new RegExp(Object.keys(textUnescapes).sort((a, b) => b.length - a.length)
  .map(token => token.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|"), "g");
function escapeText(text: string): string { return text.replace(/[\\{}$%#&_^~]/g, character => textEscapes[character]); }
function unescapeText(text: string): string {
  return text.replace(textEscapePattern, token => textUnescapes[token]);
}
type TextStyle = { text: string; color?: string; size?: "small" | "large"; boxed?: boolean };
function readTextStyle(text: string): TextStyle {
  if (text.startsWith("$") && text.endsWith("$")) {
    const body = readMarkdownText(text.slice(1, -1));
    if (body?.startsWith("{") && body.endsWith("}")) {
      let content = body.slice(1, -1);
      const color = /^\\color\{(#[a-fA-F0-9]{6})\}/.exec(content);
      if (color) content = content.slice(color[0].length);
      const size = /^\\(small|large)/.exec(content);
      if (size) content = content.slice(size[0].length);
      const boxed = content.startsWith("\\boxed{") && content.endsWith("}");
      if (boxed) content = content.slice(7, -1);
      if (content.startsWith("\\text{") && content.endsWith("}")) return { text: unescapeText(content.slice(6, -1)), color: color?.[1], size: size?.[1] as TextStyle["size"], boxed };
    }
  }
  return { text };
}
function styledText(style: TextStyle): string {
  if (!style.color && !style.size && !style.boxed) return style.text;
  let content = `${slash}text{${escapeText(style.text)}}`;
  if (style.boxed) content = `${slash}boxed{${content}}`;
  if (style.size) content = `${slash}${style.size}${content}`;
  if (style.color) content = `${slash}color{${style.color}}${content}`;
  return `$${openReviewMath(`{${content}}`)}$`;
}

export function formatRebuttal(value: string, format: RebuttalFormat, from: number, to: number, color: string = REBUTTAL_COLORS[0].value, locale: "zh" | "en" = "en"): RebuttalEdit {
  if (!Number.isFinite(from) || !Number.isFinite(to)) throw Error("selection");
  let start = Math.max(0, Math.min(value.length, Math.min(from, to)));
  let end = Math.max(0, Math.min(value.length, Math.max(from, to)));
  if (start > 0 && /[\uDC00-\uDFFF]/.test(value[start] || "") && /[\uD800-\uDBFF]/.test(value[start - 1])) start--;
  if (end < value.length && /[\uDC00-\uDFFF]/.test(value[end] || "") && /[\uD800-\uDBFF]/.test(value[end - 1] || "")) end++;
  const eol = value.includes("\r\n") ? "\r\n" : "\n";
  const selected = value.slice(start, end);
  const placeholder = locale === "zh" ? "文字" : "text";
  const replace = (text: string, selectionStart = 0, selectionEnd = text.length, begin = start, finish = end): RebuttalEdit => {
    const next = value.slice(0, begin) + text + value.slice(finish);
    if (next.length > MAX_LENGTH) throw Error("length");
    return { value: next, start: begin + selectionStart, end: begin + selectionEnd };
  };
  const block = (text: string, begin = start, finish = end): RebuttalEdit => {
    const before = value.slice(0, begin), after = value.slice(finish);
    const leading = before && !/(?:\r?\n){2}$/.test(before) ? (before.endsWith("\n") ? eol : eol + eol) : "";
    const trailing = after && !/^(?:\r?\n){2}/.test(after) ? (after.startsWith("\n") || after.startsWith("\r\n") ? eol : eol + eol) : "";
    return replace(leading + text + trailing, leading.length, leading.length + text.length, begin, finish);
  };
  if (format === "bold" || format === "italic") {
    const delimiter = format === "bold" ? "**" : "*";
    if (selected.startsWith(delimiter) && selected.endsWith(delimiter) && selected.length > delimiter.length * 2) return replace(selected.slice(delimiter.length, -delimiter.length));
    if (value.slice(start - delimiter.length, start) === delimiter && value.slice(end, end + delimiter.length) === delimiter) return replace(selected, 0, selected.length, start - delimiter.length, end + delimiter.length);
    const text = selected || placeholder;
    const leading = text.match(/^\s*/)?.[0] || "", trailing = text.match(/\s*$/)?.[0] || "";
    const core = text.slice(leading.length, text.length - trailing.length) || placeholder;
    return replace(`${leading}${delimiter}${core}${delimiter}${trailing}`, leading.length + delimiter.length, leading.length + delimiter.length + core.length);
  }
  if (["color", "small", "normal", "large", "boxed"].includes(format)) {
    const style = readTextStyle(selected || placeholder);
    if (/[\r\n]/.test(style.text) || [...style.text].length > 200) throw Error("short-text");
    if (format === "color") {
      if (!REBUTTAL_COLORS.some(item => item.value === color)) throw Error("color");
      style.color = color;
    } else if (format === "boxed") style.boxed = !style.boxed;
    else style.size = format === "normal" ? undefined : format as "small" | "large";
    return replace(styledText(style));
  }
  if (["quote", "h1", "h2", "list", "ordered"].includes(format)) {
    const lineStart = value.lastIndexOf("\n", start - 1) + 1;
    const lastCharacter = end > start && value[end - 1] === "\n" ? end - (value[end - 2] === "\r" ? 2 : 1) : end;
    let lineEnd = value.indexOf("\n", lastCharacter);
    if (lineEnd < 0) lineEnd = value.length;
    if (value[lineEnd - 1] === "\r") lineEnd--;
    const lines = (value.slice(lineStart, lineEnd) || placeholder).split(/\r?\n/);
    const marker = format === "quote" ? /^>\s?/ : format === "h1" ? /^# / : format === "h2" ? /^## / : format === "list" ? /^[-*+] / : /^\d+[.)] /;
    const remove = lines.filter(line => line.trim()).every(line => marker.test(line));
    let number = 0;
    const text = lines.map(line => {
      if (remove) return line.replace(marker, "");
      if (format === "quote") return line ? `> ${line}` : ">";
      if (!line.trim()) return line;
      if (format === "h1" || format === "h2") return `${format === "h1" ? "#" : "##"} ${line.replace(/^#{1,6}\s+/, "")}`;
      return `${format === "ordered" ? `${++number}.` : "-"} ${line.replace(/^(?:[-*+]|\d+[.)])\s+/, "")}`;
    }).join(eol);
    return block(text, lineStart, lineEnd);
  }
  if (format === "hr") {
    const before = value.slice(0, start);
    const leading = before && !/(?:\r?\n){2}$/.test(before) ? (before.endsWith("\n") ? eol : eol + eol) : "";
    const rule = `${leading}---${eol}${eol}`;
    return replace(rule, rule.length, rule.length, start, start);
  }
  if (format === "table") return block(`| ${locale === "zh" ? "方法" : "Method"} | ${locale === "zh" ? "结果" : "Result"} |${eol}| --- | --- |${eol}| ${selected.replace(/[\r\n]+/g, " ").replace(/\|/g, "\\|") || "Baseline"} | — |${eol}| ${locale === "zh" ? "我们的方法" : "Ours"} | — |`);
  if (format === "code") {
    const text = selected || (locale === "zh" ? "代码" : "code");
    const longest = Math.max(0, ...[...text.matchAll(/`+/g)].map(match => match[0].length));
    if (/[\r\n]/.test(text)) {
      const fence = "`".repeat(Math.max(3, longest + 1));
      const result = block(`${fence}${eol}${text}${eol}${fence}`);
      return { ...result, start: result.start + fence.length + eol.length, end: result.end - fence.length - eol.length };
    }
    const fence = "`".repeat(longest + 1), pad = /^[` ]|[` ]$/.test(text) ? " " : "";
    return replace(`${fence}${pad}${text}${pad}${fence}`, fence.length + pad.length, fence.length + pad.length + text.length);
  }
  if (format === "link") {
    const label = (selected || (locale === "zh" ? "链接文字" : "link text")).replace(/[\\[\]]/g, character => `\\${character}`);
    const prefix = `[${label}](`;
    return replace(`${prefix}https://)`, prefix.length, prefix.length + 8);
  }
  if (format === "math") {
    // The Math action wraps selected raw TeX. Pasted Markdown is already source
    // and must be previewed/copied as-is instead of decoded by a heuristic.
    const tex = selected || "E = mc^2";
    if (/[\r\n]/.test(tex)) {
      return block(`$$${eol}${openReviewMath(tex)}${eol}$$`);
    }
    const text = openReviewMath(tex);
    return replace(`$${text}$`, 1, 1 + text.length);
  }
  return { value, start, end };
}

export function rebuttalShortcut(event: { key: string; code?: string; ctrlKey: boolean; metaKey: boolean; shiftKey: boolean; altKey: boolean; isComposing?: boolean }): RebuttalFormat | "undo" | "redo" | undefined {
  if ((!event.ctrlKey && !event.metaKey) || event.altKey || event.isComposing) return;
  const key = event.key.toLowerCase();
  if (key === "z") return event.shiftKey ? "redo" : "undo";
  if (key === "y" && !event.shiftKey) return "redo";
  if (event.shiftKey) {
    if (event.code === "Digit9" || key === "9" || key === "(") return "quote";
    if (key === "h") return "hr";
    if (key === "b") return "boxed";
    if (key === "c") return "color";
    if (event.code === "Digit1" || key === "1" || key === "!") return "h1";
    if (event.code === "Digit2" || key === "2" || key === "@") return "h2";
    if (event.code === "Equal" || key === "+") return "large";
    if (event.code === "Minus" || key === "_" || key === "-") return "small";
    return;
  }
  if (key === "b") return "bold";
  if (key === "i") return "italic";
  if (key === "k") return "link";
}
