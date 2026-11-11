import { Marked, Renderer } from "marked";
import createDOMPurify from "dompurify";

function escapeHtml(text: string): string {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// openreview-web/client/view.js setupMarked. TeX passes through Markdown too.
const renderer = new Renderer();
renderer.html = ({ text }) => escapeHtml(text);
renderer.checkbox = ({ checked }) => checked ? "[x]" : "[ ]";
renderer.image = ({ href, title, text }) => {
  if (href.startsWith("/images/")) {
    const titleAttr = title ? `title="${title}" ` : "";
    const classAttr = href.endsWith("_icon.svg") ? 'class="icon" ' : "";
    return `<img src="${href}" alt="${text}" ${titleAttr}${classAttr}/>`;
  }
  return escapeHtml(`<img src="${href}" alt="${text}" title="${title}">`);
};
const markdown = new Marked({ renderer, gfm: true, breaks: false, async: false });
let owner: Window | undefined;
let purifier: ReturnType<typeof createDOMPurify> | undefined;

export function renderOpenReviewMarkdown(source: string): string {
  // SSR has no browser DOM. Render literal source until hydration can sanitize.
  if (typeof window === "undefined") return escapeHtml(source);
  if (owner !== window) { owner = window; purifier = createDOMPurify(window); }
  if (!purifier?.isSupported) return escapeHtml(source);
  return purifier.sanitize(markdown.parse(source) as string);
}
