# OpenReview rendering in Rebuttal Hub

The editor emits OpenReview Markdown with TeX for color, short text sizes and
boxes. Other controls insert headings, quotes, horizontal rules, lists, tables,
links and fenced code. Buttons expose keyboard shortcuts; undo/redo covers both
typing and toolbar operations. Raw HTML and external Markdown images are displayed
as text. MathJax uses OpenReview's `ui/safe` extension.

Preview uses marked 15 (GFM), DOMPurify, then MathJax 3.2.2, with OpenReview's
renderer and MathJax settings. Math is not protected from Markdown parsing.
MathJax, its extensions and fonts are served locally with the app.
Copy, Markdown download, assembly and saved versions preserve the exact source
shown in the editor; they never guess or rewrite existing TeX escapes.

Alphabetic TeX commands such as `\text`, `\rho` and `\times` can use single
backslashes. Markdown consumes one escape layer from doubled backslashes.
Control symbols such as `\{` and `\,`, and matrix row breaks, need Markdown
escaping. The math/style toolbar applies that escaping only when inserting new
TeX. Existing syntax mistakes (including unclosed bold) render as they do in
OpenReview's Markdown-enabled fields.

Format references: [OpenReview TeX support](https://docs.openreview.net/reference/openreview-tex/openreview-tex-support),
[OpenReview Markdown renderer](https://github.com/openreview/openreview-web/blob/master/client/view.js),
and [RebuttalStudio toolbar](https://github.com/runtsang/RebuttalStudio/blob/master/src/renderer/app.js).

Run `npm run check:rebuttal-render` from `web/` to verify Markdown parsing,
MathJax output, source fidelity and preview lifecycle. Production builds copy
MathJax, extensions and fonts to the versioned local vendor directory.
