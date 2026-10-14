# Community formulas and code — 2026-10-14

Community posts previously displayed LaTeX dollar delimiters as plain text and had no code-copy control. A shared renderer now covers post bodies, replies, both previews and profile reply activity. The feature uses existing dependencies and preserves the public-content boundary and private draft behavior.

## Behavior

- Inline `$...$` and display formulas render through remark-math and KaTeX, including accessible MathML. Each display `$$` delimiter occupies its own line. Literal inline/fenced code keeps dollar signs as text.
- Fenced code has on-demand highlighting and an accessible copy button. Unknown languages remain readable plain text. Copy preserves code contents, indentation, internal blank lines and Unicode; the parser's final fence newline is omitted. Feedback distinguishes success from denied/unavailable clipboard access and leaves manual selection available.
- Highlighted HTML is associated with the exact source and language. When either changes, the component immediately renders the new escaped source until the new highlighting result arrives; obsolete async work cannot display old code.
- Post and reply editors share expandable formatting examples and named Write/Preview groups. Source stays in the parent-owned draft; preview rendering never replaces the raw text.
- Public math is untrusted: HTML/image/URL math commands remain disabled, macro expansion is capped and size commands are bounded. Macro definitions do not cross formulas/posts/previews/accounts. Malformed equations remain readable without replacing the surrounding discussion.
- Existing Markdown protections remain in force: raw HTML is not activated, unsafe links are filtered, normal external links retain `noopener noreferrer nofollow`, and images show alt text instead of loading remote resources. GFM tables/task lists remain supported.
- Long display math and code use local horizontal scrolling. Memoizing unchanged bodies avoids reparsing every formula when unrelated thread controls update. Math styles and highlighting remain deferred from the homepage.

The KaTeX controls follow its official [options documentation](https://katex.org/docs/options); rendering uses the repository's installed packages, not a newly introduced dependency. The code-copy control uses the browser Clipboard API and handles a rejected promise explicitly.

## Verification

- `cd web && npm run check`: passed, including the new community Markdown suite and all existing account/draft/navigation/judging checks. New checks cover real fraction markup/MathML, inline versus literal code, escaped snippets, copy controls, bilingual examples, invalid/unclosed formulas, malicious math/links/images, bounded recursive macros, cross-render macro isolation, GFM and plaintext fallback before highlighting.
- `cd web && npm run build`: TypeScript and bundle gates passed. Initial JavaScript is **371,175 bytes raw / 118,902 bytes gzip**, across three assets; editor, math, community and highlighting stay deferred. No dependency or Python runtime wheel was added.
- `.venv/bin/python -m pytest tests/test_community.py tests/test_social_api.py -q`: **53 passed**, covering community persistence, permissions, public-profile privacy and HTTP/session integration.
- Real in-app browser, isolated loopback database: a seeded Attention discussion rendered four formulas across the post and initial reply, including a full displayed attention equation. Six highlighted lines were observed, and no rendered image element existed in community content.
- The code-copy call carried exactly the expected snippet, including Chinese, an emoji, blank lines, indentation and a literal `<script>` inside a comment. A denied clipboard promise showed the manual-copy message, never Copied; English feedback was also checked.
- Reply preview rendered three formulas and a copyable snippet. Switching back to Write and refreshing retained the exact raw draft. Changing the source produced the new snippet, and a malformed formula stayed readable. Publishing created exactly one reply; its new code and three formulas appeared in the discussion and the author's Replies activity.
- A module post was previewed, restored after refresh, then published through the actual form. The database retained its expected Markdown body (with the server's existing outer-whitespace trimming). Its three formulas and code block rendered, images stayed suppressed, and Back to practice returned to module 22.
- At **390 × 844**, document/client widths both measured **390**. Code blocks were **316** pixels wide; the long equation had a **316**-pixel local viewport for **358** pixels of content. The formula could overflow locally without widening the page. Light/dark mobile views and the desktop discussion were inspected.
- Disposable database state ended with two posts and two replies: the original fixtures plus one form-published post and one form-published reply. During the community workflow, request logs recorded zero native judge submissions and zero HTTP errors. No development or production account/content was used. The temporary server and agent-created tabs were closed after verification, and the temporary database and credential fixture were removed.

The in-app browser automation clipboard did not expose the page's native writes. Exact-payload and permission-denial checks therefore used a **test-only Clipboard API adapter with a visible QA record**, rather than claiming an OS-level clipboard round trip. All other discussion/editor/navigation interactions used the actual built app and API. Additional browser engines and production deployment were not exercised. Browser judging coverage remains 50/60 modules.
