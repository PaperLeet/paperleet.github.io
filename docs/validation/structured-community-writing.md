# Structural writing in community editors

The shared Markdown editor now offers **Section heading**, **Numbered steps** and
**Block quote** beside its existing seven formatting tools. They are available in
new posts, linked solutions, interview reports, full-page replies, inline card
comments, post edits and reply edits. The controls have Chinese and English
accessible labels and tooltips; every formatting action is a non-submit button.

## Interaction contract

- A partial selection formats its complete lines. Selecting through a newline
  excludes the following untouched line. Empty content gets a localized example.
- Heading converts existing ATX headings to level two. Numbered steps can convert
  unordered or mixed list markers into a sequence starting at one. Blank lines
  between steps or headings remain blank; quote includes its blank paragraphs.
- If every selected nonempty line already has the chosen marker, the action
  removes one level of that format. Extra separator lines isolate a newly styled
  block from surrounding prose. Repeated toggles do not accumulate separators;
  removing a style retains those paragraph boundaries.
- Existing LF and CRLF line endings and literal inline content are preserved.
  Output is bounded by the editor's existing length limit. An over-limit attempt
  leaves the source unchanged and uses the existing formatting error notice.
- **Undo formatting** restores the exact source, including the original spacing,
  backward selection and both textarea scroll offsets. Later typing invalidates
  that undo; a queued restore respects newer focus. The mounted field survives
  Write/Preview/Live preview changes and existing draft persistence is reused.
- Small card editors now use the same 40px format controls as other editors,
  increasing to 44px for coarse pointers. The wrapping toolbar keeps all ten
  tools available on narrow containers. Existing visible keyboard focus remains.

## Verification

`web/scripts/check-markdown-editor.mjs` covers exact transforms, toggles, Unicode,
LF/CRLF, reversed and newline-ended selections, empty examples and length limits.
It renders actual Markdown to confirm H2 headings, ordered steps, one continuous
multi-paragraph quote, surrounding paragraphs outside the block, and matching
article-outline anchors. Mounted React fixtures exercise each new tool across
all seven real authoring surfaces, including a linked solution: scoped draft
writes, preview markup, same field, backward-selection undo, both scroll offsets,
zero publication calls and existing affiliation/publication protections.

The full frontend checks, local and hosted builds, emitted CSS and initial/deferred
asset graph are checked locally. The build gate requires the new tools to remain
deferred. Private integration uses the reviewed public core pin and existing
backend checks. API/schema, accounts, publication receipts, dependencies, runtime,
exercise content and company-frequency evidence do not change.

Browser-rendered layout, hit testing and OS behavior have not been verified. All
release workflows remain manually disabled; this change does not publish or
deploy an application.
