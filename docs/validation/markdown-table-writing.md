# Table authoring in the shared Markdown editor

The native Table disclosure offers two, three or four columns, with two empty
body rows. An explicit preset click inserts ordinary GFM at the textarea's
current selection start, preserving selected text and surrounding source bytes.
The first header is selected for replacement. The existing Undo formatting
control restores the original source, selection direction and both scroll axes;
newer editing retires that undo operation.

Insertion validates the resulting top-level table with the same Markdown parser
as reading. It refuses code, math, HTML, links, headings, lists, quotes and existing
tables rather than silently changing their structure. Length failure leaves the
body intact. UTF-16/native textarea offsets account for emoji and CRLF/lone-CR
line endings without normalizing saved source. Busy, preview, changed owner,
hidden editor, closed disclosure and stale-route actions cannot insert. Native
Escape closes only this picker; composing and modified keys are left alone.
Pending header selection yields to a newer owner focus request or route change.

The feature is shared by posts, solutions, interviews, comments, inline comments
and post/comment editing. It uses existing scoped drafts and ordinary body
publication; no new payload, storage schema, dependency or service permission is
introduced. The disclosure expands within the toolbar so it is not clipped by
the editor frame. Presets have a 40px minimum height, 44px with a coarse pointer.

Checks use the actual React editor and authoring surfaces in JSDOM, plus pure
parser/source tests. The real interview publication fixture inserts a table in
the editor, publishes through a disposable loopback HTTP/SQLite service, retries
an intentionally lost confirmation with the same request, and reads the native
GFM table back. Existing live-preview, writing-section, shortcut, draft and table
reader checks cover adjacent behavior. Build checks include TypeScript, catalog
math, initial asset budgets and static routes. These checks do not establish
physical-browser layout, native popup or physical IME behavior. No production
write, workflow dispatch or numerical SDK execution is part of this change.
