# Author reply filtering across discussion surfaces

Full discussions and inline comment previews share a native Author only toggle
with the count of public author replies. The server-provided per-thread author
identity is the only match key, including anonymous aliases. Display names,
usernames and affiliations do not infer identity.

The filter intersects the applied comment search and keeps its order. Inline
search help and result totals name the author-only scope while it is selected.
Unsubmitted search text, private comment drafts, reply targets and editor state
are retained. Empty author results offer an explicit Show all comments action;
this keeps the query and returns focus to the stable toggle. Guest users may
read the same public discussion without creating a private editor.

Original-parent navigation can temporarily broaden the reading scope and return
to the previous author filter and trigger. The new toggle checks connected,
active controls and the parent reading context. Hidden/inert surfaces, stale
routes and a locale re-render of a retired feed cannot activate the filter.

The actual React/JSDOM discussion and card fixture covers public and anonymous
threads with identical display names, batching, search intersection, honest
counts, empty recovery, parent navigation/return, sort and locale continuity,
guests and inactive-context guards. It checks zero extra reads or writes and
unchanged URL, drafts, code, caret and both scroll axes. Existing API identity
and anonymous serialization tests remain the service contract; this refinement
adds no schema or API. Source/DOM checks do not establish physical browser
layout, native OS focus or production behavior.
