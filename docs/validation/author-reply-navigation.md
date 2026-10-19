# Author reply reading without losing the discussion

Full discussions and inline comment previews offer an author-reply entry when
public replies contain a matching post-author identity. The entry reads the first
author reply in the selected reply order. Each author row in a multi-reply sequence
has previous/next controls and an ordinal, so readers can follow updates while
keeping the rest of the conversation available. No-author threads offer no dead
entry; a single author reply needs no redundant row navigation.

The public reply author's ID must equal the public post author's ID. Names,
affiliations, viewer accounts and hidden real identities are not used. Existing
server serialization gives an anonymous post author a per-thread alias on their
own replies; the same public equality works without revealing a profile identity.

An explicit jump uses the existing reply reveal/return mechanism. It mounts the
required batch and may broaden a search/filter to expose the requested row. The
reading-return control restores the previous order, applied query, author filter,
batch and origin control when they remain available. Exhausted row bounds remain
focusable with aria-disabled and a no-op handler, allowing an exact return to a
now-exhausted source button. Account, edit and pending-action locks follow each
surface's existing reading policy. Full public discussions remain readable while
an account check is pending; inline comments retain their existing checking lock.

The hook adds no API call, listener, persistence or grade state. Retired render
callbacks, disconnected/cross-region or hidden controls, disabled states and changed URLs
cannot start author navigation. Existing newer-intent checks keep late reading
focus from taking over the learner's editor.

Validation renders the actual discussion and card components under StrictMode.
It covers 53-reply threads, both orders, several batches, ordinary and anonymous
post authors with ambiguous display names, search broadening and exact return,
zero/one-author cases, end bounds, locale continuity, account checking, stale
routes, newer editor focus and unchanged source/draft/caret/two-scroll/storage.
An isolated anonymous HTTP/SQLite fixture checks the real serialized identities
and read-only data path; production data and numerical execution are outside this
wave. JSDOM geometry/focus checks do not establish physical-browser layout or
native scrolling behavior.
