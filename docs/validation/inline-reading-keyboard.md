# Focused inline reading keyboard controls

Expanded post and reply readers now support Escape while keyboard focus is
inside that reader. The current body collapses and returns focus to its surviving
reading toggle, including when the focused code/link control leaves the DOM.
The existing nearest-position handoff retains newer focus and route priority.

The open named region joins the native Tab order and exposes its Escape shortcut.
Closed regions remain outside the Tab order. A small Esc hint appears with focus
inside a reader on non-coarse pointers, and the region has a visible focus ring.
There is no global keyboard listener or focus trap.

Only unmodified, non-repeating Escape is handled. Composing/229 keys, previously
consumed events and embedded inputs, textareas, selects or editable content retain
their behavior. A sibling comment composer does not trigger the reader. A handled
local key does not bubble to close an outer overlay as well. URLs, drafts,
publication, account state and reader content remain unchanged.

## Validation

The mounted post test first fails on the preceding source because Escape leaves
the focused reader open. Coverage then exercises both post and reply variants:
region Tab order/shortcut metadata, region and removed-code-control collapse,
modifier/IME/repeat exclusions, child prevention, editable controls, sibling and
unfocused-reader rejection, outer-event isolation, newer focus/route priority,
and unchanged draft DOM/value/selection direction/both scroll offsets/storage.
Escape also closes the safe original-text fallback after a formatter failure.

Existing rich-reader, collapse-button, community action and comment-draft flows
remain covered. Run the complete frontend suite, local/hosted builds, private
core checks and a genuine session HTTP/temporary SQLite reader/composer story
before publication. The keyboard fixtures dispatch real DOM events to mounted
controls; no browser visual QA or operating-system key automation is claimed.
