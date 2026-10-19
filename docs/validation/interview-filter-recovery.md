# Keep reading after interview filter recovery

Interview list-to-list route changes now preserve the page viewport instead of
running the App's ordinary page-top reset. Entering interviews from another page,
opening a source post, and leaving interviews keep ordinary navigation behavior.
The existing reading-return system still handles source-post returns separately.

Concept learning-state changes and recovery from empty state/difficulty/duration
filters hand focus to the persistent concept result heading before changing the
route. Saved-to-public and empty report recovery focus the persistent report
heading. Search recovery focuses and reveals the existing native search input.
Paging and explicit retries use these same persistent targets. No response-time
focus callback is introduced, so later input or scrolling keeps control when
new reports arrive. Hidden, disconnected and stale-route recovery triggers are
rejected before navigation or focus changes.

Unsubmitted interview search text remains in its existing input across non-query
filter changes, with selection and horizontal scroll intact. Explicit applied
query changes still update it. Existing account/checking/view keys retire the
input at ownership or view boundaries. No draft, query, source or preference is
added to public routes, storage or API fields.

Decorative file/lab/discussion icons load only with practice in a shared optional
chunk. An aria-hidden SVG of the same size reserves the slot while loading or if
unavailable; text labels and native controls remain immediately usable. The stop
control and core runtime/error status stay immediate. Build checks require the
icon component to stay deferred and retain the existing initial-JavaScript caps.

Validation first reproduced recovery losing focus through the actual App hash
handler. The real App fixture checks concept state/criteria/search recovery,
report saved/public scope, explicit retry, native input and heading identity,
hidden/stale triggers, no in-list page-top reset, ordinary entry/exit behavior,
and a delayed feed yielding to newer unsubmitted input/caret/scroll. Existing
workspace suites cover source/editor continuity and icon-independent controls;
interview/read-return suites cover filter/auth/history context. An anonymous
HTTP/SQLite flow retains real company/publication cohorts and unchanged database.
No physical-browser geometry, native IME or numerical SDK execution is claimed.
