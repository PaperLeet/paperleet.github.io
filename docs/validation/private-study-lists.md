# Private study lists

Saved problems remain a flat bookmark collection. Account-owned named study
lists organize independent ordered sets of exercises; creating, renaming,
reordering or deleting a list never changes bookmarks, code or judged history.
The account menu opens saved problems and lists. The problem-page folder button
opens an optional nonmodal picker; creating a list there can atomically add the
current problem. Existing lists support membership toggles. The manager supports
batch selection, retained naming/selection inputs, explicit deletion, search and
practice-state filters, and accessible up/down ordering when filters are cleared.

`#/favorites?list=<32 lowercase hex ID>` retains the existing optional `q` and
`state` filters. The same canonical context survives practice and sign-in returns.
Within a problem, named-list navigation replaces global module navigation and
labels its **full list order**, even when the return view has display filters.
Unavailable modules can be reordered or removed, but cannot be newly added.
Practice progress comes from the existing owner-validated context, never from a
list API upload. Unknown or failed progress does not become a fabricated grade.

The service must explicitly advertise `capabilities.study_lists: true`. Older
services retain normal bookmarks without sending new list requests. Signed-out
named-list links offer sign-in; a link alone does not disclose another account's
list name or contents. There is no public list publishing or automatic code import.

The public client contract is `studyListStore.ts`. GET returns the current
`owner_id`, monotonically increasing account `revision`, and private lists with
ordered problem identifiers and dates. Every write captures an operation,
expected actor/revision and random request ID. An unconfirmed response retains
the exact command for retry. Matching receipt operation/revision and resulting
state must confirm the change; newer server state is preserved on a replay.
A concurrent revision conflict requires explicit loading of the latest lists,
keeps input, and does not silently reapply an edit. Known rejected operations
return to editing. Only owner IDs/nonces, never list names or contents, enter
cross-tab invalidations. No list-content cache is written to browser storage.

Checks use actual React/JSDOM favorites, picker, manager, progress provider and
practice-navigation components; controlled held/rejected responses exercise
retry/focus/owner transitions. The App action fixture separately verifies named
practice routing and unchanged editor identity/code/caret/scroll. A task-local
loopback HTTP/SQLite bridge runs the same UI flow against disposable sessions
and the private service. Native store and backend checks cover exact receipts,
revision races, malformed payloads, isolation and retained history. Controlled
viewport rectangles test clamping and short-window placement; these checks do
not claim real-browser/device rendering, IME, OS clipboard or provider execution.
Complete build and integration receipts are kept outside the repositories.
