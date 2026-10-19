# Community recommendations

The community homepage and each board have three independent administrator-selected
slots. Public results are a separate, bounded projection of current public posts;
ordinary feed order, pagination, search, personal filters and interview counts do
not change. The default first discussion page offers a native disclosure. It reads
only after activation, caches the visit snapshot and supports explicit refresh.
Scoped/malformed reads fail visibly, stale responses retire and private practice
input is untouched. Native rendering, geometry and device behavior are not proven
by DOM checks.

Administrator preview reads use the existing public anonymous identity formatter.
Only explicit preview activation changes that snapshot; refreshing the workspace
updates placement state without silently replacing an already-previewed version.
Saving captures the previewed content hash and slot revision, records an internal
reason atomically and uses the existing immutable request receipt. Later input
remains unsubmitted when an earlier request is retried. Selection is not a content
edit or a vote. Changed content requires an explicit new preview and selection;
missing posts, inactive authors and paused boards are omitted. Withdrawing an
unavailable post remains possible and preserves audit history. Duplicate posts in
one placement, malformed commands, stale versions and unauthorized actors fail.

Public output contains at most three title/excerpt/author projections. Anonymous
profiles retain only the existing public affiliation, not account provenance.
Review reasons, administrator identity, private learner state and saved commands
never enter the public projection or placement URLs. New capabilities gate both
interfaces. Older services keep their existing navigation and audit list.

Checks cover actual workbench and community components in React/JSDOM, real
loopback HTTP/session/SQLite writes and reads, lost acknowledgements, audited
withdrawal, external author edits/removal and explicit current-version reapproval.
Mail/identity providers are controlled, and malformed/late responses are injected.
No production account, browser automation, physical focus/IME/clipboard, deployment
or workflow dispatch is part of this verification.

Compact content context identifies ordinary discussions, module discussions,
solutions and interview experiences from existing public post metadata. Current
module links lead to the corresponding discussion/solution list; retired modules
retain their original identity without an actionable practice link. Interview
context contains catalog company, role and stage only, excluding outcome,
location, date and question lists. Administrator preview uses the same projection.
Clients validate the optional discriminated context before display; legacy absent
context preserves the original presentation. Context changes remain subject to
existing content-version reapproval. No extra detail, statistics or catalog reads
are introduced by the context component.

Reading returns now bind a recommendation origin separately from the ordinary
feed row of the same post. Only an actual same-tab, unmodified title activation
on an eligible feed arms that region. The existing owner-scoped, six-hour
metadata record accepts this optional field only for an unfiltered first-page
community/board title; no title, body, author or private code is stored.

A confirmed return opens and reads that recommendation scope, waits for a valid
current response, then uses the existing interaction-aware reading controller.
Keyboard origin restores the exact link; pointer origin does not force focus.
New focus or manual collapse yields an explicit return action, which can reopen
the native disclosure. A new refresh must settle again before restoration. Read
failures remain retryable and cannot masquerade as a missing item. A genuinely
missing selection has its own notice and never falls back to an identical feed
row. Account/scope changes retire pending responses; older records and ordinary
feed return behavior remain covered by the shared reading checks. Geometry in
these tests is controlled DOM geometry, not a native browser layout claim.
