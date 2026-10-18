# Private review notifications

The deferred inbox adds a typed `review` row with a system sender and bounded literal
feedback. The public interface validates the resource identifier, phase, version,
decision and Unicode text bounds before creating local links. Report outcomes never
contain internal notes or reporter identity. When the author already deleted a reported post, the outcome explicitly says the content is unavailable, rather than claiming that it was kept or newly removed. Candidate feedback records the reviewed
version and links to current submission state. Expanding native details retains its
state across locale changes and does not send a read acknowledgement.

Services advertise literal `inbox_reviews: true`. Compatible clients request
`inbox?summary=2` and `inbox?reviews=1`, whose snapshot confirms `reviews_included: true`.
Captured read/read-all commands opt in with `include_reviews: true`. Legacy summary,
snapshot and read requests exclude reviews, including when a legacy cutoff spans a
hidden review event. Both modes share the existing monotonically increasing event
sequence, pagination and observed watermark. No third inbox tab or separate unread
counter is required.

The private service upgrades the previous event-kind constraint in a serialized
transaction and preserves IDs, read flags, timestamps and the sequence high-water
mark even after deleted events. Migration failure restores the original schema/data.
Notice metadata has no foreign key to removable posts, so removal feedback survives
content deletion. Notifications and review/audit mutations commit together, and exact
review retries do not emit duplicates. Startup does not backfill prior decisions.

`web/scripts/check-review-notifications.mjs` exercises the real panel/bell in
React/JSDOM, literal feedback, safe links, capability fallback, captured reads, lost
responses, newer arrivals, account retirement and adjacent private source/draft state.
Private tests verify actual decisions, SQLite migration/rollback, retry deduplication,
privacy, active-account authorization, strict HTTP projection and legacy/read watermarks.
Build checks keep feedback content/styles deferred and preserve initial JS budgets.
These checks do not establish rendered browser geometry, physical focus, native IME,
clipboard, live mail-provider behavior or production deployment.
