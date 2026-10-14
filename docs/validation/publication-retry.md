# Publication retry after a lost success response

Validated on 2026-10-14 against an updated private service and production frontend builds. A baseline regression test sent the same post/request ID twice and obtained two different posts. The previous service ignored the identifier; the failing test was retained and now passes.

The frontend persists a pending request scoped by account, operation and module/post. It includes the reviewed normalized source and a secure random identifier. Retrying the same source reuses that identifier. On a changed draft, the primary action explicitly says Publish as new content, while Retry previous publication sends the original source and retains newer work. Acknowledgements clear only their own pending request and only matching draft source. Existing publication metadata and private drafts are never attached to public content automatically.

The corresponding private API commits content and its owner-scoped receipt atomically. Same-key/same-content retries replay the existing row; changed operation, target or normalized source returns 409. Deleted-resource tombstones prevent old requests from recreating content. Authentication and rate limiting continue to apply. This needs the updated backend; legacy writes without request IDs remain compatible but do not become idempotent.

## Real browser evidence

A disposable fixture used the actual private account/community handlers. After the first successful post and first successful reply had committed to SQLite, the fixture replaced each success response with HTTP 503, simulating response loss. It recorded request identifiers/statuses without logging content or access codes.

- The post draft and pending request survived same-tab reload. After changing the draft, Retry previous publication recovered the original post while the newer draft remained available through Resume draft.
- The reply draft and pending request survived reload. Changing the body and selecting a new reply target exposed the explicit new-content choice. Retrying the original reply recovered its actual row while preserving the new body and target.
- Bob inherited neither Alice's private draft nor her pending request. Returning to Alice restored them.
- After the original reply was confirmed, explicitly publishing the preserved new draft created the intended second reply with its selected parent.
- Final audit: **1 post, 2 replies, 1 targeted reply, 3 receipts**, from **5 community POST attempts**. Identifier-grouped statuses were **[503, 201], [503, 201], [201]**. The retried requests created no duplicates. Native judge POSTs were zero.
- At 390 × 844, the pending-source review, warning and draft controls had document width 390. Source blocks wrap/scroll without horizontal page overflow.

The fixture's response wrapper and build snapshot were corrected before these measured flows; no production service or real user data was touched. Temporary credentials were never printed or committed.

## Checks and limitations

Frontend checks cover persisted identity in a fresh module/document, unchanged-source reuse, explicit new-source IDs, owner/post/parent isolation, late acknowledgement protection, malformed storage, source escaping and storage-blocked in-page retry. Service/API checks cover concurrency, process reinitialization, owner isolation, changed content, removed content, deleted parents, inactive accounts, invalid IDs, legacy clients, old-database initialization and rollback of both content and receipt. A retired module does not invalidate a previously committed post's replay.

Full frontend checks, TypeScript and both builds pass. Initial JS remains within unchanged 400 KiB raw / 130 KiB gzip limits: Pages **391,768 / 125,783 bytes**, hosted **396,066 / 127,026**. Community remains deferred. Numeric algorithms and browser judge bundles were unchanged; the full native curriculum gate was not repeated.

Tab storage is not cross-device persistence. If it is blocked, the current page retains an in-memory request ID and shows that reload/closing recovery is not guaranteed. Clearing storage or closing a tab can lose the pending identity. This feature does not deduplicate intentionally new publication IDs or older clients without an ID, and it does not recover a database restored inconsistently with its receipts. Nothing was pushed or deployed.

A follow-up layout check rendered the same recovery component and repository styles with a long continuous ASCII title. At a 390-pixel viewport, document width stayed 390 and the title wrapped inside its panel. The final change only adds CSS wrapping/min-width protection; both production builds passed again.
