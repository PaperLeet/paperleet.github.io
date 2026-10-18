# Moderation explanations intended for recipients

The capability `moderation_user_messages: true` enables a separate optional command
field `user_message` with at most 500 Unicode characters. When the field is present,
removal requires a nonblank value. Legacy commands omit it and keep their original
fingerprint, response and permission rules. Unknown command keys remain rejected.
The UI separates this explanation from the required internal note, previews its
recipient roles and retires removal consent after visible-text edits. A pending
operation displays its original captured decision/audience/text; newer edits remain
unsent and cannot alter a retry. No note, evidence or recipient identity is auto-filled.

Explanations are stored in additive side tables so the original review and notice
column layouts continue to accept legacy inserts. Review, message, removal, notice
and receipt writes share the originating transaction. Failures roll everything back.
Read-only projections attach the optional field only when a message exists. Old
notice readers ignore the additional field; their established feedback contract is
unchanged. Report history joins the first review whose sequence covers that own
report, so a later report batch never replaces an earlier explanation. No startup
backfill or public-profile/community projection is added.

The deferred native disclosures render literal text and retain open state across
locale changes and refresh. Existing list/error/owner guards apply. Invalid optional
text blocks confirmation instead of rendering a malformed response. Initial JS and
CSS build guards keep this feature deferred without increasing existing budgets.

`check-moderation-user-messages.mjs` exercises actual moderation, reporter history
and inbox components, audience/consent, Unicode bounds, stale-content refresh,
uncertain-response exact retries, newer drafts, legacy capability fallback, missing
source content and private code/caret/draft preservation. Private tests cover real
HTTP authorization, recipient privacy, first-review history, schema compatibility,
legacy commands, exact retries and transaction rollback on either message-table error.
These are source/TypeScript/React-JSDOM/disposable HTTP-SQLite checks, not native
browser layout, physical focus, IME, clipboard, live mail or production verification.
