# Private report history

The avatar menu and successful report confirmation open **My reports**. All,
pending and reviewed filters use canonical, share-free account routes, with
20 records per page. Each card shows the title at reporting time, the user's
own reason/details, review status and first review time. Original-post links
preserve the exact filter/page and the existing owner-bound reading return.

New reports by other people can reopen a moderation case without resetting an
earlier user's processed report. Author deletion makes the original post
unavailable; it does not manufacture a review or discard the report record.
Reviewer identities, internal review notes, other reporters and case totals
are absent from this projection. Public profiles and shared post links contain
no private report context.

The deferred page requires an active account and the literal
`community_report_history` capability. Account/route changes abort and retire
old reads. Concurrent focus/visibility refreshes coalesce; there is no polling.
Failed, foreign-owner or partial responses retain the last confirmed page.
Stable report keys retain native expanded details and current focus during
background refresh. Only an explicit retry focuses the page heading.

Validation: `check-report-history.mjs` exercises the actual community route,
25-row pagination, own-only projection, deleted posts, independent review
status, stale reads, owner changes and unchanged private editor contents/caret.
It also accepts a disposable loopback HTTP/cookie fixture for the same UI.
`check-reading-return.mjs` covers exact history filter/page returns through the
existing reading hooks and rejects guest-owned metadata. Build checks require
a unique deferred page and styles under the unchanged initial bundle budget.
Private service/HTTP tests verify authorization, query bounds, first-review
timestamps, pagination and OAuth continuations.

These are source, React/JSDOM and disposable HTTP/SQLite checks. They do not
assert rendered browser layout, native focus trapping or device behavior.
