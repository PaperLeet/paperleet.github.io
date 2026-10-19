# Explicit search feedback and draft restoration

Shared search fields now show a labelled **Search** action. When the input would
apply a different query, an inline status distinguishes it from the current
results. This status is generic: it never repeats query contents in a live region
and its text stays stable while typing. Inputs equivalent after the existing
normalization do not create false pending feedback.

With a nonempty applied query, **Restore** discards only the unsubmitted edit and
returns focus to the same search field. It does not call the parent's apply
handler, request data, reset pagination or alter matching test rows. Clear and
Escape retain their previous semantics, including clearing an applied filter.
Disabled fields explain that the input is kept; restoration is blocked while
disabled or during composition. Scope changes and owner remounts still retire
old drafts. A submit remains pending until the caller confirms its applied query.

The shared component serves community and solution feeds, interview and company
search, writing drafts, the study concept picker, discussion replies and test
details. Its existing query grammar, Unicode limits, native composition and
legacy Enter guards remain unchanged. No service API or storage schema changes.

Run from `web/`:

```sh
node scripts/check-community-search.mjs
node scripts/check-interview-search.mjs
node scripts/check-report-test-search.mjs
```

The new StrictMode group covers pending/confirmed state, accessible descriptions,
visible localized submit text, callback-free restoration, disabled/composition
gates and owner/scope retirement. Actual interview integration additionally
checks a saved, filtered page-three route without extra requests; actual
ReportView checks preserved matching row identities, expansion, source and
immutable reports. Existing caret, private draft, storage and routing checks run
alongside them. Captions and restoration controls wrap within the field, and
coarse-pointer actions retain at least 44px targets.

These are React/JSDOM and contract checks. Native screen-reader announcements,
physical geometry and OS-level composition behavior need real-device review.
