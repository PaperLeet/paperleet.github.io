# Private account problem favorites

The library heading and avatar menu open `#/favorites`. Problem module cards and the practice toolbar save canonical problem IDs, including when practice uses a numeric route. Bookmark buttons are siblings of module-card links, never interactive children of links.

The dedicated list supports local search, practice-status filtering, refresh, and immediate undo without removing or moving the focused row. Search/status are encoded in bounded return URLs and retained through practice and sign-in. Missing catalog entries can still be removed; catalog failures are reported separately from removed problems. Favorite status does not grant completion or change code, grades, or browser reports.

The account API provides private, server-owned snapshots with monotonic per-owner revisions. Clients share account reads and cross-tab invalidations carry only owner/timing metadata. A stale read/write acknowledgement cannot replace a newer snapshot. Mutation retries retain the original explicit `active` direction. An acknowledgement must confirm both the target ID and desired membership. Session errors refresh account state; a changed owner never receives the previous owner's rendered list. No bookmark content is placed in browser storage or public profiles.

Verification:

- `node scripts/check-problem-favorites.mjs`: actual React page/buttons, read deduplication, undo/focus stability, search, stale responses, distinct concurrent writes, failed reads/writes, unconfirmed acknowledgements, strict owner validation, guest/capability states, cross-tab refresh, account changes, practice/sign-in continuations.
- `node scripts/check-workspace-actions.mjs`: actual App route selection, canonical IDs on numeric routes, exact favorite return filters, unchanged code.
- The complete `npm run check`, production build and immutable startup-size gates remain mandatory.
- The private repository tests the real HTTP/session/SQLite path, actor/origin validation, replay, revisions, independent accounts, concurrent writes, removed entries and zero practice-record writes.

Automated DOM and HTTP verification does not constitute manual browser visual QA. Production verification is read-only; no user content or favorites are created as release probes.
