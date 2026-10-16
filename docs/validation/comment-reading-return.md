# Comment reading returns

Community cards support inline like/undo, direct comment composition, and newest-three previews expanded three at a time. This change makes contextual reading reversible in both those cards and the full discussion.

Selecting an original comment or a confirmed publication exposes one return action at the destination. Returning restores the previous batch, order, author filter, viewport offset and actual originating control. A newer focus, scroll, input or navigation intent cancels pending restoration before it can collapse the view. History is instance-local, account-scoped through the existing component keys, bounded to 12 steps, and cleared on collapse or manual filtering. No new requests, URL changes or stored drafts are created by reading navigation.

An unavailable original control falls back to its surviving row; a deleted origin falls back to the labelled replies section with an explicit notice. Return is disabled during full-discussion editing or account/loading checks. Existing unpublished text, caret and editor scroll survive both parent-context and posted-reply round trips. Anonymous authors keep the existing affiliation and neutral identity presentation.

## Validation

- `node scripts/check-community-card-actions.mjs`: actual React card controls; notice-aware deterministic viewport geometry, newer intent before batch restoration, bounded history, duplicate same-thread cards, publication return, draft/caret/editor scroll and account isolation.
- `node scripts/check-discussion-replies.mjs`: actual full discussion; author-only/newest restoration, active reply-editor lock, confirmed publication, and refreshed deleted destination/origin cases.
- Isolated real React + AccountProvider + HTTP/session + temporary SQLite integration: persisted inline like/undo and publication, original/posted comment round trips, unchanged composer state, collapse/reopen, active account switch and anonymous affiliation. Synthetic local accounts only; no production writes.

DOM tests verify focus, state and deterministic positioning calculations. They are not a browser visual review. The full frontend checks and both local and hosted production builds are also required before release.
