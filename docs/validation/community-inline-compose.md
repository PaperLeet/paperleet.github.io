# Community list interactions

Community and interview report cards support likes/undo and comment previews without navigating into the full discussion. Previews load only when opened, show the newest three replies, and expand three more at a time. The full discussion remains available for long replies.

The card action row now also has a direct **Write comment** entry. It opens the existing inline composer and moves focus into the editable field, including from Markdown preview mode. Reusing an open composer keeps the owner/thread draft and avoids another read or publication. A newer focus choice, navigation, or unmount cancels the scheduled focus move. Replying to a previewed comment likewise brings its composer into view. Guests return through the existing scoped sign-in flow.

The existing account-bound mutations, fixed-direction like retries, durable publication receipts, anonymous author identity with affiliation, and shared discussion drafts remain unchanged.

Validation: `node scripts/check-community-card-actions.mjs` exercises the actual React actions, direct composer entry, latest-three expansion/collapse, focus and draft preservation, guest return route, likes/undo/retry, anonymous affiliation, parent reply targeting, uncertain publication recovery, account changes, and late requests. This is DOM interaction verification; no visual browser inspection is claimed.
