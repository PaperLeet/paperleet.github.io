# Inline comment reading order

Community and interview feed cards share a native newest/oldest comment selector. The first read remains newest-first and three comments; changing order resets only that card to three rows. Show-more continues from the first newly shown row in the selected order. Reopening the same card retains its order.

Sorting uses the loaded thread without an API request or storage write, preserves the current editor node, draft, selection and scroll, and leaves the source URL unchanged. Changing order clears the previous comment-location return chain. Account changes remount the card and reset private state as before. The control is disabled while comments or the account are unconfirmed, after a read error, or for fewer than two comments.

The select has a localized accessible name, a unique per-instance list target, visible keyboard focus, theme tokens, wrapping header layout, and a 44 px touch target on coarse pointers or narrow screens. The complete discussion link remains available. No animation or dependency is introduced.

Validation: maintained React/jsdom tests cover both orders, batching, select focus, the same editor/caret/scroll, unchanged requests and session storage, reset reading returns, retained order after reopening, independent cards, Chinese labels and account checking. An isolated HTTP/session/SQLite harness additionally exercises persisted anonymous affiliation, native sorting, real like/undo, publication and idempotent lost-acknowledgement retry. These are DOM and service checks; browser visual QA was not performed.
