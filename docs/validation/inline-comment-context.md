# Inline comment context and navigation

Community and interview previews now show the original author's name and a short Markdown-aware excerpt for replies. Opening that reference expands the required batch and focuses the original comment within the same card. Anonymous identities retain their anonymous label; missing originals display an unavailable state rather than a broken action. Author replies reuse the full discussion's author badge.

Publishing a comment exposes a **View your posted comment** action. This locates the acknowledged row even when timestamp ordering puts it outside the initial three comments, without another request. Retrying an earlier publication still preserves a newer unsent draft, editor instance, selection and scroll position.

The existing `useDiscussionReplies` controller is reused with three-row batches, newest-first ordering and an instance-specific DOM ID prefix. Full discussions retain their existing twenty-row batches, oldest-first ordering, author filter and deep-link behavior. Expanded batches continue keyboard focus at the first new row. Newer focus choices and navigation win over pending moves, and multiple cards for the same thread cannot target each other's rows.

Validation:

- `node scripts/check-community-card-actions.mjs`: parent outside the initial preview, Markdown excerpt without hidden URL destinations, anonymous name protection, author badges, missing parents, posted rows beyond the first batch, unchanged unsent drafts/caret/scroll, navigation/focus priority, unique same-thread card IDs and bilingual context.
- `node scripts/check-discussion-replies.mjs`: full-discussion batching, ordering/filtering, parent/deep-link navigation, edits, publication recovery and private composer preservation.
- Complete frontend checks, TypeScript and local/hosted production builds retain the existing loading boundaries and size budgets.

These are actual React DOM interaction checks. No visual browser inspection is claimed.
