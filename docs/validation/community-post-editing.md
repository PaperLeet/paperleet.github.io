# Author post editing

On a compatible hosted application, the author can edit a discussion's title,
Markdown body and category. The existing post keeps its URL, module association,
creation date, replies and reactions. Confirmed saves replace the visible post
immediately, mark it edited and return keyboard focus to the edit button.
Closing the editor also restores focus. Moderation delete permission alone does
not grant editing: the UI requires explicit `viewer.can_edit`, the matching
author identity and a valid opaque `edit_version`. Older backends omit these
fields and therefore show no unsupported edit action.

Edit drafts use a distinct, bounded, versioned session-storage envelope keyed
by account and post. Closing/reopening or refreshing the same tab recovers the
exact title, body, category and original base version. They are separate from
new-post and reply drafts. Blocked storage produces a copy-before-leaving notice;
closing the tab or clearing storage can lose drafts. Discard has a separate
confirmation and does not change the published post. A delayed acknowledgement
can remove only its exact saved draft, preserving newer work.

The edit request sends `expected_version` and `expected_user_id` with the three
editable fields. A conflict retains the draft and refreshes the discussion.
The editor shows the latest title/body/category and disables saving until the
author explicitly reviews that version; it never automatically rebases or merges
text. The author may merge content manually and continue. Another intervening
edit can still conflict again. Failed reads leave the draft available and the
existing discussion retry action works. No automatic forced overwrite is offered.

An error response does not prove the save failed. The matching backend accepts
a retry whose requested content is already current without another text update.
It rejects a stale request for different content. The existing post, reply
composer and account context remain separate throughout this process.

## Verification

- Full frontend checks pass, including new draft bounds, exact Unicode/Markdown,
  copied-envelope account/post isolation, blocked storage, delayed cleanup,
  legacy capability gates, stale recovered drafts and Chinese/English rendering.
- The isolated hosted production build passes original size limits: initial JS
  **402,879 raw / 129,550 gzip bytes**, below 400 KiB / 130 KiB. Concurrent
  uncommitted UI changes were excluded from this build and commit.
- A real browser used a fresh local database, genuine API/SQLite sessions and
  disposable test accounts. A guest saw no edit action; the author could edit.
  Closing the editor and refreshing recovered its unpublished title, category
  and Markdown without changing the live post or its independent reply draft.
- An injected second writer caused HTTP **409**. The old draft stayed visible;
  latest content appeared for review, and saving stayed disabled until reviewed.
  Manually merging and saving returned **200** and the same post URL.
- An injected response loss happened after the real database save. The editor
  kept its draft on **503**; retry returned **200** without another text update.
  The database retained exactly **one post, one reply and one reaction**.
  A later unchanged save also did not update text. Existing reply draft text
  remained intact and the post displayed its edited marker and saved notice.
- The final build restores keyboard focus after both close and save. A real
  390 px iframe opens the editor with document and main scroll widths both
  **390 px**, without horizontal page overflow. Markdown preview continues
  through the existing escaped math/code renderer.

Screenshots and request/database outcome summaries remain in ignored local
`reports/post-edit-*` files. QA controls and disposable credentials are not part
of the application. This iteration does not add reply editing, edit history,
cross-device edit-draft sync or a production deployment. The existing browser
judging runtime, problem catalog and numerical tests are unchanged.
