# Administrator workspace

The avatar menu opens `#/community/admin` for active administrators on a compatible service. Sections cover overview, bilingual community boards/tags, report review, member status, paper review and an operation log.

Boards/tags have permanent IDs, ordering and active/paused state. Pausing preserves historical posts. Public feeds filter by tag; post/edit/interview drafts and captured publication requests retain up to five tags. New writes require configured active items; existing paused associations can be retained or removed.

Member status changes require an exact username and a reason, revoke sessions, and protect administrator accounts. The list shows public identity and affiliation, never private mailboxes or learner source.

Paper intake approval starts generation. Candidate approval is separate and tied to the exact version/package. Testing, community review and the existing release process still apply. Browser approval does not deploy.

Every workbench write records an actor, reason and before/after state atomically with a versioned receipt. Unconfirmed operations remain in account-scoped tab storage for exact retry. Concurrent changes require explicit fresh review, while unsent form input survives refresh. Report decisions keep their existing evidence/review history. Administrator roles remain server managed.

Controlled React/JSDOM and disposable loopback HTTP/SQLite verification cover behavior; no claims of browser rendering, native focus trapping, IME/clipboard or production deployment.

## Review continuity

The server advertises `admin_review_navigation: true` only to active administrators. It enables protected single-user and single-paper reads and exact current-candidate approval filters. Old servers keep inline list review; unsupported bookmarked detail/filter views offer a compatible list without sending unsupported reads. Candidate filtering happens before pagination and matches the current version and package hash; it does not relax publication requirements.

Bounded local workbench URLs retain user search/status, paper status/approval, audit kind and page (1–501). `item` selects one protected user/paper or opens a configuration item; `create=1` opens a new board/tag. Detail views return to the original filtered page. Sign-in returns accept only this closed grammar. Notes, commands, identities and secrets are never encoded in these URLs.

Edited inputs and disclosure state are held in memory for the current administrator workspace visit, across sections, filters and pages. Leaving, reloading or account revalidation clears this transient state; this is not a durable draft backup. A draft tray returns to each edited object and reset requires explicit confirmation. User-name confirmation and paper decision consent must be re-entered after remount or fresh-version review.

A captured operation keeps its own immutable payload and request ID in owner-scoped tab storage. Acknowledgement marks only the captured visit/key/edit generation; later edits remain unsubmitted even if the row leaves a filter. Retrying an old operation cannot acknowledge a new visit's draft. Resetting inputs does not revoke saved operations or unconfirmed commands.
