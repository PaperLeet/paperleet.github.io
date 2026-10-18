# Administrator workspace

The avatar menu opens `#/community/admin` for active administrators on a compatible service. Sections cover overview, bilingual community boards/tags, report review, member status, paper review and an operation log.

Boards/tags have permanent IDs, ordering and active/paused state. Pausing preserves historical posts. Public feeds filter by tag; post/edit/interview drafts and captured publication requests retain up to five tags. New writes require configured active items; existing paused associations can be retained or removed.

Member status changes require an exact username and a reason, revoke sessions, and protect administrator accounts. The list shows public identity and affiliation, never private mailboxes or learner source.

Paper intake approval starts generation. Candidate approval is separate and tied to the exact version/package. Testing, community review and the existing release process still apply. Browser approval does not deploy.

Every workbench write records an actor, reason and before/after state atomically with a versioned receipt. Unconfirmed operations remain in account-scoped tab storage for exact retry. Concurrent changes require explicit fresh review, while unsent form input survives refresh. Report decisions keep their existing evidence/review history. Administrator roles remain server managed.

Controlled React/JSDOM and disposable loopback HTTP/SQLite verification cover behavior; no claims of browser rendering, native focus trapping, IME/clipboard or production deployment.
