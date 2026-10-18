# Paper registry, duplicate preflight and interview exercise links

The paper registry (`#/paper-registry`) lists the 64 published paper sources (65 catalog entries including one course) and their exercises. It is reachable from the library and hosted paper submission page. Search accepts names, algorithms, authors and versioned arXiv abstract/PDF links; catalog course groupings are not counted as papers. Existing IDs and saved progress are unchanged.

Hosted submission now performs a debounced, read-only `/api/paper-pool/check?arxiv_url=…` lookup while the user edits the URL. It checks both published exercises and non-rejected proposals. Editing or leaving the page cancels stale lookups. Failure offers retry and leaves submission available; the existing POST independently checks duplicates before quota consumption or creation, so the preview does not authorize or bypass intake. Authentication and institutional publishing permissions are unchanged.

Future metadata PRs receive a lightweight **Paper registry identity** workflow using the same catalog source/number validation as publication. A duplicate paper source must reuse its existing paper entry; a new exercise takes a free number against the current main. The workflow does not publish or replace the full release gates.

Interview feed cards and full interview articles use exact paper titles/short names and maintained algorithm aliases to add exercise links. Markdown code, math and existing hyperlinks remain unchanged. Recognized source links (including arXiv versions/PDFs) add their exercise to the deduplicated summary without replacing the original citation. Multiple exercises for one paper link to its first maintained exercise unless a more specific alias applies. Generic tags and fuzzy matches do not become links. The summary initially shows five exercises and expands in place. Links return to the original public article; detection does not edit the post, add structured interview associations, change statistics or send text to an external model.

Validation:
- Complete `npm run check` passed, including the 86-guide/258-prompt preparation catalog. Follow-up matcher/source-link cases and Markdown checks passed after the final additions.
- TypeScript passed. Node 22.23.3 local and production builds stayed within the unchanged startup budgets.
- New frontend checks cover source identity, every registry row, safe Markdown links/anchors, exact and overlapping names, retired/unknown concepts, citation preservation, returned article context, lookup races, retries and unmount cancellation.
- Hosted duplicate/preflight regression suite: 55 passed. GraphSAGE native/report and exact copied-worker integration: 3 passed.
- Chrome local UI verified versioned PDF search, the #100 GraphSAGE destination, inline keywords, retained code/math and expanded exercise summary. The temporary fixture was removed. No production posts, proposals or submissions were created.
