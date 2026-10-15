# Practice recommendations preserve the originating library view

A learner can enter a module through a collection or filtered library. Previously,
only its return link retained that view: the header and after-pass recommendation
still searched the entire catalog. A newer Attention draft could therefore pull
a learner out of Retrieval. The first-practice card also ignored active filters.

The shared `libraryPractice` selector now supplies library rows, first practice,
header start/resume and after-pass next-module candidates. Existing module search,
difficulty/runtime matching, collection membership, retired category/tag
normalization and whole-paper progress semantics are reused. The selected paper's
module order is preserved inside that candidate pool. Direct URLs without a
library context keep their existing global continuation.

When the current scope has no unsolved core modules, the action returns to the
same filtered view and says that this view is complete; it does not declare the
whole catalog finished. Public return URL validation and encoding are unchanged.
Private progress still waits for its existing readiness boundary; no new API,
account, score, source storage or judge behavior is introduced.

## Automated verification

The maintained frontend check now tests the old unscoped BPR-to-Attention escape,
newer outside drafts, inside drafts, completed Retrieval with unfinished global
work, an empty search, module-only bilingual search, category/tag/difficulty and
mixed runtime combinations, whole-paper started/todo transitions, direct URLs,
legacy list links, unchanged source catalog/drafts, exact filter return links,
and rendered filtered start/completion actions. It also covers all seven featured
collections against the shared module matcher for three queries and four
difficulty values. The full frontend check suite passes in an isolated reviewed
source snapshot, excluding concurrent account, UI and browser-draft work.

The standalone production build passes the unchanged initial-graph limits at
401,047 raw / 129,981 gzip bytes. The pure selection helpers are separated from
the UI; no new effect, mutable shared state, network request, or dependency is
introduced. Existing helper exports remain available for their callers.

The hosted build also passes at 406,423 raw / 131,823 gzip bytes. Both remain
inside the original 400 KiB raw / 130 KiB gzip caps. The live worktree's TypeScript
check passes as well, while its concurrent UI/account/runtime-filter changes are
preserved and excluded from this commit. Existing model, catalog, browser
runtime, immutable bundle and Python judge bytes are unchanged; the full native
catalog regression from the preceding numerical correction remains applicable.

## Actual browser flow

An independent static build on localhost was tested through the actual browser:

- An unfinished Attention draft is saved outside Retrieval. The Retrieval view
  filtered to BPR offers BPR from both the first-practice card and header.
- Header start opens module 68 with the exact collection and search continuation.
  A separate log-sigmoid learner passes 11/11 tests, eight wrong implementations
  are calibrated, and all 120 real training updates execute in CPU/WASM.
- The passed result offers **this view is complete**, returning to
  `#/collections/retrieval?q=BPR` with 1/1 saved progress. The header is disabled
  rather than routing to the outside Attention draft.
- Removing the BPR search and revisiting its saved result recommends CLIP, module
  25, inside Retrieval. Following it renders CLIP and retains the collection
  return path. Returning to the global library still resumes the Attention draft.
- English module-only search `LayerNorm and Add & Norm` offers module 28 from the
  header and preserves the reserved ampersand in the return query.
- English completed-view feedback uses the same bounded completion wording.
  A no-match query instead shows **No core modules in this view**, suggests
  adjusting filters and disables start; it never claims completed work.
- At a 390-pixel iframe viewport, the English empty state has document/main width
  390, with no horizontal document overflow. This is responsive iframe QA, not
  a claim about physical mobile hardware.

The browser warning/error log is empty. The static server records 63 requests,
zero API calls and zero POSTs; its only 404 is the disposable QA wrapper's
`/favicon.ico`. Temporary tabs and server are closed, and ignored local evidence
and the completed-scope screenshot are retained. No authenticated account,
server submission, push, production frontend/backend or image deployment occurs.
