# Custom interview practice batches

The frequent-concept workbench now lets readers select up to six available modules, arrange them with keyboard-accessible move controls, remove individual choices and clear the group. Selection stays local until a practice link is opened; choosing or reordering does not change the route, jump to the page top, write browser storage or mutate report/contributor statistics.

Starting captures the ordered module numbers in the existing bounded `queue` return route, together with the company, role, stage, publication period and learning-state filters. Selected row links use the same group. Existing module navigation, cross-paper next steps, completion returns and sign-in continuations consume that snapshot. Returning restores the group for review or further editing.

The start action uses actual workspace evidence: continue a selected draft, skip completed modules when beginning unfinished work, or review an entirely completed group. Unresolved/failed progress remains neutral and opens the first selected module. No choice awards progress. Missing captured modules stay visible and removable; a partially unavailable group cannot start. Longer legacy navigation snapshots remain accepted by the existing navigation model but require a new selection of at most six before editing here.

Selection is scoped to the current view and source. Changing accounts resets local unsaved choices through the hub's existing account key; public URL snapshots remain explicit navigation context and are evaluated against the next account's own progress. Move, removal and clear operations restore focus to an available control or stable heading, while newer focus choices and navigation take priority. Anonymous/community identity and source statistics are unchanged.

Validation:

- `check-interview-practice-group.mjs` renders actual ranking, hub, module picker and next-step components. It exercises ordered selection, the limit, unknown/long captures, exact source/auth/return routes, real progress states, keyboard focus preservation, account/source changes, unchanged source evidence and bilingual labels, without server writes or selection storage.
- Existing ranking and continuation regressions cover evidence-based ordering, status filters, curated fallbacks and legacy captured batches.
- An isolated real HTTP/session/SQLite flow uses a synthetic local report with maintained catalog modules. The actual account provider and interview hub read persisted source counts; selection/reordering make no requests, the captured group drives module navigation and restores on return, and account changes discard unsaved choices. No production writes.
- Full frontend checks, TypeScript and local/hosted production builds keep the existing loading and size budgets.

DOM and HTTP behavior are verified independently of visual browser inspection; no screenshot review is claimed.
