# Repository ownership and migration

PaperLeet is separating the public local product from the privately operated
online application. The local product owns its frontend and loopback native
judge. The private platform owns hosted UI, accounts, cloud storage, community
and operations. Neither repository should contain live user data or credentials.

## Current and target repositories

| Current | Target | Ownership |
| --- | --- | --- |
| Hongyang-Du/PaperLeet | PaperLeet/paperleet after owner transfer | Public local product and consolidated core |
| packages/stridetorch here | PaperLeet/stridetorch | Independent runtime source and releases |
| PaperLeet/backend | PaperLeet/platform by in-place rename | Private hosted frontend/backend |
| PaperLeet/paperleet.github.io | Same identity | Historical PRs and eventually static deployment |

Target names do not mean that transfer or production cutover has happened.
Keep working clone, submodule and release URLs until replacements are verified.

## Contributions during transition

Public paper packages, tests, browser judging and shared core changes still go
to this development repository until consolidation. The independent local UI
and service are maintained in Hongyang-Du/PaperLeet; its allowlisted snapshot
workflow continues until the starred product becomes the public authority.

Standalone operators, autodiff and runtime work belong in
[PaperLeet/stridetorch](https://github.com/PaperLeet/stridetorch).
Product integration keeps a vendored reviewed snapshot with an immutable commit
and source hashes in `packages/stridetorch/upstream.lock.json`. Use
`python scripts/sync_stridetorch.py --revision FULL_RUNTIME_COMMIT_SHA` to prepare
a source update; it refuses unfinished local runtime changes and does not run
upstream workflows or scripts. Run native and WASM integration checks before
merging. `python scripts/sync_stridetorch.py` checks the current snapshot offline.
This preserves existing standalone installs without adding nested submodules.
Hosted UI changes move into the private platform's
`frontend/`; public contributors need no private access or production data.

## Public contributions reach the online product

The platform pins a public core commit. A private dependency PR adopts reviewed
public contributions, then checks frontend, native judging, reports and account
progress compatibility. Catalogs, test packages, runtime assets and collection
membership come from that revision. Updating core does not overwrite private
frontend source; local UI/CLI changes need not change the hosted UI.

## History and cutover

Transfer the starred repository rather than copying it into a new identity.
Keep this repository for original PR discussions, approvals and checks. Preserve
public development commits on reachable refs when consolidating source, and
link to historical reviews. Do not squash all history into one source snapshot,
force-push existing branches or delete repositories during this transition.

Switch static publication only after the private frontend build is verified.
Preserve immutable assets for cached pages. Production paths, persistent volumes
and user data are separate from GitHub repository transfers and renames.
