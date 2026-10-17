# Saving community posts directly from a list

Community, solution-feed, embedded practice-solution and profile post cards now
offer the existing account-backed bookmark control. Previously a community reader
had to open the post detail to save it. Interview cards keep their existing single
bookmark control. Availability still requires the server capability and an actual
boolean viewer state; old/unknown responses do not invent a saved state.

## Interaction and identity

- The save action changes only after a matching owner/post/boolean confirmation.
  It uses the existing idempotent `active` endpoint and `expected_user_id` guard.
  Lost confirmations retain the original operation direction for retry, with an
  explicit server-state read alternative. Account refresh and old-request abort
  behavior remain in the shared control.
- Saving has its own local confirmation state. Concurrent likes or older comment
  detail metadata cannot replace it. Other card actions and comment writing remain
  usable while a save is pending; duplicate saves cannot be sent. A newer focus
  choice takes precedence over acknowledgement focus repair.
- In a saved feed, a confirmed unsave keeps the article and its comments mounted.
  A status note offers re-saving or **Refresh saved posts**. Explicit refresh uses
  the existing filters, pagination and last-page recovery; the comment draft stays
  in its existing account/post scope. No automatic refresh, redirect or publication
  is performed by a save.
- Feed row lifetimes include the applied request and accepted-result generation.
  Profile activity rows likewise include their view and accepted-result generation.
  Fresh server rows cannot inherit an older local confirmation even
  when a fast response batches away a loading paint. Local save confirmations do
  not change the row key or remount reading/comment editors.
- Guest save links retain the existing public post/sign-in continuation. Self-only
  anonymous profile rows keep their source-free public post routes. Account changes
  retire old card state through the existing account-keyed feed/profile roots.
  Capability withdrawal hides unavailable controls without resending an already
  pending action for the same owner.
- Account confirmation temporarily hides the control while keeping its request
  controller mounted. Hidden/unavailable controls clear focus-restoration intent.
  A completed confirmation or removed retry button cannot reclaim focus from an
  editor outside the control, or from a different current route.

## Presentation and checks

Save joins the existing action row, with an active accent state and an empty-slot
rule for unavailable controls. Recovery messages receive the full action-row width.
The unsave notice is outside the action row and can wrap. Bookmark/recovery/refresh
controls have 40px minimum height and 44px coarse-pointer targets, visible keyboard
focus and existing theme typography/colors. Their styles remain deferred.

`check-feed-bookmarks.mjs` mounts real community, solution, anonymous self-profile
and embedded practice cards. It covers valid/wrong owner/wrong post confirmations,
lost acknowledgement and exact-direction retry/read recovery, concurrent likes,
older comment metadata, explicit unsave/re-save/refresh, filter/account retirement,
guest continuation, feature/state rollout and capability withdrawal. It checks
preserved article/comment DOM, private draft, native practice code, backward
selection, both scroll offsets, source/filter URLs and storage. All recorded writes
are deliberate bookmark/like calls; rendering never probes bookmarks individually.

Existing community discovery/solution, interview saved-state and profile checks
remain in the full frontend chain. Local/hosted typed builds and emitted CSS/import
boundaries are checked; private account persistence/authorization is covered by the
reviewed-core backend tests. API/schema, dependencies, publication receipts, runtime
and content/company-frequency data are unchanged. No CI, tag, production write or
deployment is requested. Browser-rendered layout and hit testing are not verified.
