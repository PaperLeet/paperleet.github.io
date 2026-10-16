# Saved-problem practice and return

The saved-problem list retains its selected query and practice status in both the URL and App route state. This also keeps account-menu sign-in returns aligned with the visible list. Its practice links use canonical module IDs, and each row participates in the existing reading-position controller.

A private reading return binds the current account, exact favorite view and original module. The v1 metadata keeps its legacy row field; only favorite views may put a canonical module ID there, and it must equal the recorded practice ID. Guest owners, arbitrary post IDs, unrelated views, extra fields, expired data and mismatched targets are rejected. Only bounded route/position metadata is stored; code, report bodies and credentials are not transported.

Back from practice restores the original link's viewport offset after the return notice is committed. Keyboard activation restores that link's focus; pointer navigation does not take keyboard focus. Continuous next-module/numeric navigation and workspace reloads use the actual catalog-bound practice hook. A newer focus or scroll action while the list loads offers explicit recovery instead of moving the reader. A removed or filtered-out original returns to the list region with a clear problem-specific notice.

Read failures reuse the original bookmark control for loading retries, including a bounded busy state. They never turn a retry into a membership mutation. Mutation retries retain their original save/remove direction. Generic network errors offer retry; account recovery controls are reserved for expired/mismatched sessions. Removing an unavailable problem retains a focusable status button and explains that it cannot currently be saved again. Clearing an unavailable progress filter returns focus to the status selector.

Validation:

- `check-favorite-reading.mjs` renders the actual saved-problem page, shared reading controller and practice-origin hook. It covers keyboard/pointer return, next/numeric navigation, private-code preservation, notice-aware geometry, delayed data, explicit recovery after new user intent, missing rows, filter changes, late actor-bound responses, prevented/modified activation and fresh-module metadata validation.
- `check-problem-favorites.mjs` covers compact read retries, no accidental writes, direction-preserving mutation retries, account-only recovery, retired-entry status/focus, filter URL persistence and progress-filter focus.
- `check-workspace-actions.mjs` renders the actual App to check route-state synchronization and catalog-bound favorite practice rearming without changing code.
- Existing community/interview reading-return and full frontend checks remain mandatory. Production verification is read-only. Automated DOM/HTTP checks are not manual browser visual QA.
