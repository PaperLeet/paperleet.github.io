# Comfortable preparation reading scale

The frequent-concept workbench, selected practice batch and self-test reader still
had small interactive text after the main application typography update. This
stylesheet-only change aligns those deferred surfaces with the existing scale:

- Self-test hint paragraphs and selected concept titles use the 16px body token.
  Hint controls, practice actions, learning filters and explanatory copy use the
  14px small-text token. Counts, provenance details and status labels retain a
  quieter hierarchy.
- Ranking action space grows from 140px to 160px to accommodate readable labels.
  The existing narrow-screen wrapping and full-width batch toggle remain in place.
- Reordering/removing selected concepts uses 36px desktop buttons and 16px icons;
  existing 44px touch controls are retained. Batch titles can shrink and wrap,
  and the narrow-screen controls can wrap within a 132px maximum region.
- Native hints remain explicit disclosures. No application state, account data,
  source statistics, routes, code drafts, exercise content or persistence changes.

Validation uses the existing mounted ranking, batch and preparation-guide flows,
including language changes, exact source/practice returns, reorder/removal focus,
open hint continuity and draft/selection/scroll preservation, followed by both
local and hosted TypeScript/Vite builds. No new implementation-mirroring tests or
browser visual QA are claimed. Font tokens, dark/light colors, reduced-motion
behavior and dynamic-entry budgets remain shared with the application.
