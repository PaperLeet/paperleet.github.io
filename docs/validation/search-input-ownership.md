# Search input continuity

An Escape event that an enclosing control had already handled still cleared
the shared search input. Retained controls could also clear pending text or
move focus after their page or account no longer owned the action.

The shared search now respects handled events, disabled or hidden surfaces,
closed native disclosures/dialogs and the caller's current ownership check.
Escape clears only the focused active search field. Community, interview,
reply and draft searches pass their existing page/account ownership checks.
Ordinary explicit search, clear, restore and IME composition retain their
existing behavior.

The new real React/JSDOM regression reproduces the handled-Escape failure
before the fix, then verifies event ownership, retained drafts and private
editor source/caret/scroll. Actual community and interview fixtures also retain
their page while hiding or retiring its search controls, checking that those
controls neither navigate nor start reads or steal newer editor focus.
Physical browser layout and native operating-system behavior are not verified
by these fixtures.
