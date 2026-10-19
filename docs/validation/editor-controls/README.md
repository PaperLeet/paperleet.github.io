# Editor and controls validation — 2026-10-19

Screenshots are actual local browser renders with an untouched attention starter:

- `editor-default.png`: collapsed tools and no search or navigation inputs.
- `editor-actions.png`: the action disclosure, shortcuts, font size, copy, reset and download.
- `editor-find-replace.png`: Cmd+Option+F, regex capture search and replacement input; no source replacement was performed in this preview.
- `editor-symbols.png`: Cmd+Shift+O with filtered class/function navigation.
- `account-controls.png`: empty test-account form, with aligned fields and primary action.
- `library-controls.png`: consistent search fields and native dropdowns in the library.
- `editor-mobile.png`: 390 × 844 viewport, code pane and compact find controls. The viewport override was reset after this check.

Automated checks passed:

- `check-editor-search.mjs`: actual CodeMirror, matching flags/counts/navigation,
  invalid/no-match/zero-width/bounded regex, capture replacement, undo, read-only,
  editor-only shortcuts, selected-text search, palette exclusivity, focus and run isolation.
- `check-python-navigation.mjs`: Python AST parity over 178 maintained sources,
  partial source handling, Unicode/CRLF, new palette filtering/jumps, source and
  focus ownership, stable editor and caret, live language changes.
- `check-editor-tools.mjs`: actual App, default closed disclosure, Escape and
  outside dismissal, copy/reset/download recovery, font persistence and owner guards.
- Existing workspace actions, UI interactions, layout keyboard ownership,
  navigation intent, catalog workbench, library navigation and account checks.
- Hosted production compile, catalog-math and build-graph checks, with updated
  deferred-editor navigation/search markers and the unchanged JavaScript budget.
- Full static-site production build, including its build-graph and search checks.

The local API's aggregate presence/statistics counters are unavailable, unrelated
to editor behavior. No production deployment or real account write was performed.
