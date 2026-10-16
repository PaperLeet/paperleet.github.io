# Reading tab navigation

At the normal desktop split, the English problem tab list needs about 683 px
while the reading pane has about 625 px. Its trailing Discuss link was clipped
without an explicit way to reach it using a pointer.

The existing reading tabs and discussion link now sit in a native horizontal
scroll viewport. Optional deferred controls move that viewport when its content
overflows. Selecting a tab, changing language, or resizing reveals the selected
tab horizontally. Ordinary scrolling preserves the reader's chosen position.
The controller does not own reading selection, source, editor, URL, or storage.
The intrinsic list is compared against the entire strip to avoid a feedback loop
as controls take up space. Controls remain focusable at the edge; when widening
removes a focused control, focus returns to the selected tab without page scroll.
The native tab list and scroll viewport remain available if the optional chunk
cannot load.

Validation:

- `node scripts/check-reading-tabs.mjs`: actual mounted React controller,
  LocaleProvider, native focus and keyboard tab navigation with simulated JSDOM
  geometry. Covers hidden-link access, edge no-ops and focus, programmatic Result
  reveal without editor focus theft, manual scroll intent, resize and locale
  changes, stable tab/editor nodes, unchanged source/caret/editor scroll/URL,
  observer cleanup with reused refs, and a window-resize fallback.
- Existing catalog workbench and mounted App workspace checks pass with the real
  controller exported into the App fixture. The new check is in `npm run check`.
- Full frontend check passes with the project's Python environment. The macOS
  system Python 3.9 is unsuitable for the existing syntax diagnostic check;
  validation uses the supported project Python instead.
- Hosted, offline, and connected browser-only TypeScript/Vite builds retain the
  existing initial-JavaScript budget. The build graph explicitly requires this
  controller to remain deferred from the paper library.
- Actual connected-build browser checks at 1280 px: English overflow controls,
  fully visible Discuss link after pointer scrolling, and the narrow desktop
  split with keyboard End selecting and revealing Solution. Discussion links
  were inspected without publishing any community content. No mobile claim is
  made from these desktop checks. From the narrow Solution view, running the
  existing guest draft reveals Result automatically and passes all 11 GroupNorm
  tests (11 reference checks, 6 mutants). Downloaded UTF-8 source before and after
  the checks and a full reload remains byte-identical at 955 bytes.
