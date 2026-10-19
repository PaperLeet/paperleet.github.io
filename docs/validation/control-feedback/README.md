# Control and feedback polish — 2026-10-19

This pass reviews the existing library, workspace/editor, interview directory, community, account and settings controls. It follows the earlier daily challenge and editor work.

## Corrections

- Backup and achievement siblings had identical owner keys. Opening backups and refreshing account/health state could accumulate multiple native dialogs. The backup now has its own key namespace. An actual App regression exercises five background updates, complete dismissal and reopening.
- Custom JSON validation identifies the function, positional arguments, keyword arguments or expected-result field. Errors are localized, linked with `aria-describedby`, shown beside the field and focused without replacing code. Editing the offending field clears its error. Combined inputs remain limited to 32 KB.
- Custom-case controls explain workspace preparation, another running test, missing functions and runtime preparation. Ctrl/Cmd+Enter inside this panel cannot start the entire module's tests.
- External company practice ignores selections unavailable in the current company scope. It reports visible result counts and provides a clear/reset action with focus return; its attribution text is shorter.
- Search and personal-menu Escape handling excludes IME confirmation, modifier chords and repeated keys. Ordinary Escape retains its local clear/close behavior.
- OAuth sign-in once again explains optional institution verification when the server explicitly supports open publishing, and gives neutral guidance for older servers. Settings explains why backups are temporarily unavailable.
- Shared controls use consistent minimum sizing, preserved invalid-field colors on hover, neutral disabled feedback and short panel fade-ins respecting reduced-motion preferences. Editor styles stay deferred.

## Verification

- All 194 commands in `npm run check` passed across the initial run and its continuation. Python-backed syntax/navigation checks use Python 3.12; the system Python's null-byte exception behavior is incompatible with the existing syntax fixture.
- The final actual App backup/workspace regression passed after the key correction, in addition to search/menu keyboard and custom-field tests.
- Real Pyodide custom-case checks passed: maintained attention reference, tensors, keyword arguments, comparisons, exceptions and bounded Console.
- Hosted production build graph/search checks passed, retaining the existing 400 KiB raw / 130 KiB gzip initial-JS limits.
- Public path-routing production build passed all 166 static route, metadata, no-JavaScript content, asset, sitemap and 404 checks. The later backup-key-only correction is separately covered by the final hosted production build and App regression.
- Browser inspection covers desktop and 390 × 844 layouts, localized validation/focus, filter recovery, settings and native backup dismissal. Test inputs are local; no real account, publication or workspace source was changed.

Screenshots: `custom-case-validation.png`, `custom-case-mobile.png`, `interview-controls.png`, `interview-mobile.png`, `settings-controls.png`.
