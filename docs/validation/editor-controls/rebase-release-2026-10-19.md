# Latest-main integration and compact editor header

The feature work was rebased onto public main `a8b919a0`, preserving the later GitHub OAuth / shared Turnstile login work (`1dbfd54c`) and the qualified torch 0.81 runtime and study/catalog updates.

- `solution.py` now shares the Code tab and the actions disclosure shares the same header row. The separate file toolbar is gone. Tabs keep their own roving tablist; the menu remains outside it. Menu search and manual-copy recovery still locate the retained editor correctly.
- Actual App tests verify header placement, menu ownership, exact code/caret/scroll retention, copy recovery, reset and deferred-loader failure/retry. Browser inspection confirms the menu opens the scoped Find panel without changing source.
- A regression discovered while integrating newer interview search scoping cleared pending report-search input on sort/page changes. The scope now excludes order and pagination while retaining company, period, saved/public and other evidence filters. Both the actual App recovery test and the dedicated search/caret check pass.
- GitHub OAuth UI tests explicitly cover the shared challenge gate, provider-specific start endpoint, exact practice continuation, one-use token reset, rejected foreign redirect and recoverable provider failure.
- The complete frontend check command list passed in segments; filesystem-capacity interruptions were rerun. Real Pyodide custom cases passed against the rebased source.
- Production build: hosted, path routing, API origin `https://app.paperleet.com`; 166 static routes and existing JS budgets pass. Initial graph: 405,775 raw bytes / 132,605 gzip bytes.
- The complete fresh public build was copied to temporary deployment staging and verified file-for-file: 16,855 files, 557,356,515 bytes; aggregate SHA-256 `d85b907a357558b9a7799e3313f5fa92bf4b3ea7038835db40584b20e6cdf21d`. Historical asset retention then added 8,129 files across 148 generations using the maintained retention script and the current gh-pages generation manifest.

`editor-header-merged.png` shows the compact header. Authentication requests made without a challenge correctly receive a security-check requirement; the existing protection is retained. No real account consent/login or user-data mutation was performed during these checks.
