# Continue after an interrupted run

Stopping the browser experiment on module 10 (Flow Matching) previously showed
`Run incomplete` and an error alert. Its Result panel offered no direct retry.
An old cancelled server job could also label a later connection failure as
`Cancelled`, masking the new failure.

Intentional browser stops now have a neutral, bilingual `Stopped` status and
explicitly say that no new result was produced. The Result panel provides Edit
code and a retry of the current editor source for stopped, cancelled or failed
runs. A known server attempt stays a server submission; a browser attempt stays
in the browser. Reconnected cancelled server jobs retain the server retry mode.
Server retry observes account, connection, conflict and required-reload gates.
The existing run token prevents duplicate starts. Notices remain scoped to their
workspace owner and module, and do not create or replace reports. Calibration
and timing from an older report are hidden while a new run is active or has
been interrupted.

Validation:

- The actual mounted App regression first fails on the original `Run incomplete`
  status. It covers neutral cancellation, clearing recovery on a new run,
  retained editor/cursor/scroll, edits made after stopping, duplicate retry,
  browser compatibility failures, server source/mode, offline/reload gates,
  server AbortError versus intentional stop, stale queue cancellation and
  reconnected cancellation without an in-memory latest attempt, plus account
  isolation. Runtime, account and queue boundaries use isolated fixtures.
- `PYTHON=<Python 3.12 environment> npm run check` exercises the complete frontend
  suite. The Python override uses the maintained environment rather than macOS
  system Python 3.9.
- Connected public, hosted and offline production builds retain their initial
  JavaScript budgets and deferred loading boundaries.
- In the actual local offline build, module 10 was run, stopped, translated to
  English and back, and retried directly from Result. Its downloaded source
  remained byte-identical to the maintained starter: 3,875 bytes, SHA-256
  `c54a80031107be0e555ff15d17385d13deb4af1d5949c617ab7f4c422abfce74`.
  This check uses browser CPU/WASM; no hosted submission or account publication
  is needed. Desktop layout is checked at the browser's existing viewport.

The real retry completed all eight original tests: eight reference tests passed
and four mutation cases were caught. The untouched starter correctly reports
3/8 passing tests with `NotImplementedError`; it is never accepted. Preparation
took 1.66 s, first calibration 110.90 s and learner tests 0.32 s in this desktop
browser. Reloading the final build restores the source and completed report.
