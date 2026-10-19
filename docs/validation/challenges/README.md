# Challenges and company practice validation — 2026-10-19

The local preview used the hosted frontend with a separate test-data account API.
Screenshots show real rendered catalog data, not mocked browser responses:

- `library-calendar.png`: Chinese library, highlighted daily row and compact sidebar month calendar.
- `daily-challenge.png`: Chinese challenge page, primary problem, monthly list, calendar and secondary Rating.
- `daily-dark.png`: matching dark theme.
- `custom-case.png`: real browser/WASM execution using the maintained attention reference, tensor arguments, expected result and Console.
- `company-practice.png`: Amazon filtering, three attributed source entries and related practice links.

Verification:

- Hosted production build and its unchanged 400 KiB raw / 130 KiB gzip initial-JavaScript budget.
- Daily schedule and leap-year calendar helpers; real UI GET/POST contract, enrollment, completion markers, account replacement, anonymous access and compact widget.
- Real Pyodide custom cases: tensors, kwargs, comparisons, exceptions, output bounds; shared browser coordinator cancellation/deadline cleanup and separation from grades.
- Existing account, workspace, problem overview, deferred-feature and library/company navigation checks.
- Private backend daily/practice/social/login tests, plus a real native full attention suite awarded exactly once (1000 → 1012).
- Historical calendar records remain available when more than 90 subsequent days have been enrolled.

This is local validation. No production deployment or real-user enrollment was performed.
The preview’s online-presence counters are unavailable because the local test server
has no presence tracker; they are unrelated to the challenge API.
