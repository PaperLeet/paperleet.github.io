# Exercise-row company sources

Hosted general exercise rows have a separate company-source button beside the main practice link. The control and expanded region are siblings of that link; no interactive controls are nested inside it. The collapsed row retains its existing height. A shared action column aligns with the table heading, including the compact layout.

Only explicitly opened rows request `community/interviews/summary` with the exact `problem_id` and publication period. The panel validates the problem/period cohort before displaying companies, public report counts and contributor counts. These counts describe published community reports, not actual interview frequency. Periods are 30, 90, 365 days and all time. Empty, malformed and failed responses have explicit recovery.

Company links retain the current query, exercise view, page size and other filters, reset page and incompatible metadata sorting, and enter the existing company practice scope. Source links open report browsing with the same problem, company and period. Existing selected-company row actions are retained.

Opening and explicit retry use guarded synchronous focus handoff. Closing restores the launcher when focus was inside the panel; responses never move focus or reopen it. Escape yields to editable controls and composition. Account or complete source-route changes retire requests and panel state. Closing/reopening the same view retains its result; locale changes retain native controls. Optional feature failure leaves practice links usable; the shared company controls retain one retry action instead of repeating the same failure on every row.

Verification covers actual React list/App interactions, fifty lazy rows, native history Back, period cancellation and late responses, stale routes, owner changes, retry/empty states and unchanged private editor contents. The isolated HTTP flow uses anonymous JudgeHandler/InterviewService reads and disposable SQLite with an unchanged whole-database digest. These checks do not establish physical browser geometry.
