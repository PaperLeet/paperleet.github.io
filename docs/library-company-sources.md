# Exercise-row company sources

Hosted general exercise rows have a separate company-source button beside the main practice link. The control and expanded region are siblings of that link; no interactive controls are nested inside it. The collapsed row retains its existing height. A shared action column aligns with the table heading, including the compact layout.

Only explicitly opened rows request `community/interviews/summary` with the exact `problem_id` and publication period. The panel validates the problem/period cohort before displaying companies, public report counts and contributor counts. These counts describe published community reports, not actual interview frequency. Periods are 30, 90, 365 days and all time. Empty, malformed and failed responses have explicit recovery.

Company links retain the current query, exercise view, page size and other filters, reset page and incompatible metadata sorting, and enter the existing company practice scope. Source links open report browsing with the same problem, company and period. Existing selected-company row actions are retained.

Opening and explicit retry use guarded synchronous focus handoff. Closing restores the launcher when focus was inside the panel; responses never move focus or reopen it. Escape yields to editable controls and composition. Account or complete source-route changes retire requests and panel state. Closing/reopening the same view retains its result; locale changes retain native controls. Optional feature failure leaves practice links usable; the shared company controls retain one retry action instead of repeating the same failure on every row.

Verification covers actual React list/App interactions, fifty lazy rows, native history Back, period cancellation and late responses, stale routes, owner changes, retry/empty states and unchanged private editor contents. The isolated HTTP flow uses anonymous JudgeHandler/InterviewService reads and disposable SQLite with an unchanged whole-database digest. Decorative lock and results icons use the existing optional workspace-icon module. Catalog errors retain an immediate error icon, with all labels, retry, save and stop controls immediate. These checks do not establish physical browser geometry.

## Read report evidence in place

Company cards in the general exercise list now provide an explicit second disclosure for up to three public reports. Only one company's preview is open per exercise. Opening counts alone never fetches report bodies. The report widget reuses strict public feed validation, anonymous affiliation rendering, plain excerpts, expandable reading and independent refresh/retry. Unknown company entries retain the existing full-source link when the public preview protocol cannot validate them.

The new preview scope must match the caller's already validated problem/company/publication cohort and the exact general-library route. Existing selected-company preview rules stay separate and unchanged. Changing period, company, account or route retires pending report reads; closing the outer company panel clears report disclosure. Late responses cannot reopen it or move focus. Inner Escape collapses the report preview first and returns to its launcher, while editable controls and composition keep Escape. Closing counts returns to the outer launcher.

Full reports open in a new tab with their public problem/company/period source hub as the return target. The original library retains its page, filters, sort and editor drafts; local library search and private input are not copied into report API queries or new-tab source context. Shared excerpts do not load remote body images while collapsed.

Actual React/App checks cover independent source reads, one-company selection, nested Escape, locale identity, malformed/empty feeds, duplicate refresh protection, stale/owner/period/closed responses, anonymous affiliation and unchanged private editor contents. The isolated JudgeHandler/InterviewService/SQLite flow additionally checks recent DPO and a 730-day-old anonymous GAE report against exact publication and company filters, then compares the entire database digest.
