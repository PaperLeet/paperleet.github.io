# Company filters across the paper library and practice

The hosted library offers company and publication-period controls (30/90/365 days or all time). A selected company displays ranked exercise rows instead of paper rows. Search, topic, category, difficulty, real module progress and source sorting compose with this scope. Default order follows distinct contributors, then report counts, then module number. These are public community mentions, not measured interview probabilities.

`company` and `company_days` are bounded public navigation fields. The company is a syntactic identifier; unknown catalog identifiers remain explicit errors instead of broadening the list. Default365 is canonicalized away. A period requires a company. Clearing company also clears its period and preserves other filters. Reset clears both. Practice, paper breadcrumbs, navigator and signed sign-in returns retain the exact library destination.

The App owns one deferred public summary per company/period/account-validation generation. It confirms the exact company and period, global company totals, unique and coherent concept evidence and complete current-catalog linkage before recommendations become available. Missing catalog modules require catalog reload. A zero result is shown only after a valid confirmed empty response. Request errors, edition unavailability and account revalidation never become empty or completed states. Ordinary browsing adds no company-filter API request; the existing optional sidebar keeps its current single evidence request. While globally filtered, it reuses the App cohort. Its local company preview has an explicit action to apply that company to the full library.

All practice entry points share the confirmed subset: header resume, collection start, random unattempted exercise, module navigator, paper practice and post-pass next step. Company status is independent of the editor; metadata changes and failures do not disable editing/running, replace the source or steal focus. A current module that leaves the scope remains editable, with an explicit return to the filtered list. Company completion cannot escape to a different paper/course outside the cohort.

Verification:

- `node scripts/check-library-company.mjs`: closed URL/auth round trips; strict coherent and missing cohort contracts; actual App list/search/practice/navigator/post-pass flow; period cancellation, malformed response/retry, account revalidation, empty/clear recovery, one shared summary and exact unchanged private source/DOM/caret/scroll with no judge/social writes.
- With `PAPERLEET_LIBRARY_COMPANY_HTTP`, the same App runs against disposable JudgeHandler/InterviewService/SQLite summaries: recent Google reports link two modules, an older report adds a third only for all time, and Meta has a separate module. Counts, company isolation and zero UI mutations are checked.
- The private service tests bounded library and nested practice returns, malformed/duplicate/private input rejection, and real signed Google/Microsoft HTTP callback continuations. Provider network is controlled.
- Existing library/course, practice, sidebar, account and full UI checks remain in the normal check command. Company metadata, controls and styles remain deferred under the unchanged initial asset/byte budgets.

Scope: source, controlled React/JSDOM, actual loopback HTTP/SQLite and local builds. No browser automation, rendered geometry, real IME/native focus trapping/clipboard, live provider, production data or deployment claim.
