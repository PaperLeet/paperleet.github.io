# Homepage registration growth

The hosted homepage shows a small hand-drawn growth card below the other right-column cards. On narrow screens it follows the problem list. It is absent from offline editions and collection pages.

`GET /api/community/registration-growth` returns public daily aggregates, never identifiers, email addresses or profiles. Counts use permanent accounts with a private registration record, exclude test accounts, and include registered accounts regardless of their current enabled status. Additional login emails do not add registrations. Dates are grouped in UTC; missing days have zero new users. Up to 366 daily points are returned, retaining the cumulative total from before that window.

The headline shows the latest total; hover, tap or use the arrow/Home/End keys on the chart to inspect each day's new registrations. The accessible value and native tooltip also include that day's cumulative total. Data refreshes every five minutes while visible and on returning to a page after at least one minute. Failed loads never become a fabricated zero; a failed refresh retains the last series with an update warning and retry action.

The SVG uses exact observed points, with hand-drawn axes and self-hosted Patrick Hand lettering matching the README Star History chart. Font provenance and the SIL Open Font License are in `web/public/fonts/`.

Validation: `node web/scripts/check-user-growth.mjs` from the web directory (use `node scripts/check-user-growth.mjs` there), the normal production build, and backend `tests/test_registration_growth.py`. Browser checks cover light/dark themes, Chinese/English, keyboard selection and a 390px mobile viewport.
