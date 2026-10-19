# Plan practice by difficulty and estimated time

The direct exercise view offers easy/hard-first and short/long-first ordering.
All four choices sort the entire filtered exercise scope before pagination, with
exercise number ascending as the tie-breaker in either direction. Estimated time
uses the same catalog value and 25-minute fallback already displayed by the list.
These estimates and difficulties are editorial metadata, not measured runtimes,
interview frequency, account grades or prerequisites.

The closed `sort` values are `difficulty`, `difficulty-desc`, `duration` and
`duration-desc`. They require the direct `view=modules` scope and are rejected for
paper, company-evidence and course routes. Selecting a new order resets the page
while retaining size and filters; changing to papers or adding a company clears a
module-only order. Existing paper/source sorting, course order and company
contributor ranking keep their established defaults. The paper registry does not
accept module-only ordering.

Library rows, the complete practice picker, next-module traversal, sign-in and
practice returns share the same route and selector. Paging does not truncate the
practice scope. Difficulty/time order and native input identity survive locale
changes. Sorting adds no fetch, preference storage, mutation or numerical grade.

Focus/exit-focus and code-download icons join the existing optional decorative
practice-icon chunk. Same-size aria-hidden placeholders reserve their slots;
labelled native controls, code download and focus/Escape actions work immediately
if the icon chunk is loading or unavailable. Stop/runtime controls stay immediate.
The production graph retains the existing initial-JavaScript budgets.

Actual React/JSDOM App checks cover all four orders through native selects,
page-two practice entry, the entire ordered picker, next module and exact return.
Route/model checks cover closed scopes, auth round trips, locale-independent
metadata ordering and immutable normalization. Existing library/workspace/reading
checks protect progress gates and private source/caret/scroll. Both builds and an
anonymous HTTP/SQLite company-cohort journey retain source evidence. No physical
browser geometry, numerical SDK run or production deployment is claimed.

The optional table headings now share those same native sort choices. In direct
exercise view, Time and Difficulty cycle ascending, descending and number order;
Module cycles title order and number order. The paper list offers title order and
a newest/oldest/number cycle on Year. Company and course headings stay static.
Each button describes its current and next order, marks its selected column and
identifies the controlled result list. Header and select changes share one guarded
callback, reset the page and preserve native header focus through locale changes.
Hidden or stale source views cannot change the route through an old control.

The headings load from the existing optional list-controls module. Loading and
failure keep five plain labels in the same grid, without dead buttons or an extra
error panel in the heading row. Tests hold and reject that real module import,
then verify usable list/search rows and unchanged editor data. At the existing
700px breakpoint, Time/Year stays hidden along with its row metadata; the compact
visible Level label keeps the full Difficulty action description for assistive
technology. CSS uses the existing 44px target and theme tokens. No rendered-device
geometry or native keyboard simulation is claimed by the DOM checks.
