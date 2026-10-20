# Find companies within an exercise's public sources

An expanded exercise-source panel with more than four companies now provides
a local Chinese/English name filter and default, contributor, latest-report
and company-name ordering. Full-width input uses the existing company-name
matching rules. Counts and report links keep their original evidence values.

Confirmed publication dates are visible on source rows. Latest ordering uses
the report publication date for the requested exercise; unknown or foreign
exercise timestamps remain last. These dates are not interview dates.

The controls remain mounted during publication-period refreshes. Query and
ordering survive a local collapse or period change, while account, exercise
and library-view changes retire them. Typing closes an opened report preview;
clearing a query does not reopen it or fetch report text automatically. Escape
first clears a focused query, then collapses the source panel. A confirmed empty
period remains distinct from a local name filter with no matches.

The real React/JSDOM library fixture covers fifty unopened rows with no source
reads, local Unicode filtering/order without extra reads, exact source/filter
links, publication refresh, preview abortion, keyboard/IME, locale and native
field continuity, stale/hidden/account actions and unchanged private editor
state and draft storage. Normal library return-position recording is retained.
No physical browser layout or native operating-system behavior is claimed.
