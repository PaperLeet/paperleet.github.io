# Compact expanded reply lists

Expanded feed-card comments now offer **Show first 3** beside the expansion
control. Full discussions offer **Show first 20** above and below expanded
replies. Both actions return to the first batch in the current order, public
search and author filter. Short lists have no extra compact control. Expansion
remains available after compaction, including after all replies were displayed.

Compaction leaves the thread and live composer open. The draft, reply target,
native editor and unsent search input remain local and unchanged. Rows beyond
the first batch unmount to release their Markdown work; retained first-batch
rows keep their identity. Explicit compaction clears temporary parent/search
reading-return jumps. Their old expansion cannot subsequently override the
smaller view. It changes neither the route nor the server's complete reply data.

The action is unavailable while a reply editor locks the list. It checks the
current hook visit, route and active connected list/control. After rows commit,
focus and reading position return to the labelled reply region when the action
still owns the user's intent. Newer typing, scrolling or navigation takes
priority. The two footer actions use 44px targets and wrap in narrow layouts;
labels are bilingual and use the existing theme and reduced-motion rules.

Actual React/JSDOM fixtures cover the full discussion and independent inline
cards: exact first-batch cohorts and re-expansion, retained filters and pending
native search selection/scroll, stable composers and draft/caret/two-scroll
state, edit locks, hidden/inert/retired-route guards, newer focus and localized
controls without additional reads, publication or draft-storage writes. This
does not establish physical browser geometry or native operating-system
interaction. No API, schema, dependency or numerical-runtime change is added.
