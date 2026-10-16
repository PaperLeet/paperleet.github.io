# Scoped paper practice

Searching `Vaswani Sinusoidal` returns only module 29, but the Transformer detail
page previously recommended module 22. The topbar and detail page could therefore
start different modules from the same library selection.

The App now passes its existing canonical practice selection to paper details.
Both detail-page practice entries and the next-module marker use that selection,
including its search, collection, difficulty and progress bounds. A newer draft
outside the selection cannot replace its recommendation. A solved selected module
offers review even when the rest of the paper is unfinished; empty and loading
selections offer no fabricated next module. The full paper curriculum and progress
remain visible, with manual links for exploring other modules.

Completed courses retain the established, explicitly labeled "Another paper"
continuation into a fresh scope. Ordinary practice and review keep their return
context.
Single-module result and overview counts use singular English labels.

The existing library/navigation check renders Chinese and English paper pages for
scoped fresh/draft/review/empty/loading states, collection/difficulty bounds, full
curriculum availability and completed-course continuation. The actual App check
verifies that its current selection reaches paper details and changes when the
return view changes. No runtime, numerical contracts, catalog data or persistence
formats change.
