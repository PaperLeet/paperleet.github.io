# Undo a practice batch change

After adding, removing, reordering, clearing, filling or applying a time plan,
the local practice batch offers **Undo** beside its selection feedback. Undo
restores the exact previous problem numbers and order in one step. It can
restore an empty selection or an unavailable captured problem. A restored
unavailable problem keeps the existing catalog-recovery state before practice
can start. Each new meaningful edit replaces the previous undo opportunity;
empty/full no-op actions leave it intact. Undo does not create a redo history.

The snapshot belongs to the current account, filter scope and report/curated
source. Switching any of those retires it, including after returning to an
earlier scope. A normal metadata or learning-progress
refresh keeps the selection history, while completion and estimates are derived
from current data. Undo checks its live hook, route and active connected control.
After activation, focus returns to the batch heading if the disappearing button
still owns focus; newer user focus takes priority. The native button has a 44px
minimum target, themed feedback and a responsive wrapping layout.

This is an in-memory selection operation. Existing practice/share links capture
the resulting public problem order when explicitly used. No API, schema, storage,
grade, contributor evidence or editor content is changed by undo. Actual
React/StrictMode fixtures cover all edit types, queued no-ops/double clicks,
missing concepts, exact links, pending progress, account/filter/source/route
retirement, hidden/inert controls, unmount, locale-stable buttons, budget input
retention and newer editor focus with unchanged source/caret/two-scroll state.
Physical browser geometry and native operating-system interaction remain
unverified.
