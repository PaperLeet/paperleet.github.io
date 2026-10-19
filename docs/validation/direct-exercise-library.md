# Browse exercises directly from the library

The library now offers a Papers / Exercises switch outside forced course and
company views. The exercise view lists the complete catalog as individual rows,
with each exercise's title, source paper, goal, difficulty, estimated time and
personal state. The same search, category, topic, difficulty, status, density and
20/50/100-row pagination controls remain available.

The public view parameter is closed to papers/modules; default paper views omit
it. Switching views records explicit history, resets only the page and retains
filters, sorting and page size. Reset filters keeps the chosen display view.
Exercise-state filters operate per exercise even when its paper is partly solved.
An exact positive integer query matches exercise numbers in the explicit exercise
view. Default exercise order is numeric; title order uses the selected language;
source year order remains based on the parent paper. Course order and company
contributor ranking retain their existing behavior.

Practice and sign-in continuations retain view, page and page size. The workspace
back link returns directly to the exercise list. Its module picker and previous /
next controls cover the entire filtered list, rather than the current screenful
or source paper. A changed or unconfirmed filtered progress scope leaves the
current editor available and offers a return to the list. Passing-result next
steps also use that same complete scope.

The switch uses named native buttons with pressed states, visible keyboard focus,
40px desktop and 44px coarse-pointer targets. Narrow screens place it on its own
row; theme variables and reduced-motion rules follow existing controls. Stale
route controls cannot switch views. The existing decorative brand SVG is emitted as an external fingerprinted asset
with identical geometry and paint attributes, keeping the initial JavaScript
budget intact. No account field, API, schema, dependency,
judging behavior or persistent display preference is added.

Validation covers closed URL/auth round trips, all 89 catalog exercises exactly
once across pages, numeric lookup, partial-paper progress, both languages, native
focus and control identity, view-history Back, exact practice/list returns,
whole-scope picker and next-module order, course/company boundaries and unchanged
private source/drafts/caret/two-scroll. A disposable anonymous HTTP/SQLite company
flow carries the view through actual report cohorts. React/JSDOM checks do not
establish physical-browser layout, native scrolling or IME behavior.
