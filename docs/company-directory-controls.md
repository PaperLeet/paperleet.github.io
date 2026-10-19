# Company directory controls

Company cards use a common title/logo rhythm and a two-column experiences/practice action row. Account target controls occupy a separate full-width row. Ordering fields and card actions have44px native targets, themed focus and selected states, wrapping narrow layouts and reduced-motion support. The company-result region stays mounted through empty searches and has a unique ID referenced by the native ordering select.

Changing directory order retains the applied filters, unsent search text, learning state and private editor. Contributor/report ordering remains based on the existing public summary. Missing counts remain unknown; no new ranking metric or company-frequency evidence is introduced.

The interview hub checks canonical route, current account context and connected active surface before changing filters. A directory additionally checks its own subtree before ordering, recovery, report/practice navigation or target actions. Native links retain real destinations and modifier behavior while their current context remains active.

The shared target-company provider checks the current interview context before opening its management panel or refreshing on focus/storage events. Buttons and guest sign-in links reject hidden/inert/retired controls. A retired target summary cannot start child actions. Requests already deliberately initiated retain the existing captured-owner, idempotent retry and confirmed-response behavior.

## Validation

An actual React/JSDOM hub fixture reproduced an old directory order changing the URL from `#/favorites` back to the previous company list. The retained assertion now verifies that ordering and links cannot navigate, fetch or claim focus after route retirement, or when hidden, inert or aria-hidden. It also checks unique result associations and locale-stable native controls.

The account-target fixture checks the same passive guards, native guest links, live add/remove actions, exact lost-response retry, account changes, late responses and private draft/code preservation. An authenticated disposable HTTP/SQLite run verifies only the deliberate target changes persist; community content and practice remain unchanged. These checks do not establish physical-browser geometry, native operating-system interaction or production behavior.
