# Numbered community and experience pages

Community feeds, profile activity and interview experiences now offer a bounded
numbered page window. First/last pages remain visible; an ellipsis is a named
five-page jump rather than a decoration. The active page uses the same primary
colors and 44-pixel targets as the library. Selecting the active page performs no
navigation or replacement read. Existing direct entry remains for four or more
pages, with the original literal/full-width, invalid-input and IME behavior.

Page arrows and numbers share Left/Right/Home/End focus traversal with the
library. Moving focus does not fetch; native Enter/Space activates a button.
Text editing, composing/modified keys and disabled/hidden controls keep their
existing behavior. Buttons and jump forms identify the persistent result region
that receives synchronous focus after an explicit page change. Later responses
yield to newer code or composer focus.

The footer is a wrapping bordered surface. Container queries switch narrow
embedded reading panes to previous/current/next controls, independently of the
outer browser width. Viewport queries provide a small-screen fallback. Extremely
large page numbers use the compact indicator and the bounded direct-entry field;
the logical numbered window and existing safe route caps stay bounded. Result
totals remain distinct from the maximum browseable page, including capped feeds.

A regression was reproduced in the actual profile component: after moving from
`#/u/alice?page=2` to `#/favorites`, a retained Next button changed the URL back to
`#/u/alice?page=3`. Each community/interview pager now checks its parent's current
normalized route/actor context and its connected, active surface before navigation. Hidden,
inert, detached and retired-route buttons or jump forms cannot navigate or read.
The existing page/scope/owner keys establish a fresh pager after a real change;
locale changes preserve the same controls and page-entry draft. Equivalent URL
spellings remain navigable rather than being mistaken for a retired view.

Actual React/JSDOM fixtures cover exact feed/profile/experience cohorts,
numbered and gap destinations, result-region associations, focused keyboard
movement without requests, current-page no-op, bilingual DOM continuity,
last partial pages, safe caps, inert and retired-route rejection, all profile
categories and unchanged private code/composer/selection/two-scroll/storage.
The original library regression covers the extracted shared window and keyboard
helper. A separate loopback fixture connects the three actual public list views
to JudgeHandler/social_api and 45 synthetic posts in disposable SQLite, checking
exact cohorts, result focus and rejection of retired controls. The guest account
context and unrelated profile practice panels are controlled. Its UI performs
GETs only; a complete SQLite dump and foreign-key check verify unchanged data.
Fixture seeding uses a temporary setup allowance restored before those reads;
it does not test publishing limits. Production builds check deferred loading and the original initial asset
budgets. This does not claim physical-browser geometry or native input testing,
and changes no API, schema, private account state or numerical runtime.
