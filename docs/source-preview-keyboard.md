# Source-preview keyboard and control consistency

Practice evidence, company-scoped library rows, general exercise company panels,
nested report previews and ranked interview evidence now share one keyboard
decision. A plain, unhandled Escape closes the nearest active preview. Input,
textarea, native select and contenteditable descendants retain their own Escape
behavior. Composition, keyCode229, repeat and modifier combinations do not dismiss
the panel. Child code/body readers keep their existing first-Escape behavior;
their prevented or stopped events cannot dismiss an outer preview too.

The practice company selector previously bubbled Escape into its parent panel,
closing the whole experience preview. A regression reproduces that behavior in
the actual component, then checks the retained selector, selected company, route
and request count. Hidden, aria-hidden, inert, disconnected or stale previews
cannot handle this close action. Existing native opening controls receive focus
only when a closing panel actually contains focus; newer editor work stays put.
Refresh and source links retain the same route and visibility checks.

Source toolbars, publication selects, practice preview actions and company-row
actions use consistent44px minimum targets and8px radii. Expanded source toggles
receive an explicit border/color state, with reduced-motion-aware feedback and
the existing theme and typography tokens. No result, query, account, schema,
runtime or grading rule changes. The helper and styles stay deferred with source
reading rather than entering the initial library shell.

Validation uses the actual React/JSDOM practice selectors, App company rows,
general exercise panels and rich inline source readers. It covers field-default
Escape, editing/composition/modifier layers, hidden/inert controls, nearest-panel
closure, opener focus, anonymous affiliation, scoped requests and unchanged
private source/caret/scroll/draft sentinels. These are controlled event/DOM/API
checks, not physical browser layout or operating-system dropdown tests.
