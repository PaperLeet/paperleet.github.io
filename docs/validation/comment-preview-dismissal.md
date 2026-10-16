# Focused comment preview dismissal

Escape now closes the focused inline comment preview while reading. A nested
expanded reply owns its first Escape; a second Escape from its surviving reading
toggle can close the preview. Sibling cards remain open. Native selectors and the
entire comment editor retain their own keyboard behavior, including preview and
formatting controls. Composing/229, repeated, modified and consumed keys are
ignored. The listener stays on the existing preview section.

Bottom-button and keyboard dismissal share a guarded focus return after the DOM
commit. A newer focus choice or route takes priority. Otherwise the surviving
comment toggle receives focus without default page scrolling and is revealed
only when necessary, using nearest instant scrolling and the existing 100px
sticky-header clearance. Explicit top-toggle dismissal retains its usual focus.
Existing request cancellation and local editor-view restoration remain in use.
No navigation, new draft field, publication or extra detail read is added by
closing a preview.

Card action and comment expansion/collapse controls have a 40px desktop minimum;
coarse-pointer controls retain at least 44px. Existing font scale, reply contents,
sorting, account boundaries and idempotent publication keep their behavior.

## Validation

The new mounted check first fails on the preceding source because Escape leaves
the focused preview open. It then verifies two-card independence, nested reader
ownership, editor/select/IME/repeat/modifier/consumed exclusions, propagation
isolation, sticky-header focus return, newer focus/route priority, a canceled
genuine component read and late response rejection, localized shortcut metadata,
and unchanged draft storage/publication. Closing releases the editor DOM; reopening
restores the exact Unicode value, backward selection and both scroll offsets
without stealing reading focus.

Run the complete frontend suite, both build profiles, private core checks and the
existing genuine account/session HTTP/temporary SQLite community story extended
with focused dismissal. Layout in the mounted fixture uses simulated geometry.
This does not claim browser visual QA or operating-system key automation.
