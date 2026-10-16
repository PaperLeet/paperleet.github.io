# Workspace controls and local keyboard ownership

The workspace layout panel now uses a 16px heading and 14px instructions and
preset labels at the default root size. Its 360px border-box width, wrapping
reset row and explicit focus rings keep the larger bilingual controls readable.
Keyboard keycaps and secondary metadata retain their smaller scale.

Execution settings, result-location notes and result actions use 14px text.
Result actions have a 40px minimum height, with 44px touch targets. Narrow-screen
Run controls retain 44px height, and saved-submission diagnostic text uses 14px
with 1.8 line spacing. Result placement, explicit detail expansion and editor
font preferences keep their existing behavior.

Layout Escape is handled by the focused panel rather than a document keyboard
listener. It closes that panel and returns focus to the trigger. The next Escape
can still exit the workspace's focus mode. Both layers ignore composing/229,
repeat, modified and already-consumed keys. The existing outside-pointer and
keyboard-departure dismissal remain nonmodal.

Keyboard opening and closing use a short-lived focus intent after the DOM commit.
A newer focus choice or route takes priority; the former unguarded animation-frame
opening callback and unconditional selection focus return are removed. No global
keyboard listener is added. The reset explanation now describes equal reading
and code widths, replacing the obsolete results-height claim. The panel's own
Escape hint says it closes this panel.

## Validation

The new mounted fixture fails on the preceding component because composing
Escape dismisses the layout panel. It then covers local Escape ownership,
modifier/repeat/IME/consumed exclusions, ordinary keyboard opening, English and
Chinese continuity, exact control identity, outer-event isolation, newer focus
and route priority, unchanged storage while reading controls, and retained
draft DOM/value/backward selection/both scroll offsets. Explicit layout choices
still persist their existing ratios.

The actual App workspace fixture also exercises the real layout panel and
workspace shortcut hook together: ignored keys retain both panel and focus mode;
one Escape closes the panel, the next restores split view. Existing error
location/edit/rerun, source matching, submissions, result details and practice
return coverage stays intact. Its account/judge boundaries and native textarea
editor are fixtures, so it does not establish live judging, browser visual QA or
operating-system key automation.

Before release, run the complete frontend suite, both local and actual hosted
build profiles, private core checks and the existing genuine account/session
HTTP/temporary SQLite community story. No runtime, exercise, catalog, API or
storage schema is changed by this workspace patch. Required cloud gates and
exact Pages/Oracle delivery remain separate release evidence.
