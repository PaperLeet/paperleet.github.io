# Readable inbox and message controls

The shared 16px body / 14px secondary scale now reaches private-message
composition, inbox previews, permission guidance and all-read/retry controls.
The inbox heading and empty conversation heading use the existing 18px lead
token. Dates and count badges retain their quieter hierarchy.

Most inbox actions provide at least 40px desktop height, with 44px controls for
coarse pointers. Sending remains a 44px primary action. The native message field
uses the 16px body token and an 88px minimum height, while keeping its existing
180px maximum and vertical resize. The inbox title and composer footer can wrap;
the send button stays at the trailing edge when a footer line wraps. On a short
viewport or with a long error, the composer can shrink and scroll inside the
fixed panel so its send/retry controls remain reachable. History keeps its own
scrolling and its existing zero-height acknowledgement guard. Existing
viewport caps, independent history scrolling, list wrapping, keyboard focus
rings and light/dark tokens remain in use.

This is a stylesheet-only behavior change. Capability rollout, snapshot cutoffs,
account boundaries, unread arrivals, idempotent retry, message body/draft/caret/
scroll, sending rules and keyboard ownership use the existing implementation.
No application state, API, persistence field or numerical exercise is changed.

Validation uses the existing complete mounted frontend suite (including inbox,
read-all, message composition, lost acknowledgements and account aborts), both
build profiles, the corresponding private API suite and genuine temporary
account/session HTTP community flows. Existing CSS parser/build checks cover the
stylesheet; no implementation-mirroring test is added. Exact cloud and live
delivery gates remain required. Mounted geometry does not establish browser
visual QA; no visual browser or operating-system keyboard automation is claimed.
