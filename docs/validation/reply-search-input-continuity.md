# Keep pending search input while reading replies

Temporary reply navigation can broaden an applied search to reveal an author or
parent reply. Previously that applied-query change also reset the search input,
discarding text the reader had typed but had not submitted.

The reply hook now marks only its temporary reading-query changes with a local
revision. CommunitySearch keeps pending input for those changes, leaving the same
native input mounted with its selection and horizontal scroll intact. Returning
restores the earlier applied result filter independently of pending text. Any
newer text typed during the reading excursion remains the current draft. The
existing Restore button still discards pending edits without rerunning the query.

Ordinary caller-confirmed search changes, saved-view restoration, scope changes
and owner remounts retain their reset behavior. The revision adds no draft copy,
return-stack payload, persistent storage, URL field, API call or grade state.
Default CommunitySearch callers do not opt into reading continuity.

Validation first reproduced the input loss against the prior implementation.
Actual StrictMode discussions cover ordinary and anonymous author jumps and
returns, exact pending text/selection/direction/scroll, newer typing, restored
applied results and callback-free Restore. Shared search checks retain existing
submit, normalization, composition-event, disabled, scope and owner behavior.
The anonymous HTTP/SQLite fixture exercises the same actual reading controls and
checks that the entire database is unchanged. These are React/JSDOM and isolated
HTTP checks; they do not establish physical-browser layout or native IME behavior.
