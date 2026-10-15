# Author reply editing

Authors can correct an existing reply's Markdown body, preserving its ID,
permalink anchor, parent, thread and creation date. An explicit author edit
capability and opaque version are required; older backends show no edit action.
The reply shows an edited date after a confirmed save, and focus returns to its
edit button. Other members and moderators cannot rewrite an author's words.

Edit drafts are bounded to 10,000 characters and stored separately by account
and reply in tab storage. They survive closing/reopening and refresh when storage
is available, without replacing the new-reply composer draft. Blocked storage
shows a copy-before-leaving notice; tab closure/storage clearing can lose drafts.
Discard requires confirmation and changes no published text. Delayed responses
clear only their exact draft. Deleted replies cannot be recreated by retrying.

Conflicts retain the draft, refresh the discussion and show latest content for
manual review before saving. A save whose success response is lost can be retried
without another update when its content is already current. No automatic merge,
forced overwrite, history viewer or cross-device edit-draft sync is provided.

Full frontend checks and the isolated hosted production build pass, including
draft bounds/isolation, late cleanup, capability gates and stale-draft rendering.
The backend's focused community/HTTP tests pass **88 checks**, including a legacy
reply schema upgrade under concurrent initialization and two simultaneous edits
with exactly one winner. Real browser runs recover the original edit draft after
refresh, retain independent new-reply text, handle 409 review and injected 503
after the real save, then confirm a retry with 200. The original two replies and
parent relationship remain; the retry makes no extra text update. A 390 px iframe
has 390 px document/main scroll widths.

The first browser refresh encountered a failed dynamic import after a QA static
304 response. Refreshing only local Last-Modified metadata forced complete 200
asset responses and recovered the page; asset bytes and application code were
unchanged. This is a verification-environment limitation, not evidence of a
product caching fix. Screenshots and request outcomes are ignored local reports.
No production release or numerical judge change is included.
