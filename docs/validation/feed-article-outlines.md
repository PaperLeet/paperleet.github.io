# Outlines while reading articles in a feed

Expanded community posts, solution-feed cards, profile post cards and interview
reports now reuse the article reader's outline. Their callers supply the actual
post ID. The outline starts folded, including on wide screens, and appears only
when there are at least two usable Markdown headings. Replies retain their
existing compact prose reader. Expanding supplied content does not fetch it again.

Selecting a section focuses its actual heading and closes the compact outline.
The complete post, comments, comment editor and list filters stay in place.
Standalone feeds use page scrolling; embedded workspace readers use the existing
left-pane-only navigation. Scroll/resize tracking identifies the current section
while the outline is open. The existing newer-focus, route and source-revision
guards remain in force. Escape closes only the focused inline reader, returns to
its surviving disclosure and leaves sibling readers and drafts alone.

Feed article anchors combine the validated post ID, an encoded React instance ID
and the Markdown source offset. Repeated headings and duplicate instances of the
same post therefore have distinct renderer-compatible targets, including beside
a standalone article. The optional instance suffix is strictly validated; it is
local DOM state and does not change share URLs or write reading data to storage.
Standalone article anchors retain their existing format.

Inline reader lifetimes are keyed by post identity, variant and exact body.
A committed A→B→A body change retires the earlier expansion instead of silently
reopening it. Locale and unrelated parent updates preserve an unchanged reader.
Comments and their editable drafts remain mounted outside that local lifetime.

Compact outline layout now belongs to `discussion-reading.css`, shared by feed
and practice-solution readers. The three existing solution-only overrides move
there with the same display, static position, spacing and 220px navigation bound.
The practice reader retains its own 24px heading scroll margin; standalone
discussion layout remains unchanged. Wrapping, typography, keyboard focus,
44px coarse-pointer rows and reduced-motion rules stay with the article reader.

## Verification

`check-feed-outlines.mjs` mounts real community solution and interview cards with
an active inline comment draft. It checks native disclosures, duplicate heading
targets and same-post instances beside standalone reading, page/pane navigation,
current-section tracking, local Escape, observer cleanup, new focus/route priority,
locale continuity, post identity and A→B→A retirement. Comment DOM, draft text,
backward selection, both textarea scroll offsets, actions, filters, URL and storage
remain unchanged; no mutation or extra detail request is made by reading.

The existing post-preview fixture also forces the new article-reader branch to
fail, confirming safe original-text fallback, retained sibling input and retry
only on deliberate reopen. Existing solution and standalone reading checks cover
their preserved behavior. The full frontend chain includes the new fixture; both
typed builds, emitted CSS/import boundaries and the private reviewed-core tests
are checked locally. Browser-rendered geometry, hit testing and OS behavior are
not verified. No workflow, tag, production write or deployment is requested.
