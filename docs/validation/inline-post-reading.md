# In-place full-post reading

Community and interview cards already receive the full published body, but their
bounded summaries required a detail-page jump to read long prose, code, math or
tables. Both lists now offer an explicit local full-post reader. Short plain
posts show their complete summary without an unnecessary control or line clamp.

The existing inline reply reader supplies the same safe Markdown renderer,
per-instance controlled region, focus/scroll handoff and memoized bounded preview.
Post mode has its own full-post labels and keeps the existing 190/210-grapheme
preview limits. Formatting mounts only after opening. Top and bottom collapse
controls remain reachable, keep the list URL and return focus without a long
animated jump. Locale changes keep an opened body; changed text closes stale
content. Titles and explicit detail links retain their original source context.

Each full reader also reuses the existing render boundary. A formatter failure
leaves safely escaped original text and working collapse controls in that reader.
Sibling cards and the parent comment composer remain mounted. Deliberately
closing and reopening retries that renderer. Internal exception messages are not
shown in the product. Existing reply behavior and privacy boundaries are kept.

No new API, persistence schema, background detail read or dependency is added.
The backend's shared post serializer supplies the complete body to both list and
detail reads; this change exposes that already-public content on demand.

## Validation

- The actual experience-card test first fails because no full-post reader exists.
- Mounted preview checks cover code/math/table rendering only on opening, exact
  title/detail returns, native collapse focus, bilingual open-state continuity,
  complete short posts, Unicode limits, changed-body reset and zero reading HTTP
  or storage writes.
- The real Community feed check exercises its reader wiring without extra detail
  requests, route changes or draft writes. Existing sorting, paging and login
  returns remain covered.
- A deliberately failing formatter verifies safe original-text fallback, sibling
  input/card survival, collapse focus and an explicit-only retry.
- Existing full reply, quote/reading return, editor/caret/scroll, account-isolation
  and publication-retry checks remain in place. Complete frontend and both builds
  must pass before publication. Verify actual reader/composer/account/session
  behavior over HTTP and temporary SQLite; production validation stays read-only.
  No browser visual QA is claimed.
