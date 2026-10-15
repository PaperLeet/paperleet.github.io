# Discussion and reply links

Discussions expose a copy action; each reply has a local link and a copy action.
Shared URLs include only the current frontend origin/path, thread and optional
reply ID. They strip URL credentials, page query strings, search/feed state,
drafts and account data. Local reply navigation retains valid feed return context.
Only one lowercase 32-hex reply parameter is accepted; duplicate/unknown parameters,
invalid IDs, external and recursive return targets remain rejected.

Opening a reply link focuses and highlights its actual article after loading,
with a single automatic scroll per requested target. Refreshing the discussion
does not repeatedly focus it. Test-account login and account continuation retain
the target. A missing/deleted reply displays an unavailable notice and full-thread
action; other replies and independent unpublished drafts remain available.

Copy actions report the browser Clipboard API result. Rejection exposes a
read-only selectable canonical URL. In the real browser, the write promise
resolved and showed copied feedback, but a separate automation clipboard read
did not confirm the same text; OS clipboard contents are therefore not claimed
as independently verified. The exact URL preparation and denied-copy manual
field are verified. QA controls forcing rejection are not application code.

## Evidence

- Full frontend checks and isolated hosted production build pass. Initial JS is
  **403,474 raw / 129,748 gzip bytes**, within the existing 400 KiB / 130 KiB
  limits. Concurrent uncommitted UI changes were excluded.
- URL tests cover plain/filtered reply links, reserved Unicode, login round trips,
  malformed/duplicate parameters, safe origin/path construction, stripped
  credentials/search tokens, forbidden protocols and existing feed navigation.
- A real browser opens the last of 26 replies as a guest. Its DOM article ID is
  both the focused and highlighted target, within the viewport. Logging in with
  a disposable local account returns to exactly the same query and article.
- An unavailable reply target shows the fallback and keeps an existing new-reply
  draft. Returning to the full thread works; refreshing does not refocus the
  previously located article. No reply, reaction or post is mutated by sharing.
- At 390 px, locating the target and the denied-copy manual field keep viewport,
  document and main widths at **390 px**. Keyboard activation is verified.

The matching private backend also expands the bounded encoded OAuth return length
for legitimate maximum-length Unicode searches nested in feed context. Its
provider flow checks retain the same fixed origins and route/character rules.
This is a local iteration; no production deployment, actual vendor login or new
numerical judging verification is claimed. Ignored `reports/discussion-links-*`
files retain screenshots and outcomes; disposable QA credentials/data are removed.
