# Private saved discussions

Community members can save a discussion for later and find it in the Saved tab.
The personal feed supports the existing module/category/search filters and
pagination, ordered by when a post was saved. Opening a post, returning to the
list and signing in retain those filters and the page. Share controls still
produce only public post/reply links, without personal feed/search context.

Bookmarks belong to the signed-in account. They do not increment public likes,
reposts, replies or profile activity counts and do not appear as a public profile
tab. A guest can sign in from the post or Saved route and return to that context.
The account capability `community_bookmarks` and a Boolean viewer flag gate the
controls. Older services that omit the capability hide the controls; an explicit
Saved route shows an unavailable notice without accidentally requesting the
ordinary public feed as if it were a private collection.

## Request and recovery contract

The public interface adds GET/POST `community/posts/{id}/bookmark`; the setter
sends an explicit Boolean `active` and `expected_user_id`. The response identifies
the account, post and confirmed state. The client rejects mismatched account,
post or action acknowledgements. Saved feed reads use `saved=true`; identity is
always derived by the private service from the authenticated session.

The server sets the desired state instead of toggling it. A lost response keeps
the original intent for an explicit retry, so retrying Save does not Unsave.
Alternatively, Read saved state performs a fresh GET and resolves the uncertainty
without another write. A 20-second request timeout releases the interface and
offers the same recovery controls. There are no automatic mutation retries.
Closing the post aborts the client request; revisiting reads the authoritative
state, since cancellation does not imply that a server write was rolled back.

The component is keyed by account/post, ignores results after unmount, and
refreshes account identity after an expired/changed session. Other post mutations
are disabled while the bookmark request runs; bookmarks are disabled during post
or reply editing. This avoids overwriting a confirmed personal flag with an
older mutation response. All reads are no-store. No private bookmark payload is
persisted in browser storage.

## Verification

- Native service/HTTP/OAuth focused regression: 169 passed, 90.65 seconds, before
  the final added duplicate-empty-filter assertion; the final bookmark HTTP case
  also passes separately. Coverage includes private actor scope, duplicate setters,
  concurrent save requests, save-time ordering, search/category/module/following
  intersections, pagination, account suspension, deleted posts, schema upgrade,
  account deletion and historical retired modules.
- Frontend checks cover filter/login/reply round trips, strict query parsing,
  share-context stripping, acknowledgement identity/action checks and accessible
  signed-in/guest/unsupported controls. React review uses event-driven mutations,
  bounded requests, primitive fetch dependencies and DOM focus after controls
  become enabled. A discovered timeout focus loss is corrected before release.
- Real local browser, built hosted frontend and the actual private HTTP/SQLite
  service verify login return, response-lost Save with an idempotent retry, and
  response-lost Unsave resolved by a GET. A server write followed by a hung
  response reaches the 20-second client timeout, restores keyboard focus to
  recovery and resolves with a GET. Six explicit setters perform five state
  changes; no automatic retry occurs. Public counts remain unchanged.
- A 22-item saved feed supports search, page two, post navigation and returning
  to the same search/page. Signing out and back in retains the Saved page.
  Changing the session cookie before an old-account mutation returns 409,
  refreshes the displayed account and preserves the original account's 22
  bookmarks. The other account has zero bookmarks. Older capability responses
  hide Saved and show an unavailable notice on direct navigation, with zero
  saved-feed requests until the capability returns.
- Chinese and English post layouts have document/main width 390 at a 390-pixel
  viewport. A clean final collection page has no warning/error console entries;
  the deliberate failed-response scenarios produce their expected HTTP errors.
- The complete frontend check command and hosted production build pass in an
  isolated snapshot containing only this change. The initial graph is 405,261
  raw / 130,402 gzip bytes, within the existing 400 KiB / 130 KiB limits. Accounts,
  community and rich text remain deferred. Concurrent unrelated UI edits are
  excluded from this verification and commit.
- The standalone production build also passes at 404,657 raw / 130,213 gzip
  bytes. No judge implementation, runtime, catalog or teaching suite changes.

This work is local. No push, production deployment, real account migration or
external message is claimed. Private storage and API implementation stay in the
private repository; only the interface and frontend are public.
