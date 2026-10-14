# Discussion reads and confirmed reply continuity

Validated on 2026-10-14 against the production hosted build and real private account/community handlers in a disposable loopback fixture.

Previously, a failed first post read had no local retry control. Reply publication discarded the successful POST's reply object, so a subsequent failed GET left the newly published reply absent from the discussion until a page reload.

Discussion pages now offer refresh/read retry while retaining the loaded post, reply list and draft editor. Initial failures can retry without navigating away. A monotonically increasing read version and the existing mounted/account boundary prevent obsolete reads from replacing current state. Writes are unavailable during a refresh; text editing remains available, preserving unpublished work.

After a successful reply POST, the actual confirmed server row is appended before the follow-up read. Its body, author, parent and deletion permissions are retained. Reapplying the same acknowledgement cannot duplicate the row/count, and a different post's reply is ignored. A failed follow-up read explicitly distinguishes confirmed publication from stale discussion data and offers GET-only retry. A button focuses and reveals the published reply. Successful read retry clears the stale-read warning. This is not POST idempotency for a lost or unacknowledged publication response.

## Browser evidence

| Boundary | Observed result |
| --- | --- |
| Module/search context | Opened Attention's Questions board filtered by the Chinese search term 全屏蔽. Post navigation, sign-in return and Back to discussions retained that exact scope/query. |
| First GET 503 | The post showed a retry control; clicking it loaded the real post without reloading the page. |
| Read failure with an unpublished draft | A deliberate GET failure retained the exact Unicode/Markdown/code reply draft. Successful retry preserved it too. |
| Reply POST 201, then GET 503 | The actual Alice reply immediately appeared, including its Bob parent, rendered code and count. Its published draft cleared, while a publication-success/stale-read message remained. |
| Focus control | View your posted reply focused the actual reply article and revealed it. |
| GET retry after publication | The warning became Reply posted; exactly one copy of the new reply remained visible. The database contained one seeded reply plus one new targeted reply; the request log contained exactly one reply POST. |
| Mobile | At 390 × 844 the document width remained 390. Header retry controls wrapped without overflow, post content remained visible through a GET failure, and the new unpublished mobile draft stayed exact. |
| Locale | The English read-retry control worked and preserved the same draft. |
| Account isolation | Bob did not inherit Alice's unpublished mobile draft. Returning to Alice in the same tab restored that exact draft. Publicly published replies remained public as intended. |

The fixture seeded its own post/reply and synthetic accounts. Browser actions published only to this disposable service, not to production or other people. Native judge POSTs were zero. Fault controls, credentials and runtime state stayed in ignored reports/temporary directories.

## Checks and scope

`npm run check` and TypeScript pass. New confirmed-reply tests verify immediate visibility, exact server metadata, duplicate acknowledgements, cross-post isolation, context/count preservation and immutable input. Existing community navigation and account/post/parent-scoped draft checks remain passing.

Both production editions build within unchanged initial-JS limits: Pages **391,768 bytes raw / 125,782 gzip**, hosted **396,066 / 127,021**. Community remains deferred. No numeric algorithms, compatibility bundles, test tolerances or native judge behavior changed, so this UI iteration did not repeat the full native curriculum gate. The private account/community service implementation and database schema were unchanged. This validation does not claim recovery of an unacknowledged POST or pagination of long reply lists.
