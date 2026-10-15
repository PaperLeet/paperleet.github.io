# Proposal draft recovery

Validated 2026-10-15 in an isolated public snapshot based on `29d32ddefe6feb798720817eb20a6214f3d5cd3e`. No push, application/image deployment, production configuration change or production data write occurred.

## Behavior and storage boundary

The proposal title, paper URL, code URL, excerpt and practice focus previously lived only in React state. The pre-fix regression fails because no editable fields persist before navigation. They now save synchronously to the existing tab-storage mechanism after every edit and restore on refresh or return to the proposal page.

Storage is scoped to `local` or `account:<user ID>`, with a separate envelope owner check. An owner-keyed workspace remount prevents another account's fields from appearing during a transition. Unresolved/anonymous hosted contexts do not read or write a proposal draft. Newly created browser tabs have independent drafts. The UI explicitly describes storage in the current tab; this is not cross-device synchronization or a promise to recover after closing a tab.

Only five editable fields persist. The bounded envelope includes a version and opaque edit revision, not authentication data, roles, acceptance reports or publication authority. Maximum lengths match the current form: title 240, URLs 2048 each, excerpt 60000 and focus 4000 characters. Partial URLs are valid draft contents. Unicode, line breaks and whitespace are preserved without trimming. Submitted requests retain the existing five API fields; owner/revision metadata is never sent.

A successful create acknowledgement clears only the captured storage revision and exact submitted fields. New edits remain recoverable, including changing text away and back to its original value. Failed submissions keep the draft. Results arriving after the old owner workspace unmounts do not clean a restored draft. Blocked reads/writes retain in-memory editing and show a copy-before-leaving notice; failed cleanup is surfaced instead of claiming the draft disappeared.

## Automated evidence

- Full `npm run check` passes, including all existing proposal request-order/version/poll tests and the new storage checks.
- Storage checks cover exact source preservation, account/local separation, copied-owner rejection, editable-field allowlisting, each field at its maximum length, invalid/oversized records, read/write/delete failures, revision-safe and ABA-safe cleanup.
- Actual React component checks cover refresh-style unmount/remount, all five fields, no automatic POST on recovery, failed/acknowledged submissions, a newer identical-text edit, account transitions, an old owner's pending response, and honest unavailable/uncleared notices. The old component demonstrably fails the new persistence assertion.
- API payload assertions confirm exactly `code_url`, `excerpt`, `focus`, `paper_url`, `title`; draft owner and revision stay local.
- Standalone initial graph: **401,063 raw / 129,990 gzip bytes**. Hosted initial graph: **406,439 raw / 131,829 gzip bytes**. Both production builds pass inside the unchanged **400 KiB raw / 130 KiB gzip** caps. The proposal editor and draft logic remain deferred.
- No numerical runtime, problem catalog, judge test or training-loop code changed.

## Actual browser → API → storage → UI

The built frontend was served on a temporary localhost port and connected to the real reviewed private `JudgeHandler` and public `SubmissionStore`. The fixture used isolated storage and disabled external model/sandbox providers; unrelated live repository work was excluded.

| Scenario | Observed result |
| --- | --- |
| Refresh five populated fields | Every field matched exactly, including an emoji, leading/trailing spaces and a multiline code excerpt. |
| Leave for the paper library and return | The same draft restored. |
| Open a fresh tab | Its title stayed empty; the first tab kept its draft. |
| Restore/return without submitting | The request log contained zero POSTs during these steps. |
| Submit a syntactically valid HTTP source URL | The real API returned 400; refreshing retained the failed draft. |
| Correct the URL and save without further edits | API 201 opened the saved proposal; a refresh showed an empty new form. |
| Hold another real 201 response, then edit a newer draft | Storage retained the captured submitted title/focus; after acknowledgement and refresh, the newer title/focus restored. |
| Switch language after recovery | The restored-draft status followed Chinese/English without changing fields. |

The API log contains exactly three create attempts: **400, 201, 201**. Two seeded records plus two confirmed UI creations exist; the newer unsent draft does not exist in server storage. There were no generate/validate/review requests or external model calls. The final page warning/error log was empty. Hosted-account isolation is covered in component and private API tests; the browser fixture used the local API path.

Ignored evidence: `reports/proposal-drafts-{before,focused,component,web,build,hosted}.txt`, API event/stored-proposal JSON, restored/English/final DOM snapshots and `reports/proposal-drafts-browser-final.png`. The screenshot shows a restored newer draft alongside the previously saved proposal. Owned tabs, local servers, data, snapshots and alternate indexes are cleaned after verification.
