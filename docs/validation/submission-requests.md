# Proposal request ordering and draft preservation

Validated 2026-10-15 in an isolated source snapshot based on public `4385908af3a69b044cf081221651aa6c3f0f7726`. No production application, API configuration, image or database was deployed or modified.

## Problem and resulting behavior

Selecting proposal A, then B, previously let A's slower detail response replace B. The pre-fix React regression actually displayed `Late A` where `Proposal B` was expected. A successful create response also cleared all fields and opened its detail even when the author had continued editing or moved elsewhere.

The viewed proposal now has an identity for each selection, independent from saved list rows. Late reads/errors cannot take over another selection, including leaving and returning to the same ID. Background actions update their own record; when the user returns to that record, its mutation result can update the current detail. Server UTC versions are compared without dropping microsecond precision, preventing older acknowledgements from replacing newer snapshots. Changing the selected operation restarts any required poll; polls wait for the preceding request to finish and stop at terminal states.

Creation captures the submitted fields and an edit revision. Only the exact unchanged submitted draft is cleared and opened. Newer edits and navigation stay intact; an accessible success notice identifies the saved proposal in the list. The notice follows the current UI language. Immediate duplicate action clicks share one in-flight gate. An older list response cannot erase a newer saved record.

## Automated checks

`node scripts/check-submissions.mjs` renders the actual React component with controlled responses, rather than testing a parallel implementation. It covers selection races, stale errors, switching to a new form, captured POST fields, newer edits, normal exact-draft clearing, bilingual notices, same-tick duplicate mutation clicks, background actions, stale list snapshots, leave/return interleavings, sub-millisecond version ordering, sequential polling, and continued polling after an older acknowledgement.

Full `npm run check` passes with this regression script included. It also checks all 69 browser bundles, 433 maintained Python sources, library/practice navigation, private workspace behavior, code backups, reports, queueing, API origins, accounts and community paths. No catalog, runtime numerical code, judge tests or training budgets were changed by this patch.

Public submission workflow tests: **43 passed, 1 skipped**, 2.89 seconds. The existing optional external-runtime case remains skipped.

Standalone production build passes: initial graph **401,058 raw / 129,993 gzip bytes**. Hosted production build passes: **406,434 raw / 131,826 gzip bytes**. The original **400 KiB raw / 130 KiB gzip** caps remain unchanged. The proposal page stays deferred.

## Browser → API → stored data → UI

A temporary localhost server served the built frontend and proxied to the actual private `JudgeHandler` and public `SubmissionStore`, with isolated storage and deliberately disabled model/sandbox providers. The fixture imported reviewed private `c790116b77fa5a2ae68d3f14692b1f11f920df4e` source; unrelated live account/email/UI changes were excluded. The local API path was exercised here; hosted ownership and offline review policy are separately verified in the private repository.

| Boundary | Evidence |
| --- | --- |
| UI selection | Clicked A then B; held A's actual API response until B had been delivered. |
| API ordering | Recorded client response order **B, A**; B's heading and persisted focus remained selected. |
| Captured POST | Held a successful create acknowledgement, edited title/focus, and verified stored JSON still contained the original submitted values. |
| Response rendering | The saved proposal appeared in the list; newer title/focus and the current form remained. |
| Saved-detail loop | Opened the saved record and saw its captured focus; returning to the form retained the newer draft. |
| Normal submission | An unchanged submitted draft opened its saved detail; New submission then showed blank fields. |
| Language | Switched the final success notice to English and back to Chinese without losing the newer fields. |

There were two seeded records and exactly three UI-created records, with exactly three create requests. All API responses succeeded. No generate/validate/review requests or external model calls occurred. The final browser warning/error log was empty. These are isolated development records, not production proposals or a hosted-account session.

Ignored local evidence: `reports/submission-requests-{before,focused,web,build,hosted,store-tests}.txt`, request event and stored-proposal JSON, selection/English/final DOM snapshots, and `reports/submission-requests-browser-final.png`. The screenshot shows the saved row, success notice and newer unsaved fields together. Temporary tabs, servers, data and source snapshots are cleaned after verification.

This change preserves edits within the mounted proposal page; it does not introduce refresh/close recovery or claim browser-storage persistence for proposal drafts.
