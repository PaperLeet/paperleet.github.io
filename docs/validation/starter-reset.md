# Reviewed starter restoration

Validated on 2026-10-14 in production Pages and hosted builds. The previous reset button immediately replaced source, treated an empty file as having no draft, and relied on initiating a download as its only backup. Reusing the current tab's backup slot could then remove the original local source.

Reset now opens a native dialog with the reviewed source, a cancel action and an explicit backup/restore action. The default focus is **Keep current code**; Escape returns focus to the reset button. Confirming initiates the reviewed source's download before applying the change. The UI does not claim the browser has completed a file download. Pending saves, active tests, storage errors and draft conflicts disable the action. Owner, problem, starter, source and draft-update expectations must still match at confirmation. The component is keyed by owner and module, so changing either removes its review.

Browser workspace resets preserve the original source through the existing atomic draft/backup transaction. A stable preservation intent survives coalesced writes and failed-save retry; subsequent typing does not reuse the original source's archive slot. The existing acceptance history remains intact, while the exact-source check marks earlier results as stale. Signed-in resets use the account workspace's guarded update/sync path and an explicit file download, rather than writing private drafts to the anonymous browser store.

## Observed browser behavior

| Case | Evidence |
| --- | --- |
| Cancel and Escape | Both left the exact source unchanged. Escape restored focus to the reset trigger. |
| Normal local reset | Attention first passed 7/7. Review and downloaded backup matched the Unicode-marked source exactly; starter and original recoverable source survived reload, with Notes still unlocked. |
| Empty file | The review explicitly showed zero bytes. The downloaded file was zero bytes, and the empty source remained recoverable after reload. |
| Concurrent tab edit | Another tab updated source while a fresh observer held the review. Confirmation became disabled. Reviewing the latest source produced an exact backup of the new source before reset. |
| Download exception | A controlled object-URL failure kept the dialog open with an error and left source untouched; an exact source download succeeded after restoring the download API. |
| Backup-store quota fault | The reset's download remained exact. A new observer saw the pre-reset source because the draft/backup transaction rolled back. Restoring storage and retrying saved the starter and archived original source. |
| Account sync | Actual private account/draft handlers saved the restored Attention starter to SQLite; audit found one account draft, no run links, zero practice runs and zero native judge POSTs. |
| Account change | Alice's open review disappeared on navigation/account change. Bob received his own unchanged starter, no inherited dialog, and a disabled reset for pristine code. Returning to Alice restored her draft. |
| Pending account write | A sync acknowledgement changed the reviewed draft version and correctly blocked an old confirmation. The final UI also waits for pending autosave before allowing a review; its immediate disabled state was observed after editing. |
| Mobile and locale | At 390 × 844, document width stayed 390 and dialog width was 354. Long source scrolls within its block. Chinese and English review/cancel actions were exercised. |

A transient lazy editor import failure occurred during the QA session. Plain-text mode retained editing/saving, and reloading restored the editor without source loss. This iteration changes starter restoration, not the module-fetch retry implementation.

## Checks and limits

`npm run check` passes, including exact/empty-source archival, accepted-history preservation, post-reset typing, storage rollback and blocked reset states. Final TypeScript and Pages/hosted builds pass. Initial JS remains within unchanged 400 KiB raw / 130 KiB gzip limits: Pages **390,040 / 125,289 bytes**, hosted **394,338 / 126,562 bytes**. No maintained numerical algorithm, test bundle, tolerance or experiment changed; the full native curriculum gate was not repeated in this UI iteration.

QA used disposable loopback servers, fault controls and private test accounts. No production data, deployment or public publication was changed. Local backup recovery still depends on retaining site storage; signed-in code backups depend on keeping the downloaded file. A browser download request is not proof of a completed disk write.
