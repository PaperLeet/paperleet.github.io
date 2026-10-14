# Code editor recovery after a failed resource load

Validated on 2026-10-14 with production Pages and hosted builds in the Codex in-app browser. A previously observed editor import failure was reproduced with a controlled 503 on its first chunk request. The old **Retry loading** action recreated the React lazy component, but the same document rejected the same import again without another HTTP request. Source remained editable in the plain-text fallback.

Browser/module-map failure caching has implementation differences; the [MDN import reference](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/import) explains the load/link/evaluation distinction. The result above is direct evidence from this browser, not a claim about every browser version.

The workspace editor now has an explicit **Reload editor** action and explains that it refreshes the page. It becomes available only after the workspace and local learning stores are ready, drafts/results have finished saving, no workspace conflict or save failure remains, and no judge run is active. The action rechecks owner and current module before refreshing. It does not request reload in a loop, change browser cache policy, dynamically evaluate fetched JavaScript or alter the isolated judge. Other optional feature boundaries retain their existing behavior.

## Browser evidence

| Case | Observed result |
| --- | --- |
| Initial editor chunk 503 | The Pages workspace stayed usable with a controlled textarea and the guarded recovery action. |
| Browser tests in fallback | The original Attention suite completed 7/7, with reference and mutant calibration, while CodeMirror was unavailable. |
| Active worker | A deliberately nonterminating learner fixture reached its first test. Reload was disabled, the running reason appeared and Stop removed the worker. The stored earlier report remained intact. |
| Failed source save | A backup-store quota fault kept the new source in the editor and downloadable byte-for-byte; recovery was disabled with a save-error reason. Restoring storage and retrying allowed recovery. |
| Peer-tab conflict | Another tab changed Attention; reload was disabled until the conflict was resolved. Both versions remained recoverable. |
| Pages recovery | The actual recovery button refreshed the document. The editor chunk then returned 200, CodeMirror reappeared, downloaded source matched the original exact string, and 7/7 still matched the current code. |
| Hosted learning save failure | A quota fault in the learning-results store disabled editor recovery. Retrying that save enabled recovery without fabricating account results. |
| Guest/account isolation | Alice did not inherit the guest's local acceptance. Her own source synced through the real private draft handler; pending autosave disabled recovery. |
| Hosted recovery | Further transient module fetch errors occurred during this fixture session. The fallback continued to preserve source and results. Once assets were available, another explicit guarded reload restored CodeMirror, Alice's exact source and her locally cached 7/7 report. Recovery does not guarantee success while resources remain unavailable. |
| Account authority | SQLite audit found one account draft with Alice's source, no run links, zero practice runs and zero native judge POSTs. |

The deliberately injected hosted chunk fault was not the first browser network failure: no editor request reached that fixture initially. A diagnostic HTTP probe later consumed its one-shot 503; the succeeding browser document fetched 200. This record therefore uses the controlled Pages 503 as its clean before/after loading proof and the hosted flow as persistence/readiness evidence.

## Automated checks and scope

`npm run check` and TypeScript pass. Deferred-feature checks cover the guarded recovery action, disabled-state reasons, source escaping, the account-loading read-only gate, and continued text editing when recovery is blocked. Both production builds retain the unchanged 400 KiB raw / 130 KiB gzip initial-JS budget: Pages **391,768 / 125,779 bytes**, hosted **396,066 / 127,026 bytes**.

No numeric algorithm, test bundle, precision threshold or experiment changed; the full native curriculum gate was not repeated for this UI iteration. QA used disposable loopback servers, private test accounts and fault controls. No production data, deployment or publication changed. Clearing site storage still removes local data; only saved results are promised across this recovery refresh.
