# Custom cases and daily paper challenges

The public app offers `#/daily-challenge`, a highlighted daily row above the library, a compact month
calendar in the desktop sidebar, and a slim notice on the selected daily module.
The dedicated challenge page puts the problem and month list beside the calendar;
Rating and its folded rules are secondary. Completed days show check marks. The calendar starts on 2026-10-19 and resets at
00:00 UTC. Past dates remain available for practice. The version 1 selection in
`paperleet/daily_challenges.json` is immutable: its 28 modules repeat in order.
Do not insert, remove or reorder entries; a future curriculum needs a new dated
schedule version. Both browser previews and the private service read this file.

The account application calls `GET /api/challenges/daily?date=YYYY-MM-DD` for the
selection and the signed-in user's private Rating/history. Starting a rated
attempt calls `POST /api/challenges/daily` with `expected_user_id` and `date`.
The account server owns time, enrollment, completion and Rating. Anonymous users
can practice directly; browser results and custom cases never award Rating.
After a full server submission, the UI refreshes through the existing submission
invalidation channel; a refresh button and window-focus refresh are also available.

## Rating

Rating starts at 1000. A successful full native submission, after enrollment and
before midnight UTC, awards once for that day. Difficulty benchmarks are
Easy=900, Medium=1200, Hard=1500. The gain is:

`max(1, floor(32 * (1 - 1 / (1 + 10 ** ((benchmark - rating) / 400))) + 0.5))`

Failures do not deduct points. This is a personal practice Rating against fixed
difficulty benchmarks, not a multiplayer Elo ranking. The service stores the
original difficulty, before/after Rating, trusted run identity and timestamps.
The award and immutable submission commit in the same SQLite transaction.

## Custom cases

Open a module's **Result → Custom case / Console**. Choose a listed function,
enter positional arguments as a JSON array and keyword arguments as a JSON
object. An optional JSON expected value enables recursive comparison with
`atol=1e-6`, `rtol=1e-5`. Booleans do not compare equal to numbers.

Tensor arguments use descriptors, including nested descriptors:

```json
[{"__tensor__": [[1, 2], [3, 4]], "dtype": "float64"}]
```

Supported dtypes are float32, float64, int64 and bool. Lists remain lists unless
wrapped explicitly. Tensor/array outputs become JSON, and nonfinite floats are
shown as strings. Exceptions display their learner-code traceback.

The existing opaque-origin iframe and disposable Worker isolate execution from
account cookies and site storage. Its network policy permits only public runtime
assets. Each custom run receives a fresh interpreter. Input is limited to 32 KiB,
Console to 16,000 Python characters, and result tensors to 10,000 elements.
Execution has a 10-second deadline after preparation; initial runtime download has
the existing loading budget. Stop terminates the worker and removes its frame.
Custom output stays separate from test reports, accepted progress and account
submissions. Results identify changed source instead of grading edited code.

## External company exercises

The Interviews page has a separately labeled external company practice directory.
As reviewed on 2026-10-19, it indexes 50 company-tagged TrenTorch exercises across
43 distinct source company labels. Amazon includes AWS when filtering with
PaperLeet's company taxonomy; an independent source-tag selector preserves the
original labels. Search and incremental listing are supported.

These are company-inspired original scenarios, as stated by
[TrenTorch's problemset](https://trentorch.com/problemset), not verified interview
experiences. They do not create community posts, reports, company mention counts
or interview-frequency evidence. Entries contain factual index metadata and
outbound links. No statements, solutions, tests or competitor code are copied.
Related PaperLeet modules are topic suggestions, not interview evidence.

## Verification and release

Run `npm run check:challenges` in `web/` for calendar, account-isolation, UI and
real Pyodide custom-case checks. Run the existing browser judge checks after
changes to the shared coordinator, plus `npm run build`.

The matching private backend and its reviewed public core must be released
together. An older backend cannot provide enrollment or Rating; the client shows
the unavailable state and retains direct practice. Do not deploy only the new
frontend expecting the existing server to award points.
