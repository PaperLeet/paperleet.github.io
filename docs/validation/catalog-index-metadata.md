# Homepage index metadata reduction

The homepage index was already split from the full catalog. At 67 modules it
had grown to 256,934 bytes, close to its original 256 KiB raw budget. Each module
repeated paper source metadata and complete function signatures that only the
selected exercise needs. Paper overview cards already use the paper-level
metadata and the module's required/provided function names.

The exporter now leaves `paper: {title: ""}` and `functions: []` placeholders in
the index, like the existing empty statement/notes fields. Full paper metadata,
anchors and signatures remain in immutable module details and legacy
`catalog.json`. Schema version 2 and the lightweight Problem shape remain
compatible with the existing detail loader. Each module is rendered only after
its detail has loaded. Starters remain byte-for-byte exact in the index because
progress status, random unattempted selection and backup review use them before
the exercise opens. Goals, scopes, translations, test counts and runtime choices
are retained.

| Catalog at 52 papers / 67 modules | Before | After | Reduction |
| --- | ---: | ---: | ---: |
| Index UTF-8 bytes | 256,934 | 213,402 | 43,532 (16.94%) |
| gzip bytes, Python gzip default compression | 67,135 | 57,826 | 9,309 (13.87%) |

These gzip sizes measure local encoded content, not response headers, live CDN
compression or an internet latency benchmark. The original 256 KiB raw budget
is unchanged. No placeholder replaces the actual source shown in the exercise.

## Verification

- The three Python test files that directly reference catalog export/data
  (`test_catalog`, `test_dev`, `test_submissions`) pass 83 tests, one optional
  Docker check skipped, in 2.12 seconds. The catalog fixture includes deliberately
  large source anchors and signatures, checks that full details retain them,
  and verifies paper cards still have the actual paper title.
- Full frontend checks pass on an isolated snapshot of committed code plus this
  change; unrelated in-progress UI changes are excluded. Every maintained
  module's complete metadata/signatures and exact starter remain in its detail,
  and the normal loader restores those complete fields.
- Standalone production build passes: initial JavaScript remains 398,110 bytes
  raw / 127,963 gzip, within the original 400 KiB / 130 KiB limits.
- Direct before/after comparison confirms every index field except the two
  deferred fields is identical. The paper/category records, full catalog,
  all 67 immutable detail URLs, browser bundles and visualization files are
  unchanged. No algorithm, adapter, worker or native judge implementation changed.
- A fresh real browser session loads the homepage, searches SASRec and opens its
  paper overview. The request log contains the index and no module detail or
  legacy full catalog. The overview displays its real title, author, venue,
  module goals and required/provided function names.
- Opening module 67 downloads exactly its one immutable detail. Chinese and
  English views show the full paper metadata and statement. The English
  browser-only run passes 10/10 and calibrates all seven fault variants. Returning
  to the search and refreshing preserves 1/1 solved, the SASRec search and the
  source-backed progress; the refresh downloads the index without fetching the
  detail again. No API request or POST is observed. Warning/error logs are empty.

Raw QA logs and screenshots are retained under ignored `reports/index-metadata-*`
paths. This is local validation; no public push or production deployment is
claimed. Private pinned-core integration is recorded in the private repository.
