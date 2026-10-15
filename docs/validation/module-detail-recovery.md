# Recovering malformed module details

## Reproduced failure

The selected-module loader accepted a truthy `module` object without validating
its fields. In a disposable localhost build of public commit
`a764110331546f2b9f2e75135e3d42793494577c`, a valid HTTP 200 detail containing
`module.required_functions: null` passed the loader and caused a React
`Cannot read properties of null (reading map)` error with a blank page.

The QA server sent that response with a long cache lifetime. After the server
was restored, a normal page reload still used the cached bad body, made no new
module-detail GET and reproduced the blank page. No production site or cache
was altered. Before fault injection, the browser saved a native Attention
reference plus a unique QA comment as a normal guest draft.

## Change

`loadProblemDetail` validates the fields the workspace renders before returning
an immutable detail. Shared identity, starter, scope, test counts and browser
metadata must match the index; JSON object key order may differ. Paper metadata,
function signatures, English text, statements and optional source fields receive
shape checks. A lightweight index summary is not accepted as a complete detail.
This validates the rendering contract, not a cryptographic digest of the body.

The loader remembers up to 128 failed public detail URLs in page memory. The
next explicit retry uses fetch `cache: reload`, bypassing and refreshing the
bad HTTP cache entry. Further failures continue to revalidate; a valid response
restores the usual `force-cache` behavior. Healthy immutable details keep their
existing cache path. No local draft, accepted report or account state is cleared.
There are no automatic API submissions or background retry loops.

Already-cancelled loads do not start a fetch. A cancellation after transport
completion is still rejected before returning data, and cancellation does not
mark healthy content as corrupt. Legacy fully loaded data without a detail URL
continues to return without an extra request.

## Verification

- The frontend data/job check runs the actual loader over all 67 exported detail
  files. Raw response bytes retain JSON negative zero. Twenty-three malformed
  fixtures cover missing nested arrays, invalid React text values, bad paper/
  signature/translation structures, a summary in place of full detail and
  mismatched starter/test/runtime metadata. Wrong identities/paths remain rejected.
- Reordered metadata keys are accepted. Consecutive malformed/invalid-JSON
  responses produce cache modes `force-cache, reload, reload, force-cache` after
  recovery. Pre-aborted and late-aborted loads preserve their cancellation reason;
  late cancellation does not poison the healthy URL's cache policy.
- Complete frontend checks pass in an isolated snapshot excluding unrelated
  in-progress UI edits. Production standalone build: 399,721 bytes raw / 128,524
  gzip initial JS. Split-API hosted build: 405,384 / 130,481. Original 400 KiB raw
  and 130 KiB gzip limits are unchanged.
- Real browser recovery rejects the actual cached bad HTTP 200 and displays the
  existing localized error/retry screen. One click requests the corrected detail.
  The saved 26-line draft is exactly equal, reconstructed from visible editor
  lines; Attention then runs the complete browser suite, 7/7, with all three
  faults calibrated. A fresh final-build repeat confirms the same one-click
  recovery and exact draft, with empty warning/error logs. No API request or
  POST occurs.

The baseline log also contains an unrelated editor dynamic-import failure before
fault injection; the blank-page reproduction is independently identified by the
later null/map error. Recovery evidence and the final-build verification are
retained under ignored `reports/detail-load-*` paths. No public push or
production deployment is claimed. Native suites, runtime bundles and exercise
metadata are unchanged. Private pinned-core integration is recorded separately.
