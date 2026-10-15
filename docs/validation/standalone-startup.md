# Standalone browser-only startup

Validated on 2026-10-14. A production build with only `VITE_PUBLIC_SITE=1`, without
an enabled hosted application or backend URL, previously rendered a blank page.
The deployment module eagerly parsed an empty hosted URL even though no account
navigation could use it. A real static-server browser run reproduced the startup
`Invalid URL` exception before the application mounted.

Hosted-origin validation now runs only when the public site actually enables
the hosted application. Disabled integrations ignore absent or malformed unused
URLs; enabled integrations retain the same HTTPS-origin validation. No fallback
backend, account API request, redirect destination or authentication rule is added.

The deployment check now loads and renders actual modules in five configurations:
local offline, same-origin hosted, public with a published app, and public without
an app with either an empty or malformed unused URL. Closed account/community
routes normalize to the library; published-app redirects retain fixed destinations.
The URL validator still rejects empty, malformed, non-HTTPS, credential-bearing,
path/query/fragment-bearing enabled destinations.

The full frontend check chain passes in an isolated committed-source snapshot,
with these deployment files applied separately from other in-progress UI work.
The standalone production build and the existing public-with-app, same-origin
hosted and split-origin hosted builds pass the original JavaScript budget. Its real
browser runs Schedules 8/8, Attention 7/7 and LoRA 6/6, three runs each, with full
first-use reference/mutant calibration and subsequent bounded page-local reuse.
The original gradient checks and all 300 LoRA updates remain unchanged. The static
server records zero API requests and no failing asset responses. A fresh browser
tab also mounts the library without the pre-fix startup error.

This frontend startup fix does not change numerical bundles, workers, Python
grading code or test thresholds. Browser/Oracle performance observations and
deployment integration are recorded separately in the private backend repository.
No public push or production deployment is claimed.

## Build-time origin validation

The browser and Vite now share the same pure HTTPS-origin validators. Vite's
resolved build configuration validates enabled hosted-app and configured API
origins after loading environment files. An invalid configuration fails before
browser assets are emitted, with the variable name and required format. It never
echoes the supplied URL, credentials or query data. Unused hosted-app URLs remain
ignored when that integration is disabled; an empty API origin still means
same-origin requests.

The deployment check resolves the actual project Vite configuration against
temporary environment files: valid standalone/public/hosted modes, malformed and
empty enabled destinations, non-HTTPS origins, credentials, paths, queries and
fragments. It verifies base-file, mode-file, mode-local-file and shell precedence,
including a custom staging mode. Temporary files and inherited VITE variables are
restored after the checks. Existing same-origin/split-origin request credentials
and URL guards remain exercised by the API regression.

Two real Vite CLI builds independently reject a missing enabled hosted-app URL
and an API URL containing fixture credentials. Neither creates its output
directory, and neither prints the credential sentinel. The full frontend check
chain and standalone/public/hosted/split-origin production builds pass. A real
standalone browser still completes Attention 7/7 with all three faults calibrated,
without an API request. Numerical bundles, workers and Python sources remain
unchanged by this configuration-only iteration.
