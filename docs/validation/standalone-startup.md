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
