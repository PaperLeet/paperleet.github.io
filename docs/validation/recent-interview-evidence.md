# Read concepts by recent public report publication

The community-linked concept list keeps contributor ranking by default and adds
an explicit Latest reports order. Each concept uses the most recent valid
publication timestamp among its positive-count, matching public company sources.
Ties keep the original contributor order; undated or invalid calendar timestamps
come last. Curated practice does not present a report chronology.

Rows display the latest linked report date in UTC. Company-source tooltips and
accessible names add their own publication dates. The source is the backend's
MAX(public post created_at), not an interview month, content-edit time or a claim
about actual interview frequency. Existing report-period, company, role, learning
state, query, difficulty and duration filters continue to select the same cohort.

The native order select keeps keyboard focus. Switching resets the visible first
batch while retaining already mounted guide, hint and source rows by concept ID;
returning to the other order restores their open state and scroll positions.
Explicit sorting cancels older source-return recovery so a late reading callback
cannot impose an earlier position. Reveal/compact controls still bound visible
rows, retain real ranks and yield to newer editor/scroll intent.

Only the two-value display preference is stored for the current tab, with an
in-memory fallback if storage is denied. It survives source-post reading and
return, but adds no query or new public URL field, account preference, server
field, listener or API call. Source and practice links keep their existing public
filter/queue context. Changing the order does not fetch or publish anything.

Actual StrictMode checks cover chronology, strict UTC/calendar validation,
identity/count exclusions, contributor ties, invalid/undated rows, first/all
batches, locale and control identity, retained guide/hint/source DOM, old-reading
retirement, denied storage, stale/account/curated guards and unchanged private
source, selection, both scrolls, metadata, grades and URL. An actual Interviews
resource-hook flow reads an anonymous disposable HTTP/SQLite cohort containing an
older two-contributor concept and a newer one-contributor concept. The entire
database remains unchanged. React/JSDOM tests do not establish physical-browser
geometry, native focus or live-provider behavior.
