# Moderation evidence comparison

When a reported snapshot's version differs from the loaded current post,
administrators can explicitly expand **Compare with current content**. This
loads a separate component and styles; unopened comparisons do not parse or
render differences. Existing snapshot Markdown reading remains available.

The comparison covers title, body and category only. It labels old/new values,
shows body line numbers, addition/deletion markers and exact line-ending
labels. HTML, Markdown and code are literal text in the comparison; they do
not execute, fetch assets or submit a review. Identical visible fields with a
different version produce an explicit notice to inspect other information
such as interview details or paper links in the original post.

The model preserves every source character. Small changed blocks use bounded
line alignment (at most 250,000 matrix cells). Larger blocks are explicitly
labelled as complete replacements, including unmatched unchanged lines in
their counts. Three context lines surround changes; unrelated unchanged
regions are labelled as omitted. The rendered preview stops at 400 source
rows with a visible notice. Both complete raw bodies remain available in
native disclosures; no source text is silently truncated. Inputs beyond the
comparison bound use the full-source view instead of line analysis.

Opening/closing a comparison, switching locale and refreshing the current
version retain stable native disclosure, source-scroll and review-note DOM.
The original-post link opens a labelled new tab with `noopener noreferrer`,
preserving the workbench. A newly accepted content version or report revision
clears removal consent; it preserves the note and requires a new explicit
confirmation. Existing stale-version, exact-retry and account guards remain.

`check-moderation-comparison.mjs` checks exact reconstruction/numbering,
Unicode, repeated/blank lines, LF/CRLF/CR/final-newline changes, bounded work,
context gaps, raw complete sources, hostile literal text and retained DOM.
`check-community-reports.mjs` exercises the actual administrator route,
explicit lazy disclosure, zero added comparison API requests, retained notes
and sources, new-tab context and consent retirement after normal content and
report-batch refreshes. The same UI runs against disposable loopback HTTP and
SQLite fixtures. Build checks keep comparison code/styles out of the initial
library bundle under the existing budgets.

Evidence is source, React/JSDOM and disposable API behavior. No rendered
browser geometry, native focus trap, device or live-provider behavior is
claimed.
