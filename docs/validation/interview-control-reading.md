# Readable interview controls and publishing guidance

The interview hub still used 12–13px labels for filters, linked concepts,
company information and common actions after the main reading scale changed.
The institution-verification notice used 13px text and 28px actions.

This stylesheet-only refinement uses the shared 16px body scale for search and
form inputs, tab labels and primary company/resource titles. Filters, linked
concepts, descriptions and common actions use the 14px small-text scale. Dates
and other secondary metadata retain a separate hierarchy. These are rem values,
so the user's preferred browser base size remains respected.

Common controls have at least 40px desktop targets and 44px coarse-pointer
targets. Company-source disclosures retain their existing 44px narrow-screen
rule. Search clear buttons no longer shrink to 28px. Form fields use 16px text
and 44px targets; filter selects also use 16px on touch devices. Tabs stay on one
horizontally scrollable row when space is limited. Results headings, author rows,
filter groups and notice actions can wrap; long concept labels and affiliation
errors have bounded widths and break opportunities.

The publication notice has a 16px heading, 14px explanation, clear account-refresh
button and visible keyboard focus. Permission checks, account refresh, draft
ownership, anonymous affiliation, submission/retry logic and content are unchanged.
No account or publication writes are used for production verification.

The complete existing frontend checks pass with 209 PASS markers. Default and
actual hosted Pages-profile builds retain three initial JavaScript assets, the
original 409,600 raw / 133,120 gzip byte limits and deferred feature boundaries. Their
interview stylesheet has the same content hash, including the inset tab focus
ring that remains visible inside the scroll container.

Source review covers the three stylesheets and emitted CSS. Existing
mounted tests cover interview filters, linked practice, anonymous publication,
permission recovery, drafts and keyboard behavior; they are not visual layout
tests. No manual browser visual review is claimed. Cloud gates, actual deployment
actions and live asset comparisons remain separate requirements.
