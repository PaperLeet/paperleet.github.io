# Readable, compact company sources

Frequent-concept rows and the practice page now share the same source-company presentation: three visible company chips, with a native disclosure for additional sources. Company labels use the existing 14px small-text token, counts use the 13px metadata token, and touch targets reach 44px. Source labels and the evidence link on the practice page are enlarged consistently. Existing bundled organization logos remain unchanged.

Every chip keeps its exact report count and contributor tooltip. Links enter the matching company's source experiences for the same concept, retaining the ranking's role, stage, period and preparation context. The practice-page chips retain their existing past-year source scope. Expanding or collapsing reads no additional API data, changes no URL and writes no workspace or draft storage. Empty and zero-count sources produce no company claim.

The native details/summary preserves keyboard order and open state across language and count updates. Independent instances have unique controlled regions. Closing while an extra link still holds focus returns to its surviving summary; a newer focus choice or route takes priority. There is no global listener, persistent disclosure state or additional dependency. Language changes retain the existing saved language preference behavior.

## Verification

- The ranking test first fails on the previous implementation because all five source chips are displayed without a disclosure.
- `check-interview-ranking.mjs` exercises real ranking links plus native disclosure, three visible sources, all additional links, exact counts and routes, native DOM Tab order, separate instances, bilingual/count continuity, CSS-hidden closed content, hidden-link focus repair and newer focus/route guards. Disclosures are exercised with storage writes rejected.
- `check-concept-journey.mjs` exercises the actual practice-page company component through its asynchronous summary API, including malformed/stale responses, complete source links, disclosure CSS, unchanged working editor DOM/code/selection/scroll, and no additional requests when reading company sources. Existing concept/practice/login returns remain covered.
- Run the complete frontend suite, TypeScript/public/hosted builds and unchanged initial-load budgets. The new presentation and styles remain in the deferred interview/practice metadata graph.
- Before publication, verify the source → experiences → practice → source flow through real React components, account sessions, HTTP handlers and a temporary SQLite database. Synthetic reports stay local; production verification is read-only. No browser visual QA is claimed.
