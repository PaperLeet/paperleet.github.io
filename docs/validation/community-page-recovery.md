# Explicit recovery from an empty community page

Community and per-exercise feeds retain the requested page when an offset becomes
empty. A successful read no longer replaces the URL or silently changes the local
embedded page. The list distinguishes this condition from an empty search, an
empty first page, access gates and request failures. One recovery surface offers
first page and, where applicable, the last browsable page, preserving search,
board, tag, saved/following scope, solutions, publication period and sort.

A recovery or ordinary pagination action focuses and reveals the existing list
before requesting another page. It rejects hidden/disconnected controls, stale
feed routes and embedded practice routes, pending account/data states and invalid
page numbers. Data completion adds no focus or scroll operation. Search text,
selection and horizontal scroll remain on their existing input if the reader
starts typing while the response is pending. Embedded recovery moves only the
reading pane and retains the practice URL and unfinished composer/editor state.

The App retains viewport and workbench tab state for query changes within the
same community root or validated exercise-topic path, as it already does for
interviews. Entering/leaving feeds, changing the topic path and opening a post
retain normal navigation. Existing reading-return logic owns post returns.

The regression first reproduced page 99 being replaced by page 3 through the real
App. Actual App checks cover root/topic scopes, first/last recovery, zero totals,
hidden/stale controls, delayed input priority and unchanged saved source. Existing
embedded/profile paging tests add explicit empty local-page recovery with intact
composer draft, selection and both scroll positions. A disposable anonymous
HTTP/SQLite flow checks actual root/topic offset totals and exact result cohorts;
the entire database remains unchanged during UI reads. No schema, numerical
runtime, service mutation or production deployment changes are made. React/JSDOM
checks do not claim physical-browser geometry, native IME or live-provider QA.
