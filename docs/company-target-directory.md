# Browsing target companies

The signed-in company directory includes a native **My target companies only** checkbox next to ordering. It reuses the existing account target store without a second request or a new API. Search, evidence-based ordering, scoped practice links and company preparation progress still work within the visible cards. The count describes the visible matching companies, not interview frequency.

The filter is private UI state, scoped to the current account and the mounted interview hub. It stays selected while opening company preparation and returning to the directory. It does not add target IDs or filter state to public URLs, share links or browser storage. A different account starts with its own view and confirmed target list. Guests, accounts without the capability and the local edition retain the public directory.

No-target, no-search-match, unavailable catalog entries and unconfirmed target reads have distinct recovery. A failed target read cannot silently turn a target-only view into the full directory. Last confirmed targets remain authoritative while a refresh or change is pending; failed or lost-response writes retain the original retry semantics.

Removing a focused card in the filtered directory returns focus to the stable filter checkbox after the server confirms it. A newer editing action, scroll, route, account or focus change cancels that return. Passive refreshes do not claim focus. Native links and page filters keep their existing ownership checks.

The controls use 44px targets, native checkbox behavior, persistent list associations, themed selected/focus states, wrapping narrow layouts and forced-colors outlines. Validation uses actual React/JSDOM components and a disposable authenticated HTTP/SQLite fixture; it does not claim physical browser geometry, native OS interaction or production verification.
