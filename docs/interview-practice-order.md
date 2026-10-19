# Practice order for reported interview concepts

The reported-concept list offers Easy first and Short time first alongside
Contributors and Latest reports. These two choices use the exercise catalog's
difficulty and estimated minutes. Their position labels say Practice order;
they do not change report statistics or claim actual interview frequency.

Easy first orders E, M and H. Short time first accepts finite positive numeric
estimates. Missing or invalid metadata sorts after known values. Ties retain
the original contributor order, and rows without available exercises remain
readable with their public sources. Search, difficulty/time conditions and
learning-state filters continue to intersect this list.

The existing per-tab preference stores only a validated order enum. Native
selection survives locale changes and reading remounts, with memory fallback
when session storage is denied. Explicit ordering retires an older pending
source-return request and resets the visible batch to six while retaining
previously mounted guide, hint and source-preview state. Filtering, source
counts, practice grades, private code and drafts are not modified or refetched.
Hidden/inert or retired controls cannot update the preference. Curated fallback
lists retain their curriculum order and do not expose report-order controls.

Actual React/JSDOM tests check both planning orders, exact cohorts, source ties,
invalid metadata, unknown-last placement, native locale/storage/remount behavior,
individual and ancestor inactivity, retired-route locale changes, source-return
precedence and retained reading state. They are DOM/source evidence, not physical
browser layout or native operating-system focus verification. No backend schema,
API, dependency, problem or numerical-runtime changes are introduced.
