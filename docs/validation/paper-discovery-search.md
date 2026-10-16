# Paper discovery search

The library could not find an existing paper from its author or arXiv identifier.
An empty query result also made the global practice button claim that all core
modules were complete, despite having no modules in its current scope.

Search now includes the catalog's authors (string or array), venue, year, source
URL and arXiv ID alongside its existing titles, summaries, tags and module goals.
Whitespace-separated terms can match across paper metadata and the same module;
a module-specific query still narrows its paper's module list. The shared list
and practice selector retains collection, difficulty, progress and runtime bounds.

The existing strict arXiv parser recognizes abstract/PDF links, modern and legacy
identifiers, and supported host aliases. Search ignores a recognized reference's
version and URL query/fragment. This normalization only selects catalog entries:
search never fetches the supplied URL or creates an outgoing source link.
Unknown references stay empty. The visible placeholder advertises author/arXiv
search while retaining the existing accessible search label and URL state.

The topbar uses its existing scoped idle label, distinguishing an empty or
optional-only view from completion of a filtered scope or the whole catalog.
Pending progress still cannot claim completion or start a fabricated module.

Focused checks cover bibliography queries in both languages, author arrays,
combined author/year/module terms, versioned/legacy/PDF references, foreign hosts,
unknown IDs, filter isolation and unchanged module-specific matches. Actual App
mounts verify empty, scoped completion, full completion, optional-only and pending
progress labels in Chinese and English. Existing workspace action, draft, editor,
diagnostic and return-navigation checks remain in place.
