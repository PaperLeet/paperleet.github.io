# Searchable catalog pages

The official Pages build enables `VITE_PATH_ROUTING=1`. Every catalog entry has a
real directory document, so direct visits and refreshes work on GitHub Pages
without a rewrite service or a `404.html` redirect:

- Papers and the standalone course: `/papers/<paper-id>/`
- Exercises: `/problems/<category>/<module-id>/`
- Featured collections: `/collections/<collection-id>/`

`web/scripts/prerender.mjs` reads the exported catalog after Vite builds. It writes
English paper summaries, exercise scopes and statements, source links, related
exercise links, page-specific metadata, canonical URLs and social-card tags into
each HTML document. The homepage starts with twenty papers; JavaScript retains
the existing progressive loading behavior. No solutions or review notes are
added to the static exercise statements. Titles use the catalog's recorded venue
and year; no publication facts are inferred.

The same build generates `dist/sitemap.xml` and `dist/robots.txt`. Adding a paper
or module through the normal catalog export automatically adds its page and
sitemap entry. The sitemap contains only canonical document URLs, without hash
fragments, filters, account data or guessed modification dates. The social-card
PNG is rendered from the committed SVG and shared by the page-specific previews.

The application still uses its established internal route and return-context
contracts. `AppLink` emits normal catalog hrefs, and a deferred History API adapter
handles navigation and browser traversal. Old `#/papers/...`, `#/10`, and module
ID bookmarks remain usable and normalize to the public path. Existing dynamic
community, profile, account and messaging links remain root hash routes: these
are not statically exported or included in the sitemap. Standalone/local builds
keep their previous routing unless the flag is explicitly enabled.

`searchMetadata.ts` is shared by the generator and client navigation, so changing
pages also updates the title, description and canonical instead of leaving the
previous paper's metadata behind. Query parameters preserve library return
context but do not enter canonical URLs or the sitemap.

Validation runs in the production Pages quality gate: all generated URLs are
served from a strict directory-based HTTP server, with checks for HTTP 200,
distinct metadata, meaningful content without JavaScript, root-based assets,
safe HTML escaping, sitemap completeness, a valid social image, and real 404s for
missing documents. Navigation checks cover legacy links, history, modifiers,
prevented clicks, reading-position capture order, context and metadata changes.

After publishing, verify the live URLs and submit `https://paperleet.com/sitemap.xml`
in the existing Search Console property. Request indexing for representative
paper and exercise URLs. Google decides whether and when to index a page and how
to display its title and snippet.
