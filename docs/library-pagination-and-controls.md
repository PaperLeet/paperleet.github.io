# Paginated library and control surfaces

The searchable paper list now uses explicit pages instead of cumulative reveal batches. The catalog and practice recommendation scope remain complete: paging changes only the rows on screen. Course and company views page modules; the general library pages papers.

The canonical public route carries optional `page` (1–10000) and `page_size` (20, 50 or 100). Default values are omitted. Native browser history records explicit page changes, including production path routes. Search, category, topic, difficulty, progress and sorting changes reset the page, while retaining the selected size. Page-size changes retain the first visible cohort. Practice links, sign-in returns and continuations retain the exact paged source. Empty or outdated pages offer explicit recovery, without silently rewriting history or reducing the full practice scope.

The list has a compact previous/next control above the rows and numbered pages, range totals and per-page selection below. Large lists offer a bounded numbered window and validated direct jump. Native disabled/selected states, page/list labels, keyboard focus at the result summary, composition-safe search clearing and jump validation make paging predictable. Versioned page-scoped position storage restores the original row on return, yielding to newer typing, focus or scrolling. Old infinite-list positions are ignored.

Shared control styles align field radii, focus visibility, placeholders and disabled/busy feedback. Library search, filters, sort, density, active filters and page controls receive larger hit areas, clearer grouping and consistent borders. Layouts wrap on narrow viewports, preserve touch targets and honor reduced motion. Optional collection discovery reserves its layout space while loading; collection controls are 44 pixels. Rich pagination has an immediate working previous/next fallback if its optional chunk is loading or unavailable. Collection artwork and filter chips load independently of the first usable list. Passive submission metrics and detailed runtime progress also load separately; the stop control remains immediately available throughout progress loading or failure.

Validation uses actual React/JSDOM components, real App hash history, production path navigation and disposable HTTP/SQLite flows. These checks cover complete filtered scope, page sizes and ranges, page-aware practice/login returns, Back/Forward, invalid and stale routes, pending progress, search and IME guards, per-page anchor restoration, locale continuity and unchanged private code, caret, scroll and draft sentinels. Four production editions retain the existing initial-JavaScript budgets. No physical browser layout, native IME/focus/clipboard or live-provider verification is claimed. This change adds no schema, API, dependency or numerical-runtime changes.

The top pager also accepts a direct page number, so readers can move beyond adjacent
pages without travelling to the footer. It shares the footer's integer, full-width
number, IME and Escape behavior. Invalid input stays at the field with its error;
a successful explicit jump hands focus to the existing results summary. A single
page retains its simple indicator. The footer marks the active page with the
primary button colors. Search, select, clear and filter controls now share a
44-pixel interior target and aligned outer heights; focus and hover feedback stays
visible in both themes without modifying editor inputs or numerical execution.

The footer now pairs its arrows with visible Previous/Next labels on wider
screens. The page strip accepts Left/Right and Home/End to move focus without
navigating; native Enter/Space activates the focused button. Disabled and hidden
buttons are skipped, and input editing, composition, modified keys and inactive
surfaces are left alone. Page inputs select their existing number on focus for
quick replacement. Shared page fields and actions have 44-pixel targets and
consistent radii; validation keeps the error color during hover and focus.
General form fields retain a 44-pixel minimum height after shared styles load.
Arrow labels give way to compact icons below 700 pixels. Library search/filter
fields have hover feedback, and a focused row action keeps
its parent row highlighted. These additions are covered by the library component
regression check; actual browser layout and native keyboard behavior still need
visual verification.

The compact pager above the list also exposes the same 20/50/100 page-size
selector as the footer. It retains the visible cohort and updates both controls
through the existing route state. Its controls wrap instead of extending beyond
narrow layouts. Topic and sorting fields reserve space for their chevrons, and
category chips, active filters and library header links share 44-pixel targets.
The component regression covers synchronized sizes, cohort retention, locale
continuity and inactive top-pager guards. Styles have source review only; no
physical browser layout verification is claimed for this refinement.

Display options now occupy their own row above the result summary and compact
pager. The DOM order matches that hierarchy, so keyboard navigation reaches the
view, sort and density controls before page controls. The count and quick pager
share a separate wrapping region; at narrow widths the count sits above the
pager. View, density, sort and reset controls use matching outer heights. Search
padding and selected segmented controls follow the same field rhythm. Native
page-size and jump inputs have a focus halo, with reduced-motion support.

Across shared account, community and interview fields, hover styling excludes
invalid and keyboard-focused inputs so it cannot replace their feedback color.
The current refinement is checked with actual library pagination/filter and
shared page-jump regression fixtures plus production builds. Physical browser
geometry and native operating-system interactions remain unverified.
