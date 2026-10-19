# Library pagination and control polish

The existing paper, exercise, course and saved-problem lists retain their URL paging, 20/50/100 page sizes, direct page entry, stale-page recovery and exact practice return position.

Numbered-page gaps now offer named native buttons that jump into the omitted page range. They advance up to five pages, bounded by the hidden range; tooltips and accessible names state the exact destination. Page windows stay bounded for large catalogs. Small screens retain previous/current/next and direct entry.

The footer uses natural wrapping instead of an unconditional separate options row. Current pages have a clearer surface, and arrow tooltips explain their actions. Sorting, view switches, density controls and removable filters use consistent 44px targets, aligned baselines, focus states and reduced-motion feedback. Layout uses existing theme and typography tokens.

Validation uses actual React StrictMode/JSDOM list navigation, including gap destinations, resulting rows, locale-preserved button identity, list associations and result-heading focus; existing library and saved-problem paging checks cover URL/history/filter/return behavior. Builds check TypeScript, static routes and initial bundle budgets. No physical browser geometry or native pointer/IME validation is claimed.
