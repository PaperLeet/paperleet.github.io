# Roomier reading and clearer hierarchy

The primary paper list now uses 18 px titles, 17 px body copy, 16 px primary controls, 15 px supporting text, 14 px metadata and 13 px captions at the browser's normal 16 px base. These are rem values; the root stays at 100% and follows the reader's browser preference. Body text explicitly adopts the shared body token instead of falling back to the root size. No zoom, transform, font download or new dependency is involved.

The owning library stylesheet gives comfortable rows an 88 px minimum with 18 px vertical padding, and compact rows a 64 px minimum. Both keep the same readable title size. Rows can grow naturally for wrapped titles. Filters get wider native selects at 16 px; existing responsive wrapping, one-column layout, horizontal tab scrolling and full-width mobile controls remain available. Header navigation stays at 17 px on desktop and 16 px on small screens. Problem headings use 22 px section and 19 px subsection text with 17 px prose, improving hierarchy over the former 18/16 px headings.

The shared scale also increases existing community, profile, achievements, feedback and interview copy. Colors continue to use the existing theme tokens. Editor font preferences and document state are unchanged. The changes are limited to CSS and this note; account/API, execution, navigation and persistence behavior are unchanged.

Validate using the existing frontend checks, both edition builds and compiled CSS/manifest inspection. CSS inspection establishes declarations and bundle boundaries, not rendered geometry or browser visual QA. Native clipboard permission prompts are not exercised, and no production user records are written.
