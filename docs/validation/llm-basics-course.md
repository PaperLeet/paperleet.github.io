# LLM Basics as one course

LLM Basics opens its 18 existing exercises directly at `#/collections/llm-basics`. It no longer lists the training/generation course as a nested entry. Source papers, module identifiers, saved code and accepted progress remain unchanged; other collections still list papers.

The course keeps its declared exercise order, resumes unfinished drafts, filters status per exercise, and retains search/difficulty/topic filters in practice return links. The workspace returns directly to LLM Basics and offers all 18 modules in its picker, including transitions between source papers. Scroll restoration accepts module anchors as well as paper anchors. Legacy `lists/llm-basics` links remain supported.

Validation:

- Bilingual navigation checks cover direct exercise links, exact course order, partial-paper progress filtering, pending progress, ordered recommendations and cross-paper continuation.
- Local and production-hosted TypeScript/Vite builds pass the unchanged startup budgets. Hosted initial JavaScript: 408,368 bytes raw / 132,188 bytes gzip.
- Chrome preview confirms the flat 18-exercise list, course return link and whole-course module picker, including opening Attention from the foundation exercise. The local hosted preview cannot authenticate against the production API; no production submissions were made.

The accompanying classic-foundations PR was merged with fresh exports for StrideTorch 0.66. Native `paperleet ci --budget 600` passes all 85 modules, 11,349 executions over three repetitions, in 269.11 seconds. Real Pyodide 0.29.4 / NumPy 2.2.5 runs pass ResNet (9 tests, 6 mutants), BatchNorm (11 tests, 6 mutants) and VAE (10 tests, 7 mutants).

Full frontend `npm run check` passes. Before publication, current main advanced StrideTorch to 0.67; the source was integrated and all 85 exports regenerated. All three new exercises pass complete real WASM reference/mutant/learner checks again on 0.67, and the catalog, immutable bundle and course-navigation checks pass after regeneration.

After the concurrent administrator navigation update, the hosted Node 22.23.3 build exceeded the 130 KiB gzip limit by 126 bytes (409,560 raw / 133,246 gzip). Collection descriptions were already absent from the UI; removing those unused bilingual fields from the initial metadata reduces the exact Node 22.23.3 hosted graph to **408,681 raw / 132,850 gzip**. Names, source membership, routes, rendered content and the 400/130 KiB budgets are unchanged. Catalog and course-navigation regressions pass.
