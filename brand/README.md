# PaperLeet brand assets

The PaperLeet mark is a smiling black cat emoji with a soft, abstract front-facing head, closed crescent eyes, warm coral cheeks, and an open smile with a small coral tongue. Its cream circular badge keeps the cat visible on both light and dark backgrounds. The geometry and colors match `web/src/Brand.tsx`.

| Asset | Canvas | Use |
| --- | --- | --- |
| `paperleet-logo-light.svg` | 320 × 80 | Full wordmark on light backgrounds; transparent canvas |
| `paperleet-logo-dark.svg` | 320 × 80 | Full wordmark on dark backgrounds; transparent canvas |
| `paperleet-mark.svg` | 96 × 96 | Standalone black cat on its cream circular badge |
| `paperleet-cat.svg` | 96 × 96 | Black cat head on a transparent canvas; use on light backgrounds |
| `../favicon.svg` | 64 × 64 | Browser icon with the same cream badge |
| `paperleet-brand-preview.svg` | 1200 × 640 | Light and dark logo sheet with 96, 32, and 16 px icon samples |

| Color | Value |
| --- | --- |
| Black cat | `#161619` |
| Closed eyes, smile, and badge | `#f3efe6` |
| Cheeks and tongue | `#f09996` |
| Light-theme wordmark | `#262626` |
| Dark-theme wordmark | `#efeff1` |

The cat stays black in both interface themes. Keep the cream badge on dark backgrounds; do not invert the head to white. SVG assets contain native paths and shapes with no external references or runtime fonts. Wordmark and preview-label glyphs are outlined from Arial Bold; the website header uses its system font stack.
