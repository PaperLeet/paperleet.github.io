# PaperLeet brand assets

The PaperLeet mark is Naiwa: a round yellow face, green eyes, and a straight mouth.
This uses the first round-head version, before the eyes were enlarged. Light and
dark themes share the same transparent mark and the existing PaperLeet wordmark.

| Asset | Canvas | Use |
| --- | --- | --- |
| `paperleet-naiwa.png` | 256 × 256 | Transparent standalone mark |
| `paperleet-mark.svg` | 96 × 96 | Self-contained SVG wrapper around the selected PNG |
| `paperleet-logo-light.svg` | 320 × 80 | Wordmark for light backgrounds |
| `paperleet-logo-dark.svg` | 320 × 80 | Wordmark for dark backgrounds |
| `../favicon.svg` | 64 × 64 | Self-contained icon |
| `../favicon.png` | 96 × 96 | Browser and search icon |
| `../favicon.ico` | 16, 32, 48, 96 px | Browser icon sizes |
| `paperleet-brand-preview.svg` | 1200 × 640 | Light/dark wordmarks and small-size samples |
| `social-card.svg`, `social-card.png` | 1200 × 630 | Social sharing card |

The selected original is kept in `docs/brand/naiwa-round-v1.png`. It was generated
with the built-in imagegen tool from the user's reference and subsequent design
feedback. The final instruction changed only the head to a front-facing circle,
preserving the yellow color, green eyes, dark outlines, and straight mouth.
Production files remove excess transparent padding (1000 × 1000 crop at x=125,
y=122) and resize without changing the drawing. SVG exports embed the PNG; they
are not traced vector artwork. Existing wordmark glyphs remain outlined paths.
The navigation uses a 96 px copy at `web/src/assets/paperleet-naiwa.png`, imported
by Vite so its URL changes when the image changes.

`paperleet-cat.svg` is a legacy asset retained for old links. It is no longer the
active brand mark.
