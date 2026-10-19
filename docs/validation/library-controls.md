# Library pagination and control refinement

The library footer groups the visible result range and numbered navigation above
page-size and direct-jump controls. Direct entry is available whenever there is
more than one page, including an obsolete requested page. A single-page view
keeps only the relevant controls. Clicking the current page does not navigate.

Below 520 CSS pixels the numbered window becomes Previous / current page / Next;
the native jump field still reaches every page. Page controls retain 44px targets.
Active topic, difficulty, status and sort controls follow their existing URL state.
Search icons no longer shrink beside long text; select indicators do not intercept
native select clicks. Narrow filter groups put the topic on its own row. Shared
buttons have consistent minimum height, disabled/pressed feedback and reduced
motion support. List-row focus and hover emphasize the destination without
changing row geometry.

Validation for this change uses the actual React components in JSDOM:

- `check-library-pagination.mjs`: bounded rows, page/size changes, small-catalog
  direct jump, obsolete-page recovery, invalid/full-width/IME input, Escape,
  accessible list association, exact return URLs, locale continuity and preserved
  private code, selection, scroll and storage.
- `check-catalog-workbench.mjs`: density persistence, URL-synchronized difficulty
  and its active treatment, topic aliases and keyboard navigation.
- Existing library-navigation, community-pagination, interview-browse,
  ui-interactions and library-company checks verify shared controls and routes.
- TypeScript, catalog math and the production build validate the emitted assets.

These checks do not provide physical-browser layout, native select popup or
physical IME verification. This UI wave does not execute numerical SDK probes,
change catalog data, dispatch Actions, or deploy a site.
