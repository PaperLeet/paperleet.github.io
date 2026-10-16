# Read complete replies inside community and interview lists

Long and formatted replies previously exposed only a four-line narrative excerpt. Code, formulae and tables required opening the discussion. Each visible reply now offers a native, bilingual **Read more / Show less** control when its bounded preview omits content or formatting. Plain replies within the 280-grapheme limit render completely, without a remaining CSS line clamp. The existing link opens the exact reply in its discussion and retains the list's return context.

Full Markdown mounts only after explicit expansion. It reuses the existing community renderer: highlighted, copyable code; KaTeX with untrusted macros; GFM tables; safe links; escaped raw HTML; image alt text instead of external image requests. The full view has controls above and below its body. Closing from the bottom returns focus to its compact control and brings that control into view only when needed, with an immediate scroll. A newer focus choice or route takes priority. Region IDs and disclosure state are per mounted reply; different body text starts collapsed. Language changes preserve the open view.

This is a frontend-only change. Expanding or collapsing a reply does not fetch the thread again, navigate, publish, or alter draft storage. It preserves the current composer DOM, draft, selection, selection direction and both scroll offsets. Closing the whole comment section still unmounts its DOM and retains the previously tested editor resume behavior. Anonymous author presentation and affiliation are unchanged.

## Verification

- The new card assertion first fails on the previous implementation because no in-place reading control exists.
- `node scripts/check-community-card-actions.mjs` exercises real native controls with React/jsdom: lazy rich reading, code/math/table rendering, image/raw-HTML exclusion, focus return and newer intent, complete short replies, editor preservation, and no additional requests or storage writes. The original like, comment publication, uncertain acknowledgement, sorting, account-isolation and collapsed-editor tests remain active.
- `node scripts/check-post-preview.mjs` checks formatting detection, complete medium-length Chinese replies, computed unclipped CSS, Unicode grapheme truncation, on-demand full rendering, closed/open control order, offscreen collapse, independent identical replies and region IDs, bilingual state, changed content and route guards. Existing narrative previews, post summaries, source links and failure recovery remain covered.
- Run the complete frontend suite, TypeScript build, public build, hosted build and build budgets before publication. The renderer remains in the deferred community/interview graph.
- Exercise the same flow against the real account/session HTTP API and a temporary SQLite database before releasing; synthetic content stays local. Production verification reads exact deployed assets and health only. No browser visual QA is claimed.
