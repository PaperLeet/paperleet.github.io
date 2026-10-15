# Interview experiences and preparation

Open **面经与准备 / Interviews** from the header. The hub has an experience feed,
a report-based concept ranking and a company index. Company, role, interview stage,
publication period, search, sort and page are carried in shareable local URLs.
Opening a thread, signing in or practicing a linked module preserves the hub context.

Signed-in members can share an experience with company, role, stage, optional outcome,
month and location, Markdown reflection and up to twelve concepts actually discussed.
Drafts and unconfirmed publication receipts are scoped to the account and current tab.
Preview does not publish. Retrying an unchanged request uses the same publication ID;
changing a draft cannot overwrite a previous post. Closing a tab is not a promise of
cross-device or permanent draft recovery. Keep a copy when browser storage is blocked.

Experiences are ordinary community threads: replies, saved posts, author edits,
version-conflict review and deletion continue to work. Metadata edits update the same
thread and the resulting counts. The standard community publication limits apply.

## What the counts mean

Only active authors' published, structured experiences count. The selected period uses
the report's publication date, not a guessed interview date. Concepts are ranked first
by distinct contributors across companies, then report mentions. A user's repeated
reports do not increase that contributor count. Editing, deleting or disabling content
changes the evidence. Removed catalog modules remain in historical posts but disappear
from current practice rankings. Company tags on problem pages link to this evidence.

These are attributed, self-reported concepts, **not official company questions or
official interview frequencies**. The empty state has no invented users or counts.
Curated preparation paths are curriculum recommendations and do not contribute to
the company statistics.

The bilingual taxonomy, preparation paths and official resource links live in
`paperleet/interview_catalog.json`, packaged with the public Python core so both app
editions and the account service use the same values. Official resources link to
Amazon Jobs, Microsoft Careers and OpenAI. Role-specific processes should be confirmed
with the recruiting team. The browser-only edition exposes the preparation resources
and links to the account application for community data.

## Validation

`web/scripts/check-interviews.mjs` checks routes, canonical metadata, account isolation,
drafts, publication recovery and edit versions. `check-interview-ui.mjs` renders the
actual components in an isolated DOM and exercises loading/error/empty states,
filter and account races, preview, drafts, retry IDs, cleanup failures, concept limits,
company evidence and metadata conflict review. These checks do not access a running
browser or write production content. The build verifies that the new pages remain
deferred and the existing initial-JavaScript budgets remain unchanged.
