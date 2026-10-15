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

## Company preparation spaces

Company cards open a preparation overview with that company's official resources,
community experiences and report-based concept ranking. The resource library contains
17 entries covering 16 companies. It distinguishes interview guides from careers
portals. Career entries link to official role/application information and make no claim
about specific interview rounds or question frequency. DeepSeek's recruitment entry
is linked from its official home page. Sources have a review date; recruiters remain
the source for a specific role's current arrangements. Companies without a listed
source show an explicit empty state rather than another company's resources.

Choose a role to narrow the curated preparation paths. These are editorial curriculum
recommendations, independent of the company evidence. In-page preparation/resource
buttons scroll and focus the destination heading while preserving the hash route and
respecting reduced-motion preferences. Filter chips can be removed independently;
returning from an empty pagination page preserves all the other filters.

Read requests have a 15-second timeout. Null, empty or primitive successful responses
show a retryable data error instead of an indefinite loader. Retry preserves the view,
late or aborted responses cannot overwrite it, and changing language translates the
error without issuing another request. `check-company-preparation.mjs` exercises the
actual components and controlled timers for these flows, without production writes.
