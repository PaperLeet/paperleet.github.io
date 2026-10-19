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
20 entries covering 16 companies. It distinguishes interview guides from careers
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

## Private target companies / 我的目标公司

In the account application, use **Set target / 设为目标** on a company card or
preparation overview to save up to twelve companies. **My target companies / 我的目标公司**
is a collapsed shortcut list above the hub filters. Its links open company preparation
with the current role, period and other applicable report filters. Search, saved-report
mode and captured practice batches are cleared for this navigation.

Targets belong to the signed-in account and sync through the private account service;
they are not published to profiles, company counts or affiliation. Other tabs refresh
on focus or a storage invalidation containing only an account ID and nonce. The
selection itself is not kept in local storage. An uncertain save retains the last
confirmed state and retries the original add/remove direction. Switching accounts
retires in-flight requests and hides the previous account's list immediately.

Guests can sign in from a company card; signing in does not automatically add a target.
Read-only editions and older services without the capability omit these controls.
Removed catalog companies can still be removed from the private list.

`check-company-target-store.mjs` checks ownership, revisions, concurrent responses and
retry behavior. `check-company-targets.mjs` exercises the actual hub in React/JSDOM,
including StrictMode, native links, account retirement and source/draft preservation;
`--readonly` checks the browser-only edition. Private service tests cover real HTTP,
separate sessions, authorization, concurrent capacity and idempotent updates. These
checks do not prove native layout, device focus or real identity-provider behavior.

### Target-company capacity and recovery

When twelve targets are already confirmed, the company control opens the shortcut
manager without sending another add. The manager states the limit and lets you remove
one. Read failures also keep an actionable status control that opens the retry area.
Neither action silently changes targets or the interview filters.

A compatible service returns `400 / company_targets_full` only for an atomic, definite
capacity rejection. The UI labels that addition as not saved, distinct from a lost or
malformed acknowledgement. Refresh the current list or confirm a removal before
retrying the original add; it never adds automatically. A definite rejected addition
can be canceled locally. An uncertain request retains its original direction and
cannot be discarded as though the server had rejected it. Legacy services without
the specific code keep the existing uncertain-result recovery.

Opening management and canceling a rejected addition move focus synchronously to
the native summary. Background reads and late saves do not take focus from newer
editing. Failed-add notices and captured intent stay in memory for the current
account/provider lifetime; they are not cross-device drafts. `check-company-target-recovery.mjs` checks the real hub, capacity races, late/pre-rejection reads, explicit
recovery, locale and account transitions in React/JSDOM; it also supports a disposable
real HTTP/SQLite bridge. It does not prove native browser geometry or device focus.

## Personal learning states in company comparison

An applied comparison can show All learning states, Not started, In progress or
Passed, using the viewer's actual workspace evidence. Public company/contributor
counts and source order are independent of this personal filter. Missing exercise
details never count as unstarted; the page states how many source concepts are
unavailable and the All state keeps their source links. Completion is summarized
only when the full current comparison has loaded metadata and confirmed progress.

Row actions open practice directly with an ordered, filtered six-concept snapshot.
Practice, sign-in, the module navigator and return keep the pair, public scope,
learning filter and captured queue. Later grades do not rewrite that queue. Changing
only the learning filter retains an existing selected batch and its order, including
items outside the new list; unapplied company-search text also stays intact.

Study filters are canonical only in concept views or applied company comparisons.
A shared view uses the opener's own progress; shared captured batches clear the
sender's learning-state filter. The URL contains public identifiers and filter enums,
never grades, counts, code or report drafts. Progress/catalog errors keep the applied
filter, show a recovery action and allow an explicit return to All.

`check-comparison-learning.mjs` checks the actual hub, filters, row practice, navigator,
next-step, owner/locale transitions and exact returns in React/JSDOM, with an optional
disposable real HTTP/SQLite public-summary bridge. It does not establish native
browser geometry, physical focus, IME, clipboard or live-provider behavior.

## Question and follow-up records

The interview composer has an optional **Record a question and follow-ups / 整理一条问题与追问**
panel below the report editor. Keep the actual question, your approach and follow-up
reflection in separate inputs, optionally naming an already selected concept.
Opening the panel, typing or choosing a concept does not publish, insert text or
change the report's linked concepts. Generic questions need no exercise association.

Insert commits one ordinary Markdown record at the report cursor without replacing
the selection or existing text. The body remains the canonical published content:
records participate in ordinary reading, search, editing and sharing. Counts still
come only from explicitly selected concepts. A length or incomplete Markdown/fence
boundary failure keeps both source and inputs intact. Undo restores the prior report,
record input, selection direction and both textarea scroll offsets; newer body or
record edits disable that undo. Clear fields has a separate explicit confirmation.

Uninserted inputs stay in the owner-bound draft in the current tab. The writing draft
center labels, previews/searches and resumes them. They are not automatically part
of a publication payload. Insert or clear them before publishing the current draft.
An explicit original-request retry can still confirm an earlier publication; its
cleanup cannot erase a newer private record. If the optional panel fails to load,
a plain editor preserves the inputs and can insert without a concept label.

`check-question-records.mjs` exercises parsing, cold/storage failure recovery, draft
center resumption, actual composer insertion/undo, locale/owner transitions, fallback
editing, captured publication retry and safe Markdown readback. The optional
disposable HTTP/SQLite bridge proves a single idempotent public post, searchable
inserted content and private uninserted content. No new API field or service schema
is introduced; native browser geometry/focus/IME/clipboard is not established by
these component and HTTP checks.
