# Compact interview browsing filters

The interview hub keeps company, role and publication period beside the search
field. Stage and outcome, practice difficulty and duration, and location/month
inputs share one native **More filters** disclosure. Each select has a visible
label. The disclosure reports applied secondary fields and unapplied context
edits; applied filter chips remain outside it and can be removed while closed.

Collapsing never applies, resets or discards inputs. The explicit bottom collapse
button returns focus to the summary only if focus is still in that disclosure.
No asynchronous focus or automatic opening follows a filter response. An initial
visit with applied secondary fields starts expanded; later route changes respect
the reader's chosen open/closed state. Account changes retire unfinished inputs.

The location/month form keeps its existing explicit apply, range validation,
composition guard and capability gates. There is one form and no nested details.
Public response scopes, ranking order and all canonical browse/practice links
retain their previous meaning. Readonly editions expose only their usable
practice filters, and the company index has no empty advanced panel.

Validation from `web/`:

```sh
node scripts/check-interview-browse-filters.mjs
node scripts/check-interview-browse-filters.mjs --readonly
node scripts/check-interview-context.mjs
node scripts/check-interview-practice-filters.mjs
node scripts/check-interview-filter-support.mjs
```

The first two checks render the actual hub under StrictMode with bounded API
fixtures and fail on service writes. They exercise collapse/reopen, applied and
unapplied state, stable native controls, range errors, simulated composition,
locale and view changes, account retirement, exact routes, and preservation of a
separate private editor. The existing checks additionally cover live disposable
HTTP/SQLite context and practice flows through the release harness.

These are React/JSDOM and contract checks. Physical layout, native focus scrolling
and OS input-method behavior require a separate browser/device review.

Advanced-filter feedback refinement (2026-10-19): an explicit Apply identifies
the first invalid location/month field, links it to the translated error and
focuses that native input. Validation uses the existing location/month contract;
the form opts out of browser-specific validation bubbles so the same localized
message and field association are used for out-of-range months. An unrelated
edit keeps a still-valid error diagnosis; fixing a field updates or clears it
without moving focus. Restore and clear actions put focus on the retained
location input if their own button was focused. No error, restore or clear action
can claim a closed/hidden/inert/disconnected/retired form.

Escape from an active advanced-filter text field, summary or action collapses
the panel and returns focus to its summary while retaining unapplied input,
selection, validation and native field identity. Native selects, month/date/time
pickers, comboboxes, dialogs, composition, modified/repeated/handled keys and
retired controls keep their own keys. The hub supplies its current account/view
guard; the panel also checks the exact rendered URL before local actions.

Real React/JSDOM fixtures verify these actions in hosted and self-study editions
and retain private code, selection, scroll, storage and source scope. CSS gets
source review and production build checks. Native picker/IME interaction and
physical browser layout are not claimed as verified. No API, schema, dependency,
grading or numerical runtime change is introduced.

A partial month reported by the native input as `badInput` is also rejected
before applying, so an unfinished picker value cannot silently become an
unbounded date filter. The JSDOM fixture supplies that validity flag explicitly;
it does not establish physical picker behavior. The disposable HTTP fixture
checks invalid input makes no request, then applies a corrected September range
to the real feed and distinct-author summary with one matching synthetic report
and no frontend writes.
