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
