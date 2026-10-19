# Study prompt reading continuity

For the current guide instance, collapsing a settled study guide keeps its native
hint disclosures mounted. Reopening and changing language reuse that content and
the reader's expanded hints without another import or a loading flash. This is
page-local reading state; it does not create progress records, grades or reports.

The footer provides **Collapse all hints**, scoped to that guide, and **Collapse
study prompts**. The latter returns focus to the guide summary only when focus
is still inside that guide, revealing a covered summary through the existing
reading viewport helper. Retained controls refuse actions while their outer
guide is closed. Other guides and a newer editor focus stay independent.

New topics immediately retire the previous body and hints. A closed unseen topic
does not load content. Collapsing during a pending read cancels its presentation
and deadline; late results cannot install content or affect a later topic. A
settled error remains available after reopening and requires explicit retry.
The existing bounded timeout, note unlock and source/practice links are retained.

Run from `web/`:

```sh
node scripts/check-preparation-guides.mjs
```

Two additional groups render real guides under StrictMode and exercise paired
guide isolation, unique native controls, retained DOM/hints, locale changes,
explicit keyboard collapse, hidden-control guards, deferred topic switches,
failure/retry, pending cancellation and late results. They preserve a separate
private draft, selection direction, both scroll positions, mount identity,
storage and URL. All 89 guides, 267 bilingual prompts, curricula, handbook
selection, practice/login returns and existing timeout/notes tests also run.

These checks use React/JSDOM. They do not verify physical viewport geometry,
screen-reader announcements or OS-level input/focus behavior on a real device.
