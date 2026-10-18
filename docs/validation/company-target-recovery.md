# Target-company recovery

Definite capacity rejections are distinct from uncertain transport outcomes. Only a
literal status `400` and code `company_targets_full` classify as a definite rejected add.
Other/legacy errors retain captured retry direction and cannot be locally canceled
as capacity failures. The service raises a ValueError-compatible capacity subtype;
the route adds the code while preserving status `400`, no-store, owner/role/origin
checks and closed command schemas. Capacity failures leave revision and targets
unchanged inside the existing immediate transaction. No schema or cap is changed.

A read started before a rejection cannot authorize retry. A later confirmed snapshot
or mutation recomputes available space without adding automatically. A locally full
card opens the native private manager and explains the limit. Read/account failures
keep a direct route to status/retry. Explicit cancel only clears a definite rejected
add; it writes nothing. New account generations retire all requests/notices.

The new check renders actual hub/provider/card/summary components and checks stale
capacity, full-list management, refresh, removal and original-add retry, cancel, lost
acknowledgement, locale/caret/draft continuity, read failure and owner isolation.
Private tests prove exact HTTP code, unchanged revision/state, repeated rejection,
idempotent success after removal, unknown/invalid command distinction, cross-owner
privacy, no-store, origin and unaffected practice/public content.

Scope: TypeScript, actual React/JSDOM and disposable HTTP/SQLite. No production
writes or native browser layout, physical focus, IME, clipboard or provider claims.
