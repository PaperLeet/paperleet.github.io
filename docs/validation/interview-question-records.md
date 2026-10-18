# Interview question records

One parent-owned optional draft field holds selected concept ID plus prompt,
approach and follow-ups. It is bounded, Unicode-validated and account/tab scoped.
It survives cold storage restoration, in-page quota failure, draft-center search
and resumption. It is excluded from publication payloads and receipt signatures.
Private scratch participates in durability and cleanup equality so an old receipt
cannot erase a newer record even when the public body has not changed. A separate
newer-draft hint exposes the explicit original retry while the new-draft submit is
blocked. No automatic post, association, grade or code attachment is added.

Insertion preserves all existing source, uses a codepoint-safe cursor boundary and
ordinary Markdown headings. Parser checks prove generated headings are top-level
and a temporary boundary is not swallowed by unfinished code, HTML or math. The
boundary is never published. Length/position/module/empty failures preserve draft
and body. An optional concept must be selected and loaded; a generic record needs
none. Undo has exact before/after guards and restores scratch, caret direction and
two-scroll state. New record edits, including concept choice, block overwrite.

The optional panel is lazy. Failure fallback remains editable, can retry in place
and can explicitly insert without a concept label. Clear confirmation focuses the
stable native summary. Explicit insertion/undo uses the existing focus return with
a newer-user-action/route/visibility/owner guard; background completion cannot
override subsequent input. Existing outline tools retain their behavior.

Actual React/JSDOM tests cover composer, draft center, retry and safe Markdown
reading. The optional real HTTP/SQLite path uses synthetic local accounts and a
maintained catalog concept, confirms one post after a lost acknowledgement and
checks public search/counts without exposing uninserted scratch. This does not
prove native layout, physical focus, IME/clipboard or live provider behavior.
