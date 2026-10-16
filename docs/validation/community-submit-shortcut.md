# Community submit shortcut

Opt publishing Markdown editors into Command/Control + Enter. New posts, full discussions, inline comments, interview reports and post/reply edits use their existing form submission paths. A bilingual keyboard hint and `aria-keyshortcuts` expose the command; the hint joins existing descriptions without replacing writing guidance or formatting errors.

The command uses the focused textarea’s associated native form and first submit control. It respects native required-field validation and disabled controls, including fieldsets, hidden forms, edit-version review and account readiness. The existing submit handlers continue to own publication identifiers, expected account IDs, version checks, draft preservation and confirmation. Plain/Shift/Alt Enter, composition, legacy IME key code 229, and held-key repeats cannot accidentally submit.

Workspace run capture yields to Markdown editors, while the code editor keeps its run command. Handled publishing keys still bubble for reading-position and activity observers; only their native default is prevented. The workspace run guard also ignores legacy IME confirmation.

Validation: `check-community-submit-shortcut.mjs` mounts actual Markdown editors, native forms and the actual workspace shortcut hook. It covers submitter identity, validation, disabled/hidden/preview forms, opt-in, focus and form isolation, draft/caret/scroll, accessible bilingual hints and code-run ownership. Existing Markdown tests verify all five full authoring surfaces, while card and full-discussion tests publish through keyboard paths and retain their context and retry checks.

Isolated real React + AccountProvider + HTTP/session + SQLite integration covers native keyboard publication, a lost acknowledgement after the server commits, an idempotent keyboard retry, rapid duplicate retry commands, preserved unsent drafts and owner isolation. All test publications use synthetic local accounts. Full frontend checks, both production build configurations, complete cloud gates and actual deployment verification are required. DOM checks do not claim browser visual review.
