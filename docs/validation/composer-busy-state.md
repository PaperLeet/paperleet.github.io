# Preserve writing context during asynchronous requests

Publishing, editing and inline comment requests now pass a separate `busy` flag to the shared Markdown editor. Busy textareas use native `readOnly` rather than `disabled`, so an in-flight keyboard submission retains its existing focus, draft, selection and scroll. Native selection and copying remain available. See [MDN disabled semantics](https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Attributes/disabled) for the focus distinction.

The toolbar, preview switch and submit shortcut stay locked while a request is pending. A bilingual status replaces the submission hint and is associated with the textarea; its indicator animates only when reduced motion is not requested. Account recovery and unavailable editors retain their separate native disabled state. Completing a request never introduces a new focus effect or overrides a newer user focus choice.

Coverage uses actual React editors and native forms in `check-community-submit-shortcut.mjs`: pending selection/scroll/focus preservation, immutable draft, accessible descriptions, copy key default, blocked duplicate submission, unlocked retry, hard-disable precedence and newer focus priority. Existing inline, discussion and editing checks remain in the full frontend suite.

An isolated React + AccountProvider + real HTTP/session + SQLite integration holds the first acknowledgement after the server commits. It checks the pending editor, deliberately loses the acknowledgement, then retries from the still-focused field without focus repair. The retry retains the publication identifier and creates only one persisted comment. Anonymous affiliation, immediate comment visibility, reading context, unsent drafts and account isolation remain covered. No production content is written; DOM verification does not claim browser visual review.

Required validation: full frontend checks, local and hosted builds within existing bundle caps, frozen clean-revision HTTP integration, unchanged native/browser-WASM/backend release gates, quality-gated Pages and exact served artifact/API revision verification.
