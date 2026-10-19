# Editor and shared control interactions

The filename is part of the Code tab. A single action disclosure shares the same row beside the workspace tabs; there is no separate file toolbar. Find, replace,
function navigation, line navigation and keyboard help open on demand. Font size,
copy, reset and download live in the action disclosure. Native disclosure, field
and select semantics preserve Tab, Enter, Escape and platform keyboard behavior.

## Editor shortcuts

| Action | Windows / Linux | macOS |
| --- | --- | --- |
| Find | Ctrl+F | Cmd+F (Ctrl+F also accepted) |
| Replace | Ctrl+H | Cmd+Option+F |
| Next / previous match | Enter / Shift+Enter, F3 / Shift+F3 | Enter / Shift+Enter, F3 / Shift+F3 |
| Case / whole word / regex in Find | Alt+C / W / R | Cmd+Option+C / W / R |
| Select all matches in Find | Alt+Enter | Option+Enter |
| Replace all with Replace expanded | Ctrl+Enter | Ctrl+Enter |
| Go to line (also line:column) | Ctrl+G | Ctrl+G |
| Go to function or class | Ctrl+Shift+O | Cmd+Shift+O |
| Keyboard help | F1, Ctrl+Shift+P | F1, Cmd+Shift+P |
| Select next occurrence | Ctrl+D | Cmd+D |
| Toggle comment | Ctrl+/ | Cmd+/ |
| Move line | Alt+Up / Down | Option+Up / Down |
| Indent / outdent | Tab / Shift+Tab | Tab / Shift+Tab |
| Run tests from code | Ctrl+Enter | Cmd+Enter (Ctrl+Enter also accepted) |
| Close editor overlay | Escape | Escape |

Search is scoped to the code editor. Browser Find remains available outside it.
Search inputs and navigation palettes do not invoke Run or Submit. The replacement
shortcut only replaces while the replacement row is visible. Read-only workspaces
cannot replace code. CodeMirror owns matching, history and capture-group expansion
(`$1`, `$2`, `$&`); a replace-all is one undo transaction. Invalid regex and no-match
states appear inline. Counts cap at `1000+` to keep large match lists bounded.
Changing language updates the custom panel labels without recreating editor DOM,
query input or the current selection. Function navigation validates the exact
current source before jumping, including deferred source updates.

Shared form controls have consistent radii, select chevrons, focus rings, disabled
and invalid states, and coarse-pointer sizes. Native selects remain native, so
arrow keys and type-ahead work without a separate custom menu implementation.

## Validation

`node scripts/check-editor-search.mjs` exercises the actual mounted CodeMirror:
hidden default, keyboard scope, match counts, flags, invalid and zero-width regex,
capture replacement, undo, read-only guards, selection prefill, run isolation,
line/symbol/help commands, stable locale focus and bounded counts.

The existing Python navigation check retains AST parity over all 178 maintained
sources and now tests the on-demand palette. Existing editor-tool, workspace,
keyboard ownership and catalog-navigation checks cover the action disclosure,
copy/reset recovery, retained drafts, caret, selection and scroll position.

Search, navigation and tool code/styles remain deferred with the editor. The
existing initial-JavaScript build budget is unchanged. No production release is
performed by these local UI changes.

References: [VS Code editing](https://code.visualstudio.com/docs/editing/codebasics),
[VS Code keybindings](https://code.visualstudio.com/docs/reference/default-keybindings),
[CodeMirror search API](https://codemirror.net/docs/ref/#search).
