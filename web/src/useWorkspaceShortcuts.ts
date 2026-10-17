import { useEffect, useRef } from "react";

type ShortcutEvent = Pick<KeyboardEvent, "key" | "metaKey" | "ctrlKey" | "altKey" | "shiftKey" | "repeat" | "isComposing" | "defaultPrevented"> & {keyCode?:number};
export function isRunShortcut(event: ShortcutEvent) {
  return event.key === "Enter" && (event.metaKey || event.ctrlKey) && !event.altKey && !event.shiftKey && !event.repeat && !event.isComposing && event.keyCode!==229 && !event.defaultPrevented;
}

type Actions = { enabled: boolean; canRun: boolean; focused: boolean; onRun: () => void; onExitFocus: () => void };
export function useWorkspaceShortcuts(actions: Actions) {
  const latest = useRef(actions);
  useEffect(() => { latest.current = actions; });
  useEffect(() => {
    // Discussion editors own their publishing chord, even inside a problem page.
    const blocked = (event: KeyboardEvent) => !latest.current.enabled || (event.target as HTMLElement | null)?.closest('.markdown-editor, .discussion-outline-filter, .community-code-block, .submission-completion-notice, .submission-completion-fallback, [role="dialog"], dialog[open]') || !!document.querySelector('dialog[open]');
    // Capture the run chord before the editor treats Enter as a new line.
    const handleRun = (event: KeyboardEvent) => {
      if (blocked(event) || !isRunShortcut(event)) return;
      event.preventDefault();
      event.stopPropagation();
      if (latest.current.canRun) latest.current.onRun();
    };
    // Escape bubbles so editor completion menus can handle it first.
    const handleEscape = (event: KeyboardEvent) => {
      if (blocked(event) || event.key !== "Escape" || event.defaultPrevented || event.repeat || event.isComposing || event.keyCode===229 || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey || !latest.current.focused) return;
      event.preventDefault();
      latest.current.onExitFocus();
    };
    window.addEventListener("keydown", handleRun, true);
    window.addEventListener("keydown", handleEscape);
    return () => { window.removeEventListener("keydown", handleRun, true); window.removeEventListener("keydown", handleEscape); };
  }, []);
}
