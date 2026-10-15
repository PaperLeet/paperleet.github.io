import { useBrowserWorkspace } from './useBrowserWorkspace';
import type { BrowserOwner } from './browserWorkspaceStore';

/** Guest drafts stay in this origin's browser cache, separate from every account. */
export function useGuestWorkspace(owner?: BrowserOwner) {
  return useBrowserWorkspace(owner);
}
