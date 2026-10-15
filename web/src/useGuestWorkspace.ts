import { useMemo, useRef, useSyncExternalStore } from 'react';
import { createGuestWorkspace, type GuestOwner } from './guestWorkspace';

export function useGuestWorkspace(owner?: GuestOwner) {
  const active = useRef<ReturnType<typeof createGuestWorkspace> | undefined>(undefined);
  const workspace = useMemo(() => {
    const next = createGuestWorkspace(owner, () => active.current === next);
    return next;
  }, [owner]);
  active.current = workspace;
  const view = useSyncExternalStore(workspace.subscribe, workspace.getSnapshot, workspace.getSnapshot);
  return { ...workspace, ...view };
}
