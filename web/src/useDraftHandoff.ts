import { useEffect, useState } from 'react';
import { claimHandoff, consumeHandoff, HANDOFF_TTL, saveHandoff, type DraftHandoff } from './draftHandoff';

export function useDraftHandoff(owner: string) {
  const [pending,setPending] = useState<DraftHandoff>();
  const [error,setError] = useState('');
  useEffect(() => {
    if (owner === 'unresolved') return;
    try { setPending(claimHandoff(sessionStorage,owner)); setError(''); }
    catch { setPending(undefined); setError('storage'); }
  },[owner]);
  useEffect(() => {
    if (!pending) return;
    const timer = setTimeout(() => discard(pending.id), Math.max(0,HANDOFF_TTL-(Date.now()-pending.created)));
    return () => clearTimeout(timer);
  },[pending?.id,pending?.created]);
  function prepare(problem: string, code: string) {
    if (owner !== 'anonymous') return false;
    try {
      const value: DraftHandoff = {version:1,id:crypto.randomUUID(),problem,code,created:Date.now()};
      saveHandoff(sessionStorage,value);setPending(value);setError('');return true;
    } catch { setError('storage');return false; }
  }
  function discard(id: string) {
    try { consumeHandoff(sessionStorage,id); }
    catch { setError('storage'); }
    setPending(previous => previous?.id === id ? undefined : previous);
  }
  return { pending: pending?.owner === owner && owner.startsWith('user:') ? pending : undefined, error, prepare, discard };
}
