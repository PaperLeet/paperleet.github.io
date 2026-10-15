import { useEffect, useRef } from 'react';
import { subscribePracticeChanges } from './practiceChanges';

/** Revalidate server-owned views on a commit hint, or when returning to an old tab. */
export function usePracticeInvalidation(refresh: () => void, owner?: string, problem?: string) {
  const callback = useRef(refresh);
  callback.current = refresh;
  useEffect(() => {
    let lastRefresh = Date.now();
    let dirty = false;
    const refreshIfVisible = () => {
      if (document.visibilityState === 'hidden' || navigator.onLine === false) return;
      if (!dirty && Date.now() - lastRefresh < 5_000) return;
      dirty = false;
      lastRefresh = Date.now();
      callback.current();
    };
    const unsubscribe = subscribePracticeChanges(change => {
      if (!owner || change.owner !== owner || change.kind !== 'submission' ||
          (problem && change.problem && change.problem !== problem)) return;
      dirty = true;
      refreshIfVisible();
    });
    window.addEventListener('focus', refreshIfVisible);
    window.addEventListener('online', refreshIfVisible);
    document.addEventListener('visibilitychange', refreshIfVisible);
    return () => {
      unsubscribe();
      window.removeEventListener('focus', refreshIfVisible);
      window.removeEventListener('online', refreshIfVisible);
      document.removeEventListener('visibilitychange', refreshIfVisible);
    };
  }, [owner, problem]);
}
