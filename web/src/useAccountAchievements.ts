import { useCallback, useEffect, useState } from 'react';
import { readAccountAchievements, type AccountAchievements } from './accountAchievements';
import { socialApi, useAccount } from './social';
import { useI18n } from './i18n';

export function announceAchievementChange(userId:string) {
  window.dispatchEvent(new CustomEvent('paperleet-achievements-changed',{detail:userId}));
}

/** Scope every response to its authenticated owner, including in-flight saves. */
export function useAccountAchievements(trigger:string, enabled = true) {
  const {user,capabilities,loading:accountLoading} = useAccount();
  const {locale} = useI18n();
  const userId = user?.id;
  const supported = !!capabilities?.account_achievements;
  const [result,setResult] = useState<AccountAchievements>();
  const [error,setError] = useState<{userId:string;message:string}>();
  const [revision,setRevision] = useState(0);
  const refresh = useCallback(() => setRevision(value => value + 1),[]);
  const accept = useCallback((next:AccountAchievements) => setResult(current =>
    current?.user_id === next.user_id && current.revision > next.revision ? current : next),[]);
  useEffect(() => {
    if (!userId || !supported || !enabled || accountLoading) return;
    const controller = new AbortController();
    setError(undefined);
    socialApi<unknown>('account/achievements',undefined,controller.signal)
      .then(value => { if (!controller.signal.aborted) accept(readAccountAchievements(value,userId)); })
      .catch(() => { if (!controller.signal.aborted) setError({userId,message:locale === 'zh' ? '账号勋章同步失败，请重试。' : 'Could not sync account achievements. Please try again.'}); });
    return () => controller.abort();
  },[userId,supported,enabled,accountLoading,trigger,revision,locale,accept]);
  useEffect(() => {
    if (!userId || !supported || !enabled) return;
    const visibleRefresh = () => { if (document.visibilityState === 'visible') refresh(); };
    const changed = (event:Event) => { if ((event as CustomEvent).detail === userId) refresh(); };
    window.addEventListener('focus',visibleRefresh);
    document.addEventListener('visibilitychange',visibleRefresh);
    window.addEventListener('paperleet-achievements-changed',changed);
    // Reconcile other-device passes while this page remains open.
    const timer = window.setInterval(visibleRefresh,60_000);
    return () => {
      window.removeEventListener('focus',visibleRefresh);
      document.removeEventListener('visibilitychange',visibleRefresh);
      window.removeEventListener('paperleet-achievements-changed',changed);
      window.clearInterval(timer);
    };
  },[userId,supported,enabled,refresh]);
  return {snapshot:supported && enabled && !accountLoading && userId && result?.user_id === userId ? result : undefined,accept,
    error:error && error.userId === userId ? error.message : '',refresh,supported,accountLoading};
}
