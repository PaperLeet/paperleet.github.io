import { useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react';
import AppLink from './AppLink';
import { appLocation } from './appLocation';
import { signInHref } from './accountNavigation';
import { useAccount } from './social';
import { useI18n } from './i18n';
import { rebuttalHref } from './rebuttalCloud';
import { createRebuttalJoin, sharingErrorText } from './rebuttalSharing';
import { forgetRebuttalShareToken, readRebuttalShareToken, rebuttalJoinHref, rememberRebuttalShareToken } from './rebuttalShareHandoff';

export default function RebuttalJoin({ initialToken, invalid }: { initialToken?: string; invalid?: boolean }) {
  const { t, locale } = useI18n();
  const account = useAccount();
  const [token, setToken] = useState<string>(), [handoffError, setHandoffError] = useState<'storage' | 'missing'>();
  const incoming = useRef(initialToken), captured = useRef(false), lastIncoming = useRef<string | undefined>(undefined), lastInvalid = useRef(false), current = useRef({ owner: '', token, enabled: false });
  const owner = account.user?.id || '';
  const available = account.capabilities?.rebuttal_invite_links === true;
  current.current = { owner, token, enabled: !account.loading && available && !!owner && !invalid && (!initialToken || initialToken === token) };
  const store = useMemo(() => createRebuttalJoin({ owner, token: token || '', canAct: () => current.current.enabled && current.current.owner === owner && current.current.token === token }), [owner, token]);
  const state = useSyncExternalStore(store.subscribe, store.getSnapshot, store.getSnapshot);
  function capture() {
    try {
      if (invalid) { forgetRebuttalShareToken(sessionStorage); setHandoffError('missing'); return; }
      if (incoming.current) rememberRebuttalShareToken(sessionStorage, incoming.current);
      const value = readRebuttalShareToken(sessionStorage);
      setToken(value); setHandoffError(value ? undefined : 'missing'); incoming.current = undefined;
    } catch (error) { setHandoffError(error instanceof Error && ['expired', 'invalid'].includes(error.message) ? 'missing' : 'storage'); setToken(undefined); }
  }
  useEffect(() => {
    const changed = initialToken && initialToken !== lastIncoming.current || invalid && !lastInvalid.current;
    lastIncoming.current = initialToken; lastInvalid.current = !!invalid;
    if (!captured.current || changed) {
      incoming.current = initialToken;
      captured.current = true; capture();
    }
    if (initialToken || invalid) {
      const url = new URL(location.href); url.hash = rebuttalJoinHref;
      window.history.replaceState(window.history.state, '', url);
      window.dispatchEvent(new window.HashChangeEvent('hashchange'));
    }
  }, [initialToken, invalid]);
  useEffect(() => { store.activate(); if (owner && token && available && !account.loading) void store.load(); }, [store, owner, token, available, account.loading]);
  useEffect(() => () => store.dispose(), [store]);
  useEffect(() => {
    if (state.invalid) { try { forgetRebuttalShareToken(sessionStorage); } catch { /* The invalid token can grant no access. */ } }
  }, [state.invalid]);
  useEffect(() => () => {
    const route = appLocation.hash;
    if (!route.startsWith('#/rebuttal/join') && !route.startsWith('#/account')) {
      try { forgetRebuttalShareToken(sessionStorage); } catch { /* The bounded handoff expires automatically. */ }
    }
  }, []);
  function leave() { try { forgetRebuttalShareToken(sessionStorage); } catch { /* Leave still works when storage is blocked. */ } appLocation.hash = rebuttalHref(); }
  async function accept() {
    const result = await store.accept();
    if (!result || current.current.owner !== owner || current.current.token !== token || !current.current.enabled) return;
    try { forgetRebuttalShareToken(sessionStorage); } catch { setHandoffError('storage'); return; }
    appLocation.hash = rebuttalHref(result.paperId);
  }
  const preview = state.preview;
  return <main className="rst-shell hreb-shell hreb-join"><h1>{t('论文协作邀请', 'Paper invitation')}</h1>
    {handoffError ? <div className="hreb-error" role="alert">{handoffError === 'storage' ? t('无法暂存邀请链接。请允许此网站使用浏览器存储后重试。', 'Cannot retain this invitation. Allow browser storage for this site and retry.') : t('邀请链接已失效，请重新打开邀请链接。', 'This invitation is unavailable. Open the invitation link again.')}
      {handoffError === 'storage' && <button className="rst-button" onClick={capture}>{t('重试', 'Retry')}</button>}</div>
      : account.loading || !token || initialToken && initialToken !== token ? <p role="status">{t('正在读取邀请…', 'Loading invitation…')}</p>
      : !account.user ? <AppLink className="rst-button rst-primary" href={signInHref(rebuttalJoinHref)}>{t('登录以查看邀请', 'Sign in to view invitation')}</AppLink>
      : !available ? <p className="hreb-notice">{t('邀请链接正在更新，请稍后刷新。', 'Invitation links are updating. Refresh shortly.')}</p>
      : <>{state.error && <p className="hreb-error" role="alert">{sharingErrorText(state.error, t)}</p>}
        {state.loading ? <p role="status">{t('正在读取邀请…', 'Loading invitation…')}</p> : preview && !state.invalid && <article className="hreb-join-paper"><h2>{preview.link.title}</h2><p>{preview.link.owner.display_name} (@{preview.link.owner.username})</p>
          <p>{[preview.link.venue, preview.link.role === 'editor' ? t('可编辑', 'Editor') : t('只读', 'Viewer')].filter(Boolean).join(' · ')}</p>
          <p>{t('有效期至', 'Expires')} {new Date(preview.link.expires_at).toLocaleDateString(locale === 'zh' ? 'zh-CN' : 'en-US')}</p>
          <button className="rst-button rst-primary" disabled={state.busy} onClick={() => void accept()}>{state.busy ? t('正在接受…', 'Accepting…') : state.pending ? t('重试接受', 'Retry acceptance') : t('接受邀请', 'Accept invitation')}</button>
        </article>}
        {state.error && !state.pending && !state.invalid && <button className="rst-button" disabled={state.busy || state.loading} onClick={() => void store.load()}>{t('刷新邀请', 'Refresh invitation')}</button>}
      </>}
    <button className="rst-button" disabled={state.busy} onClick={leave}>{t('返回我的论文', 'Back to my papers')}</button>
  </main>;
}
