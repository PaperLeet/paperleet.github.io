import { useState, type ReactNode } from 'react';
import { useI18n } from './i18n';
import type { AccountUser } from './social';

type AccessProps = {
  user: AccountUser | null;
  onSignOut: () => Promise<void>;
  onRefresh: () => Promise<void>;
  refreshFailed?: boolean;
  children?: ReactNode;
};

export function AccountAccessGate({ user, onSignOut, onRefresh, refreshFailed, children }: AccessProps) {
  return user?.status === 'waitlisted'
    ? <Waitlist user={user} onSignOut={onSignOut} onRefresh={onRefresh} refreshFailed={refreshFailed} />
    : children;
}

export function Waitlist({ user, onSignOut, onRefresh, refreshFailed }: Omit<AccessProps, 'children' | 'user'> & { user: AccountUser }) {
  const { t, locale, setLocale } = useI18n();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  async function act(action: () => Promise<void>) {
    if (busy) return;
    setBusy(true);
    setError('');
    try { await action(); }
    catch { setError(t('暂时无法完成操作，请稍后重试。', 'We could not complete this action. Please try again shortly.')); }
    finally { setBusy(false); }
  }
  return <main className="social-page waitlist-page">
    <header className="waitlist-header">
      <strong>PaperLeet</strong>
      <button type="button" onClick={() => setLocale(locale === 'zh' ? 'en' : 'zh')} aria-label={t('切换为英文', 'Switch to Chinese')}>{locale === 'zh' ? 'English' : '中文'}</button>
    </header>
    <section className="social-panel waitlist-card" aria-labelledby="waitlist-title">
      <h1 id="waitlist-title">{t('已加入等待名单', 'You’re on the waitlist')}</h1>
      <p className="social-description">{t('你的 PaperLeet 账号已创建。', 'Your PaperLeet account has been created.')}</p>
      <p className="waitlist-email">{user.email}</p>
      <p className="social-description">{t('目前仅向学校／企业账户开放使用。你的账号已加入等待名单，暂时无法使用练习、判题和社区功能。', 'Access is currently open to school and company accounts only. Your account is on the waitlist, so practice, judging and community features are not available yet.')}</p>
      <p className="social-description">{t('后续开放个人账户后，你可以使用同一个账号登录，无需重复注册。', 'When access opens to personal accounts, you can sign in with this same account without registering again.')}</p>
      <div className="waitlist-actions" aria-busy={busy}>
        <button type="button" className="button primary" disabled={busy} onClick={() => void act(onSignOut)}>{t('退出并切换账号', 'Sign out and switch accounts')}</button>
        <button type="button" className="button" disabled={busy} onClick={() => void act(onRefresh)}>{t('刷新状态', 'Refresh status')}</button>
      </div>
      {error && <p className="social-notice error" role="alert">{error}</p>}
      {!error && refreshFailed && <p className="social-notice error" role="alert">{t('暂时无法刷新状态，请稍后重试。你的等待名单记录已保留。', 'Your status could not be refreshed. Please try again shortly; your waitlist entry is saved.')}</p>}
    </section>
  </main>;
}
