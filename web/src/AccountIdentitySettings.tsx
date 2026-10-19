import { useEffect, useRef, useState } from 'react';
import { Mail, ShieldCheck } from 'lucide-react';
import { useI18n } from './i18n';
import { errorMessage, socialApi, SocialNotice, useAccount } from './social';
import type { AccountUser, AffiliationRecord } from './social';
import { resolveAffiliation } from './affiliationNames';
import './account-identity.css';

export default function AccountIdentitySettings({ user }: { user: AccountUser }) {
  return <IdentityContents key={user.id} user={user} />;
}

function IdentityContents({ user }: { user: AccountUser }) {
  const { t } = useI18n();
  const { setUser, capabilities } = useAccount();
  const active = useRef(true), pending = useRef(false);
  const [busy, setBusy] = useState(false), [error, setError] = useState(''), [notice, setNotice] = useState('');
  const [email, setEmail] = useState(''), [code, setCode] = useState('');
  const [request, setRequest] = useState<{ email: string; challenge: string; retryAt: number }>();
  const [now, setNow] = useState(Date.now), [disable, setDisable] = useState<string>();
  useEffect(() => { active.current = true; return () => { active.current = false; }; }, []);
  const remaining = request ? Math.max(0, Math.ceil((request.retryAt - now) / 1000)) : 0;
  useEffect(() => { if (!remaining) return; const timer = setInterval(() => setNow(Date.now()), 1000); return () => clearInterval(timer); }, [remaining > 0]);
  function message(e: unknown) {
    const value = errorMessage(e);
    return value.startsWith('Sign in again before changing login methods')
      ? t('请先退出并重新登录，再于 15 分钟内更改登录方式。', 'Sign out and sign in again, then change your login methods within 15 minutes.') : value;
  }
  async function perform(action: () => Promise<void>) {
    if (pending.current || !active.current) return;
    pending.current = true; setBusy(true); setError(''); setNotice('');
    try { await action(); } catch (e) { if (active.current) setError(message(e)); }
    finally { pending.current = false; if (active.current) setBusy(false); }
  }
  function accept(next: AccountUser) {
    if (!active.current) return;
    if (next.id !== user.id) throw new Error(t('账号已变化，请刷新重试。', 'Your account changed. Refresh and try again.'));
    setUser(next);
  }
  async function send() {
    if (remaining) return;
    await perform(async () => {
      const address = (request?.email || email).trim().toLowerCase();
      const data = await socialApi<{ challenge: string; retry_after: number }>('account/login-emails/send-code', { email: address, expected_user_id: user.id });
      if (!active.current) return;
      setRequest({ email: address, challenge: data.challenge, retryAt: Date.now() + data.retry_after * 1000 }); setNow(Date.now()); setCode('');
    });
  }
  async function verify() {
    if (!request) return;
    await perform(async () => {
      const data = await socialApi<{ user: AccountUser }>('account/login-emails/verify', { email: request.email, code, challenge: request.challenge, expected_user_id: user.id });
      if (!active.current) return;
      accept(data.user); setRequest(undefined); setEmail(''); setCode('');
      setNotice(t('登录邮箱已绑定。可将它设为主要邮箱，原账号的数据全部保留。', 'Login email linked. You can make it primary; all data stays with this account.'));
    });
  }
  function change(id: string, action: 'primary' | 'disable') {
    void perform(async () => {
      const data = await socialApi<{ user: AccountUser }>('account/login-methods/change', { id, action, expected_user_id: user.id });
      if (!active.current) return;
      accept(data.user); setDisable(undefined);
      setNotice(action === 'primary' ? t('主要登录邮箱已更新，其他设备已退出。', 'Primary login email updated. Other devices were signed out.') : t('登录方式已停用，其他设备已退出。', 'Login method disabled. Other devices were signed out.'));
    });
  }
  return <>
    <section className="social-panel account-login-methods" aria-labelledby="login-methods-heading">
      <h3 id="login-methods-heading"><Mail size={16} aria-hidden="true" /> {t('登录方式与个人邮箱', 'Login methods & personal email')}</h3>
      <p>{t('建议绑定长期使用的个人邮箱。毕业或换工作不会改变账号、帖子和做题记录。机构认证邮箱不会自动获得登录权限。', 'Link a permanent personal email to keep access after graduation or a job change. Your account, posts and practice stay intact. Affiliation verification does not enable sign-in.')}</p>
      <ul className="account-method-list">{user.login_methods?.map(method => <li key={method.id}>
        <strong>{method.email}</strong>
        <span>{method.kind === 'email' ? t('邮箱验证码', 'Email code') : method.kind === 'github' ? 'GitHub' : method.kind === 'google' ? 'Google' : 'Microsoft'} · {method.is_primary ? t('主要邮箱', 'Primary') : method.enabled ? t('已启用', 'Enabled') : t('已停用', 'Disabled')}</span>
        {method.enabled && !method.is_primary && <div className="account-method-actions">
          {method.verified && <button type="button" className="text-button" disabled={busy} onClick={() => change(method.id, 'primary')}>{t('设为主要邮箱', 'Make primary')}</button>}
          <button type="button" className="text-button" disabled={busy} onClick={() => setDisable(method.id)}>{t('停用登录', 'Disable sign-in')}</button>
        </div>}
        {disable === method.id && <div className="identity-confirm" role="group" aria-label={t('确认停用登录', 'Confirm disabling sign-in')}>
          <p>{t('停用此登录方式并退出其他设备；机构认证历史和账号数据将保留。', 'Disable this login method and sign out other devices. Affiliation history and account data remain.')}</p>
          <button className="button subtle" type="button" disabled={busy} onClick={() => change(method.id, 'disable')}>{t('确认停用', 'Confirm disable')}</button>
          <button className="text-button" type="button" disabled={busy} onClick={() => setDisable(undefined)}>{t('取消', 'Cancel')}</button>
        </div>}
      </li>)}</ul>
      {capabilities?.login_email_binding_available && <form className="social-form" onSubmit={event => { event.preventDefault(); void (request ? verify() : send()); }}>
        <label>{t('添加登录邮箱（建议个人邮箱）', 'Add a login email (personal recommended)')}<input type="email" autoComplete="email" required maxLength={254} value={request?.email || email} disabled={busy || Boolean(request)} onChange={event => setEmail(event.target.value)} /></label>
        {request && <>
          <p role="status">{t('验证码已发送，10 分钟内有效。', 'Code sent. It expires in 10 minutes.')}</p>
          <label>{t('绑定验证码', 'Linking code')}<input required autoComplete="one-time-code" inputMode="numeric" pattern="[0-9]{6}" maxLength={6} value={code} disabled={busy} onChange={event => setCode(event.target.value.replace(/\D/g, ''))} /></label>
          <div className="account-method-actions"><button type="button" className="text-button" disabled={busy || remaining > 0} onClick={() => void send()}>{remaining ? t(`${remaining} 秒后可重发`, `Resend in ${remaining}s`) : t('重发验证码', 'Resend code')}</button>
            <button type="button" className="text-button" disabled={busy} onClick={() => { setRequest(undefined); setCode(''); }}>{t('更换邮箱', 'Change email')}</button></div>
        </>}
        <button type="submit" className="button" disabled={busy}>{busy ? t('正在处理…', 'Working…') : request ? t('确认绑定邮箱', 'Link email') : t('发送绑定验证码', 'Send linking code')}</button>
      </form>}
      <p className="account-fineprint">{t('更改登录方式需在重新登录后 15 分钟内完成。完整邮箱仅本人可见；绑定不会合并两个已有账号。', 'Change login methods within 15 minutes of signing in. Full emails remain private. Linking never merges existing accounts.')}</p>
      {error && <SocialNotice error={error} />}{notice && <p role="status">{notice}</p>}
    </section>
    <section className="social-panel account-affiliations" aria-labelledby="affiliations-heading">
      <h3 id="affiliations-heading"><ShieldCheck size={16} aria-hidden="true" /> {t('机构与经历', 'Affiliations & history')}</h3>
      <p>{t('可认证多个机构，并选择帖子旁展示的主要机构。毕业或离职后标为历史机构，认证记录仍然保留。', 'Verify multiple affiliations and choose the primary one shown beside posts. Mark an affiliation as past after graduation or leaving a job; its verification history remains.')}</p>
      {user.affiliations?.length ? user.affiliations.map(item => <AffiliationEditor key={`${user.id}:${item.id}:${item.revision}`} item={item} userId={user.id} />) : <p>{t('尚未添加机构。可在下方验证学校或企业邮箱。', 'No affiliations yet. Verify a school or company email below.')}</p>}
    </section>
  </>;
}

function AffiliationEditor({ item, userId }: { item: AffiliationRecord; userId: string }) {
  const { t, locale } = useI18n(); const { setUser } = useAccount();
  const organization = resolveAffiliation(item) || item;
  const [draft, setDraft] = useState({ id: item.id, revision: item.revision, position: item.position, start_date: item.start_date, end_date: item.end_date, status: item.status, is_primary: item.is_primary });
  const [busy, setBusy] = useState(false), [error, setError] = useState('');
  const active = useRef(true), pending = useRef(false);
  useEffect(() => { active.current = true; return () => { active.current = false; }; }, []);
  async function save() {
    if (pending.current) return; pending.current = true; setBusy(true); setError('');
    try {
      const data = await socialApi<{ user: AccountUser }>('account/affiliations/update', { ...draft, expected_user_id: userId });
      if (!active.current) return;
      if (data.user.id !== userId) throw new Error(t('账号已变化，请刷新重试。', 'Your account changed. Refresh and try again.'));
      setUser(data.user);
    } catch (e) { if (active.current) setError(errorMessage(e)); }
    finally { pending.current = false; if (active.current) setBusy(false); }
  }
  return <details className="affiliation-editor"><summary><strong>{locale === 'en' ? organization.name_en || organization.name : organization.name}</strong><span>{item.status === 'past' ? t('历史机构', 'Past') : item.is_primary ? t('当前 · 主要', 'Current · Primary') : t('当前机构', 'Current')}</span></summary>
    <form className="social-form" onSubmit={event => { event.preventDefault(); void save(); }}>
      <p className="private-email">{item.email}</p><small>{t('邮箱验证于 ', 'Mailbox verified on ')}{item.verified_at.slice(0, 10)}</small>
      <p>{t('认证证明当时可使用此邮箱；以下身份与日期由你填写。', 'Verification confirms mailbox access at that time; you provide the role and dates below.')}</p>
      <label>{t('身份／职位', 'Role / position')}<input maxLength={100} disabled={busy} value={draft.position} onChange={e => setDraft({ ...draft, position: e.target.value })} /></label>
      <label>{t('机构状态', 'Affiliation status')}<select value={draft.status} disabled={busy} onChange={e => setDraft({ ...draft, status: e.target.value as 'current' | 'past', is_primary: e.target.value === 'past' ? false : draft.is_primary, end_date: e.target.value === 'current' ? '' : draft.end_date })}><option value="current">{t('当前机构', 'Current')}</option><option value="past">{t('历史机构（已毕业／离职）', 'Past (graduated / left)')}</option></select></label>
      <label>{t('开始日期（选填）', 'Start date (optional)')}<input type="date" disabled={busy} value={draft.start_date} onChange={e => setDraft({ ...draft, start_date: e.target.value })} /></label>
      {draft.status === 'past' && <label>{t('结束日期（选填）', 'End date (optional)')}<input type="date" disabled={busy} value={draft.end_date} onChange={e => setDraft({ ...draft, end_date: e.target.value })} /></label>}
      {draft.status === 'current' && <label className="social-checkbox"><input type="checkbox" checked={draft.is_primary} disabled={busy || item.is_primary} onChange={e => setDraft({ ...draft, is_primary: e.target.checked })} /><span>{t('作为帖子旁的主要机构', 'Primary affiliation beside posts')}</span></label>}
      {error && <SocialNotice error={error} />}<button type="submit" className="button subtle" disabled={busy}>{busy ? t('正在保存…', 'Saving…') : t('保存机构信息', 'Save affiliation')}</button>
    </form>
  </details>;
}
