import { useEffect, useRef, useState } from 'react';
import { ShieldCheck } from 'lucide-react';
import AffiliationBadge from './AffiliationBadge';
import { useI18n } from './i18n';
import { errorMessage, SocialError, SocialNotice, socialApi, useAccount } from './social';
import type { AccountUser } from './social';

export default function AffiliationVerification({ user }: { user: AccountUser }) {
  const { t } = useI18n();
  const { capabilities, setUser } = useAccount();
  const [email, setEmail] = useState(''), [code, setCode] = useState('');
  const [request, setRequest] = useState<{ email: string; challenge: string; retryAt: number }>();
  const [busy, setBusy] = useState(false), [error, setError] = useState('');
  const [now, setNow] = useState(Date.now);
  const active = useRef(true);
  useEffect(() => { active.current = true; return () => { active.current = false; }; }, []);
  const remaining = request ? Math.max(0, Math.ceil((request.retryAt - now) / 1000)) : 0;
  useEffect(() => {
    if (!request || remaining === 0) return;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [request, remaining]);
  const verified = user.affiliation_verified === true || Boolean(user.affiliation);
  async function send() {
    if (busy || remaining) return;
    setBusy(true); setError('');
    try {
      const address = email.trim().toLowerCase();
      const data = await socialApi<{ sent: boolean; challenge: string; retry_after: number }>('account/affiliation/send-code', {
        email: address, expected_user_id: user.id,
      });
      if (!active.current) return;
      setRequest({ email: address, challenge: data.challenge, retryAt: Date.now() + data.retry_after * 1000 });
      setNow(Date.now()); setCode('');
    } catch (e) {
      if (active.current) setError(e instanceof SocialError && e.code === 'email_quota_exhausted'
        ? t('当前邮件额度已用完，请稍后再认证；你仍可继续使用账号。', 'Email capacity is full. Try verification later; you can continue using your account.') : errorMessage(e));
    } finally { if (active.current) setBusy(false); }
  }
  async function verify() {
    if (busy || !request) return;
    setBusy(true); setError('');
    try {
      const data = await socialApi<{ user: AccountUser }>('account/affiliation/verify', {
        email: request.email, challenge: request.challenge, code: code.trim(), expected_user_id: user.id,
      });
      if (!active.current) return;
      if (data.user.id !== user.id) throw new Error(t('账号已变化，请刷新后重试。', 'Your account changed. Refresh and try again.'));
      setUser(data.user);
      setRequest(undefined); setCode('');
    } catch (e) { if (active.current) setError(errorMessage(e)); }
    finally { if (active.current) setBusy(false); }
  }
  return <section className="social-panel affiliation-verification" aria-labelledby="affiliation-heading">
    <h3 id="affiliation-heading"><ShieldCheck size={16} aria-hidden="true" /> {t('学校／企业邮箱认证', 'School / company email verification')}</h3>
    {verified ? <>
      <p role="status">{t('学校／企业邮箱已认证，无需重复验证。', 'Your school/company email is verified. No additional verification is needed.')}</p>
      <AffiliationBadge affiliation={user.affiliation} institution={user.institution} />
      {user.affiliation_email && <p className="private-email">{user.affiliation_email}</p>}
    </> : <>
      <p>{t('选填：认证后可展示已验证的学校／企业归属。未认证也可登录、练习、发帖和回复。认证邮箱不会替换原来的登录账号。', 'Optional: verify your affiliation to display your verified school/company. You can sign in, practice, post and reply without affiliation verification. Your original sign-in account stays the same.')}</p>
      {capabilities?.affiliation_verification_available === true ? <form className="social-form" onSubmit={event => { event.preventDefault(); void (request ? verify() : send()); }}>
        <label>{t('学校／企业邮箱', 'School / company email')}<input type="email" required autoComplete="email" maxLength={254} value={request?.email ?? email} disabled={busy || Boolean(request)} onChange={event => setEmail(event.target.value)} /></label>
        <small>{t('支持 .edu、.edu.cn 和已审核的学校／企业域名。完整邮箱仅本人可见，机构名称会公开；邮箱认证仅证明邮箱归属。', 'Supports .edu, .edu.cn and reviewed school/company domains. Your full email stays private; your affiliation is public. Verification proves mailbox control only.')}</small>
        {request && <>
          <p role="status">{t('验证码已发送，10 分钟内有效。', 'Code sent. It expires in 10 minutes.')}</p>
          <label>{t('验证码', 'Verification code')}<input required inputMode="numeric" autoComplete="one-time-code" pattern="[0-9]{6}" maxLength={6} value={code} disabled={busy} onChange={event => setCode(event.target.value.replace(/\D/g, ''))} /></label>
          <div className="social-actions"><button type="button" className="text-button" disabled={busy || remaining > 0} onClick={() => void send()}>{remaining > 0 ? t(`${remaining} 秒后可重发`, `Resend in ${remaining}s`) : t('重发验证码', 'Resend code')}</button>
            <button type="button" className="text-button" disabled={busy} onClick={() => { setRequest(undefined); setCode(''); setError(''); }}>{t('更换邮箱', 'Change email')}</button></div>
        </>}
        <SocialNotice error={error} />
        <button className="button" type="submit" disabled={busy}>{busy ? t('正在处理…', 'Working…') : request ? t('确认认证', 'Verify email') : t('发送认证验证码', 'Send verification code')}</button>
      </form> : <p>{t('邮箱认证暂不可用，你仍可继续使用账号。', 'Email verification is temporarily unavailable. You can continue using your account.')}</p>}
    </>}
  </section>;
}
