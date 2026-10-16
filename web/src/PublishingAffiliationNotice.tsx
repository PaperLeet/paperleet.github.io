import { useState } from 'react';
import { ShieldCheck } from 'lucide-react';
import { useI18n } from './i18n';
import { errorMessage, useAccount } from './social';
import { canPublishCommunity } from './communityPublishing';
import './publishing-affiliation.css';

export default function PublishingAffiliationNotice() {
  const {user, capabilities, refresh, loading, error: accountError} = useAccount();
  const {t} = useI18n();
  const [checking, setChecking] = useState(false), [error, setError] = useState('');
  if (!user || canPublishCommunity(user, capabilities)) return null;
  async function check() {
    if (checking || loading) return;
    setChecking(true); setError('');
    try { await refresh(); } catch (e) { setError(errorMessage(e)); }
    finally { setChecking(false); }
  }
  return <div className="publishing-affiliation-notice" role="status">
    <ShieldCheck size={18} aria-hidden="true"/>
    <div><strong>{t('验证机构邮箱后可发布', 'Verify a school or work email to publish')}</strong>
      <p>{t('发帖和回复需要已验证的学校或企业邮箱。所属机构将公开展示，完整邮箱不会公开。请使用学校或企业账号登录；当前草稿会保留。', 'Posts and replies require a verified school or work email. Your affiliation will be public; your full email stays private. Sign in with your school or work account. Your current draft is kept.')}</p>
      <div className="publishing-affiliation-actions"><a href="#/account">{t('账号设置', 'Account settings')}</a><button type="button" disabled={checking || loading} onClick={() => void check()}>{checking || loading ? t('正在检查…', 'Checking…') : t('刷新账号', 'Refresh account')}</button></div>
      {(error || accountError) && <p className="publishing-affiliation-error" role="alert">{error || accountError}</p>}
    </div>
  </div>;
}
