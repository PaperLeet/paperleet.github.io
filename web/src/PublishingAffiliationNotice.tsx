import AppLink from './AppLink';
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
  const legacy=capabilities?.community_open_publishing!==true;
  if (!user || canPublishCommunity(user, capabilities)) return null;
  async function check() {
    if (checking || loading) return;
    setChecking(true); setError('');
    try { await refresh(); } catch (e) { setError(errorMessage(e)); }
    finally { setChecking(false); }
  }
  return <div className="publishing-affiliation-notice" role="status">
    <ShieldCheck size={18} aria-hidden="true"/>
    <div><strong>{legacy?t('当前服务要求认证后发布', 'This service requires verification to publish'):t('当前账号暂不可发布','Publishing is currently unavailable for this account')}</strong>
      <p>{legacy?t('此版本服务仍要求已验证的学校或企业邮箱。请更新服务或前往账号设置认证；当前草稿会保留。', 'This service version still requires a verified school or work email. Update the service or verify in account settings. Your current draft is kept.'):t('正常登录的账号无需学校／公司认证即可发帖。请刷新账号状态，或前往账号设置重新登录；当前草稿会保留。','Active signed-in accounts can post without school/company verification. Refresh your account or sign in again in account settings. Your current draft is kept.')}</p>
      <div className="publishing-affiliation-actions"><AppLink href="#/account">{t('账号设置', 'Account settings')}</AppLink><button type="button" disabled={checking || loading} onClick={() => void check()}>{checking || loading ? t('正在检查…', 'Checking…') : t('刷新账号', 'Refresh account')}</button></div>
      {(error || accountError) && <p className="publishing-affiliation-error" role="alert">{error || accountError}</p>}
    </div>
  </div>;
}
