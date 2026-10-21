import { useI18n } from './i18n';
import type { AffiliationRecord } from './social';
import { resolveAffiliation } from './affiliationNames';
import './account-identity.css';

export default function AffiliationHistory({ affiliations }: { affiliations?: AffiliationRecord[] }) {
  const { t, locale } = useI18n();
  if (!affiliations?.length) return null;
  // The current institution is already on the identity card, so the record stays folded.
  return <section className="affiliation-history" aria-label={t('机构经历', 'Affiliation history')}><details>
    <summary><span>{t('机构经历', 'Affiliation history')}</span><span className="affiliation-history-count">{affiliations.length}</span></summary>
    <ul>{affiliations.map(item => {
      const organization = resolveAffiliation(item) || item;
      return <li key={item.id}>
      <strong>{locale === 'en' ? organization.name_en || organization.name : organization.name}</strong>
      <span>{item.status === 'past' ? t('历史机构', 'Past affiliation') : t('当前机构', 'Current affiliation')}{item.is_primary ? t(' · 主要机构', ' · Primary') : ''}</span>
      {item.position && <span>{item.position}</span>}
      {(item.start_date || item.end_date) && <span>{item.start_date || '…'} — {item.status === 'current' ? t('至今', 'Present') : item.end_date || t('已结束', 'Ended')}</span>}
      <small>{t('邮箱验证于 ', 'Mailbox verified on ')}<span className="affiliation-history-date">{item.verified_at.slice(0, 10)}</span></small>
    </li>; })}</ul>
    <p>{t('邮箱认证证明验证时的邮箱控制权；任职、在读状态和日期由用户填写。', 'Mailbox verification confirms access at the time of verification. Roles, current status and dates are provided by the user.')}</p>
  </details></section>;
}
