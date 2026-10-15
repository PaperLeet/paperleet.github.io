import { useI18n } from './i18n';
import { resolveAffiliation } from './affiliationNames';
import OrganizationLogo from './OrganizationLogo';
import type { Affiliation, Institution } from './social';

export default function AffiliationBadge({ affiliation, institution }: { affiliation?: Affiliation | null; institution?: Institution | null }) {
  const { locale, t } = useI18n();
  const organization = resolveAffiliation(affiliation, institution);
  if (!organization) return null;
  const name = locale === 'en' ? organization.name_en || organization.name : organization.name;
  return <span className="institution-badge" aria-label={t(`所属机构：${name}`, `Affiliation: ${name}`)}>
    <OrganizationLogo id={organization.id} size={16} /><span className="affiliation-name">{name}</span>
  </span>;
}
