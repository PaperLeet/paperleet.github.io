import type { AccountCapabilities, AccountUser } from './social';

/** Server eligibility is authoritative. Legacy responses need verified public affiliation. */
export function canPublishCommunity(user: AccountUser | null | undefined, capabilities?: AccountCapabilities): boolean {
  if (!user || user.status && user.status !== 'active') return false;
  if (user.is_test_user) return capabilities?.auth_mode === 'test' && user.can_publish_community !== false;
  if (typeof user.can_publish_community === 'boolean') return user.can_publish_community;
  if (user.email_verified !== true) return false;
  // Display-name catalogs and domain recognition must never confer publishing permission.
  return !!user.affiliation?.domain?.trim() && !!user.affiliation.name?.trim()
    || user.institution_verified === true && !!user.institution?.id && !!user.institution.name?.trim();
}

export function affiliationRequired(error: unknown): boolean {
  return !!error && typeof error === 'object' && 'code' in error && error.code === 'affiliation_required';
}
