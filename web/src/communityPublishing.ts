import type { AccountCapabilities, AccountUser } from './social';

/** Current services allow active accounts without affiliation; honor explicit server denial. */
export function canPublishCommunity(user: AccountUser | null | undefined, capabilities?: AccountCapabilities): boolean {
  if (!user || user.status && user.status !== 'active') return false;
  if (user.is_test_user) return capabilities?.auth_mode === 'test' && user.can_publish_community !== false;
  if (typeof user.can_publish_community === 'boolean') return user.can_publish_community;
  if (capabilities?.community_open_publishing === true) return true;
  if (user.email_verified !== true) return false;
  // Display-name catalogs and domain recognition must never confer publishing permission.
  return !!user.affiliation?.domain?.trim() && !!user.affiliation.name?.trim()
    || user.institution_verified === true && !!user.institution?.id && !!user.institution.name?.trim();
}

export function affiliationRequired(error: unknown): boolean {
  return !!error && typeof error === 'object' && 'code' in error && error.code === 'affiliation_required';
}
