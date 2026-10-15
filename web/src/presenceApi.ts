import { apiFetch } from './apiClient';
import { publicSite, hostedAppAvailable, validateHostedAppUrl } from './deployment';

export const presenceHeartbeatPath = '/api/presence/heartbeat';
type PresenceDeployment = { publicSite: boolean; hostedAppAvailable: boolean; hostedAppUrl?: string };
type PresenceRequest = (path: string, options?: RequestInit) => Promise<Response>;

/** The public catalog may send heartbeats, without enabling its account APIs. */
export function createPresenceRequest(
  deployment: PresenceDeployment,
  accountApi: PresenceRequest = apiFetch,
  browserFetch: typeof fetch = fetch,
): PresenceRequest | undefined {
  if (deployment.publicSite && !deployment.hostedAppAvailable) return;
  const origin = deployment.publicSite ? validateHostedAppUrl(deployment.hostedAppUrl || '') : '';
  return (path, options = {}) => {
    if (path !== presenceHeartbeatPath) throw Error('Expected the presence heartbeat endpoint');
    return deployment.publicSite
      ? browserFetch(origin + path, { ...options, credentials: 'include' })
      : accountApi(path, options);
  };
}

export const presenceRequest = createPresenceRequest({
  publicSite,
  hostedAppAvailable,
  hostedAppUrl: import.meta.env.VITE_HOSTED_APP_URL,
});
