import { communityContinuation } from './communityNavigation';
import { libraryContinuation } from './libraryNavigation';

const continuationPrefix = "account/continue/";
const errorPrefix = "account/error/";
const loginErrors = new Set(["identity", "expired", "cancelled", "unavailable", "rate_limited"]);

/** Allow only known shapes of local hash routes; never an external destination. */
function returnRoute(route: string): string | undefined {
  if (route.startsWith('#')) return undefined;
  if (route.includes('?')) return communityContinuation(route) ?? libraryContinuation(route);
  if (route === "" || route === "community" || route === "submit") return route;
  if (/^[1-9]\d*$/.test(route)) return route;
  if (/^community\/problem\/[a-z0-9_-]+\/[a-z0-9_-]+$/.test(route)) return route;
  if (/^community\/[a-f0-9]{32}$/.test(route)) return route;
  if (/^u\/[a-z0-9_]{3,24}$/.test(route)) return route;
  if (/^(papers|collections|lists)\/[a-z0-9_-]+$/.test(route)) return route;
  if (/^(rl|architecture|diffusion|finetuning|optimization|representation|world_models|retrieval)\/[a-z0-9_-]+$/.test(route)) return route;
  return undefined;
}

export function signInHref(destination: string): string {
  let route = destination;
  if (destination.startsWith("#")) {
    try {
      const raw = destination.replace(/^#\/?/, ""), query = raw.indexOf('?');
      route = query < 0 ? decodeURIComponent(raw) : decodeURIComponent(raw.slice(0, query)) + raw.slice(query);
    }
    catch { return "#/account"; }
  }
  const valid = returnRoute(route);
  return valid === undefined ? "#/account" : `#/${continuationPrefix}${encodeURIComponent(valid)}`;
}

// App's existing hash decoder has already decoded this route exactly once.
export function accountReturnHref(route: string): string | undefined {
  if (route.startsWith(errorPrefix)) {
    const remainder = route.slice(errorPrefix.length), separator = remainder.indexOf('/');
    if (separator < 0 || !loginErrors.has(remainder.slice(0, separator))) return undefined;
    const valid = returnRoute(remainder.slice(separator + 1));
    return valid === undefined ? undefined : `#/${valid}`;
  }
  if (!route.startsWith(continuationPrefix)) return undefined;
  const valid = returnRoute(route.slice(continuationPrefix.length));
  return valid === undefined ? undefined : `#/${valid}`;
}

export function isAccountRoute(route: string): boolean {
  return route === "account" || route.startsWith(continuationPrefix) || route.startsWith(errorPrefix);
}

export function organizationLoginError(route: string): string | undefined {
  if (!route.startsWith(errorPrefix)) return undefined;
  const code = route.slice(errorPrefix.length).split('/')[0];
  return loginErrors.has(code) ? code : undefined;
}
