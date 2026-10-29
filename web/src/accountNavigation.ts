import {readAdminRoute} from './adminNavigation';
import {readProfileRoute,profileFeedHref} from './profileNavigation';
import { communityContinuation } from './communityNavigation';
import { favoriteHref,readFavoriteRoute,libraryContinuation } from './libraryNavigation';
import {readInterviewRoute,interviewHref} from './interviewNavigation';

const continuationPrefix = "account/continue/";
const errorPrefix = "account/error/";
const loginErrors = new Set(["identity", "expired", "cancelled", "unavailable", "rate_limited"]);

/** Allow only known shapes of local hash routes; never an external destination. */
function returnRoute(route: string): string | undefined {
  if (route.startsWith('#')) return undefined;
  if(/^daily-challenge(?:\?date=\d{4}-\d{2}-\d{2})?$/.test(route))return route;
  const favorites=readFavoriteRoute(route);if(favorites)return favoriteHref(favorites).slice(2);
  const profile=readProfileRoute(route);if(profile)return profileFeedHref(profile.username,profile).slice(2);
  const interviews=readInterviewRoute(route);
  if(interviews)return interviewHref(interviews).slice(2);
  if (/^submit\?(?:proposal=[a-f0-9]{32}|view=pool)$/.test(route)) return route;
  if (readAdminRoute(route)) return route;
  if (route.includes('?')) return communityContinuation(route) ?? libraryContinuation(route);
  if (["community/reports","community/moderation","","community","submit","achievements","favorites","drafts","paper-registry"].includes(route)) return route;
  if (/^(?:[1-9]\d*|community\/(?:problem\/[a-z0-9_-]+\/[a-z0-9_-]+|[a-f0-9]{32})|submission\/[a-f0-9]{32}|u\/[a-z0-9_]{3,24}|messages(?:\/[a-z0-9_]{3,24})?|(?:papers|collections|lists|foundations|rl|architecture|diffusion|finetuning|optimization|representation|world_models|retrieval|self_evolving|network_biology|multimodal|robotics|ai_infra|diffusion_rl)\/[a-z0-9_-]+)$/.test(route)) return route;
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
