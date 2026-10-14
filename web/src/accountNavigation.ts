const continuationPrefix = "account/continue/";

/** Allow only known shapes of local hash routes; never an external destination. */
function returnRoute(route: string): string | undefined {
  if (route === "" || route === "community" || route === "submit") return route;
  if (/^[1-9]\d*$/.test(route)) return route;
  if (/^community\/problem\/[a-z0-9_-]+\/[a-z0-9_-]+$/.test(route)) return route;
  if (/^community\/[a-f0-9]{32}$/.test(route)) return route;
  if (/^u\/[a-z0-9_]{3,24}$/.test(route)) return route;
  if (/^(papers|collections|lists)\/[a-z0-9_-]+$/.test(route)) return route;
  if (/^(rl|architecture|diffusion|finetuning|optimization|representation|world_models)\/[a-z0-9_-]+$/.test(route)) return route;
  return undefined;
}

export function signInHref(destination: string): string {
  let route = destination;
  if (destination.startsWith("#")) {
    try { route = decodeURIComponent(destination.replace(/^#\/?/, "")); }
    catch { return "#/account"; }
  }
  const valid = returnRoute(route);
  return valid === undefined ? "#/account" : `#/${continuationPrefix}${encodeURIComponent(valid)}`;
}

// App's existing hash decoder has already decoded this route exactly once.
export function accountReturnHref(route: string): string | undefined {
  if (!route.startsWith(continuationPrefix)) return undefined;
  const valid = returnRoute(route.slice(continuationPrefix.length));
  return valid === undefined ? undefined : `#/${valid}`;
}

export function isAccountRoute(route: string): boolean {
  return route === "account" || route.startsWith(continuationPrefix);
}
