/** The public site has no account API; protected routes stay on the hosted app. */
export const publicSite = import.meta.env.VITE_PUBLIC_SITE === "1";
export function validateHostedAppUrl(value: string): string {
  const url = new URL(value);
  if (url.protocol !== "https:" || url.username || url.password || url.pathname !== "/" || url.search || url.hash)
    throw Error("The hosted application must be an HTTPS origin");
  return url.origin;
}
const hostedApp = publicSite ? validateHostedAppUrl(import.meta.env.VITE_HOSTED_APP_URL || "") : "";
export function hostedAppHref(hash = "#/"): string {
  if (!publicSite) return hash;
  const url = new URL(hostedApp);
  url.hash = hash.startsWith("#/") ? hash : "#/";
  return url.href;
}
export function isHostedRoute(hash: string): boolean {
  return /^#\/(account(?:\/|$)|community(?:\/|$)|u\/|submit(?:$|\?))/.test(hash);
}
export function startPublicNavigation(): boolean {
  if (!publicSite) return true;
  const redirect = () => {
    if (isHostedRoute(location.hash)) location.replace(hostedAppHref(location.hash));
  };
  if (isHostedRoute(location.hash)) { redirect(); return false; }
  document.addEventListener("click", event => {
    if (!(event instanceof MouseEvent) || event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey || event.defaultPrevented) return;
    const anchor = (event.target as Element | null)?.closest("a");
    const hash = anchor?.getAttribute("href") || "";
    if (isHostedRoute(hash)) { event.preventDefault(); location.assign(hostedAppHref(hash)); }
  }, true);
  window.addEventListener("hashchange", redirect);
  return true;
}
