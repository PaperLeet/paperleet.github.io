import { decodeProblemHash } from "./reports";

/** The public site has no account API; protected routes stay on the hosted app. */
export const publicSite = import.meta.env.VITE_PUBLIC_SITE === "1";
export const hostedEdition = import.meta.env.VITE_APP_EDITION === "hosted";
export const hostedAppAvailable = hostedEdition || (publicSite && import.meta.env.VITE_HOSTED_APP_AVAILABLE === "1");
export function validateHostedAppUrl(value: string): string {
  const url = new URL(value);
  if (url.protocol !== "https:" || url.username || url.password || url.pathname !== "/" || url.search || url.hash)
    throw Error("The hosted application must be an HTTPS origin");
  return url.origin;
}
const hostedApp = publicSite && hostedAppAvailable ? validateHostedAppUrl(import.meta.env.VITE_HOSTED_APP_URL || "") : "";
export function hostedAppHref(hash = "#/"): string {
  if (!publicSite) return hash;
  if (!hostedAppAvailable) return "#/";
  const url = new URL(hostedApp);
  url.hash = hash.startsWith("#/") ? hash : "#/";
  return url.href;
}
export function isHostedRoute(hash: string): boolean {
  if (!hash.startsWith("#")) return false;
  return /^(account(?:\/|\?|$)|community(?:\/|\?|$)|u\/|submit(?:$|\?))/.test(decodeProblemHash(hash));
}
export function startPublicNavigation(): boolean {
  if (hostedEdition) return true;
  if (!publicSite) {
    const library = () => {
      if (isHostedRoute(location.hash) && !/^submit(?:$|\?)/.test(decodeProblemHash(location.hash))) location.hash = "#/";
    };
    library();
    window.addEventListener("hashchange", library);
    return true;
  }
  if (!hostedAppAvailable) {
    const library = () => { if (isHostedRoute(location.hash)) location.hash = "#/"; };
    library();
    window.addEventListener("hashchange", library);
    return true;
  }
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
