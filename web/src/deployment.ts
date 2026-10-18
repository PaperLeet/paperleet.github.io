import {appLocation} from './appLocation';
import { decodeProblemHash } from "./reports";
import { validateHostedAppUrl } from "./deploymentOrigins";
export { validateHostedAppUrl } from "./deploymentOrigins";

/** The public site has no account API; protected routes stay on the hosted app. */
export const publicSite = import.meta.env.VITE_PUBLIC_SITE === "1";
export const hostedEdition = import.meta.env.VITE_APP_EDITION === "hosted";
export const hostedAppAvailable = hostedEdition || (publicSite && import.meta.env.VITE_HOSTED_APP_AVAILABLE === "1");
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
  return /^(account(?:\/|\?|$)|community(?:\/|\?|$)|messages(?:\/|\?|$)|u\/|submit(?:$|\?)|drafts(?:$|\?))/.test(decodeProblemHash(hash));
}
export function startPublicNavigation(): boolean {
  if (hostedEdition) return true;
  if (!publicSite) {
    const library = () => {
      if (isHostedRoute(appLocation.hash) && !/^submit(?:$|\?)/.test(decodeProblemHash(appLocation.hash))) appLocation.hash = "#/";
    };
    library();
    window.addEventListener("hashchange", library);
    return true;
  }
  if (!hostedAppAvailable) {
    const library = () => { if (isHostedRoute(appLocation.hash)) appLocation.hash = "#/"; };
    library();
    window.addEventListener("hashchange", library);
    return true;
  }
  const redirect = () => {
    if (isHostedRoute(appLocation.hash)) location.replace(hostedAppHref(appLocation.hash));
  };
  if (isHostedRoute(appLocation.hash)) { redirect(); return false; }
  document.addEventListener("click", event => {
    if (!(event instanceof MouseEvent) || event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey || event.defaultPrevented) return;
    const anchor = (event.target as Element | null)?.closest("a");
    const hash = anchor?.getAttribute("href") || "";
    if (isHostedRoute(hash)) { event.preventDefault(); location.assign(hostedAppHref(hash)); }
  }, true);
  window.addEventListener("hashchange", redirect);
  return true;
}
