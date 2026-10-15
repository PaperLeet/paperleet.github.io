/** Shared by the browser and Vite's resolved deployment configuration. */
function httpsOrigin(value: string, message: string): string {
  let url: URL;
  try { url = new URL(value); } catch { throw Error(message); }
  if (url.protocol !== "https:" || url.username || url.password || url.pathname !== "/" || url.search || url.hash)
    throw Error(message);
  return url.origin;
}

export function validateHostedAppUrl(value: string): string {
  return httpsOrigin(value, "The hosted application must be an HTTPS origin");
}

export function validateApiOrigin(value: string): string {
  return httpsOrigin(value, "The API must be an HTTPS origin");
}

export function validateDeploymentEnvironment(environment: Record<string, string | undefined>): void {
  const publishedApp = environment.VITE_PUBLIC_SITE === "1" &&
    (environment.VITE_APP_EDITION === "hosted" || environment.VITE_HOSTED_APP_AVAILABLE === "1");
  if (publishedApp) {
    try { validateHostedAppUrl(environment.VITE_HOSTED_APP_URL || ""); }
    catch { throw Error("Invalid VITE_HOSTED_APP_URL: set an HTTPS origin for the enabled hosted application"); }
  }
  if (environment.VITE_API_ORIGIN) {
    try { validateApiOrigin(environment.VITE_API_ORIGIN); }
    catch { throw Error("Invalid VITE_API_ORIGIN: use an HTTPS origin, or leave it empty for same-origin requests"); }
  }
}
