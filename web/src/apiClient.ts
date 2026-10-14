/** A fixed deployment origin; cookies stay on the API host and out of JS. */
export function validateApiOrigin(value: string): string {
  const url = new URL(value);
  if (url.protocol !== "https:" || url.username || url.password || url.pathname !== "/" || url.search || url.hash)
    throw Error("The API must be an HTTPS origin");
  return url.origin;
}
const configuredOrigin = import.meta.env.VITE_API_ORIGIN || "";
const apiOrigin = configuredOrigin ? validateApiOrigin(configuredOrigin) : "";
export const apiCredentials: RequestCredentials = apiOrigin ? "include" : "same-origin";
export function apiUrl(path: string): string {
  if (!path.startsWith("/api/") || /[\r\n\\]/.test(path)) throw Error("Expected an API path");
  return apiOrigin + path;
}
export function apiFetch(path: string, options: RequestInit = {}) {
  return fetch(apiUrl(path), { credentials: apiCredentials, ...options });
}
