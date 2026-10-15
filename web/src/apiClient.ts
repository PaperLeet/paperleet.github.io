/** A fixed deployment origin; cookies stay on the API host and out of JS. */
import { validateApiOrigin } from "./deploymentOrigins";
export { validateApiOrigin } from "./deploymentOrigins";
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
