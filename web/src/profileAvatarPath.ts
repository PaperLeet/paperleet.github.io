/** Only the API's validated avatar route may be displayed on public pages. */
export function safeAvatarPath(value: unknown): string | undefined {
  return typeof value === "string" && !/[\r\n]/.test(value) && /^\/api\/avatars\/[a-f0-9]{32}\/[a-f0-9]{64}\.jpg$/.test(value) ? value : undefined;
}
