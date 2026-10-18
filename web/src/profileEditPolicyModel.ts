export type ProfileEditPolicy = {
  username: { interval_seconds: number; next_change_at: string | null };
  display_name: { interval_seconds: number; next_change_at: string | null };
  avatar: { daily_limit: number; remaining: number; reset_at: string };
};

const object = (value: unknown): value is Record<string, unknown> => !!value && typeof value === "object" && !Array.isArray(value);
const timestamp = (value: unknown): value is string => typeof value === "string" && Number.isFinite(Date.parse(value));

/** Older deployments have no policy; never imply that they enforce these limits. */
export function readProfileEditPolicy(value: unknown): ProfileEditPolicy | undefined {
  if (!object(value) || !object(value.username) || !object(value.display_name) || !object(value.avatar)) return;
  for (const [field, seconds] of [["username", 2_592_000], ["display_name", 604_800]] as const) {
    const rule = value[field];
    if (!object(rule) || rule.interval_seconds !== seconds || !(rule.next_change_at === null || timestamp(rule.next_change_at))) return;
  }
  const avatar = value.avatar;
  if (avatar.daily_limit !== 3 || !Number.isSafeInteger(avatar.remaining) || Number(avatar.remaining) < 0 ||
      Number(avatar.remaining) > 3 || !timestamp(avatar.reset_at)) return;
  return value as unknown as ProfileEditPolicy;
}
