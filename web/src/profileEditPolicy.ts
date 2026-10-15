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

export function profileEditLimitMessage(value: unknown, english: boolean): string | undefined {
  if (!object(value) || value.code !== "profile_edit_limit" ||
      !["username", "display_name", "avatar"].includes(String(value.field)) || !timestamp(value.next_change_at)) return;
  const names = english ? { username: "Username", display_name: "Display name", avatar: "Avatar" }
    : { username: "用户名", display_name: "昵称", avatar: "头像" };
  const name = names[value.field as keyof typeof names];
  const date = new Date(value.next_change_at).toLocaleString(english ? "en-US" : "zh-CN", { timeZoneName: "short" });
  return english ? `${name} cannot be changed again until ${date}. Your edits are kept on this page.`
    : `${name}暂时不能再次修改，可在 ${date} 后更改。当前编辑内容仍保留在本页。`;
}

export function currentProfileEditLimitMessage(value: unknown) {
  let english = false;
  try { english = localStorage.getItem("paperleet.locale") === "en"; } catch { /* Default language. */ }
  return profileEditLimitMessage(value, english);
}
