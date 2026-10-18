import { currentLocale } from "./locale";
export {readProfileEditPolicy, type ProfileEditPolicy} from "./profileEditPolicyModel";

const object = (value: unknown): value is Record<string, unknown> => !!value && typeof value === "object" && !Array.isArray(value);
const timestamp = (value: unknown): value is string => typeof value === "string" && Number.isFinite(Date.parse(value));

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
  return profileEditLimitMessage(value, currentLocale() === "en");
}
