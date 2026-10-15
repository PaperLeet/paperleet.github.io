import { useState } from "react";
import { LogOut } from "lucide-react";
import { useI18n } from "./i18n";
import { Avatar, errorMessage, profileHref, socialApi, useAccount } from "./social";
import type { AccountUser } from "./social";

export default function AccountLogout({ user }: { user: AccountUser }) {
  const { setUser } = useAccount();
  const { t } = useI18n();
  const [busy, setBusy] = useState(false), [error, setError] = useState("");
  async function logout() {
    if (busy) return;
    setBusy(true); setError("");
    try {
      await socialApi("account/logout", {});
      setUser(null);
    } catch (e) { setError(errorMessage(e)); }
    finally { setBusy(false); }
  }
  return <div className="account-nav-actions">
    <a className="account-nav" href={profileHref(user)} title={t("我的主页", "My profile")}>
      <Avatar user={user} size={29} />
    </a>
    <button type="button" className="account-logout-nav" disabled={busy}
      onClick={() => void logout()} title={t("退出当前设备", "Sign out on this device")}
      aria-label={t("退出登录", "Log out")} aria-busy={busy}>
      <LogOut size={15} aria-hidden="true" />
      <span>{busy ? t("正在退出…", "Signing out…") : t("退出登录", "Log out")}</span>
    </button>
    {error && <span className="account-nav-error" role="alert">{error}</span>}
  </div>;
}
