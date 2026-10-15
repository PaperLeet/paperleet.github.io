import { useState } from "react";
import { LogOut } from "lucide-react";
import { useI18n } from "./i18n";
import { errorMessage, socialApi, useAccount } from "./social";

export default function AccountLogout() {
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
  return <div className="account-menu-logout">
    <button type="button" className="account-menu-row account-menu-signout" disabled={busy}
      onClick={() => void logout()} title={t("退出当前设备", "Sign out on this device")}
      aria-label={t("退出登录", "Log out")} aria-busy={busy}>
      <LogOut size={15} aria-hidden="true" />
      <span>{busy ? t("正在退出…", "Signing out…") : t("退出登录", "Log out")}</span>
    </button>
    {error && <p className="account-menu-error" role="alert">{error}</p>}
  </div>;
}
