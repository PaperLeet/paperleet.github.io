import { useEffect, useState } from "react";
import { Globe2 } from "lucide-react";
import { useI18n } from "./i18n";
import { useAccount } from "./social";
import { createPresenceHeartbeat, presenceStorageKey, presenceVisitorId } from "./presence";
import { presenceRequest } from "./presenceApi";
import "./online-presence.css";

function browserVisitorId() {
  try {
    return presenceVisitorId(window.localStorage);
  } catch {
    return presenceVisitorId();
  }
}

export default function OnlinePresence() {
  const { t, locale } = useI18n();
  const { user, loading } = useAccount();
  const [online, setOnline] = useState<number | null>(null);

  useEffect(() => {
    setOnline(null);
    if (loading || !presenceRequest) return;
    const heartbeat = createPresenceHeartbeat({
      visitorId: browserVisitorId,
      onChange: setOnline,
      fetch: presenceRequest,
    });
    function syncVisibility() {
      if (document.visibilityState === "visible" && navigator.onLine) heartbeat.resume();
      else heartbeat.pause();
    }
    function syncVisitor(event: StorageEvent) {
      if (event.key !== presenceStorageKey) return;
      // A different tab may have initialized the shared browser ID first.
      heartbeat.pause();
      syncVisibility();
    }
    syncVisibility();
    document.addEventListener("visibilitychange", syncVisibility);
    window.addEventListener("online", syncVisibility);
    window.addEventListener("offline", syncVisibility);
    window.addEventListener("storage", syncVisitor);
    return () => {
      heartbeat.dispose();
      document.removeEventListener("visibilitychange", syncVisibility);
      window.removeEventListener("online", syncVisibility);
      window.removeEventListener("offline", syncVisibility);
      window.removeEventListener("storage", syncVisitor);
    };
  }, [loading, user?.id]);

  const numberLocale = locale === "zh" ? "zh-CN" : "en-US";
  const fullCount = online === null ? "—" : online.toLocaleString(numberLocale);
  const count = online === null ? "—" : online.toLocaleString(numberLocale,
    online >= 10_000 ? { notation: "compact", maximumFractionDigits: 1 } : undefined);
  const label = online === null ? t("在线人数暂不可用", "Online count unavailable")
    : t(`${fullCount} 人在线`, `${fullCount} online`);
  const explanation = t(
    "最近 90 秒内浏览过页面的人数估计，包含访客；同一登录账号只计一次。",
    "Estimated viewers active in the last 90 seconds, including guests. Each signed-in account is counted once.",
  );
  const tooltip = online === null
    ? `${t("在线人数暂不可用。", "Online count unavailable. ")} ${explanation}`
    : explanation;

  return (
    <span
      className={`online-presence${online === null ? " unavailable" : ""}`}
      title={tooltip}
      aria-label={`${label}. ${tooltip}`}
      tabIndex={0}
    >
      <Globe2 className="online-presence-icon" size={14} strokeWidth={1.8} aria-hidden="true" />
      <span className="online-presence-value" aria-hidden="true"><b>{count}</b><span className="online-presence-label">{t("人在线", "online")}</span></span>
      <span className="online-presence-tooltip" role="tooltip">{tooltip}</span>
    </span>
  );
}
