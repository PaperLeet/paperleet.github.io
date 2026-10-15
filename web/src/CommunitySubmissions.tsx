import { Code2 } from "lucide-react";
import { useI18n } from "./i18n";
import "./community-submissions.css";

export default function CommunitySubmissions({ count }: { count?: number }) {
  const { t, locale } = useI18n();
  const available = count !== undefined && Number.isSafeInteger(count) && count >= 0;
  const numberLocale = locale === "zh" ? "zh-CN" : "en-US";
  const fullCount = available ? count.toLocaleString(numberLocale) : "—";
  const compactCount = available ? count.toLocaleString(numberLocale,
    count >= 10_000 ? { notation: "compact", maximumFractionDigits: 1 } : undefined) : "—";
  const label = available
    ? t(`全社区累计提交 ${fullCount} 次`, `${fullCount} all-time community submissions`)
    : t("全社区累计提交次数暂不可用", "All-time community submissions unavailable");

  return <span className={`community-submissions${available ? "" : " unavailable"}`} role="img" aria-label={label}>
    <Code2 className="community-submissions-icon" size={14} strokeWidth={1.8} aria-hidden="true" />
    <span className="community-submissions-value" aria-hidden="true"><b>{compactCount}</b><span className="community-submissions-label">{t("次提交", "submissions")}</span></span>
  </span>;
}
