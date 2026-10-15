import { Code2 } from "lucide-react";
import { useI18n } from "./i18n";
import NavMetric from "./NavMetric";

export default function CommunitySubmissions({ count }: { count?: number }) {
  const { t, locale } = useI18n();
  const available = count !== undefined && Number.isSafeInteger(count) && count >= 0;
  const numberLocale = locale === "zh" ? "zh-CN" : "en-US";
  const fullCount = available ? count.toLocaleString(numberLocale) : "—";
  const compactCount = available ? count.toLocaleString(numberLocale,
    count >= 10_000 ? { notation: "compact", maximumFractionDigits: 1 } : undefined) : "—";
  const label = available
    ? t(`累计提交 ${fullCount} 次`, `${fullCount} total submissions`)
    : t("累计提交次数暂不可用", "Total submissions unavailable");

  return <NavMetric className={`community-submissions${available ? "" : " unavailable"}`} label={label} value={compactCount}
    icon={<Code2 size={14} strokeWidth={1.8} aria-hidden="true" />} />;
}
