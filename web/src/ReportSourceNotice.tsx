import { useEffect, useState } from "react";
import { useI18n } from "./i18n";
import { fingerprintSource, reportSourceState, type ReportSourceState, type SourceFingerprint } from "./reportSource";
import type { Report } from "./types";

export function useReportSource(report: Report | undefined, source: string) {
  const [fingerprint, setFingerprint] = useState<SourceFingerprint>();
  const hash = report?.source_sha256;
  useEffect(() => {
    if (!hash) return;
    let cancelled = false;
    // Keep hashing out of the editor's input path, and cancel superseded work.
    const timer = setTimeout(() => {
      void fingerprintSource(source).then((value) => { if (!cancelled) setFingerprint(value); });
    }, 150);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [source, hash]);
  return reportSourceState(report, source, fingerprint);
}

export function ReportSourceNotice({ state }: { state: ReportSourceState }) {
  const { t } = useI18n();
  if (state === "none") return null;
  const text = state === "current"
    ? t("结果对应当前代码。", "These results match the current code.")
    : state === "changed"
      ? t("代码已修改；以下是上次提交的结果，请重新运行验证当前代码。", "Code changed. These are results for an earlier submission; run tests again to check the current code.")
      : state === "checking"
        ? t("正在核对结果对应的代码…", "Checking which code these results belong to…")
        : t("无法确认此结果是否对应当前代码，请重新运行验证。", "The code version for these results cannot be confirmed. Run tests again to check the current code.");
  return <p className={`report-source-notice ${state}`} role="status">{text}</p>;
}
