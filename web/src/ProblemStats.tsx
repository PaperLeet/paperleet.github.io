import { useCallback, useEffect, useRef, useState } from "react";
import { useI18n } from "./i18n";
import { publicSite } from "./deployment";
import { apiFetch } from "./apiClient";

export type ProblemStatistics = {
  attempted_users: number;
  solved_users: number;
  total_submissions: number;
  accepted_submissions: number;
  failed_submissions: number;
  acceptance_rate: number | null;
};
export type ProblemStatsMap = Record<string, ProblemStatistics>;

const emptyStats: ProblemStatistics = {
  attempted_users: 0,
  solved_users: 0,
  total_submissions: 0,
  accepted_submissions: 0,
  failed_submissions: 0,
  acceptance_rate: null,
};

// An absent entry means zero submissions only after the public aggregate loaded.
export function getProblemStats(stats: ProblemStatsMap | undefined, id: string) {
  return stats ? stats[id] ?? emptyStats : undefined;
}

function parseStats(value: unknown): ProblemStatsMap {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("Invalid practice statistics");
  }
  const countFields = ["attempted_users", "solved_users", "total_submissions", "accepted_submissions", "failed_submissions"] as const;
  for (const entry of Object.values(value)) {
    if (!entry || typeof entry !== "object" || countFields.some((field) => !Number.isSafeInteger(entry[field]) || entry[field] < 0) ||
      !(entry.acceptance_rate === null || (typeof entry.acceptance_rate === "number" && Number.isFinite(entry.acceptance_rate) && entry.acceptance_rate >= 0 && entry.acceptance_rate <= 100))) {
      throw new Error("Invalid practice statistics");
    }
  }
  return value as ProblemStatsMap;
}

export function useProblemStats(refreshKey?: string) {
  const [stats, setStats] = useState<ProblemStatsMap>();
  const [loading, setLoading] = useState(true);
  const [available, setAvailable] = useState<boolean | null>(null);
  const [error, setError] = useState<string | null>(null);
  const request = useRef<AbortController | null>(null);
  const refresh = useCallback(async () => {
    if (publicSite) { setLoading(false); setAvailable(false); setError(null); return; }
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    setLoading(true);
    setError(null);
    const timeout = window.setTimeout(() => controller.abort(), 12_000);
    try {
      const response = await apiFetch("/api/practice/stats", { signal: controller.signal, credentials: "omit", cache: "no-store" });
      if (!response.ok) throw new Error("Practice statistics are unavailable");
      const data = await response.json();
      if (request.current !== controller) return;
      if (typeof data?.available !== "boolean") throw new Error("Invalid practice statistics");
      const next = data.available ? parseStats(data.stats) : undefined;
      setStats(next);
      setAvailable(data.available);
    } catch {
      if (request.current !== controller) return;
      setStats(undefined);
      setAvailable(null);
      setError("Practice statistics are unavailable");
    } finally {
      window.clearTimeout(timeout);
      if (request.current === controller) {
        setLoading(false);
        request.current = null;
      }
    }
  }, []);
  useEffect(() => {
    void refresh();
    return () => {
      request.current?.abort();
      request.current = null;
    };
  }, [refresh, refreshKey]);
  return { stats, refresh, loading, error, available };
}

function formatCount(value: number | undefined, locale: string) {
  return value === undefined ? "—" : value.toLocaleString(locale === "zh" ? "zh-CN" : "en-US");
}
function formatRate(stats: ProblemStatistics | undefined, locale: string) {
  return stats?.acceptance_rate == null ? "—" : `${stats.acceptance_rate.toLocaleString(locale === "zh" ? "zh-CN" : "en-US", { minimumFractionDigits: 1, maximumFractionDigits: 1 })}%`;
}

export function ProblemStats({ stats, loading = false }: { stats?: ProblemStatistics; loading?: boolean }) {
  const { t, locale } = useI18n();
  const metrics = [
    ["attempted_users", t("尝试人数", "Attempted users"), formatCount(stats?.attempted_users, locale)],
    ["solved_users", t("通过人数", "Solved users"), formatCount(stats?.solved_users, locale)],
    ["total_submissions", t("提交次数", "Submissions"), formatCount(stats?.total_submissions, locale)],
    ["accepted_submissions", t("通过提交", "Accepted submissions"), formatCount(stats?.accepted_submissions, locale)],
    ["failed_submissions", t("未通过提交", "Failed submissions"), formatCount(stats?.failed_submissions, locale)],
    ["acceptance_rate", t("通过率", "Acceptance"), formatRate(stats, locale)],
  ];
  const content = <dl className="problem-stats" aria-label={t("题目统计", "Problem statistics")} aria-busy={loading}
    title={stats ? t("仅统计正式提交；人数按账号去重，通过率 = 通过提交 / 全部提交。", "Full submissions only. User counts are unique; acceptance = accepted / total submissions.") : loading ? t("正在加载统计", "Loading statistics") : t("统计暂不可用", "Statistics unavailable")}>
    {metrics.map(([key, label, value]) => <div key={key} data-stat={key}><dt>{label}</dt><dd>{value}</dd></div>)}
  </dl>;
  return stats ? content : <details className="stats-availability"><summary>{loading ? t("正在加载统计…", "Loading statistics…") : t("公开统计暂不可用", "Public statistics unavailable")}</summary><p>{t("统计仅包含服务端确认的正式提交；本机练习进度仍会保留。", "Statistics include server-verified submissions. Your browser learning progress is still saved.")}</p>{content}</details>;
}

export function ProblemStatsSummary({ stats }: { stats?: ProblemStatistics }) {
  const { t, locale } = useI18n();
  return <div className="module-stats" aria-label={t("题目统计", "Problem statistics")}>
    <span>{t("尝试人数", "Attempted users")} <b>{formatCount(stats?.attempted_users, locale)}</b></span>
    <span title={t("通过提交 / 全部提交", "Accepted / total submissions")}>{t("通过率", "Acceptance")} <b>{formatRate(stats, locale)}</b></span>
  </div>;
}
