import {useI18n} from "./i18n";
import type {ProblemStatistics} from "./ProblemStats";

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
  return stats ? content : <details className="stats-availability"><summary>{loading ? t("正在加载统计…", "Loading statistics…") : t("公开统计暂不可用", "Public statistics unavailable")}</summary><p>{t("统计仅包含服务端确认的正式提交；本机练习进度仍会保留。", "Statistics include server-verified submissions. Your practice progress is still saved on this device.")}</p>{content}</details>;
}

export function ProblemStatsSummary({ stats }: { stats?: ProblemStatistics }) {
  const { t, locale } = useI18n();
  return <div className="module-stats" aria-label={t("题目统计", "Problem statistics")}>
    <span>{t("尝试人数", "Attempted users")} <b>{formatCount(stats?.attempted_users, locale)}</b></span>
    <span title={t("通过提交 / 全部提交", "Accepted / total submissions")}>{t("通过率", "Acceptance")} <b>{formatRate(stats, locale)}</b></span>
  </div>;
}
