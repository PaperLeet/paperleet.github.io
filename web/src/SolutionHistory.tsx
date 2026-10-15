import { useEffect, useRef, useState } from "react";
import { apiFetch } from "./apiClient";
import { useI18n } from "./i18n";
import { createSolutionHistoryReader, emptySolutionHistory, SolutionHistoryIdentityError } from "./solutionHistoryData";
import type { SolutionHistoryState } from "./solutionHistoryData";
import type { Problem } from "./types";
import "./solution-history.css";

type Props = { owner: string; problem: Problem; revision: number };

export default function SolutionHistory({ owner, problem, revision }: Props) {
  const { t, locale } = useI18n();
  const [refresh, setRefresh] = useState(0);
  const scope = JSON.stringify([owner, problem.id, revision, refresh]);
  const [view, setView] = useState<{ scope: string; state: SolutionHistoryState }>();
  const [copy, setCopy] = useState<{ scope: string; id: string; ok: boolean }>();
  const reader = useRef<{ scope: string; value: ReturnType<typeof createSolutionHistoryReader> } | undefined>(undefined);
  const signedIn = owner.startsWith("user:") && owner.length > 5;
  // React effects run after rendering. Hide the old account's source immediately on a scope change.
  const state = view?.scope === scope ? view.state : { ...emptySolutionHistory(), loading: signedIn };

  useEffect(() => {
    if (!signedIn) {
      setView(undefined);
      setCopy(undefined);
      return;
    }
    const value = createSolutionHistoryReader(owner.slice(5), problem.id, async (path, signal) => {
      const response = await apiFetch(path, { signal, cache: "no-store" });
      if (response.status === 401 || response.status === 403) throw new SolutionHistoryIdentityError("Submission history access changed");
      if (!response.ok) throw new Error("Submission history unavailable");
      return response.json();
    }, (next) => setView({ scope, state: next }));
    reader.current = { scope, value };
    void value.loadMore();
    return () => {
      value.dispose();
      if (reader.current?.value === value) reader.current = undefined;
    };
  }, [owner, problem.id, scope, signedIn]);

  async function copyCode(id: string, code: string) {
    try {
      await navigator.clipboard.writeText(code);
      setCopy({ scope, id, ok: true });
    } catch {
      setCopy({ scope, id, ok: false });
    }
  }

  if (!signedIn) return <section className="solution-history">
    <p>{t("登录后提交的代码和结果会保存在账号下。", "Sign in to save submitted code and results to your account.")}</p>
  </section>;

  return <section className="solution-history" aria-label={t("我的提交", "My submissions")} aria-busy={state.loading}>
    <div className="solution-history-heading">
      <p>{t("这里保留每次提交时的代码和判题结果。", "Each submission keeps the code and judge result from that attempt.")}</p>
      <button type="button" disabled={state.loading} onClick={() => setRefresh((value) => value + 1)}>{t("刷新", "Refresh")}</button>
    </div>
    {state.error && <div className="solution-history-error" role="alert">
      <p>{t("暂时无法加载提交记录，请重试。", "Submission history could not be loaded. Please try again.")}</p>
      <button type="button" onClick={() => { if (reader.current?.scope === scope) void reader.current.value.loadMore(); }}>{t("重试", "Retry")}</button>
    </div>}
    {!state.loading && !state.error && !state.runs.length && <p className="solution-history-empty">
      {t("这道题还没有提交记录。点击 Submit 后，代码和结果会显示在这里。", "No submissions for this problem yet. Use Submit to save your code and results here.")}
    </p>}
    <ol className="solution-history-list">
      {state.runs.map((run) => <li key={run.id}>
        <details>
          <summary>
            <span className={`solution-history-result ${run.accepted ? "accepted" : "failed"}`}>{run.accepted ? t("通过", "Accepted") : t("未通过", "Not accepted")}</span>
            <time dateTime={run.created_at}>{new Date(run.created_at).toLocaleString(locale === "zh" ? "zh-CN" : "en-US")}</time>
            <span className="solution-history-count">{t("测试", "Tests")} {run.report.tests.filter((test) => test.passed).length}/{run.report.tests.length}</span>
          </summary>
          <div className="solution-history-source">
            <div className="solution-history-source-heading">
              <strong>{t("提交时的代码", "Submitted code")}</strong>
              <button type="button" onClick={() => void copyCode(run.id, run.code)}>{t("复制代码", "Copy code")}</button>
            </div>
            {copy?.scope === scope && copy.id === run.id && <p className="solution-history-copy" role="status">{copy.ok ? t("已复制", "Copied") : t("无法自动复制，请在下方选择代码后复制。", "Copy is unavailable. Select the code below to copy it.")}</p>}
            {run.report.error && <p className="solution-history-run-error">{run.report.error}</p>}
            <pre tabIndex={0} aria-label={t("提交时的 Python 代码（只读）", "Submitted Python code (read only)")}><code>{run.code}</code></pre>
          </div>
        </details>
      </li>)}
    </ol>
    {state.loading && <p className="solution-history-loading" role="status">{t("正在加载提交记录…", "Loading submissions…")}</p>}
    {!state.loading && !state.error && state.hasMore && state.runs.length > 0 && <button className="solution-history-more" type="button" onClick={() => { if (reader.current?.scope === scope) void reader.current.value.loadMore(); }}>
      {t("加载更早的提交", "Load older submissions")}
    </button>}
  </section>;
}
