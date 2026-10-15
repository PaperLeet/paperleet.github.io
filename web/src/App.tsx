import { useEffect, useRef, useState } from "react";
import { Highlight } from "./SyntaxHighlight";
import { Community, UserProfile, Account, Submissions, Visuals, ArtifactCharts, Prose, PythonEditor, SolutionHistory } from "./deferredPages";
import {
  ArrowDownToLine,
  ArrowLeft,
  ArrowRight,
  BookOpen,
  Check,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Code2,
  Copy,
  ExternalLink,
  FileCode2,
  FlaskConical,
  LockKeyhole,
  Moon,
  MessageCircle,
  Play,
  Sun,
  Terminal,
  X,
  XCircle,
} from "lucide-react";
import { isPassed, layers, status } from "./types";
import type { Category, Paper, Problem, Report } from "./types";
import { PaperDetail, PaperLibrary, paperModules } from "./PaperViews";
import { collectionIdFromRoute } from "./paperCollections";
import { decodeProblemHash } from "./reports";
import { LocaleProvider, useI18n } from "./i18n";
import { AccountNav, AccountProvider, useAccount } from "./social";
import { usePracticeWorkspace } from "./PracticeWorkspace";
import { BrowserRunProgress } from "./BrowserRunProgress";
import { browserProgressText } from "./browserProgress";
import Brand from "./Brand";
import RepositoryLink from "./RepositoryLink";
import OnlinePresence from "./OnlinePresence";
import { BilingualText } from "./BilingualText";
import { getProblemStats, ProblemStats, useProblemStats } from "./ProblemStats";
import ResizableSplit from "./ResizableSplit";
import { recommendPractice, nextPractice } from "./practice-recommendation";
import { PracticeNextStep } from "./PracticeJourney";
import { loadProblemDetail } from "./catalogData";
import { useQueuedJudge } from "./useQueuedJudge";
import { jobErrorText } from "./judgeJobs";
import { useBrowserJudge } from "./useBrowserJudge";
import { browserReason } from "./browserJudge";
import { browserBudget } from './browserBudgets';
import { topicHref, topicIdFromRoute, topicFromProblem } from "./communityTopics";
import { accountReturnHref, isAccountRoute, signInHref } from "./accountNavigation";
import { publicSite, hostedAppHref, hostedAppAvailable } from "./deployment";
import { reportTestLabel, type ExperimentView } from "./experimentResults";
import { ReportSourceNotice, useReportSource } from "./ReportSourceNotice";
import type { ReportSourceState } from "./reportSource";
import TestComparison from "./TestComparison";
import { BrowserResultActions } from './BrowserResultActions';
import { matchesBrowserSubmission } from './browserVerification';
import { BrowserSyntaxNotice } from './BrowserSyntaxNotice';
import { useBrowserLearning } from './useBrowserLearning';
import { browserLearningView } from './browserLearning';
import { BrowserLearningNotice } from './BrowserLearningNotice';
import { ResetCodeButton } from './ResetCodeButton';
import { learnerTraceLocation, type EditorFocusRequest, type SourceLocation } from './sourceLocation';
import { SourceLocationButton } from './SourceLocationButton';
import "./community.css";

const difficulty = (d: string) =>
  ({ E: "Easy", M: "Medium", H: "Hard" })[d] || d;
const editorFontSizes = [13, 14, 15, 16, 17, 18, 20];
type Theme = "light" | "dark";
function initialTheme(): Theme {
  try {
    const saved = localStorage.getItem("paperleet.theme");
    if (saved === "light" || saved === "dark") return saved;
  } catch {
    // Browser storage can be unavailable in private or embedded contexts.
  }
  try {
    return window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";
  } catch {
    return "light";
  }
}
function initialEditorFontSize() {
  try {
    const saved = Number(localStorage.getItem("paperleet.editor.fontSize"));
    if (editorFontSizes.includes(saved)) return saved;
  } catch {
    // The editor remains usable when browser storage is unavailable.
  }
  return 15;
}
function getId() {
  return decodeProblemHash(location.hash);
}
function download(name: string, content: string) {
  const url = URL.createObjectURL(
    new Blob([content], { type: "text/plain;charset=utf-8" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}
function ReportView({
  report,
  problem,
  sourceState,
  source,
  onLocate,
  children,
}: {
  report?: Report;
  problem: Problem;
  sourceState: ReportSourceState;
  source: string;
  onLocate?: (location: SourceLocation) => void;
  children?: React.ReactNode;
}) {
  const { t, locale } = useI18n();
  const labels = t(
    "接口契约|参考对齐|数学性质|真实行为",
    "Contract|Reference|Properties|Behavior",
  ).split("|");
  if (!report)
    return (
      <div className="empty-state">
        <div className="empty-icon">
          <FlaskConical size={28} />
        </div>
        <h3>{t("用测试验证你的实现", "Validate your implementation")}</h3>
        <p>
          {t(
            "在右侧编写代码，然后点击「运行测试」。",
            "Write your code on the right, then select Run tests.",
          )}
          <br />
          {t(
            "测试结果和失败原因会直接显示在这里。",
            "Results and failure details will appear here.",
          )}
        </p>
      </div>
    );
  const passed = report.tests.filter((test) => test.passed).length,
    total = report.tests.length,
    allPassed = !report.error && total > 0 && passed === total;
  return (
    <div className="report-view">
      <div className={`result-head ${allPassed ? "success" : "failure"}`}>
        <div>
          {allPassed ? <CheckCircle2 size={24} /> : <XCircle size={24} />}
          <h3>
            {report.error
              ? t("判题未完成", "Run incomplete")
              : isPassed(problem, report)
                ? sourceState !== "current" ? t("上次测试通过", "Previous tests passed") : report.runtime === "browser-numpy-wasm" ? t("浏览器通过", "Browser tests passed") : "Accepted"
                : allPassed
                  ? t("所选测试通过", "Selected tests passed")
                  : t("需要再检查一下", "Tests need attention")}
          </h3>
        </div>
        <span>
          {passed} / {total} {t("项通过", "tests passed")}
        </span>
      </div>
      <ReportSourceNotice state={sourceState} />
      {report.error && <pre className="failure-message">{report.error}</pre>}
      {report.error && <SourceLocationButton location={learnerTraceLocation(report.error,report.impl,source,sourceState)} onLocate={onLocate} />}
      {children}
      <div className="report-meta">
        <span>{t("最近一次运行", "Latest run")}</span>
        <span>
          {report.created_at
            ? new Date(report.created_at).toLocaleString(
                locale === "zh" ? "zh-CN" : "en-US",
              )
            : ""}
        </span>
      </div>
      <div className="test-layers">
        {layers.map((l, i) => {
          const [a, b] = report.summary[l] || [0, 0];
          return (
            <div key={l}>
              <div>
                <b>{l}</b>
                <span>
                  {a}/{b}
                </span>
              </div>
              <small>{labels[i]}</small>
              <div className="mini-progress">
                <i style={{ width: `${b ? (a / b) * 100 : 0}%` }} />
              </div>
            </div>
          );
        })}
      </div>
      {report.tests.map((test, i) => (
        <details
          key={`${test.nodeid || test.id}-${i}`}
          className={`test-detail ${test.passed ? "passed" : "failed"}`}
          open={!test.passed}
        >
          <summary>
            {test.passed ? <CheckCircle2 size={15} /> : <XCircle size={15} />}
            <span>{reportTestLabel(test)}</span>
            <b>{test.layer}</b>
            <small>{(test.duration_s || 0).toFixed(2)}s</small>
          </summary>
          <div className="test-body">
            <p className="test-why">
              <b>{t("为什么检查这个性质", "Why this property matters")}</b>
              {test.why ||
                t("见本题测试说明。", "See the problem's test specification.")}
            </p>
            {test.comparison && <TestComparison comparison={test.comparison} />}
            {!test.passed && <SourceLocationButton location={learnerTraceLocation(test.message,report.impl,source,sourceState)} onLocate={onLocate} />}
            {test.message && (test.comparison ? <details className="failure-trace"><summary>{t("完整错误信息", "Full error details")}</summary><pre className="failure-message">{test.message}</pre></details> : (
              <pre className="failure-message">{test.message}</pre>
            ))}
            {test.artifacts && Object.keys(test.artifacts).length > 0 && (
              <ArtifactCharts data={test.artifacts} />
            )}
          </div>
        </details>
      ))}
      {report.mutation && (
        <section className="mutation">
          <h4>{t("错误变体检查", "Mutation checks")}</h4>
          {Object.entries(report.mutation).map(([id, caught]) => (
            <div key={id}>
              <b>
                {caught.length ? "✓" : "×"} {id}
              </b>
              <span>
                {caught.length
                  ? caught.join(", ")
                  : t("当前测试未捕获", "Not caught by current tests")}
              </span>
            </div>
          ))}
        </section>
      )}
    </div>
  );
}
export default function App() {
  return (
    <LocaleProvider>
      <AccountProvider>
        <Workspace />
      </AccountProvider>
    </LocaleProvider>
  );
}
function Workspace() {
  const { locale, setLocale, t } = useI18n();
  const { user, health, healthError, refreshHealth, capabilities } =
    useAccount();
  const practice = usePracticeWorkspace();
  const runToken = useRef<object | undefined>(undefined);
  const { saved, update } = practice;
  const online = !!health?.judge.ready && !healthError;
  const hosted = health?.mode === "hosted";
  const browserWorkspace = practice.owner === 'local' || practice.owner === 'anonymous';
  const testAuth = capabilities?.auth_mode === "test";
  const [problems, setProblems] = useState<Problem[]>([]),
    [categories, setCategories] = useState<Category[]>([]),
    [papers, setPapers] = useState<Paper[]>([]),
    [loadError, setLoadError] = useState(""),
    [id, setId] = useState(getId),
    [tab, setTab] = useState("description"),
    [rightTab, setRightTab] = useState("code"),
    [preparingRun, setPreparingRun] = useState(false),
    [toast, setToast] = useState(""),
    [font, setFont] = useState(initialEditorFontSize),
    [theme, setTheme] = useState<Theme>(initialTheme);
  const [details, setDetails] = useState<Record<string, Problem>>({});
  const [detailError, setDetailError] = useState("");
  const [detailRetry, setDetailRetry] = useState(0);
  const [labView, setLabView] = useState<ExperimentView>("auto");
  const [submissionRevision, setSubmissionRevision] = useState(0);
  function openLab(view: ExperimentView = "auto") {
    setLabView(view);
    setTab("lab");
  }
  const { stats: problemStats, refresh: refreshProblemStats, loading: statsLoading } = useProblemStats(id);
  const sourceProblem = problems.find(
    (p) => p.id === id || String(p.number) === id,
  );
  const localize = (p: Problem): Problem =>
    locale === "en" && p.translations?.en ? { ...p, ...p.translations.en } : p;
  const loadedProblem = sourceProblem?.detail_url ? details[sourceProblem.detail_url] : sourceProblem;
  const problem = loadedProblem ? localize(loadedProblem) : undefined;
  const submissionPage = id === "submit";
  const communityPath = id.split('?')[0];
  const communityPage = communityPath === "community" || communityPath.startsWith("community/");
  const communityTopicId = topicIdFromRoute(communityPath);
  const communityProblem = communityTopicId ? problems.find((p) => p.id === communityTopicId) : undefined;
  const accountPage = isAccountRoute(id);
  const profileUsername = id.startsWith("u/") ? id.slice(2) : "";
  const current = problem ? saved[problem.id] || {} : {};
  const selectedPaper = papers.find((paper) =>
    sourceProblem ? paper.id === sourceProblem.paper_id : id === `papers/${paper.id}`,
  );
  const siblingModules = selectedPaper
    ? paperModules(selectedPaper, problems)
    : sourceProblem
      ? [sourceProblem]
      : [];
  const moduleIndex = siblingModules.findIndex((p) => p.id === problem?.id);
  const learning = useBrowserLearning(hosted ? practice.owner : 'unresolved');
  const learningView = browserLearningView(practice.owner,problems,saved,learning.records);
  const learningSaved = learningView.saved;
  const notesUnlocked = !!problem && status(problem,learningSaved[problem.id]) === 'done';
  const learningNotice = hosted && <BrowserLearningNotice count={learningView.localPasses} failed={learning.failed} saving={learning.saving} onRetry={learning.retry} />;
  const progressReady = practice.ready && !practice.loading && learning.ready;
  const resumeProblem = progressReady ? (selectedPaper ? recommendPractice(siblingModules, learningSaved) : undefined) || recommendPractice(problems, learningSaved) : undefined;
  const nextProblem = problem && progressReady
    ? nextPractice(problem, selectedPaper?.module_ids || [], problems, learningSaved)
    : undefined;
  const browser = useBrowserJudge(practice.owner);
  const queueReady = !publicSite && practice.ready && !practice.loading && problems.length > 0 && (!hosted || !!user);
  const judge = useQueuedJudge(practice.owner, queueReady, (report, runOwner) => {
    const entry = problems.find((p) => p.id === report.problem);
    if (!entry || !practice.owns(runOwner)) return;
    practice.applyReport(entry, report, runOwner);
    browser.clear(entry.id);
    learning.dismiss(entry,runOwner);
    void refreshProblemStats();
    if ([entry.id, String(entry.number)].includes(getId())) setTab("tests");
    setSubmissionRevision(value => value + 1);
    setToast(runOwner.startsWith("user:") ? t("提交完成，解答已保存到账号", "Submission complete. Solution saved to your account.") : t("测试已完成", "Run complete"));
  });
  useEffect(() => { runToken.current = undefined; setPreparingRun(false); }, [practice.owner]);
  const report = problem ? browser.reports[problem.id] || learningView.records[problem.id]?.report || current.report : undefined;
  const syntaxFailure = problem ? browser.syntaxErrors?.[problem.id] : undefined;
  const hasCurrentSyntaxError = !!syntaxFailure && syntaxFailure.source === (current.code ?? problem?.starter);
  const reportSource = useReportSource(report, current.code ?? problem?.starter ?? "");
  const currentSource = current.code ?? problem?.starter ?? '';
  const [editorFocus, setEditorFocus] = useState<(EditorFocusRequest & { owner: string; problem: string })>();
  useEffect(() => {
    setEditorFocus(previous => previous && previous.owner === practice.owner && previous.problem === problem?.id && previous.source === currentSource ? previous : undefined);
  }, [practice.owner, problem?.id, currentSource]);
  function locateSource(location: SourceLocation) {
    if (!problem || !practice.ready || !practice.owns(practice.owner) || ![problem.id,String(problem.number)].includes(getId())) return;
    setRightTab('code');
    setEditorFocus({ owner: practice.owner, problem: problem.id, source: currentSource, location });
  }
  function editorFocused(request: EditorFocusRequest) {
    setEditorFocus(previous => previous === request ? undefined : previous);
  }
  const browserAvailable = !!problem?.browser_judge;
  const browserHere = browser.problem === problem?.id;
  const running = preparingRun || judge.active || browser.active;
  const editorRecoveryReason = running
    ? t('请先停止测试或等待运行完成，再重新加载。', 'Stop the tests or wait for the run to finish before reloading.')
    : Object.keys(practice.conflicts).length
      ? t('请先解决草稿冲突，再重新加载。', 'Resolve draft conflicts before reloading.')
    : practice.error
      ? t('工作区同步失败，请先重试同步。', 'Workspace sync failed. Retry syncing first.')
    : learning.failed
      ? t('请先重试保存本机学习记录，再重新加载。', 'Retry saving browser learning progress before reloading.')
    : !practice.ready || practice.loading || !learning.ready
      ? t('正在加载工作区，暂不能重新加载编辑器。', 'Wait for the workspace to finish loading before reloading the editor.')
    : practice.pending || practice.saving || learning.saving
      ? t('正在保存草稿和学习记录，保存完成后可重新加载。', 'Saving drafts and learning progress. Reloading becomes available when saving finishes.')
    : '';
  const browserStatus = browserProgressText(browser.progress || { stage: browser.stage || 'loading' }, locale === 'en');
  const queuedHere = judge.pending?.problem === problem?.id;
  const runStatus = judge.reconnecting ? t("连接中断，正在恢复任务状态…", "Reconnecting to your existing job…")
    : judge.job?.status === "queued" ? t(`正在排队，前方 ${judge.job.ahead} 个任务`, `Queued · ${judge.job.ahead} jobs ahead`)
    : judge.job?.status === "running" ? t("正在执行测试…", "Running tests…")
    : t("正在提交判题任务…", "Submitting your job…");
  function loadCatalog() {
    return fetch(
      `${import.meta.env.BASE_URL}data/index.json`,
      { cache: "no-cache" },
    )
      .then(async (r) => {
        if (!r.ok)
          throw Error(
            t(
              "暂时无法加载题库，请稍后刷新。",
              "The problem catalog is unavailable. Please refresh shortly.",
            ),
          );
        return r.json();
      })
      .then((data) => {
        setProblems(Array.isArray(data) ? data : data.problems);
        const loadedPapers: Paper[] = Array.isArray(data.papers) ? [...data.papers] : [];
        // Shuffle on catalog load, keeping the order stable during browsing and filtering.
        for (let i = loadedPapers.length - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1));
          [loadedPapers[i], loadedPapers[j]] = [loadedPapers[j], loadedPapers[i]];
        }
        setPapers(loadedPapers);
        setCategories(Array.isArray(data.categories) ? data.categories : []);
        setLoadError("");
      })
      .catch((e) => setLoadError(e.message));
  }
  useEffect(() => {
    void loadCatalog();
    const h = () => {
      setId(getId());
      setTab("description");
      setRightTab("code");
      window.scrollTo(0, 0);
    };
    window.addEventListener("hashchange", h);
    return () => window.removeEventListener("hashchange", h);
  }, []);
  useEffect(() => {
    setDetailError("");
    if (!sourceProblem?.detail_url || details[sourceProblem.detail_url]) return;
    const controller = new AbortController();
    const key = sourceProblem.detail_url;
    void loadProblemDetail(sourceProblem, import.meta.env.BASE_URL, fetch, controller.signal)
      .then((detail) => {
        if (!controller.signal.aborted) setDetails((previous) => ({ ...previous, [key]: detail }));
      })
      .catch(() => {
        if (!controller.signal.aborted) setDetailError(key);
      });
    return () => controller.abort();
  }, [sourceProblem?.detail_url, detailRetry]);
  useEffect(() => {
    setPreparingRun(false);
    setToast("");
    setTab("description");
  }, [practice.owner]);
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    document.documentElement.style.colorScheme = theme;
    document.querySelector('meta[name="theme-color"]')?.setAttribute(
      "content",
      theme === "dark" ? "#1a1a1a" : "#ffffff",
    );
  }, [theme]);
  function selectTheme(nextTheme: Theme) {
    setTheme(nextTheme);
    try {
      localStorage.setItem("paperleet.theme", nextTheme);
    } catch {
      // Theme selection still works for this session without browser storage.
    }
  }
  useEffect(() => {
    if (!toast) return;
    const timer = setTimeout(() => setToast(""), 6000);
    return () => clearTimeout(timer);
  }, [toast]);
  async function run(requestedMode: "browser" | "server", verification?: Report) {
    if (!problem || running || runToken.current || !practice.ready) return;
    const runOwner = practice.owner;
    const token = {};
    runToken.current = token;
    const ownsRun = () => practice.owns(runOwner) && runToken.current === token;
    const submittedCode = current.code ?? problem.starter;
    setPreparingRun(true);
    try {
      if (verification) {
        const matches = await matchesBrowserSubmission(problem,verification,submittedCode);
        if (!ownsRun()) return;
        if (!matches) { setToast(t('代码版本已变化，请重新运行后确认。', 'The source version changed. Run tests again before verifying.')); return; }
      }
      if (problem.browser_judge && requestedMode !== "server") {
        const outcome = await browser.run(problem, submittedCode);
        if (!ownsRun()) return;
        if (outcome.type === "result") {
          // Hosted completions and shared statistics remain server-authoritative.
          if (!user) practice.applyReport(problem, outcome.report, runOwner);
          else learning.record(problem,outcome.report,runOwner);
          if ([problem.id,String(problem.number)].includes(getId())) setTab("tests");
          setToast(t("浏览器测试已完成", "Browser run complete"));
          return;
        }
        if (outcome.type === 'syntax_error') {
          if ([problem.id,String(problem.number)].includes(getId())) setTab('tests');
          setToast(t('请先修正 Python 语法错误。','Fix the Python syntax error before running tests.'));
          return;
        }
        return;
      }
      if (!ownsRun()) return;
      if (publicSite) {
        if (!hostedAppAvailable) {
          setToast(t("当前版本暂未开放服务端提交。未登录的代码仅在当前页面保留。", "Server submissions are not available yet. Guest code lasts only while this page stays open."));
          return;
        }
        location.assign(hostedAppHref(signInHref(id)));
        return;
      }
      if (hosted && (!user || !online || practice.conflicts[problem.id])) {
        setToast(!user ? t("需要登录后使用服务端判题。", "Sign in to use server judging.")
          : practice.conflicts[problem.id] ? t("请先解决草稿版本冲突。", "Resolve the draft conflict first.")
          : t("服务端判题暂时不可用，请稍后重试。", "Server judging is unavailable. Try again shortly."));
        return;
      }
      if (!queueReady) { setToast(t("工作区正在加载，请稍后提交。", "The workspace is loading. Submit again shortly.")); return; }
      if (hosted) await practice.flush();
      if (!ownsRun()) return;
      await judge.submit(problem.id, submittedCode);
    } catch (e) {
      if (!ownsRun()) return;
      if (e instanceof DOMException && e.name === "AbortError") {
        setToast(t("浏览器运行已停止", "Browser run stopped"));
        return;
      }
      setToast(e instanceof Error ? e.message : t("暂时无法运行，请稍后重试。", "Unable to run. Please try again shortly."));
    } finally {
      if (ownsRun()) { runToken.current = undefined; setPreparingRun(false); }
    }
  }
  function signInToSubmit() {
    if (publicSite && !hostedAppAvailable) return;
    location.assign(hostedAppHref(signInHref(id)));
  }
  async function copy(text: string) {
    try {
      await navigator.clipboard.writeText(text);
      setToast(t("已复制", "Copied"));
    } catch {
      setToast(
        t(
          "剪贴板不可用，请手动复制。",
          "Clipboard unavailable. Please copy manually.",
        ),
      );
    }
  }
  const tabs = [
    ["description", t("题目", "Description")],
    ["explanation", t("讲解", "Explanation")],
    ["lab", t("实验", "Lab")],
    ["tests", t("测试", "Tests")],
    ["notes", t("坑点", "Notes")],
    ["solution", t("参考解", "Solution")],
  ];
  const webReadme = problem?.readme.replace(
    /##\s*6[.、．]?\s*(?:怎么运行|编写与运行|在网页中运行|Write and run|Run your implementation|Running)[^\n]*\n[\s\S]*?(?=\n##\s*7|$)/,
    `## 6. ${t("编写与运行", "Write and run")}\n\n${problem?.browser_judge ? t("在右侧完成实现，点击「运行测试」可在浏览器执行完整测试；全部通过后解锁坑点。登录后点击 Submit，服务端会保存提交时的代码和判题结果，可在「我的提交」中查看。未登录时，代码和进度仅在当前页面保留，刷新或离开后不保存。", "Complete the implementation and select Run tests to run the full suite in your browser. Passing unlocks Notes. Sign in and use Submit to save the submitted code and server result in My submissions. Guest code and progress last only while this page stays open and are cleared on refresh or exit.") : t("在右侧完成实现，登录后点击 Submit，由服务端执行完整测试并保存提交时的代码和结果，可在「我的提交」中查看。未登录的代码仅在当前页面保留。", "Complete the implementation, sign in, and use Submit to run the full suite on the server. The submitted code and result are kept in My submissions. Guest code lasts only while this page stays open.")}\n`,
  );
  return (
    <div className={`app ${problem ? "workspace-view" : ""}`}>
      <header className="topbar">
        <div className="nav-inner">
          <Brand />
          <nav>
            <a
              className={
                !problem &&
                !submissionPage &&
                !communityPage &&
                !accountPage &&
                !profileUsername
                  ? "active"
                  : ""
              }
              href="#"
            >
              <BilingualText zh="论文题库" en="Papers" />
            </a>
            <button
              className={`continue-link ${problem ? "active" : ""}`}
              disabled={!problem && !resumeProblem}
              onClick={() => {
                if (!problem && resumeProblem) location.hash = `/${resumeProblem.number}`;
              }}
            >
              {problem ? <BilingualText zh="工作区" en="Workspace" /> : progressReady && problems.length > 0 && !resumeProblem ? <BilingualText zh="核心模块已完成" en="Core modules complete" /> : resumeProblem && status(resumeProblem, learningSaved[resumeProblem.id]) === "started" ? <BilingualText zh="继续练习" en="Continue" /> : <BilingualText zh="开始练习" en="Start practice" />}
            </button>
            {(!publicSite || hostedAppAvailable) && <a className={submissionPage ? "active" : ""} href={hostedAppHref("#/submit")}>
              <BilingualText zh="提交论文" en="Submit paper" />
            </a>}
            {hostedAppAvailable && <a
              className={communityPage || profileUsername ? "active" : ""}
              href={hostedAppHref("#/community")}
            >
              <BilingualText zh="社区" en="Community" />
            </a>}
          </nav>
          <div className="nav-actions">
            <RepositoryLink />
            <OnlinePresence />
            <button
              className="language-toggle"
              aria-label={t("切换到英文", "Switch to Chinese")}
              onClick={() => setLocale(locale === "zh" ? "en" : "zh")}
            >
              <span className={locale === "zh" ? "selected" : ""}>中</span>
              <i>/</i>
              <span className={locale === "en" ? "selected" : ""}>EN</span>
            </button>
            <div
              className="theme-switcher"
              role="group"
              aria-label={t("外观", "Appearance")}
            >
              <button
                type="button"
                className={theme === "light" ? "selected" : ""}
                aria-label={t("浅色主题", "Light theme")}
                aria-pressed={theme === "light"}
                title={t("浅色主题", "Light theme")}
                onClick={() => selectTheme("light")}
              >
                <Sun size={15} aria-hidden="true" />
              </button>
              <button
                type="button"
                className={theme === "dark" ? "selected" : ""}
                aria-label={t("深色主题", "Dark theme")}
                aria-pressed={theme === "dark"}
                title={t("深色主题", "Dark theme")}
                onClick={() => selectTheme("dark")}
              >
                <Moon size={15} aria-hidden="true" />
              </button>
            </div>
            <AccountNav returnTo={id} />
          </div>
        </div>
      </header>
      {communityPage ? (
        <Community
          route={id}
          topicId={communityTopicId}
          topic={communityProblem ? topicFromProblem(communityProblem) : undefined}
          catalogLoading={!problems.length && !loadError}
          catalogError={loadError}
          reloadCatalog={loadCatalog}
          postId={
            communityPath === "community" || communityTopicId ? undefined : communityPath.slice("community/".length)
          }
        />
      ) : accountPage ? (
        <Account returnTo={accountReturnHref(id)} />
      ) : profileUsername ? (
        <UserProfile key={profileUsername} username={profileUsername} />
      ) : submissionPage ? (
        <Submissions
          key={`${health?.mode || "pending"}:${user?.id || "anonymous"}`}
          onPublished={loadCatalog}
        />
      ) : loadError ? (
        <main className="load-error">
          <Terminal size={35} />
          <h2>{t("题库暂时不可用", "Catalog unavailable")}</h2>
          <p>{loadError}</p>
          <button className="button" onClick={loadCatalog}>
            {t("重新加载", "Reload")}
          </button>
        </main>
      ) : !problems.length ? (
        <main className="loading">
          <span className="spinner" />
          {t("正在加载论文题库…", "Loading papers…")}
        </main>
      ) : sourceProblem && !problem ? (
        <main className={detailError === sourceProblem.detail_url ? "load-error" : "loading"}>
          {detailError === sourceProblem.detail_url ? <>
            <h2>{t("题目暂时无法加载", "Module unavailable")}</h2>
            <p>{t("你的草稿仍然保留，请重试加载题目。", "Your draft is preserved. Try loading this module again.")}</p>
            <button className="button" onClick={() => setDetailRetry((value) => value + 1)}>{t("重试加载", "Retry loading")}</button>
            <a className="text-button" href="#">{t("返回题库", "Back to papers")}</a>
          </> : <><span className="spinner" />{t("正在加载这道题…", "Loading this module…")}</>}
        </main>
      ) : !problem ? (
        <>{learningNotice}{selectedPaper ? (
          <PaperDetail
            paper={selectedPaper}
            problems={problems}
            saved={learningSaved}
            stats={problemStats}
          />
        ) : (
          <PaperLibrary
            key={collectionIdFromRoute(id) || "all-papers"}
            collectionId={collectionIdFromRoute(id)}
            progressReady={progressReady}
            categories={categories}
            papers={papers}
            problems={problems}
            saved={learningSaved}
          />
        )}</>
      ) : (
        <>
          <div className="workspace-toolbar">
            <a
              className="back-link"
              href={selectedPaper ? `#/papers/${selectedPaper.id}` : "#"}
            >
              <ArrowLeft size={15} />
              {selectedPaper
                ? `${selectedPaper.number}. ${selectedPaper.short_title}`
                : t("论文题库", "Papers")}
            </a>
            <span className="toolbar-divider" />
            <b>
              {problem.number}. {problem.title}
            </b>
            <div className="problem-navigation">
              <button
                className="icon-button"
                aria-label={t("上一个模块", "Previous module")}
                disabled={moduleIndex <= 0}
                onClick={() =>
                  (location.hash = `/${siblingModules[moduleIndex - 1].number}`)
                }
              >
                <ChevronLeft size={15} />
              </button>
              <span>
                {moduleIndex + 1} / {siblingModules.length}
              </span>
              <button
                className="icon-button"
                aria-label={t("下一个模块", "Next module")}
                disabled={moduleIndex >= siblingModules.length - 1}
                onClick={() =>
                  (location.hash = `/${siblingModules[moduleIndex + 1].number}`)
                }
              >
                <ChevronRight size={15} />
              </button>
            </div>
          </div>
          <ResizableSplit
            as="main"
            className="workspace"
            axis="horizontal"
            storageKey="paperleet.layout.workspace.v1"
            initial={0.5}
            minFirst={320}
            minSecond={360}
            collapseAt={850}
            label={t("调整题面与代码区宽度", "Resize description and code panels")}
            resetLabel={t("拖动或使用方向键调整，双击或按 Enter 恢复默认", "Drag or use arrow keys to resize; double-click or press Enter to reset")}
            firstId="problem-description-panel"
            secondId="problem-editor-panel"
          >
            <section className="description-pane" id="problem-description-panel">
              <div className="pane-tabs">
                {tabs.map(([key, label]) => (
                  <button
                    key={key}
                    className={tab === key ? "selected" : ""}
                    onClick={() => key === "lab" ? openLab() : setTab(key)}
                  >
                    {key === "description" ? (
                      <FileCode2 size={14} />
                    ) : key === "lab" ? (
                      <FlaskConical size={14} />
                    ) : key === "notes" &&
                      !notesUnlocked ? (
                      <LockKeyhole size={12} />
                    ) : null}
                    {label}
                    {key === "tests" && (report || syntaxFailure) && (
                      <i
                        aria-hidden="true"
                        className={
                          !hasCurrentSyntaxError && reportSource === 'current' && isPassed(problem, report)
                            ? "green-dot"
                            : "yellow-dot"
                        }
                      />
                    )}
                  </button>
                ))}
                {hostedAppAvailable && <a className="module-discussion-tab" href={topicHref(problem.id)}>
                  <MessageCircle size={14} />{t("讨论", "Discuss")}
                </a>}
              </div>
              <div className="description-scroll" key={problem.id}>
                {learningNotice}
                {tab === "tests" ? (
                  <>
                  {syntaxFailure && <BrowserSyntaxNotice failure={syntaxFailure} source={currentSource} onLocate={practice.ready ? locateSource : undefined} />}
                  {(!syntaxFailure || report) && <ReportView report={report} problem={problem} sourceState={reportSource} source={currentSource} onLocate={practice.ready ? locateSource : undefined}>
                    {hosted && report?.runtime === 'browser-numpy-wasm' && <BrowserResultActions problem={problem} report={report} sourceState={reportSource} signedIn={!!user} online={online} ready={progressReady} conflict={!!practice.conflicts[problem.id]} syncError={!!practice.error} busy={running} onSignIn={signInToSubmit} onVerify={() => void run('server',report)} onRefresh={() => void refreshHealth()} onLab={() => openLab('run')} />}
                    {!running && reportSource === "current" && progressReady && isPassed(problem, report) && (
                      <PracticeNextStep next={nextProblem} samePaper={!!nextProblem && nextProblem.paper_id === problem.paper_id} onNotes={() => setTab("notes")} onLab={() => openLab("run")} />
                    )}
                  </ReportView>}
                  </>
                ) : tab === "lab" ? (
                  <Visuals key={`${problem.id}:${labView}`} problem={problem} report={report} sourceState={reportSource} initialView={labView} />
                ) : tab === "notes" ? (
                  !notesUnlocked ? (
                    <div className="empty-state">
                      <div className="empty-icon">
                        <LockKeyhole size={25} />
                      </div>
                      <h3>
                        {t(
                          "通过测试后解锁坑点",
                          "Pass the tests to unlock notes",
                        )}
                      </h3>
                      <p>
                        {t(
                          "完成全部测试后，查看论文与常见实现之间的细节差异。",
                          "Complete every test to explore subtle differences between the paper and common implementations.",
                        )}
                      </p>
                      <button
                        className="button"
                        onClick={() => setTab("tests")}
                      >
                        {t("查看测试进度", "View test progress")}{" "}
                        <ArrowRight size={15} />
                      </button>
                    </div>
                  ) : (
                    <Prose
                      content={problem.notes}
                      problemId={problem.id}
                      onViz={() => openLab("concept")}
                    />
                  )
                ) : tab === "solution" ? (
                  <section className="solution-panel">
                    <h2>{t("参考实现", "Reference implementation")}</h2>
                    <p>
                      {t(
                        "建议先用测试结果定位问题，再对照参考解。",
                        "Use test results to find the issue before checking the reference.",
                      )}
                    </p>
                    <details>
                      <summary>
                        <BookOpen size={16} />
                        {t(
                          "展开参考解（包含答案）",
                          "Reveal reference solution (spoilers)",
                        )}
                      </summary>
                      {problem.solution ? (
                        <Highlight code={problem.solution} />
                      ) : (
                        <div className="notice">
                          {t(
                            "参考实现尚未提供。",
                            "No reference implementation is available yet.",
                          )}
                        </div>
                      )}
                    </details>
                  </section>
                ) : (
                  <>
                    <div className="problem-heading">
                      <div className="heading-kicker">
                        <FileCode2 size={18} />
                        <span>
                          {t("模块", "MODULE")} {problem.number}
                        </span>
                        {notesUnlocked && (
                          <span className="accepted">
                            <CheckCircle2 size={14} />
                            {t("已通过", "Solved")}
                          </span>
                        )}
                      </div>
                      <h1>
                        {problem.number}. {problem.title}
                      </h1>
                      <div className="heading-tags">
                        <span className={`difficulty ${problem.difficulty}`}>
                          {difficulty(problem.difficulty)}
                        </span>
                        <span>PyTorch</span>
                        <span>{problem.estimated_minutes || 25} min</span>
                        {problem.parent && (
                          <a
                            href={`#/${problems.find((p) => p.id === problem.parent)?.number || problem.parent}`}
                          >
                            {t("前置题", "Prerequisite")}:{" "}
                            {problems.find((p) => p.id === problem.parent)
                              ?.number || "—"}
                          </a>
                        )}
                      </div>
                      <ProblemStats stats={getProblemStats(problemStats, problem.id)} loading={statsLoading} />
                    </div>
                    <a
                      className="paper-card"
                      href={
                        problem.paper?.url ||
                        `https://arxiv.org/abs/${problem.paper?.arxiv}`
                      }
                      target="_blank"
                      rel="noreferrer"
                    >
                      <BookOpen size={19} />
                      <div>
                        <small>
                          {t("阅读论文", "READ THE PAPER")}{" "}
                          <span>{problem.paper?.venue}</span>
                        </small>
                        <strong>{problem.paper?.title}</strong>
                        <p>
                          {Array.isArray(problem.paper?.authors)
                            ? problem.paper.authors.join(", ")
                            : problem.paper?.authors}
                        </p>
                      </div>
                      <ExternalLink size={14} />
                    </a>
                    {problem.module && (
                      <section className="module-scope-card">
                        <h3>{t("本模块的目标", "Your implementation goal")}</h3>
                        <p>{t(problem.module.goal, problem.module.goal_en)}</p>
                        <p className="module-scope-detail">
                          {t(problem.module.scope, problem.module.scope_en)}
                        </p>
                        <div className="module-functions">
                          <span>{t("需要实现", "Implement")}</span>
                          {problem.module.required_functions.map((name) => (
                            <code key={name}>{name}</code>
                          ))}
                        </div>
                        {!!problem.module.provided_functions.length && (
                          <div className="module-functions provided-functions">
                            <span>{t("已提供", "Provided")}</span>
                            {problem.module.provided_functions.map((name) => (
                              <code key={name}>{name}</code>
                            ))}
                          </div>
                        )}
                      </section>
                    )}
                    {tab === "explanation" && (
                      <Visuals key={problem.id} problem={problem} report={report} sourceState={reportSource} initialView="concept" />
                    )}
                    {locale === "en" && !problem.translations?.en && (
                      <div className="notice">
                        English problem content is not available yet. The
                        original text appears below.
                      </div>
                    )}
                    <Prose
                      content={webReadme || ""}
                      problemId={problem.id}
                      onViz={() => openLab("concept")}
                    />
                    {tab === "description" && (
                      <button
                        className="lab-callout"
                        onClick={() => openLab(problem.vizData?.kind === "toy_experiment" ? "auto" : "concept")}
                      >
                        <FlaskConical size={21} />
                        <span>
                          <b>{problem.vizData?.kind === "toy_experiment" ? t("查看核心算法实验", "Explore the core experiment") : t("把公式放进实验台", "Explore the formulas")}</b>
                          <small>
                            {problem.vizData?.kind === "toy_experiment" ? t("对比参考实现与自己的运行结果", "Compare the reference with your run results") : t(
                              "调整参数，观察结果怎样变化",
                              "Adjust parameters and see what changes",
                            )}
                          </small>
                        </span>
                        <ArrowRight size={18} />
                      </button>
                    )}
                  </>
                )}
              </div>
            </section>
            <section className="editor-pane" id="problem-editor-panel">
              <div className="pane-tabs editor-tabs">
                <button
                  className={rightTab === "code" ? "selected" : ""}
                  onClick={() => setRightTab("code")}
                >
                  <Code2 size={15} />
                  {t("代码", "Code")}
                </button>
                <button
                  className={rightTab === "lab" ? "selected" : ""}
                  onClick={() => setRightTab("lab")}
                >
                  <FlaskConical size={14} />
                  {t("实验台", "Lab")}
                </button>
                {user && <button className={rightTab === "history" ? "selected" : ""} onClick={() => setRightTab("history")}>
                  {t("我的提交", "My submissions")}
                </button>}
                <span className="editor-language">Python 3 · PyTorch</span>
              </div>
              {rightTab === "history" && user ? (
                <SolutionHistory key={`${practice.owner}:${problem.id}`} owner={practice.owner} problem={problem} revision={submissionRevision} />
              ) : rightTab === "lab" ? (
                <div className="editor-lab">
                  <Visuals key={problem.id} problem={problem} report={report} sourceState={reportSource} initialView="auto" />
                </div>
              ) : (
                <>
                  <ResizableSplit
                    className="editor-split"
                    axis="vertical"
                    storageKey="paperleet.layout.editor.v1"
                    initial={0.72}
                    minFirst={180}
                    minSecond={100}
                    label={t("调整代码与测试结果区高度", "Resize code and test results panels")}
                    resetLabel={t("拖动或使用方向键调整，双击或按 Enter 恢复默认", "Drag or use arrow keys to resize; double-click or press Enter to reset")}
                    firstId="problem-code-panel"
                    secondId="problem-results-panel"
                  >
                  <div className="editor-code-panel" id="problem-code-panel">
                  <div className="editor-tools">
                    <span>
                      <FileCode2 size={13} />
                      solution.py
                    </span>
                    <div>
                      <label className="editor-font-size">
                        <span aria-hidden="true">Aa</span>
                        <select
                          className="editor-font-size-select"
                          value={font}
                          title={t("代码字号", "Code font size")}
                          aria-label={t("代码字号", "Code font size")}
                          onChange={(event) => {
                            const size = Number(event.target.value);
                            setFont(size);
                            try {
                              localStorage.setItem(
                                "paperleet.editor.fontSize",
                                String(size),
                              );
                            } catch {
                              // Keep the selected size for this session.
                            }
                          }}
                        >
                          {editorFontSizes.map((size) => (
                            <option key={size} value={size}>{size} px</option>
                          ))}
                        </select>
                      </label>
                      <button
                        className="icon-button"
                        title={t("复制代码", "Copy code")}
                        aria-label={t("复制代码", "Copy code")}
                        onClick={() => copy(current.code ?? problem.starter)}
                      >
                        <Copy size={14} />
                      </button>
                      <button
                        className="icon-button"
                        title={t("下载实现", "Download code")}
                        aria-label={t("下载实现", "Download code")}
                        onClick={() =>
                          download(
                            `problem-${problem.number}.py`,
                            current.code ?? problem.starter,
                          )
                        }
                      >
                        <ArrowDownToLine size={14} />
                      </button>
                      <ResetCodeButton key={`${practice.owner}:${problem.id}`} owner={practice.owner} problem={problem} saved={current}
                        ready={practice.ready} busy={practice.pending || practice.saving || running} conflict={!!practice.conflicts[problem.id]} failed={!!practice.error} signedIn={!!user}
                        onReset={expected => {
                          if (!practice.resetToStarter(problem,expected)) return false;
                          setToast(t("已恢复初始代码，已提交的解答仍可在提交记录中查看。", "Starter restored. Submitted solutions remain in your submission history."));
                          return true;
                        }} />
                    </div>
                  </div>
                  <div className="code-editor">
                    <PythonEditor
                      key={`${practice.owner}:${problem.id}`}
                      readOnly={!practice.ready}
                      value={current.code ?? problem.starter}
                      focusRequest={editorFocus?.owner === practice.owner && editorFocus.problem === problem.id && editorFocus.source === currentSource ? editorFocus : undefined}
                      onFocusHandled={editorFocused}
                      theme={theme === "dark" ? "dark" : "light"}
                      onChange={(value) => update(problem.id, { code: value })}
                      recovery={{blocked:!!editorRecoveryReason,reason:editorRecoveryReason,reload:() => {
                        if (!editorRecoveryReason && practice.owns(practice.owner) && [problem.id,String(problem.number)].includes(getId())) location.reload();
                      }}}
                      font={font}
                      label={`${problem.title} ${t("Python 代码编辑器", "Python code editor")}`}
                    />
                  </div>
                  <div className="editor-status">
                    <span>
                      <Check size={12} />
                      {practice.loading
                        ? t("正在加载工作区…", "Loading workspace…")
                        : !user
                          ? t('未登录 · 刷新或离开后代码不保留', 'Not signed in · code is cleared on refresh or exit')
                        : hosted
                          ? !user
                            ? t(
                                "未登录的代码仅在当前页面保留",
                                "Guest code lasts only while this page stays open",
                              )
                            : practice.error
                              ? t(
                                  "同步失败，草稿保留在当前浏览器",
                                  "Sync failed · draft kept in this browser",
                                )
                              : practice.conflicts[problem.id]
                                ? t(
                                    "草稿版本冲突，请选择保留的版本",
                                    "Draft conflict · choose a version below",
                                  )
                                : practice.saving || practice.pending
                                  ? t("正在同步草稿…", "Syncing draft…")
                                  : t("已同步到账号", "Synced to your account")
                          : current.code !== undefined
                            ? t("代码仅在当前页面保留", "Code is kept only on this page")
                            : t("初始代码", "Starter code")}
                    </span>
                    <span>
                      UTF-8 <i>·</i> Python
                    </span>
                  </div>
                  </div>
                  <div className="editor-results-panel" id="problem-results-panel">
                  {problem.browser_judge && <div className="browser-judge-panel">
                    {browser.active && (browserHere
                      ? <BrowserRunProgress progress={browser.progress || { stage: browser.stage || 'loading' }} onStop={browser.stop} />
                      : <div role="status">{t("另一道题正在浏览器运行", "Another module is running in the browser")} <button onClick={browser.stop}>{t("停止运行", "Stop run")}</button></div>)}
                    <small>{t('运行测试会在浏览器中执行；点击 Submit 后，服务端判题并将解答保存到账号。', 'Run tests executes in your browser. Submit runs on the server and saves your solution to your account.')}</small>
                    {browserBudget(problem).total > 120_000 && <small>{t(`完整实验最多运行 ${browserBudget(problem).total / 60_000} 分钟；可随时停止。`, `The full experiment can take up to ${browserBudget(problem).total / 60_000} minutes. You can stop at any time.`)}</small>}
                    {browser.reasons[problem.id] && <p role="status">{browserReason(browser.reasons[problem.id], locale === "en")} {t('可以修改后重试，或登录后提交到服务端。', 'Edit and retry, or sign in to submit on the server.')}</p>}
                    {browser.calibrations[problem.id] && report?.runtime === "browser-numpy-wasm" && <small>{t("兼容性校验", "Compatibility calibration")}: {browser.calibrations[problem.id].reference_passed} {t("项参考测试通过", "reference tests passed")} · {Object.keys(browser.calibrations[problem.id].mutants_caught).length} {t("个错误实现被抓住", "mutants caught")}</small>}
                    {browser.diagnostics[problem.id] && <details><summary>{t("运行时详情", "Runtime details")}</summary><code>{browser.diagnostics[problem.id]}</code></details>}
                    {report?.runtime === "browser-numpy-wasm" && <p role="status">{t("本次结果来自浏览器 · NumPy / WASM", "Results from your browser · NumPy / WASM")}</p>}
                  </div>}
                  {!problem.browser_judge && <div className="server-judge-panel"><strong>{t('服务端 CPU 判题', 'Server CPU judging')}</strong><small>{t('在网页完成实现后提交，服务端执行完整测试并返回结果。', 'Implement on the web, then submit for the complete suite and feedback from the server.')}</small>{hosted && !user && <small>{t('登录后可提交并保存解答；未登录的代码仅在当前页面保留。', 'Sign in to submit and save your solution. Guest code lasts only while this page stays open.')}</small>}</div>}
                  {judge.active && <div className="practice-notice judge-queue-notice" role="status">
                    <span><b>{queuedHere ? runStatus : t("另一道题正在排队或判题", "Another module has a queued or running job")}</b><br />
                      {t("可以继续编辑；本次判题使用提交时的代码。刷新后可继续查看任务。", "You can keep editing. This run uses the submitted code; refresh to reconnect.")}</span>
                    {!queuedHere && <a href={`#/${judge.pending?.problem}`}>{t("查看任务", "View job")}</a>}
                    {judge.job?.status === "queued" && <button onClick={() => void judge.cancel()}>{t("取消排队", "Leave queue")}</button>}
                  </div>}
                  {judge.error && <div className="practice-notice" role="alert">{jobErrorText(judge.error, locale)}</div>}
                  {practice.error && <div className="practice-notice" role="alert">
                    <span>{practice.error}</span>
                    <button disabled={practice.saving} onClick={practice.retry}>
                      {browserWorkspace ? t('重试本机保存','Retry browser save') : t("重试同步", "Retry sync")}
                    </button>
                  </div>}
                  {!judge.active && judge.job?.status === "cancelled" && <div className="practice-notice" role="status">{t("已取消排队，本次提交未执行。", "Removed from the queue. This submission was not executed.")}</div>}
                  {(practice.conflicts[problem.id] ||
                    (!health && healthError) ||
                    (hosted && !browserAvailable && (!user || !online))) && (
                    <div className="practice-notice" role="status">
                      {practice.conflicts[problem.id] ? (
                        <>
                          <span>
                            {t(
                              browserWorkspace ? "另一标签页修改了这道题。两个版本均已保留，请核对后选择。" : "其他设备更新了这道题的草稿。选择保留哪个版本；当前代码尚未覆盖服务器。",
                              browserWorkspace ? "Another tab changed this module. Both versions are kept; review them before choosing." : "Another device updated this draft. Choose which version to keep; your current code has not overwritten the server.",
                            )}
                          </span>
                          <button
                            disabled={practice.saving}
                            onClick={() =>
                              practice.resolveConflict(problem.id, false)
                            }
                          >
                            {browserWorkspace ? t('使用另一版本','Use the other version') : t("使用服务器版本", "Use server draft")}
                          </button>
                          <button
                            disabled={practice.saving}
                            onClick={() =>
                              practice.resolveConflict(problem.id, true)
                            }
                          >
                            {t("保留当前版本", "Keep this draft")}
                          </button>
                        </>
                      ) : !health ? (
                        <>
                          <span>
                            {t(
                              "无法确认服务配置，工作区尚未加载。",
                              "Service configuration is unavailable. Your workspace has not been loaded.",
                            )}
                          </span>
                          <button onClick={() => void refreshHealth()}>
                            {t("重试", "Retry")}
                          </button>
                        </>
                      ) : !user ? (
                        <>
                          <span>
                            {testAuth
                              ? t(
                                  "使用管理员提供的内测账号和访问码登录后，即可运行测试。",
                                  "Sign in with your administrator-issued test account and access code to run tests.",
                                )
                              : t(
                                  "验证学校或公司邮箱并登录后，即可使用在线判题。",
                                  "Verify your school or company email and sign in to use online judging.",
                                )}
                          </span>
                          <a href={signInHref(id)}>
                            {t("登录以运行", "Sign in to run")}
                          </a>
                        </>
                      ) : (
                        <>
                          <span>
                            {t(
                              "判题沙箱尚未就绪，请稍后重试。草稿仍会同步。",
                              "The judge sandbox is not ready. Please try again shortly; your drafts will still sync.",
                            )}
                          </span>
                          <button onClick={() => void refreshHealth()}>
                            {t("刷新状态", "Refresh status")}
                          </button>
                        </>
                      )}
                    </div>
                  )}
                  {!user && <div className="practice-notice" role="status"><span>{t('未登录时不保存代码和练习记录。请先登录再开始需要保存的解答；登录跳转后当前代码不会自动带入。', 'Guest code and practice records are not saved. Sign in before starting a solution you want to keep; your current code will not transfer through sign-in.')}</span>{hostedAppAvailable && <button onClick={signInToSubmit}>{t('登录', 'Sign in')}</button>}</div>}
                  <div className="console">
                    <ReportSourceNotice state={reportSource} />
                    <div className="console-header">
                      <span>
                        <Terminal size={14} />
                        {t("测试结果", "Test results")}
                      </span>
                      {(report || syntaxFailure) && (
                        <button
                          className="text-button"
                          onClick={() => setTab("tests")}
                        >
                          {t("查看详情", "View details")}{" "}
                          <ArrowRight size={13} />
                        </button>
                      )}
                    </div>
                    <p>
                      {running
                        ? browserHere ? browserStatus : queuedHere ? runStatus : t("正在处理另一道题的判题任务。", "A judge job for another module is in progress.")
                        : syntaxFailure
                          ? hasCurrentSyntaxError
                            ? t('语法检查失败。查看详情定位错误，修改后重新运行。','Syntax check failed. View details, fix the source, and run again.')
                            : t('代码已修改，请重新运行语法检查与测试。','Code changed. Run syntax checks and tests again.')
                        : report
                          ? `${t("上次结果：", "Last result: ")}${report.tests.filter((test) => test.passed).length} / ${report.tests.length} ${t("项测试通过。", "tests passed.")}`
                          : t(
                              "完成实现后，点击下方按钮运行测试。",
                              "Complete the implementation, then run the tests below.",
                            )}
                    </p>
                  </div>
                  </div>
                  </ResizableSplit>
                  <div className="editor-bottom">
                    <span className="editor-run-status" role="status">
                      {running ? (
                        <>
                          <span className="spinner" />
                          {judge.job?.status === "queued" ? t("正在排队…", "Queued…") : t("正在判题…", "Running tests…")}
                        </>
                      ) : hosted && !user && !browserAvailable ? (
                        <a href={signInHref(id)}>
                          {t("登录后运行", "Sign in to run")}
                        </a>
                      ) : online || browserAvailable ? (
                        <>
                          <i className="green-dot" />
                          {browserAvailable
                            ? online
                              ? t("浏览器 / 服务端可用", "Browser / server ready")
                              : t("浏览器测试可用", "Browser tests ready")
                            : t("判题服务已连接", "Judge connected")}
                        </>
                      ) : (
                        <>
                          <i className="yellow-dot" />
                          {t("判题服务未连接", "Judge offline")}
                        </>
                      )}
                    </span>
                    <div className="editor-run-actions">
                      {(browserAvailable || (!hosted && !publicSite)) && <button
                        className="button subtle run-button"
                        disabled={running || !practice.ready}
                        onClick={() => void run(browserAvailable ? 'browser' : 'server')}
                      >
                        <Play size={15} fill="currentColor" />
                        {t("运行测试", "Run tests")}
                      </button>}
                      {hostedAppAvailable && <button
                        className="button run-button"
                        disabled={running || !practice.ready || (!!user && (!queueReady || !online || !!practice.conflicts[problem.id]))}
                        onClick={() => user ? void run('server') : signInToSubmit()}
                      >
                        {user ? t("提交", "Submit") : t("登录后提交", "Sign in to submit")}
                      </button>}
                    </div>
                  </div>
                </>
              )}
            </section>
          </ResizableSplit>
        </>
      )}
      {toast && (
        <div role="status" className="toast">
          <CheckCircle2 size={16} />
          {toast}
          <button
            className="icon-button"
            aria-label={t("关闭提示", "Dismiss notification")}
            onClick={() => setToast("")}
          >
            <X size={13} />
          </button>
        </div>
      )}
    </div>
  );
}
