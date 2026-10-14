import { useEffect, useRef, useState } from "react";
import { Highlight } from "./SyntaxHighlight";
import { Community, UserProfile, Account, Submissions, Visuals, ArtifactCharts, Prose, PythonEditor } from "./deferredPages";
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
  Globe2,
  LockKeyhole,
  Moon,
  MessageCircle,
  Play,
  RotateCcw,
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
import { getProblemStats, ProblemStats, useProblemStats } from "./ProblemStats";
import ResizableSplit from "./ResizableSplit";
import { recommendPractice, nextPractice } from "./practice-recommendation";
import { PracticeNextStep } from "./PracticeJourney";
import { loadProblemDetail } from "./catalogData";
import { useQueuedJudge } from "./useQueuedJudge";
import { jobErrorText } from "./judgeJobs";
import { useBrowserJudge } from "./useBrowserJudge";
import { browserReason, type JudgeMode } from "./browserJudge";
import { browserBudget } from './browserBudgets';
import { topicHref, topicIdFromRoute, topicFromProblem } from "./communityTopics";
import { accountReturnHref, isAccountRoute, signInHref } from "./accountNavigation";
import { reportTestLabel, type ExperimentView } from "./experimentResults";
import { ReportSourceNotice, useReportSource } from "./ReportSourceNotice";
import type { ReportSourceState } from "./reportSource";
import TestComparison from "./TestComparison";
import { BrowserResultActions } from './BrowserResultActions';
import { matchesBrowserSubmission } from './browserVerification';
import { useDraftHandoff } from './useDraftHandoff';
import { DraftHandoffNotice } from './DraftHandoffNotice';
import { BrowserSyntaxNotice } from './BrowserSyntaxNotice';
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
  children,
}: {
  report?: Report;
  problem: Problem;
  sourceState: ReportSourceState;
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
            "在右侧编写代码，然后点击「运行全部测试」。",
            "Write your code on the right, then select Run all tests.",
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
  const handoff = useDraftHandoff(practice.owner);
  const runToken = useRef<object | undefined>(undefined);
  const { saved, update } = practice;
  const online = !!health?.judge.ready && !healthError;
  const hosted = health?.mode === "hosted";
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
    [guide, setGuide] = useState(false),
    [font, setFont] = useState(initialEditorFontSize),
    [theme, setTheme] = useState<Theme>(initialTheme);
  const [details, setDetails] = useState<Record<string, Problem>>({});
  const [detailError, setDetailError] = useState("");
  const [detailRetry, setDetailRetry] = useState(0);
  const [labView, setLabView] = useState<ExperimentView>("auto");
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
  const progressReady = practice.ready && !practice.loading;
  const resumeProblem = progressReady ? (selectedPaper ? recommendPractice(siblingModules, saved) : undefined) || recommendPractice(problems, saved) : undefined;
  const nextProblem = problem && progressReady
    ? nextPractice(problem, selectedPaper?.module_ids || [], problems, saved)
    : undefined;
  const browser = useBrowserJudge(practice.owner);
  const judge = useQueuedJudge(practice.owner, progressReady && problems.length > 0, (report, runOwner) => {
    const entry = problems.find((p) => p.id === report.problem);
    if (!entry || !practice.owns(runOwner)) return;
    practice.applyReport(entry, report, runOwner);
    browser.clear(entry.id);
    void refreshProblemStats();
    if ([entry.id, String(entry.number)].includes(getId())) setTab("tests");
    setToast(t("判题已完成", "Run complete"));
  });
  useEffect(() => { runToken.current = undefined; setPreparingRun(false); }, [practice.owner]);
  const [judgeMode, setJudgeMode] = useState<JudgeMode>("auto");
  const report = problem ? browser.reports[problem.id] || current.report : undefined;
  const syntaxFailure = problem ? browser.syntaxErrors?.[problem.id] : undefined;
  const hasCurrentSyntaxError = !!syntaxFailure && syntaxFailure.source === (current.code ?? problem?.starter);
  const reportSource = useReportSource(report, current.code ?? problem?.starter ?? "");
  const handoffProblem = problems.find(entry => entry.id === handoff.pending?.problem);
  const handoffSaved = handoffProblem ? saved[handoffProblem.id] || {} : {};
  const browserAvailable = !!problem?.browser_judge && judgeMode !== "server";
  const browserHere = browser.problem === problem?.id;
  const running = preparingRun || judge.active || browser.active;
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
  async function run(requestedMode: JudgeMode = judgeMode, verification?: Report) {
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
          if (!hosted) practice.applyReport(problem, outcome.report, runOwner);
          if ([problem.id,String(problem.number)].includes(getId())) setTab("tests");
          setToast(t("浏览器测试已完成", "Browser run complete"));
          return;
        }
        if (outcome.type === 'syntax_error') {
          if ([problem.id,String(problem.number)].includes(getId())) setTab('tests');
          setToast(t('请先修正 Python 语法错误。','Fix the Python syntax error before running tests.'));
          return;
        }
        if (requestedMode === "browser") return;
      }
      if (!ownsRun()) return;
      if (hosted && (!user || !online || practice.conflicts[problem.id])) {
        setToast(!user ? t("需要登录后使用服务端判题。", "Sign in to use server judging.")
          : practice.conflicts[problem.id] ? t("请先解决草稿版本冲突。", "Resolve the draft conflict first.")
          : t("服务端判题暂时不可用，请稍后重试。", "Server judging is unavailable. Try again shortly."));
        return;
      }
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
  function signInWithDraft() {
    if (practice.error) {
      setToast(t('草稿尚未保存，请先下载代码备份再登录。', 'Your draft could not be saved. Download a code backup before signing in.'));
      return;
    }
    if (!problem || !progressReady || !handoff.prepare(problem.id,current.code ?? problem.starter)) return;
    location.hash = signInHref(id).slice(1);
  }
  function importHeldDraft(backup: boolean) {
    const held = handoff.pending;
    if (!held?.owner || !handoffProblem || !progressReady || !practice.owns(held.owner) || problem?.id !== held.problem || practice.conflicts[held.problem]) return false;
    if (backup) download(`paperleet-${held.problem.replaceAll('/','-')}-before-import.py`,handoffSaved.code || '');
    const applied = update(held.problem,{code:held.code},{owner:held.owner,code:handoffSaved.code,updated:handoffSaved.updated});
    if (!applied) return false;
    handoff.discard(held.id);
    setToast(t('草稿已导入；重新运行后可确认账号成绩。', 'Draft imported. Run tests again to verify account results.'));
    return true;
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
    `## 6. ${t("编写与运行", "Write and run")}\n\n${hosted && problem?.browser_judge ? t("本题可直接在浏览器 CPU/WASM 运行完整测试，无需 GPU。未登录时草稿保存在当前浏览器；登录后可确认导入本题代码，再由服务端判题计入账号成绩并解锁坑点。", "Run this complete suite directly on your browser CPU/WASM; no GPU is needed. Signed-out drafts stay in this browser. After signing in, choose whether to import this module’s code, then use server judging to record account results and unlock Notes.") : hosted ? (testAuth ? t("使用内测账号登录后，可在右侧完成实现并运行测试。草稿与判题结果会私密同步到账号；全部通过后解锁坑点。", "Sign in with your test account, implement the module, and run tests. Drafts and judging results sync privately to your account; passing all tests unlocks Notes.") : t("受邀用户登录后，可在右侧完成实现并运行测试。草稿与判题结果会私密同步到账号；全部通过后解锁坑点。", "Invited members can sign in, implement the module, and run tests. Drafts and judging results sync privately to your account; passing all tests unlocks Notes.")) : t("在右侧编辑器中完成实现，点击「运行全部测试」。测试结果会在「测试」标签页显示；全部通过后解锁坑点。草稿会自动保存在当前浏览器。", "Complete the implementation in the editor and select Run all tests. Results appear in the Tests tab; passing all tests unlocks Notes. Drafts are saved automatically in this browser.")}\n`,
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
              {t("论文题库", "Papers")}
            </a>
            <button
              className={`continue-link ${problem ? "active" : ""}`}
              disabled={!problem && !resumeProblem}
              onClick={() => {
                if (!problem && resumeProblem) location.hash = `/${resumeProblem.number}`;
              }}
            >
              {problem ? t("工作区", "Workspace") : progressReady && problems.length > 0 && !resumeProblem ? t("核心模块已完成", "Core modules complete") : resumeProblem && status(resumeProblem, saved[resumeProblem.id]) === "started" ? t("继续练习", "Continue") : t("开始练习", "Start practice")}
            </button>
            <a className={submissionPage ? "active" : ""} href="#/submit">
              {t("提交论文", "Submit paper")}
            </a>
            <a
              className={communityPage || profileUsername ? "active" : ""}
              href="#/community"
            >
              {t("社区", "Community")}
            </a>
            <button onClick={() => setGuide(true)}>
              {t("使用指南", "Guide")}
            </button>
          </nav>
          <div className="nav-actions">
            <span
              className={`connection ${online ? "connected" : ""}`}
              title={t(
                "网页直接提交代码并显示测试结果",
                "Run code and view test results in this workspace",
              )}
            >
              <i />
              {online
                ? t("判题服务已连接", "Judge connected")
                : t("判题服务未连接", "Judge offline")}
            </span>
            <button
              className="language-toggle"
              aria-label={t("切换到英文", "Switch to Chinese")}
              onClick={() => setLocale(locale === "zh" ? "en" : "zh")}
            >
              <Globe2 size={15} />
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
                <span className="theme-label">{t("浅色", "Light")}</span>
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
                <span className="theme-label">{t("深色", "Dark")}</span>
              </button>
            </div>
            <AccountNav returnTo={id} />
          </div>
        </div>
      </header>
      {handoff.pending && handoffProblem && user && <DraftHandoffNotice key={`${practice.owner}:${handoff.pending.id}`} pending={handoff.pending} problem={handoffProblem} current={handoffSaved} username={user.username} open={problem?.id === handoff.pending.problem} ready={progressReady} conflict={!!practice.conflicts[handoff.pending.problem]} onImport={importHeldDraft} onDiscard={() => handoff.discard(handoff.pending!.id)} />}
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
        selectedPaper ? (
          <PaperDetail
            paper={selectedPaper}
            problems={problems}
            saved={saved}
            stats={problemStats}
            onGuide={() => setGuide(true)}
          />
        ) : (
          <PaperLibrary
            key={collectionIdFromRoute(id) || "all-papers"}
            collectionId={collectionIdFromRoute(id)}
            progressReady={practice.ready && !practice.loading}
            categories={categories}
            papers={papers}
            problems={problems}
            saved={saved}
            onGuide={() => setGuide(true)}
          />
        )
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
                      status(problem, current) !== "done" ? (
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
                <a className="module-discussion-tab" href={topicHref(problem.id)}>
                  <MessageCircle size={14} />{t("讨论", "Discuss")}
                </a>
              </div>
              <div className="description-scroll" key={problem.id}>
                {tab === "tests" ? (
                  <>
                  {syntaxFailure && <BrowserSyntaxNotice failure={syntaxFailure} source={current.code ?? problem.starter} />}
                  {(!syntaxFailure || report) && <ReportView report={report} problem={problem} sourceState={reportSource}>
                    {hosted && report?.runtime === 'browser-numpy-wasm' && <BrowserResultActions problem={problem} report={report} sourceState={reportSource} signedIn={!!user} online={online} ready={progressReady} conflict={!!practice.conflicts[problem.id]} syncError={!!practice.error} busy={running} error={handoff.error} onSignIn={signInWithDraft} onVerify={() => void run('server',report)} onRefresh={() => void refreshHealth()} onLab={() => openLab('run')} />}
                    {!running && reportSource === "current" && progressReady && !(hosted && report?.runtime === "browser-numpy-wasm") && isPassed(problem, report) && (
                      <PracticeNextStep next={nextProblem} samePaper={!!nextProblem && nextProblem.paper_id === problem.paper_id} onNotes={() => setTab("notes")} onLab={() => openLab("run")} />
                    )}
                  </ReportView>}
                  </>
                ) : tab === "lab" ? (
                  <Visuals key={`${problem.id}:${labView}`} problem={problem} report={report} sourceState={reportSource} initialView={labView} />
                ) : tab === "notes" ? (
                  status(problem, current) !== "done" ? (
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
                        {status(problem, current) === "done" && (
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
                <span className="editor-language">Python 3 · PyTorch</span>
              </div>
              {rightTab === "lab" ? (
                <div className="editor-lab">
                  <Visuals key={problem.id} problem={problem} report={report} sourceState={reportSource} initialView="concept" />
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
                      <button
                        className="icon-button"
                        title={t(
                          "恢复初始代码（先下载备份）",
                          "Reset code (download backup first)",
                        )}
                        disabled={!practice.ready}
                        aria-label={t(
                          "恢复初始代码并备份",
                          "Reset starter with backup",
                        )}
                        onClick={() => {
                          const hasDraft =
                            !!current.code && current.code !== problem.starter;
                          if (hasDraft)
                            download(
                              `problem-${problem.number}-backup.py`,
                              current.code!,
                            );
                          update(problem.id, { code: problem.starter });
                          setToast(
                            hasDraft
                              ? t(
                                  "已恢复初始代码，原代码已下载备份。",
                                  "Starter restored. Your previous code was downloaded as a backup.",
                                )
                              : t("已恢复初始代码。", "Starter restored."),
                          );
                        }}
                      >
                        <RotateCcw size={14} />
                      </button>
                    </div>
                  </div>
                  <div className="code-editor">
                    <PythonEditor
                      key={`${practice.owner}:${problem.id}`}
                      readOnly={!practice.ready}
                      value={current.code ?? problem.starter}
                      theme={theme === "dark" ? "dark" : "light"}
                      onChange={(value) => update(problem.id, { code: value })}
                      font={font}
                      label={`${problem.title} ${t("Python 代码编辑器", "Python code editor")}`}
                    />
                  </div>
                  <div className="editor-status">
                    <span>
                      <Check size={12} />
                      {practice.loading
                        ? t("正在加载工作区…", "Loading workspace…")
                        : hosted
                          ? !user
                            ? t(
                                "草稿仅保存在此浏览器，登录后使用独立工作区",
                                "Browser-only draft · sign in for your account workspace",
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
                            ? t("草稿已自动保存", "Draft saved automatically")
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
                    <label>{t("判题位置", "Run tests in")}{" "}
                      <select aria-label={t("判题位置", "Run tests in")} value={judgeMode} disabled={running} onChange={(event) => setJudgeMode(event.target.value as JudgeMode)}>
                        <option value="auto">{t("自动（优先浏览器）", "Auto (browser first)")}</option>
                        <option value="browser">{t("仅浏览器", "Browser only")}</option>
                        <option value="server">{t("服务端", "Server")}</option>
                      </select>
                    </label>
                    <small>{judgeMode === 'server' ? t('将提交当前代码，由服务端 CPU 执行完整测试。', 'Submit the current source for the complete suite on the server CPU.') : t("原型：本机 CPU / WASM 运行完整测试，无需 GPU。", "Prototype: the complete suite runs on your CPU / WASM; no GPU needed.")}</small>
                    {judgeMode !== 'server' && browserBudget(problem).total > 120_000 && <small>{t(`完整实验包含模型训练与错误实现校准，最多运行 ${browserBudget(problem).total / 60_000} 分钟；可随时停止后切换服务端。`, `The full experiment includes model training and mutant calibration, capped at ${browserBudget(problem).total / 60_000} minutes. You can stop and switch to Server.`)}</small>}
                    {hosted && <small>{t("浏览器结果仅供本机反馈；服务端确认后计入账号进度与统计。", "Browser results are local feedback; server verification records account progress and statistics.")}</small>}
                    {browser.reasons[problem.id] && <p role="status">{browserReason(browser.reasons[problem.id], locale === "en")}。{judgeMode === "auto" ? t("自动模式回退到服务端，请查看下方状态。", "Auto mode falls back to the server; see the status below.") : t("可切换至服务端运行。", "Select Server to run this suite.")}</p>}
                    {browser.calibrations[problem.id] && report?.runtime === "browser-numpy-wasm" && <small>{t("兼容性校验", "Compatibility calibration")}: {browser.calibrations[problem.id].reference_passed} {t("项参考测试通过", "reference tests passed")} · {Object.keys(browser.calibrations[problem.id].mutants_caught).length} {t("个错误实现被抓住", "mutants caught")}</small>}
                    {browser.diagnostics[problem.id] && <details><summary>{t("运行时详情", "Runtime details")}</summary><code>{browser.diagnostics[problem.id]}</code></details>}
                    {report?.runtime === "browser-numpy-wasm" && <p role="status">{t("本次结果来自浏览器 · NumPy / WASM", "Results from your browser · NumPy / WASM")}</p>}
                  </div>}
                  {!problem.browser_judge && <div className="server-judge-panel"><strong>{t('服务端 CPU 判题', 'Server CPU judging')}</strong><small>{t('在网页完成实现后提交，服务端执行完整测试并返回结果。', 'Implement on the web, then submit for the complete suite and feedback from the server.')}</small>{hosted && !user && <small>{t('登录后可提交；现在可以先阅读题面、编辑和保存本机草稿。', 'Sign in to submit. You can read, edit and save a browser draft first.')}</small>}</div>}
                  {judge.active && <div className="practice-notice judge-queue-notice" role="status">
                    <span><b>{queuedHere ? runStatus : t("另一道题正在排队或判题", "Another module has a queued or running job")}</b><br />
                      {t("可以继续编辑；本次判题使用提交时的代码。刷新后可继续查看任务。", "You can keep editing. This run uses the submitted code; refresh to reconnect.")}</span>
                    {!queuedHere && <a href={`#/${judge.pending?.problem}`}>{t("查看任务", "View job")}</a>}
                    {judge.job?.status === "queued" && <button onClick={() => void judge.cancel()}>{t("取消排队", "Leave queue")}</button>}
                  </div>}
                  {judge.error && <div className="practice-notice" role="alert">{jobErrorText(judge.error, locale)}</div>}
                  {!judge.active && judge.job?.status === "cancelled" && <div className="practice-notice" role="status">{t("已取消排队，本次提交未执行。", "Removed from the queue. This submission was not executed.")}</div>}
                  {(practice.error ||
                    practice.conflicts[problem.id] ||
                    (!health && healthError) ||
                    (hosted && !browserAvailable && (!user || !online))) && (
                    <div className="practice-notice" role="status">
                      {practice.conflicts[problem.id] ? (
                        <>
                          <span>
                            {t(
                              "其他设备更新了这道题的草稿。选择保留哪个版本；当前代码尚未覆盖服务器。",
                              "Another device updated this draft. Choose which version to keep; your current code has not overwritten the server.",
                            )}
                          </span>
                          <button
                            onClick={() =>
                              practice.resolveConflict(problem.id, false)
                            }
                          >
                            {t("使用服务器版本", "Use server draft")}
                          </button>
                          <button
                            onClick={() =>
                              practice.resolveConflict(problem.id, true)
                            }
                          >
                            {t("保留当前版本", "Keep this draft")}
                          </button>
                        </>
                      ) : practice.error ? (
                        <>
                          <span>{practice.error}</span>
                          <button onClick={practice.retry}>
                            {t("重试同步", "Retry sync")}
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
                                  "在线内测仅向受邀且已验证机构邮箱的用户开放判题。",
                                  "Online judging is available to invited members with a verified institutional email.",
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
                    <span>
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
                          {t("可以运行", "Ready")}
                        </>
                      ) : (
                        <>
                          <i className="yellow-dot" />
                          {t("判题服务暂时不可用", "Judge unavailable")}
                        </>
                      )}
                    </span>
                    <button
                      className="button run-button"
                      disabled={
                        running ||
                        !practice.ready ||
                        (!browserAvailable && hosted &&
                          (!user ||
                            !online ||
                            !!practice.conflicts[problem.id]))
                      }
                      onClick={() => void run()}
                    >
                      <Play size={15} fill="currentColor" />
                      {running
                        ? judge.job?.status === "queued" ? t("排队中", "Queued") : t("运行中", "Running")
                        : t("运行全部测试", "Run all tests")}
                    </button>
                  </div>
                </>
              )}
            </section>
          </ResizableSplit>
        </>
      )}
      {guide && (
        <div className="modal-backdrop" onClick={() => setGuide(false)}>
          <section
            role="dialog"
            aria-modal="true"
            aria-labelledby="guide-title"
            className="guide-modal"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              className="icon-button modal-close"
              aria-label={t("关闭指南", "Close guide")}
              onClick={() => setGuide(false)}
            >
              <X size={19} />
            </button>
            <div className="eyebrow">
              <BookOpen size={16} />
              PAPERLEET
            </div>
            <h2 id="guide-title">
              {t("实现论文里的核心模块", "Implement a paper’s core modules")}
            </h2>
            <p>
              {t(
                "阅读论文、编写代码、运行测试都在这个工作区完成。",
                "Read the paper, write code, and run tests in this workspace.",
              )}
            </p>
            <ol>
              <li>
                <b>{t("选择论文与模块", "Choose a paper and module")}</b>
                <small>
                  {t(
                    "先了解论文的核心思想和练习范围，再选择模块，阅读接口与约定。",
                    "Read the paper’s core idea and selected scope, then choose a module and review its contract.",
                  )}
                </small>
              </li>
              <li>
                <b>{t("在编辑器中实现", "Implement in the editor")}</b>
                <small>
                  {t(
                    "草稿自动保存；实验台可以帮助你观察公式的行为。",
                    "Drafts are saved automatically. Use the lab to explore how the formulas behave.",
                  )}
                </small>
              </li>
              <li>
                <b>{t("运行全部测试", "Run all tests")}</b>
                <small>
                  {t(
                    "检查失败原因并修改代码。全部通过后，解锁这道题的坑点。",
                    "Read failure details and revise your code. Passing all tests unlocks the problem's notes.",
                  )}
                </small>
              </li>
              <li>
                <b>{t("提交新论文", "Contribute a paper")}</b>
                <small>
                  {t(
                    "提交论文和官方代码来源，生成候选实现与测试；通过验证和人工评审后才会加入题库。",
                    "Submit a paper and official code source. Generated code and tests enter the catalog only after validation and human review.",
                  )}
                </small>
              </li>
            </ol>
            <button className="button" onClick={() => setGuide(false)}>
              {t("开始练习", "Start practicing")} <ArrowRight size={15} />
            </button>
          </section>
        </div>
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
