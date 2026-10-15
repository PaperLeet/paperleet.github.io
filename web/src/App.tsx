import {loadCatalogIndex} from './catalogIndex';
import { selectLibraryPractice } from './libraryPractice';
import { useEffect, useRef, useState } from "react";
import { navigateTabs } from "./tabNavigation";
import { Highlight } from "./SyntaxHighlight";
import { PreparationGuide, InterviewPracticeNextStep, Interviews, ProblemCompanyTags, Community, UserProfile, Account, SolutionHistory, SubmissionResultPage, Submissions, Visuals, WorkspaceTestResults, Prose, PythonEditor, WorkspaceBackup, QueueAdmissionNotice, ModuleNavigator, ResetCodeButton, BrowserDraftBackups, SettingsPage, Achievements, AchievementObserver, BrowserResultActions, JudgeRuntimeDetails, WorkspaceLayoutMenu } from "./deferredPages";
import {
  ArrowDownToLine,
  ArrowLeft,
  ArrowRight,
  AlertCircle,
  BookOpen,
  Check,
  CheckCircle2,
  ChevronDown,
  Maximize2,
  Minimize2,
  Code2,
  Copy,
  ExternalLink,
  FileCode2,
  FlaskConical,
  LockKeyhole,
  Loader2,
  MessageCircle,
  Play,
  Terminal,
  X,
  XCircle,
} from "lucide-react";
import { isPassed, status } from "./types";
import type { Category, Paper, Problem, Report } from "./types";
import { PaperDetail, PaperLibrary, paperModules } from "./PaperViews";
import { collectionIdFromRoute } from "./paperCollections";
import { decodeProblemHash } from "./reports";
import { LocaleProvider, useI18n } from "./i18n";
import { AccountNav, AccountProvider, useAccount } from "./social";
import { usePracticeWorkspace } from "./PracticeWorkspace";
import { BrowserRunProgress } from "./BrowserRunProgress";
import Brand from "./Brand";
import RepositoryLink from "./RepositoryLink";
import { OnlinePresence } from "./deferredPages";
import CommunitySubmissions from "./CommunitySubmissions";
import { BilingualText } from "./BilingualText";
import { getProblemStats, ProblemStats, useProblemStats } from "./ProblemStats";
import { ResizableSplit } from "./deferredPages";
import { recommendPractice, nextPractice } from "./practice-recommendation";
import { useQueuedJudge } from "./useQueuedJudge";
import { jobErrorText } from "./judgeJobs";
import { reloadApp } from './appRecovery';
import { useBrowserJudge } from "./useBrowserJudge";
import { browserReason } from "./browserMessages";
import { browserBudget } from './browserBudgets';
import { topicHref, topicIdFromRoute, topicFromProblem } from "./communityTopics";
import { accountReturnHref, isAccountRoute, signInHref } from "./accountNavigation";
import { publicSite, hostedAppHref, hostedAppAvailable } from "./deployment";
import { type ExperimentView } from "./experimentResults";
import { useReportSource } from "./ReportSourceNotice";
import { useWorkspaceShortcuts } from "./useWorkspaceShortcuts";
import { matchesBrowserSubmission } from './browserVerification';
import { useBrowserLearning } from './useBrowserLearning';
import { browserLearningView } from './browserLearning';
import { submissionDestination } from './submissionNavigation';
import { BrowserLearningNotice } from './BrowserLearningNotice';
import { type EditorFocusRequest, type SourceLocation } from './sourceLocation';
import { libraryHref, libraryReturnHref, readLibraryRoute, routePath, withLibraryContext } from './libraryNavigation';
import "./community.css";

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
  const { locale, t } = useI18n();
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
  const [resultNotice, setResultNotice] = useState<{owner:string; href:string}>();
  const [mobilePane, setMobilePane] = useState<"read" | "code">("read");
  const [focusMode, setFocusMode] = useState(false);
  const [layoutRequest, setLayoutRequest] = useState<{horizontal:{ratio:number};vertical?:{ratio:number}}>();
  const [resultsCollapsed, setResultsCollapsed] = useState(true);
  const [resultsRequested,setResultsRequested]=useState(false);
  function selectReadingTab(nextTab: string) {
    setTab(nextTab); setMobilePane("read"); setFocusMode(false);
    requestAnimationFrame(() => {
      document.querySelector(".description-scroll")?.scrollTo(0, 0);
      if (window.matchMedia("(max-width: 850px)").matches && document.querySelector(".mobile-workspace-switch")) window.scrollTo(0, 0);
    });
  }
  function showTests() {
    setResultsRequested(true);setResultsCollapsed(false);setRightTab('code');setMobilePane('code');
    requestAnimationFrame(()=>{
      if(!problem||![problem.id,String(problem.number)].includes(routePath(getId())))return;
      document.getElementById('problem-results-toggle')?.focus({preventScroll:true});
    });
  }
  function openEditor() {
    setRightTab("code"); setMobilePane("code");
    requestAnimationFrame(() => {
      document.querySelector<HTMLElement>('#problem-editor-panel [contenteditable="true"], #problem-editor-panel textarea')?.focus({ preventScroll: true });
      if (window.matchMedia("(max-width: 850px)").matches) document.querySelector(".mobile-workspace-switch")?.scrollIntoView({ block: "start" });
    });
  }
  const [details, setDetails] = useState<Record<string, Problem>>({});
  const [detailError, setDetailError] = useState("");
  const [detailRetry, setDetailRetry] = useState(0);
  const [labView, setLabView] = useState<ExperimentView>("auto");
  const [backupOpen, setBackupOpen] = useState(false);
  useEffect(() => { setBackupOpen(false); }, [practice.owner]);
  useEffect(() => { if (!user && rightTab === "history") setRightTab("code"); }, [user, rightTab]);
  const [submissionRevision, setSubmissionRevision] = useState(0);
  function openLab(view: ExperimentView = "auto") {
    setLabView(view);
    selectReadingTab("lab");
  }
  const path = routePath(id), libraryReturn = libraryReturnHref(id), libraryView = readLibraryRoute(id);
  const { stats: problemStats, totalSubmissions, refresh: refreshProblemStats, loading: statsLoading } = useProblemStats(path);
  const sourceProblem = problems.find(
    (p) => p.id === path || String(p.number) === path,
  );
  const localize = (p: Problem): Problem =>
    locale === "en" && p.translations?.en ? { ...p, ...p.translations.en } : p;
  const loadedProblem = sourceProblem?.detail_url ? details[sourceProblem.detail_url] : sourceProblem;
  const problem = loadedProblem ? localize(loadedProblem) : undefined;
  const submissionPage = path === "submit";
  const submissionResultId = /^submission\/([a-f0-9]{32})$/.exec(path)?.[1];
  const communityPath = id.split('?')[0];
  const communityPage = communityPath === "community" || communityPath.startsWith("community/");
  const communityTopicId = topicIdFromRoute(communityPath);
  const communityProblem = communityTopicId ? problems.find((p) => p.id === communityTopicId) : undefined;
  const accountPage = isAccountRoute(id);
  const interviewPage = path === "interviews";
  const settingsPage = path === "settings";
  const achievementsPage = path === "achievements";
  const settingsReturn = useRef("#");
  useEffect(() => {
    if (!settingsPage && !achievementsPage) settingsReturn.current = location.hash || "#";
  }, [id, settingsPage, achievementsPage]);
  const profileUsername = id.startsWith("u/") ? id.slice(2) : "";
  const current = problem ? saved[problem.id] || {} : {};
  const selectedPaper = papers.find((paper) =>
    sourceProblem ? paper.id === sourceProblem.paper_id : path === `papers/${paper.id}`,
  );
  const siblingModules = selectedPaper
    ? paperModules(selectedPaper, problems)
    : sourceProblem
      ? [sourceProblem]
      : [];
  const learning = useBrowserLearning(hosted ? practice.owner : 'unresolved');
  const learningView = browserLearningView(practice.owner,problems,saved,learning.records);
  const learningSaved = learningView.saved;
  const notesUnlocked = !!problem && status(problem,learningSaved[problem.id]) === 'done';
  const learningNotice = hosted && <BrowserLearningNotice count={learningView.localPasses} failed={learning.failed} saving={learning.saving} onRetry={learning.retry} />;
  const progressReady = practice.ready && !practice.loading && learning.ready;
  const practiceReturn = libraryView ? libraryHref(libraryView.collectionId, libraryView.filters) : libraryReturn;
  const practiceScope = selectLibraryPractice(papers, problems, learningSaved, libraryView || (libraryReturn ? readLibraryRoute(libraryReturn) : undefined));
  const scopedSiblings = siblingModules.filter(module => practiceScope.problems.some(candidate => candidate.id === module.id));
  const idlePracticeLabel = !practiceScope.problems.some(module => !module.module.optional) ? t("当前范围暂无核心模块", "No core modules in this view") : practiceReturn && practiceReturn !== "#" ? t("本范围核心模块已完成", "Core modules complete in this view") : t("核心模块已完成", "Core modules complete");
  const resumeProblem = progressReady ? (selectedPaper ? recommendPractice(scopedSiblings, learningSaved) : undefined) || recommendPractice(practiceScope.problems, learningSaved) : undefined;
  const nextProblem = problem && progressReady
    ? nextPractice(problem, selectedPaper?.module_ids || [], practiceScope.problems, learningSaved)
    : undefined;
  const browser = useBrowserJudge(practice.owner);
  const queueReady = !publicSite && practice.ready && !practice.loading && problems.length > 0 && (!hosted || !!user);
  const judge = useQueuedJudge(practice.owner, queueReady, (report, runOwner) => {
    const entry = problems.find((p) => p.id === report.problem);
    if (!entry || !practice.owns(runOwner)) return;
    practice.applyReport(entry, report, runOwner);
    browser.clear(entry.id);
    learning.dismiss(entry,runOwner);
    const destination = submissionDestination(report, runOwner, entry, getId());
    const resultHref = destination?.href;
    if (resultHref && destination.redirect) {
      setResultNotice(undefined);
      setToast('');
      location.hash = resultHref;
    } else if (resultHref) {
      setResultNotice({owner:runOwner, href:resultHref});
    } else {
      void refreshProblemStats();
      setSubmissionRevision(value => value + 1);
      setToast(runOwner.startsWith("user:") ? t("提交完成，解答已保存到账号", "Submission complete. Solution saved to your account.") : t("测试已完成", "Run complete"));
    }
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
    if (!problem || !practice.ready || !practice.owns(practice.owner) || ![problem.id,String(problem.number)].includes(routePath(getId()))) return;
    setResultsCollapsed(true);openEditor();
    setEditorFocus({ owner: practice.owner, problem: problem.id, source: currentSource, location });
  }
  function editorFocused(request: EditorFocusRequest) {
    setEditorFocus(previous => previous === request ? undefined : previous);
  }
  const browserAvailable = !!problem?.browser_judge;
  const browserHere = browser.problem === problem?.id;
  const running = preparingRun || judge.active || browser.active;
  const canRun = !!problem && !running && practice.ready && !(!browserAvailable && hosted && (!user || !online || !!practice.conflicts[problem.id]));
  const workspaceAttention = !!(judge.error || practice.error || (!health && healthError) || (problem && (practice.conflicts[problem.id] || browser.reasons[problem.id])));
  const resultsVisible = !!(resultsRequested || report || syntaxFailure || running || workspaceAttention || (!health && healthError) || (hosted && !browserAvailable && (!user || !online)) || judge.job?.status === "cancelled");
  useEffect(() => { setResultsCollapsed(true);setResultsRequested(false); }, [problem?.id, practice.owner]);
  useEffect(() => { if (workspaceAttention || running) setResultsCollapsed(false); }, [workspaceAttention, running, problem?.id, practice.owner]);
  useWorkspaceShortcuts({ enabled: !!problem && !backupOpen, canRun, focused: focusMode, onRun: () => void run(), onExitFocus: () => setFocusMode(false) });
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
  const queuedHere = judge.pending?.problem === problem?.id;
  const runStatus = judge.waiting ? judge.waiting.attempting ? t("正在确认能否入队…", "Confirming admission…") : t("正在等待判题名额…", "Waiting for a judge slot…")
    : judge.reconnecting ? t("连接中断，正在恢复任务状态…", "Reconnecting to your existing job…")
    : judge.job?.status === "queued" ? t(`正在排队，前方 ${judge.job.ahead} 个任务`, `Queued · ${judge.job.ahead} jobs ahead`)
    : judge.job?.status === "running" ? t("正在执行测试…", "Running tests…")
    : t("正在提交判题任务…", "Submitting your job…");
  function loadCatalog() {
    return loadCatalogIndex(import.meta.env.BASE_URL)
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
      .catch(() => setLoadError(t('暂时无法加载题库，请稍后刷新。','The problem catalog is unavailable. Please refresh shortly.')));
  }
  useEffect(() => {
    void loadCatalog();
    const h = () => {
      setId(getId());
      setTab("description");
      setRightTab("code");
      setMobilePane("read");
      setFocusMode(false);
      setLayoutRequest(undefined);
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
    void import('./catalogData').then(({loadProblemDetail})=>loadProblemDetail(sourceProblem,import.meta.env.BASE_URL,fetch,controller.signal))
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
    setResultNotice(undefined);
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
  useEffect(() => {
    const sync = (event: StorageEvent) => {
      if (event.key === 'paperleet.theme' || event.key === null) setTheme(initialTheme());
      if (event.key === 'paperleet.editor.fontSize' || event.key === null) setFont(initialEditorFontSize());
    };
    window.addEventListener('storage', sync);
    return () => window.removeEventListener('storage', sync);
  }, []);
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
  async function run(requestedMode: "browser" | "server" = "browser", verification?: Report) {
    if (!problem || running || runToken.current || !practice.ready) return;
    const runOwner = practice.owner;
    const token = {};
    runToken.current = token;
    const ownsRun = () => practice.owns(runOwner) && runToken.current === token;
    const submittedCode = current.code ?? problem.starter;
    setResultsCollapsed(false);
    setRightTab("code");
    setMobilePane("code");
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
          setToast(t("浏览器测试已完成", "Browser run complete"));
          return;
        }
        if (outcome.type === 'syntax_error') {
          setToast(t('请先修正 Python 语法错误。','Fix the Python syntax error before running tests.'));
          return;
        }
        return;
      }
      if (!ownsRun()) return;
      if (publicSite) {
        if (!hostedAppAvailable) {
          setToast(t("当前版本暂未开放服务端提交。", "Server submissions are not available yet."));
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
    ["notes", t("坑点", "Notes")],
    ["solution", t("参考解", "Solution")],
  ];
  const webReadme = problem?.readme.replace(
    /##\s*6[.、．]?\s*(?:怎么运行|编写与运行|在网页中运行|Write and run|Run your implementation|Running)[^\n]*\n[\s\S]*?(?=\n##\s*7|$)/,
    `## 6. ${t("编写与运行", "Write and run")}\n\n${problem?.browser_judge ? t("在右侧完成实现，点击「运行测试」在浏览器执行，通过后解锁坑点。登录后 Submit 可保存解答和结果到「我的提交」。", "Write code on the right, then Run tests in your browser to unlock Notes. Sign in and Submit to save your solution and server result in My submissions.") : t("在右侧完成实现，登录后 Submit。服务端会判题并保存解答和结果，可在「我的提交」中查看。", "Write code on the right, then sign in and Submit. The server judges your solution and saves it in My submissions.")}\n\n${t("游客草稿自动保存在当前浏览器，刷新后恢复，不会自动并入账号。", "Guest drafts save in this browser and survive reloads, separately from your account.")}\n`,
  );
  const workspaceSaveState = practice.error || (problem && practice.conflicts[problem.id]) ? "warning"
    : practice.loading || !practice.ready ? "loading"
    : practice.saving || practice.pending ? "saving" : "saved";
  return (
    <div className={`app ${problem ? "workspace-view" : ""}`} data-mobile-pane={mobilePane} data-focus-mode={focusMode || undefined}>
      <header className="topbar">
        <div className="nav-inner">
          <Brand />
          <nav aria-label={t("主导航", "Main navigation")}>
            <a
              className={
                !problem &&
                !submissionPage &&
                !communityPage &&
                !interviewPage &&
                !accountPage &&
                !settingsPage &&
                !achievementsPage &&
                !profileUsername
                  ? "active"
                  : ""
              }
              href={libraryReturn?.startsWith('#/interviews') ? '#' : libraryReturn || '#'}
              aria-current={!problem && !submissionPage && !communityPage && !interviewPage && !accountPage && !settingsPage && !achievementsPage && !profileUsername ? 'page' : undefined}
            >
              <BilingualText zh="论文题库" en="Papers" />
            </a>
            <button
              className={`continue-link ${problem ? "active" : ""}`}
              aria-current={problem ? 'page' : undefined}
              disabled={!problem && !resumeProblem}
              onClick={() => {
                if (!problem && resumeProblem) location.hash = withLibraryContext(resumeProblem.number, practiceReturn).slice(1);
              }}
            >
              {problem ? <BilingualText zh="工作区" en="Workspace" /> : progressReady && problems.length > 0 && !resumeProblem ? <BilingualText zh="核心模块已完成" en="Core modules complete" /> : resumeProblem && status(resumeProblem, learningSaved[resumeProblem.id]) === "started" ? <BilingualText zh="继续练习" en="Continue" /> : <BilingualText zh="开始练习" en="Start practice" />}
            </button>
            {(!publicSite || hostedAppAvailable) && <a className={submissionPage ? "active" : ""} aria-current={submissionPage ? 'page' : undefined} href={hostedAppHref("#/submit")}>
              <BilingualText zh="提交论文" en="Submit paper" />
            </a>}
            {hostedAppAvailable && <a
              className={communityPage || profileUsername ? "active" : ""}
              aria-current={communityPage ? 'page' : undefined}
              href={hostedAppHref("#/community")}
            >
              <BilingualText zh="社区" en="Community" />
            </a>}
            <a className={interviewPage ? "active" : ""} aria-current={interviewPage ? "page" : undefined} href="#/interviews"><BilingualText zh="面经" en="Interviews" /></a>
          </nav>
          <div className="nav-actions">
            <a className="privacy-link" href={`${import.meta.env.BASE_URL}privacy.html`}>{t("隐私", "Privacy")}</a>
            <RepositoryLink />
            <OnlinePresence />
            <CommunitySubmissions count={totalSubmissions} />
            <AccountNav returnTo={id} theme={theme} onTheme={selectTheme} onBackup={() => setBackupOpen(true)} backupReady={practice.owner !== 'unresolved' && problems.length > 0} />
          </div>
        </div>
      </header>
      {backupOpen && <WorkspaceBackup key={practice.owner} owner={practice.owner} problems={problems} saved={saved} backups={practice.backups} conflicts={practice.conflicts}
        recovery={{blocked:!!editorRecoveryReason,reason:editorRecoveryReason,reload:() => {
          if (!editorRecoveryReason && practice.owns(practice.owner)) location.reload();
        }}}
        ready={practice.ready && !practice.loading} blocked={running ? t('请先停止测试或等待运行完成。', 'Stop the tests or wait for the run to finish.')
          : practice.error ? t('请先重试工作区保存；仍可下载备份。', 'Retry workspace saving first. Backup downloads remain available.')
          : Object.keys(practice.conflicts).length ? t('请先解决工作区草稿冲突。', 'Resolve workspace draft conflicts first.')
          : practice.pending || practice.saving ? t('请等待草稿保存完成。', 'Wait for drafts to finish saving.') : ''}
        onClose={() => setBackupOpen(false)} onDownload={download}
        onReplace={(entry, source, expected) => {
          if (running || !practice.owns(expected.owner) || !practice.ready || practice.pending || practice.saving || practice.error || Object.keys(practice.conflicts).length) return false;
          const replaced = practice.replaceCode(entry,source,expected);
          if (replaced) setToast(t('代码已载入，请留意保存状态并重新运行测试。', 'Code loaded. Check the save status and run tests again.'));
          return replaced;
        }} />}
      {submissionResultId ? (
        <SubmissionResultPage runId={submissionResultId} problems={problems}
          nextProblemFor={entry => nextPractice(entry, papers.find(paper => paper.id === entry.paper_id)?.module_ids || [], problems, learningSaved)} />
      ) : achievementsPage ? (
        <Achievements problems={problems} papers={papers} saved={learningSaved} ready={progressReady && problems.length > 0} returnTo={settingsReturn.current} onRetry={() => { void loadCatalog(); practice.retry(); learning.retry(); void refreshHealth(); }} error={loadError || practice.error || (!health && healthError ? t('暂时无法读取工作区，请重试连接。','The workspace is unavailable. Try connecting again.') : undefined)} workspaceName={user ? t(`${user.display_name} 的学习工作区`,`${user.display_name}’s learning workspace`) : t('当前浏览器工作区','Current browser workspace')} />
      ) : settingsPage ? (
        <SettingsPage theme={theme} onTheme={selectTheme} returnTo={settingsReturn.current}
          onBackup={() => setBackupOpen(true)}
          backupReady={practice.owner !== 'unresolved' && problems.length > 0}
          accountHref={user ? hostedAppHref('#/account') : undefined}
          workspaceName={user ? t(`${user.display_name} 的账号工作区`, `${user.display_name}’s account workspace`) : t('当前浏览器工作区', 'Current browser workspace')} />
      ) : interviewPage ? (
        <Interviews route={id} problems={problems} catalogError={loadError} onRetryCatalog={()=>void loadCatalog()} saved={learningSaved} progressReady={progressReady} progressError={practice.error || (learning.failed ? "unavailable" : undefined)} onRetryProgress={()=>{practice.retry();learning.retry();}} />
      ) : communityPage ? (
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
      ) : !sourceProblem && !selectedPaper && !libraryView ? (
        <main className="route-not-found"><BookOpen size={32} /><h1>{t("找不到这篇论文或模块", "Paper or module not found")}</h1><p>{t("链接可能已失效，或内容已经调整。可以返回题库搜索论文和模块。", "This link may be outdated or the content may have changed. Search the library to find a paper or module.")}</p><code>{path.slice(0,120)}</code><a className="button" href={libraryReturn || "#"}><ArrowLeft size={15} />{t("返回题库", "Back to papers")}</a></main>
      ) : sourceProblem && !problem ? (
        <main className={detailError === sourceProblem.detail_url ? "load-error" : "loading"}>
          {detailError === sourceProblem.detail_url ? <>
            <h2>{t("题目暂时无法加载", "Module unavailable")}</h2>
            <p>{t("你的草稿仍然保留，请重试加载题目。", "Your draft is preserved. Try loading this module again.")}</p>
            <button className="button" onClick={() => setDetailRetry((value) => value + 1)}>{t("重试加载", "Retry loading")}</button>
            <button className="text-button" disabled={!!editorRecoveryReason} onClick={()=>{if(!editorRecoveryReason&&practice.owns(practice.owner))reloadApp();}}>{t('刷新并恢复草稿','Reload and restore draft')}</button>
            {editorRecoveryReason&&<small>{editorRecoveryReason}</small>}
            <a className="text-button" href={libraryReturn || '#'}>{t("返回题库", "Back to papers")}</a>
          </> : <><span className="spinner" />{t("正在加载这道题…", "Loading this module…")}</>}
        </main>
      ) : !problem ? (
        <>{learningNotice}{selectedPaper ? (
          <PaperDetail
            paper={selectedPaper}
            problems={problems}
            saved={learningSaved}
            stats={problemStats}
            returnTo={libraryReturn}
            progressReady={progressReady}
          />
        ) : (
          <PaperLibrary
            key={collectionIdFromRoute(id) || "all-papers"}
            collectionId={collectionIdFromRoute(id)}
            route={id}
            onFilters={href => {
              if (!readLibraryRoute(href)) return;
              try { history.replaceState(history.state,'',href); } catch { /* Filtering and explicit return links remain usable if history writes are blocked. */ }
              setId(decodeProblemHash(href));
            }}
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
              href={libraryReturn?.startsWith('#/interviews') ? libraryReturn : selectedPaper ? withLibraryContext(`papers/${selectedPaper.id}`,libraryReturn) : libraryReturn || '#'}
            >
              <ArrowLeft size={15} />
              {libraryReturn?.startsWith('#/interviews') ? t('面经','Interviews') : selectedPaper
                ? `${selectedPaper.number}. ${selectedPaper.short_title}`
                : t("论文题库", "Papers")}
            </a>
            <span className="toolbar-divider" />
            <b title={locale === 'en' ? problem.translations?.en?.title || problem.title : problem.title}>
              {problem.number}. {locale === 'en' ? problem.translations?.en?.title || problem.title : problem.title}
            </b>
            <div className="workspace-run-controls">
              {(running || !practice.ready || practice.loading || (!online && !browserAvailable)) && <span className="workspace-run-state" role="status">
                {running ? <><span className="spinner" />{judge.active ? runStatus : browser.active ? t("运行中…", "Running…") : t("正在准备…", "Preparing…")}</> : !practice.ready || practice.loading ? t("正在准备工作区…", "Preparing workspace…") : t("判题服务未连接", "Judge offline")}
              </span>}
              {browser.active && <button className="workspace-stop-button" type="button" onClick={browser.stop}><X size={14}/>{t("停止","Stop")}</button>}
              <div className="editor-run-actions">
                {(browserAvailable || (!hosted && !publicSite)) && <button
                  className="button subtle run-button" aria-keyshortcuts="Meta+Enter Control+Enter"
                  disabled={running || !practice.ready}
                  onClick={() => void run(browserAvailable ? 'browser' : 'server')}
                ><Play size={15} fill="currentColor" />{t("运行测试", "Run tests")}</button>}
                {hostedAppAvailable && <button
                  className="button run-button"
                  disabled={running || !practice.ready || (!!user && (!queueReady || !online || !!practice.conflicts[problem.id]))}
                  onClick={() => user ? void run('server') : signInToSubmit()}
                >{user ? t("提交", "Submit") : t("登录后提交", "Sign in to submit")}</button>}
              </div>
            </div>
            <WorkspaceLayoutMenu onLayout={(ratio, reset) => {
              setFocusMode(false);
              setLayoutRequest({horizontal:{ratio},vertical:reset?{ratio:.72}:undefined});
              if (reset) setResultsCollapsed(!running && !workspaceAttention);
            }} />
            <button className="workspace-focus-toggle" aria-pressed={focusMode} title={t("专注写代码，按 Esc 返回分栏", "Focus on code; press Esc to restore both panels")} onClick={() => { if (!focusMode) openEditor(); setFocusMode(value => !value); }}>{focusMode ? <Minimize2 size={14} /> : <Maximize2 size={14} />}{focusMode ? t("退出专注", "Exit focus") : t("专注模式", "Focus mode")}</button>
            <ModuleNavigator key={problem.id} current={problem} modules={siblingModules} allModules={problems} saved={learningSaved} ready={progressReady} returnTo={libraryReturn} />
          </div>
          <div className="mobile-workspace-switch" role="group" aria-label={t("练习视图", "Practice view")}>
            <button aria-pressed={mobilePane === "read"} aria-controls="problem-description-panel" onClick={() => selectReadingTab(tab)}><BookOpen size={16}/>{t('题面','Problem')}</button>
            <button aria-pressed={mobilePane === "code"} aria-controls="problem-editor-panel" onClick={openEditor}><Code2 size={16}/>{t('编写代码','Write code')}{(report||syntaxFailure)&&<span aria-hidden="true" className={`mobile-result-dot ${!hasCurrentSyntaxError&&reportSource==='current'&&isPassed(problem,report)?'passed':''}`}/>}</button>
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
            layout={layoutRequest?.horizontal}
            label={t("调整题面与代码区宽度", "Resize description and code panels")}
            resetLabel={t("拖动或使用方向键调整，双击或按 Enter 恢复默认", "Drag or use arrow keys to resize; double-click or press Enter to reset")}
            firstId="problem-description-panel"
            secondId="problem-editor-panel"
          >
            <section className="description-pane" id="problem-description-panel">
              <div className="pane-tabs" role="tablist" aria-label={t('题目内容','Problem content')} onKeyDown={navigateTabs}>
                {tabs.map(([key, label]) => (
                  <button
                    key={key}
                    id={`reading-tab-${key}`}
                    role="tab"
                    tabIndex={tab === key ? 0 : -1}
                    className={tab === key ? "selected" : ""}
                    aria-selected={tab === key}
                    aria-controls="problem-reading-content"
                    onClick={() => key === "lab" ? openLab() : selectReadingTab(key)}
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
                  </button>
                ))}
                {hostedAppAvailable && <a className="module-discussion-tab" href={topicHref(problem.id)}>
                  <MessageCircle size={14} />{t("讨论", "Discuss")}
                </a>}
              </div>
              <div className="description-scroll" id="problem-reading-content" role="tabpanel" aria-labelledby={`reading-tab-${tab}`} tabIndex={0} key={`${problem.id}:${tab}`}>
                {learningNotice}
                {tab === "lab" ? (
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
                        onClick={() => showTests()}
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
                    <ProblemCompanyTags problemId={problem.id} />
                      <div className="heading-tags">
                        <span className={`difficulty ${problem.difficulty}`}>
                          {({ E: t("简单", "Easy"), M: t("中等", "Medium"), H: t("困难", "Hard") } as Record<string,string>)[problem.difficulty] || problem.difficulty}
                        </span>
                        <span>PyTorch</span>
                        <span>{problem.estimated_minutes || 25} min</span>
                        {problem.parent && (
                          <a
                            href={problems.find(p => p.id === problem.parent) ? withLibraryContext(problems.find(p => p.id === problem.parent)!.number,libraryReturn) : `#/${problem.parent}`}
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
                    {tab === "description" && <PreparationGuide key={problem.id} problemId={problem.id} onNotes={()=>selectReadingTab("notes")} notesUnlocked={notesUnlocked&&progressReady} onLab={()=>openLab("concept")}/>}
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
              <div className="pane-tabs editor-tabs" role="tablist" aria-label={t('工作区视图','Workspace view')} onKeyDown={navigateTabs}>
                <button
                  id="editor-tab-code"
                  role="tab"
                  tabIndex={rightTab === 'code' ? 0 : -1}
                  aria-controls="editor-code-view"
                  className={rightTab === "code" ? "selected" : ""}
                  aria-selected={rightTab === "code"}
                  onClick={() => setRightTab("code")}
                >
                  <Code2 size={15} />
                  {t("代码", "Code")}
                </button>
                <button
                  id="editor-tab-lab"
                  role="tab"
                  tabIndex={rightTab === 'lab' ? 0 : -1}
                  aria-controls="editor-lab-view"
                  className={rightTab === "lab" ? "selected" : ""}
                  aria-selected={rightTab === "lab"}
                  onClick={() => setRightTab("lab")}
                >
                  <FlaskConical size={14} />
                  {t("实验台", "Lab")}
                </button>
                {user && <button id="editor-tab-history" role="tab" aria-selected={rightTab==="history"} tabIndex={rightTab==="history"?0:-1} aria-controls="editor-history-view" className={rightTab === "history" ? "selected" : ""} onClick={() => setRightTab("history")}>
                  {t("我的提交", "My submissions")}
                </button>}
                <span className="editor-language">Python 3 · PyTorch</span>
              </div>
              <div className="editor-history" id="editor-history-view" role="tabpanel" aria-labelledby="editor-tab-history" hidden={rightTab!=='history' || !user}>
                {rightTab === 'history' && user && <SolutionHistory key={`${practice.owner}:${problem.id}`} owner={practice.owner} problem={problem} revision={submissionRevision} />}
              </div>
              <div className="editor-lab" id="editor-lab-view" role="tabpanel" aria-labelledby="editor-tab-lab" tabIndex={0} hidden={rightTab!=='lab'}>
                {rightTab === 'lab' && <Visuals key={problem.id} problem={problem} report={report} sourceState={reportSource} initialView="auto" />}
              </div>
              <div className="editor-workbench" data-results-visible={resultsVisible} id="editor-code-view" role="tabpanel" aria-labelledby="editor-tab-code" hidden={rightTab!=='code'}>
                  <ResizableSplit
                    className="editor-split"
                    axis="vertical"
                    storageKey="paperleet.layout.editor.v1"
                    initial={0.72}
                    layout={layoutRequest?.vertical}
                    secondCollapsed={resultsCollapsed || !resultsVisible}
                    minFirst={180}
                    minSecond={100}
                    label={t("调整代码与测试结果区高度", "Resize code and test results panels")}
                    resetLabel={t("拖动或使用方向键调整，双击或按 Enter 恢复默认", "Drag or use arrow keys to resize; double-click or press Enter to reset")}
                    firstId="problem-code-panel"
                    secondId="problem-results-panel"
                  >
                  <div className="editor-code-panel" id="problem-code-panel">
                  <div className="editor-tools">
                    <span className="editor-filename">
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
                        if (!editorRecoveryReason && practice.owns(practice.owner) && [problem.id,String(problem.number)].includes(routePath(getId()))) location.reload();
                      }}}
                      font={font}
                      label={`${problem.title} ${t("Python 代码编辑器", "Python code editor")}`}
                    />
                  </div>
                  <div className="editor-status" data-state={workspaceSaveState}>
                    <span>
                      {workspaceSaveState === "warning" ? <AlertCircle size={12} /> : workspaceSaveState === "loading" || workspaceSaveState === "saving" ? <Loader2 className="save-spinner" size={12} /> : <Check size={12} />}
                      {practice.loading
                        ? t("正在加载工作区…", "Loading workspace…")
                        : !practice.ready
                          ? t("工作区暂不可用", "Workspace unavailable")
                        : practice.error
                          ? browserWorkspace
                            ? t("本地保存失败，请重试", "Local save failed · retry")
                            : t("同步失败，草稿保留在当前浏览器", "Sync failed · draft kept in this browser")
                        : practice.conflicts[problem.id]
                          ? t("草稿版本冲突，请选择保留的版本", "Draft conflict · choose a version below")
                        : practice.saving || practice.pending
                          ? browserWorkspace
                            ? t("正在保存到当前浏览器…", "Saving in this browser…")
                            : t("正在同步草稿…", "Syncing draft…")
                        : browserWorkspace
                          ? current.code !== undefined
                            ? t("已保存到当前浏览器", "Saved in this browser")
                            : t("初始代码 · 编辑后自动保存到本地", "Starter code · edits save locally")
                          : t("已同步到账号", "Synced to your account")}
                    </span>
                    <span>
                      UTF-8 <i>·</i> Python
                    </span>
                  </div>
                  </div>
                  <div className="editor-results-panel" id="problem-results-panel" hidden={!resultsVisible}>
                  <button id="problem-results-toggle" type="button" className="results-pane-toggle" aria-label={resultsCollapsed?t('展开测试结果','Expand test results'):t('收起测试结果','Minimize test results')} aria-expanded={!resultsCollapsed} aria-controls="problem-results-content" onClick={() => setResultsCollapsed(value => !value)}>
                    <Terminal size={14} aria-hidden="true"/><b>{t('测试结果','Test results')}</b>
                    <span>{running ? t('运行中…','Running…') : workspaceAttention ? t('需要处理','Needs attention') : hasCurrentSyntaxError ? t('语法错误','Syntax error') : reportSource === 'changed' || syntaxFailure ? t('代码已修改','Code changed') : report && reportSource !== 'current' ? t('上次运行','Previous run') : report ? `${report.tests.filter(test => test.passed).length} / ${report.tests.length}` : t('等待运行','Ready to run')}</span>
                    <ChevronDown size={14} aria-hidden="true"/>
                  </button>
                  <div className="editor-results-content" id="problem-results-content" hidden={resultsCollapsed}>
                  {problem.browser_judge && (browser.active || browser.reasons[problem.id] || browser.diagnostics[problem.id] || report?.runtime === "browser-numpy-wasm") && <div className="browser-judge-panel">
                    {browser.active && (browserHere
                      ? <BrowserRunProgress progress={browser.progress || { stage: browser.stage || 'loading' }} onStop={browser.stop} />
                      : <div role="status">{t("另一道题正在浏览器运行", "Another module is running in the browser")} <button onClick={browser.stop}>{t("停止运行", "Stop run")}</button></div>)}
                    {browser.active && browserHere && browserBudget(problem).total > 120_000 && <small>{t(`完整实验最多运行 ${browserBudget(problem).total / 60_000} 分钟；可随时停止。`, `The full experiment can take up to ${browserBudget(problem).total / 60_000} minutes. You can stop at any time.`)}</small>}
                    {browser.reasons[problem.id] && <p role="status">{browserReason(browser.reasons[problem.id], locale === "en")} {t('可以修改后重试，或登录后提交到服务端。', 'Edit and retry, or sign in to submit on the server.')}</p>}
                    {(report?.runtime === "browser-numpy-wasm" || browser.diagnostics[problem.id]) && <JudgeRuntimeDetails server={false} browserResult={report?.runtime === "browser-numpy-wasm"} calibration={browser.calibrations[problem.id]} diagnostics={browser.diagnostics[problem.id]} measurement={report === browser.reports[problem.id] ? browser.measurements[problem.id] : undefined}/>}
                  </div>}
                  {judge.waiting && <QueueAdmissionNotice waiting={judge.waiting} changed={queuedHere && judge.waiting.source !== currentSource} href={!queuedHere && judge.pending ? withLibraryContext(judge.pending.problem,libraryReturn) : undefined} onStop={() => { judge.stopWaiting(); }} onDownload={() => download(`paperleet-submission.py`,judge.waiting!.source)} />}
                  {judge.active && !judge.waiting && <div className="practice-notice judge-queue-notice" role="status">
                    <span><b>{queuedHere ? runStatus : t("另一道题正在排队或判题", "Another module has a queued or running job")}</b><br />
                      {t("可以继续编辑；本次判题使用提交时的代码。刷新后可继续查看任务。", "You can keep editing. This run uses the submitted code; refresh to reconnect.")}</span>
                    {!queuedHere && <a href={`#/${judge.pending?.problem}`}>{t("查看任务", "View job")}</a>}
                    {judge.job?.status === "queued" && <button onClick={() => void judge.cancel()}>{t("取消排队", "Leave queue")}</button>}
                  </div>}
                  {judge.error && <div className="practice-notice" role="alert">
                    <span>{jobErrorText(judge.error, locale)}{judge.reloadRequired && editorRecoveryReason && <small>{editorRecoveryReason}</small>}</span>
                    {judge.reloadRequired && <button disabled={!!editorRecoveryReason} onClick={() => {
                      if (!editorRecoveryReason && practice.owns(practice.owner)) reloadApp();
                    }}>{t('刷新并恢复草稿', 'Reload and restore draft')}</button>}
                  </div>}
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
                              browserWorkspace ? "另一标签页更新了这道题的草稿。请选择保留当前代码或使用另一版本；也可先下载当前代码。" : "其他设备更新了这道题的草稿。选择保留哪个版本；当前代码尚未覆盖服务器。",
                              browserWorkspace ? "Another tab updated this draft. Keep your current code or use the other version; you can download your current code first." : "Another device updated this draft. Choose which version to keep; your current code has not overwritten the server.",
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
                  {!running&&(report||syntaxFailure||resultsRequested)&&<WorkspaceTestResults key={`${practice.owner}:${problem.id}`} problem={problem} report={report} syntax={syntaxFailure} source={currentSource} sourceState={reportSource} canRun={canRun} running={running}
                    onEdit={()=>{setResultsCollapsed(true);openEditor();}} onRun={()=>void run()} onLocate={practice.ready?locateSource:undefined}>
                    {hosted&&report?.runtime==='browser-numpy-wasm'&&<BrowserResultActions problem={problem} report={report} sourceState={reportSource} signedIn={!!user} online={online} ready={progressReady} conflict={!!practice.conflicts[problem.id]} syncError={!!practice.error} busy={running} onSignIn={signInToSubmit} onVerify={()=>void run('server',report)} onRefresh={()=>void refreshHealth()} onLab={()=>openLab('run')}/>}
                    {reportSource==='current'&&progressReady&&isPassed(problem,report)&&<InterviewPracticeNextStep current={problem} problems={problems} saved={learningSaved} next={nextProblem} returnTo={libraryReturn} onNotes={()=>selectReadingTab('notes')} onLab={()=>openLab('run')}/>}
                  </WorkspaceTestResults>}
                  </div>
                  </div>
                  </ResizableSplit>

              </div>
            </section>
          </ResizableSplit>
        </>
      )}
      <AchievementObserver key={practice.owner} problems={problems} papers={papers} saved={learningSaved} ready={progressReady && problems.length > 0} />
      {resultNotice?.owner === practice.owner && (
        <div role="status" className="toast">
          <CheckCircle2 size={16} />
          {t('提交已完成', 'Submission complete')}
          <a href={resultNotice.href} onClick={() => setResultNotice(undefined)}>{t('查看结果', 'View result')}</a>
          <button className="icon-button" aria-label={t('关闭提示', 'Dismiss notification')} onClick={() => setResultNotice(undefined)}><X size={13} /></button>
        </div>
      )}
      {toast && !resultNotice && (
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
