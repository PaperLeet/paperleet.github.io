import { useEffect, useRef, useState } from "react";
import type { ChangeEvent, ReactNode } from "react";
import RebuttalEditor from "./RebuttalEditor";
import { assemblyErrorText, assemblyScope, useAssemblyDraft } from "./rebuttalAssemblyDraft";
import { toOpenReviewMarkdown } from "./rebuttalFormatting";
import Preview from "./RebuttalPreview";
import { ArrowDownToLine, ArrowLeft, ArrowRight, Check, ChevronRight, ClipboardList, Copy, FileText, FlaskConical, ListChecks, Loader2, MessageSquare, Plus, Quote, Search, Settings2, Sparkles, Trash2, Upload, UserRound, History, X } from "lucide-react";
import { useI18n } from "./i18n";
import { apiFetch } from "./apiClient";
import { socialApi, useAccount } from "./social";
import { assembleResponse, countWords, demoProject, reviewerOverview, saveResponseSnapshot, importReviewFile, makeId, newProject, readWorkspace, splitReviews, validateWorkspace, writeWorkspace, removeConcern, removeExperiment, restoreRemovedContent } from "./rebuttalModel";
import type { Experiment, Issue, RebuttalProject, Reviewer, Workspace, RemovedContent } from "./rebuttalModel";
import RebuttalLibrary from "./RebuttalLibrary";
import { readUsers, writeUsers, createLocalUser, selectLocalUser, workspaceStorage } from "./rebuttalUsers";
import type { UserDirectory } from "./rebuttalUsers";
import "./rebuttal.css";

export type HostedStudioSession = {
  project: RebuttalProject; readOnly: boolean; saveStatus: "saved" | "saving" | "error" | "conflict";
  onChange: (project: RebuttalProject) => void; onBack: () => void; onShare: () => void;
  onRefresh: () => void; role: "owner" | "editor" | "viewer";
  onPersonalDraft?: (bridge: { flush: () => Promise<boolean>; hasPending: () => boolean } | null) => void;
};

type Stage = "overview" | "reviews" | "experiments" | "issues" | "export" | "history";
type AIStatus = { configured: boolean; model: string | null };
type AnalyzedIssue = Omit<Issue, "id" | "status" | "outline" | "response"> & { experiment?: { title: string; protocol: string; metric: string } };

function download(name: string, content: string, type = "text/plain") {
  const url = URL.createObjectURL(new Blob([content], { type: `${type};charset=utf-8` }));
  const a = document.createElement("a"); a.href = url; a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function Field({ label, children }: { label: string; children: ReactNode }) {
  return <label className="rst-field"><span>{label}</span>{children}</label>;
}
function initialWorkspace() {
  const storage = { getItem: (key: string) => localStorage.getItem(key), setItem: (key: string, value: string) => localStorage.setItem(key, value) };
  const users = readUsers(storage);
  const local = readWorkspace(workspaceStorage(storage, users.directory.activeUserId));
  return { ...local, directory: users.directory, userError: users.error };
}
function paperFromUrl() {
  return typeof location === "undefined" ? null : new URLSearchParams(location.search).get("paper");
}
function paperUrl(id?: string) {
  history.pushState(null, "", id ? `/rebuttal/?paper=${encodeURIComponent(id)}` : "/rebuttal/");
}
function hasReviewContent(review: Reviewer) {
  return Boolean(review.text.trim() || review.summary?.trim() || review.aiSummary || review.strengths?.trim() || review.rating?.trim() || review.confidence?.trim());
}
function assistanceSource(project: RebuttalProject, issue: Issue) {
  return JSON.stringify({ context: project.context, title: issue.title, quote: issue.quote, outline: issue.outline,
    evidence: project.experiments.filter(e => e.issueId === issue.id && e.status === "done" && e.result.trim()).map(e => [e.title, e.result]) });
}

export default function RebuttalStudio({ hosted }: { hosted?: HostedStudioSession }) {
  const { t, locale } = useI18n();
  const { user: accountUser, loading: accountLoading, capabilities } = useAccount();
  const [initial] = useState(() => hosted ? {
    workspace: { version: 1 as const, activeProjectId: hosted.project.id, projects: [hosted.project] },
    error: false, userError: false,
    directory: { version: 1 as const, activeUserId: "cloud", users: [{ id: "cloud", name: "", createdAt: new Date().toISOString() }] },
  } : initialWorkspace());
  const readOnly = hosted?.readOnly || false;
  const hostedRef = useRef(hosted); hostedRef.current = hosted;
  const lastSubmittedProject = useRef(hosted?.project);
  async function localRequest(path: string, options: RequestInit = {}) {
    const body = options.body ? { ...JSON.parse(String(options.body)), project_id: hostedRef.current?.project.id, expected_user_id: accountUser?.id } : undefined;
    const response = await apiFetch(path, { ...options, body: body ? JSON.stringify(body) : undefined,
      headers: { "Content-Type": "application/json", ...options.headers }, cache: "no-store", redirect: "error" });
    const value = await response.json();
    if (!response.ok) throw Error(typeof value?.error === "string" ? value.error : t("服务暂不可用", "Service unavailable"));
    return value;
  }
  const [workspace, setWorkspace] = useState<Workspace>(() => { const id = paperFromUrl(); return initial.workspace.projects.some(p => p.id === id) ? { ...initial.workspace, activeProjectId: id! } : initial.workspace; });
  const [directory, setDirectory] = useState<UserDirectory>(initial.directory);
  const [storageBlocked, setStorageBlocked] = useState(initial.error);
  const [inPaper, setInPaper] = useState(() => !!hosted || initial.workspace.projects.some(p => p.id === paperFromUrl()));
  const [newUserName, setNewUserName] = useState("");
  const [snapshotId, setSnapshotId] = useState("");
  const [legacyHistory, setLegacyHistory] = useState(false);
  const explicitAssemblyScope = useRef(false);
  const user = directory.users.find(u => u.id === directory.activeUserId)!;
  const activeUserId = useRef(user.id);
  activeUserId.current = user.id;
  const latestWorkspace = useRef(workspace);
  latestWorkspace.current = workspace;
  const [saveError, setSaveError] = useState(initial.error);
  const [saved, setSaved] = useState(false);
  const [stage, setStage] = useState<Stage>("overview");
  const [selectedId, setSelectedId] = useState("");
  const [reviewerFilter, setReviewerFilter] = useState("all");
  const [query, setQuery] = useState("");
  const [expandedSummaries, setExpandedSummaries] = useState<string[]>([]);
  const [settings, setSettings] = useState(false);
  const [busy, setBusy] = useState("");
  const [notice, setNotice] = useState("");
  const [ai, setAi] = useState<AIStatus>({ configured: false, model: null });
  const [suggestion, setSuggestion] = useState<{ ownerId: string; projectId: string; issueId: string; text: string; base: string; source: string } | null>(null);
  const [removed, setRemoved] = useState<(RemovedContent & { ownerId: string }) | null>(null);
  const [experimentDraft, setExperimentDraft] = useState<{ issueId: string; title: string; newIssueTitle: string } | null>(null);
  const [exportReviewer, setExportReviewer] = useState("all");
  const backupInput = useRef<HTMLInputElement>(null);
  const reviewInput = useRef<HTMLInputElement>(null);
  const mounted = useRef(true);
  const project = workspace.projects.find(p => p.id === workspace.activeProjectId)!;
  const reviewDraft = project.reviews;
  const meaningfulReviews = project.reviews.filter(review => hasReviewContent(review) || project.issues.some(issue => issue.reviewerId === review.id));
  const issues = project.issues.filter(issue => (reviewerFilter === "all" || issue.reviewerId === reviewerFilter) && `${issue.title} ${issue.quote}`.toLowerCase().includes(query.toLowerCase()));
  const selected = issues.find(issue => issue.id === selectedId) || issues[0];
  const exportReviewerId = exportReviewer === "all" ? undefined : exportReviewer.slice("reviewer-".length);
  const versionScopeAvailable = exportReviewer === "all" || project.reviews.some(review => `reviewer-${review.id}` === exportReviewer);
  const versionsAvailable = !hosted || capabilities?.rebuttal_reviewer_versions === true;
  const draftCloud = !!hosted;
  const draftAvailable = !hosted || !!accountUser && !accountLoading && capabilities?.rebuttal_assembly_drafts === true;
  const draftSharedReadOnly = readOnly;
  const draft = useAssemblyDraft({ owner: hosted ? !accountLoading && accountUser?.id || "" : user.id, paperId: project.id, scope: exportReviewer, cloud: draftCloud, available: draftAvailable, enabled: draftAvailable && inPaper && (stage === "export" || stage === "history" && !legacyHistory && versionScopeAvailable), request: hosted ? socialApi : undefined });
  useEffect(() => { hostedRef.current?.onPersonalDraft?.({ flush: flushPersonalDrafts, hasPending: draft.hasPending }); return () => hostedRef.current?.onPersonalDraft?.(null); }, [draft.flushAll, draft.hasPending, draft.failedScope]);
  const generated = toOpenReviewMarkdown(assembleResponse(project, exportReviewerId));
  const assembled = draftAvailable && !draft.state.ready ? "" : draft.state.source ?? generated;
  const words = countWords(assembled);
  const pending = project.issues.filter(issue => (!exportReviewerId || issue.reviewerId === exportReviewerId) && !issue.response.trim()).length;
  const activeReviewer = project.reviews.find(r => r.id === reviewerFilter) || meaningfulReviews[0] || project.reviews[0];
  const visibleExperiments = project.experiments.filter(e => reviewerFilter === "all" || project.issues.some(i => i.id === e.issueId && i.reviewerId === reviewerFilter));
  const historyScope = legacyHistory ? "legacy" : exportReviewer;
  const historyScopes = project.reviews.map(review => ({ scope: `reviewer-${review.id}`, name: review.name }));
  for (const item of project.snapshots || []) {
    if (item.scope?.startsWith("reviewer-") && !historyScopes.some(group => group.scope === item.scope)) historyScopes.push({ scope: item.scope, name: `${item.reviewerName || item.scope.slice("reviewer-".length)} ${t("（已移除）", "(removed)")}` });
  }
  historyScopes.push({ scope: "all", name: t("所有审稿人", "All reviewers") });
  if (project.snapshots?.some(item => item.scope === undefined)) historyScopes.push({ scope: "legacy", name: t("旧版本", "Legacy versions") });
  const historyVersions = (project.snapshots || []).filter(item => legacyHistory ? item.scope === undefined : item.scope === exportReviewer);
  const snapshot = historyVersions.find(item => item.id === snapshotId) || historyVersions.at(-1);
  const historyName = historyScopes.find(group => group.scope === historyScope)?.name || "";
  const versionLabel = (label: string) => label.replace(/^Version (\d+)$/, "V$1");
  const canSaveVersion = !draftSharedReadOnly && versionsAvailable && versionScopeAvailable && !(stage === "history" && legacyHistory) && draftAvailable && draft.state.ready && draft.state.status !== "loading" && !!assembled.trim();
  const selectedExperiments = project.experiments.filter(experiment => experiment.issueId === selected?.id);
  const experimentIssues = project.issues.filter(issue => issue.reviewerId === activeReviewer?.id);
  const suggestionChanged = !!(suggestion && selected && (selected.response !== suggestion.base || assistanceSource(project, selected) !== suggestion.source));

  useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
  useEffect(() => {
    const controller = new AbortController();
    localRequest("/api/rebuttal/status", { signal: AbortSignal.any([controller.signal, AbortSignal.timeout(5000)]) })
      .then(value => setAi(value as AIStatus)).catch(() => {});
    return () => controller.abort();
  }, [settings]);
  useEffect(() => {
    if (hostedRef.current) {
      const changed = workspace.projects.find(p => p.id === workspace.activeProjectId);
      if (!readOnly && changed && changed !== lastSubmittedProject.current) {
        lastSubmittedProject.current = changed;
        hostedRef.current.onChange(changed);
      }
      return;
    }
    if (storageBlocked) return;
    // Save this render immediately so navigation cannot cancel a pending debounce.
    let ok = false;
    try { ok = writeWorkspace(workspaceStorage(localStorage, user.id), workspace); } catch { /* Storage access itself can be denied. */ }
    setSaveError(!ok); setSaved(ok);
  }, [workspace, storageBlocked, user.id, readOnly]);
  useEffect(() => {
    function navigate() {
      if (hostedRef.current) return;
      const id = paperFromUrl();
      const found = latestWorkspace.current.projects.find(p => p.id === id);
      setSettings(false);
      setInPaper(!!found);
      if (found) { setWorkspace(current => ({ ...current, activeProjectId: found.id })); setStage("overview"); }
    }
    window.addEventListener("popstate", navigate);
    return () => window.removeEventListener("popstate", navigate);
  }, []);
  useEffect(() => { if (!notice) return; const timer = setTimeout(() => setNotice(""), 7000); return () => clearTimeout(timer); }, [notice]);
  useEffect(() => { setReviewerFilter("all"); setExportReviewer("all"); explicitAssemblyScope.current = false; setLegacyHistory(false); setSelectedId(""); setSuggestion(null); setQuery(""); setStage("overview"); setSnapshotId(""); setExpandedSummaries([]); setExperimentDraft(null); setRemoved(null); }, [project.id, user.id]);
  useEffect(() => { if (stage !== "experiments") setExperimentDraft(null); }, [stage]);
  useEffect(() => {
    if (!settings) return;
    const previous = document.activeElement as HTMLElement | null;
    const modal = document.querySelector<HTMLElement>(".rst-modal");
    modal?.querySelector<HTMLElement>("button, input, textarea, select")?.focus();
    function keydown(event: KeyboardEvent) {
      if (event.key === "Escape" && !busy) { setSettings(false); }
      if (event.key !== "Tab" || !modal) return;
      const nodes = [...modal.querySelectorAll<HTMLElement>("button:not(:disabled), input:not(:disabled), textarea:not(:disabled), select:not(:disabled), summary, a[href]")];
      const first = nodes[0], last = nodes.at(-1);
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
    }
    document.addEventListener("keydown", keydown);
    return () => { document.removeEventListener("keydown", keydown); previous?.focus(); };
  }, [settings, busy]);

  async function flushPersonalDrafts() {
    const ok = await draft.flushAll();
    if (!ok && mounted.current) { const scope = draft.failedScope(); if (scope) setExportReviewer(scope); setStage("export"); setNotice(t("个人草稿尚未保存，请重试或下载。", "Your personal draft is unsaved. Retry or download it.")); }
    return ok;
  }
  function updateProject(change: Partial<RebuttalProject> | ((value: RebuttalProject) => RebuttalProject), id = project.id) {
    if (readOnly || hostedRef.current?.readOnly) return;
    setWorkspace(current => ({ ...current, projects: current.projects.map(p => p.id === id ? { ...(typeof change === "function" ? change(p) : { ...p, ...change }), updatedAt: new Date().toISOString() } : p) }));
  }
  function updateIssue(id: string, change: Partial<Issue>) { updateProject(p => ({ ...p, issues: p.issues.map(issue => issue.id === id ? { ...issue, ...change } : issue) })); }
  function updateExperiment(id: string, change: Partial<Experiment>) { updateProject(p => ({ ...p, experiments: p.experiments.map(item => item.id === id ? { ...item, ...change } : item) })); }
  function enterPaper(id: string) {
    if (busy || readOnly) return;
    setWorkspace(current => ({ ...current, activeProjectId: id })); setInPaper(true); setStage("overview"); paperUrl(id);
  }
  async function backToLibrary() { if (!await flushPersonalDrafts()) return; if (hosted) { hosted.onBack(); return; } setInPaper(false); setSettings(false); paperUrl(); }
  function addProject(sample = false, title?: string, venue?: string) {
    if (busy || hosted || readOnly) return;
    if (workspace.projects.length >= 100) { setNotice(t("每个用户最多保存 100 篇论文。", "Each user can keep up to 100 papers.")); return; }
    const next = sample ? demoProject() : { ...newProject(), title: title?.trim() || t("未命名论文", "Untitled paper"), venue: venue?.trim() || "OpenReview" };
    setWorkspace(current => ({ ...current, activeProjectId: next.id, projects: [...current.projects, next] }));
    setInPaper(true); setStage("overview"); paperUrl(next.id);
  }
  async function switchUser(id: string, nextDirectory = selectLocalUser(directory, id)) {
    if (busy || id === user.id) return;
    if (!await flushPersonalDrafts()) return;
    try {
      if (initial.userError) { setNotice(t("用户列表无法读取，请先导出当前备份。", "The user list cannot be read. Export your current backup first.")); return; }
      // A damaged destination must not become the active profile and lock out healthy data.
      const next = readWorkspace(workspaceStorage(localStorage, id));
      if (next.error) { setNotice(t("该用户的数据无法读取，未切换用户。", "This user's data cannot be read. Your current user is unchanged.")); return; }
      if (storageBlocked || !writeWorkspace(workspaceStorage(localStorage, user.id), latestWorkspace.current)) { setNotice(t("当前修改尚未保存，请先导出备份。", "Current changes are unsaved. Export a backup first.")); return; }
      if (!writeUsers(localStorage, nextDirectory)) { setNotice(t("无法保存用户设置", "Could not save user settings")); return; }
      setDirectory(nextDirectory); setWorkspace(next.workspace); setStorageBlocked(false); setSaveError(false); setSaved(true);
      setRemoved(null); setSuggestion(null); setSettings(false); setInPaper(false); paperUrl();
    } catch { setNotice(t("浏览器存储不可用，请先导出备份。", "Browser storage is unavailable. Export a backup first.")); }
  }
  function addUser() {
    try { const next = createLocalUser(directory, newUserName); switchUser(next.activeUserId, next); setNewUserName(""); }
    catch (error) { setNotice(error instanceof Error ? error.message : t("无法创建用户", "Could not create user")); }
  }
  function showStage(next: Stage) {
    setExperimentDraft(null);
    if ((next === "experiments" || next === "issues") && !meaningfulReviews.some(review => review.id === reviewerFilter)) {
      const firstReviewer = meaningfulReviews[0] || project.reviews[0];
      if (firstReviewer) setReviewerFilter(firstReviewer.id);
    }
    if (next === "export") {
      if (stage !== "history" || !versionScopeAvailable) setExportReviewer(reviewerFilter === "all" || project.reviews.some(review => review.id === reviewerFilter) ? assemblyScope(reviewerFilter) : project.reviews[0] ? `reviewer-${project.reviews[0].id}` : "all");
      setLegacyHistory(false);
    }
    if (next === "history" && stage !== "history") {
      if (stage !== "export" && !explicitAssemblyScope.current) {
        const reviewer = project.reviews.find(review => review.id === reviewerFilter) || project.reviews[0];
        setExportReviewer(reviewer ? `reviewer-${reviewer.id}` : "all");
      }
      setLegacyHistory(false); setSnapshotId("");
    }
    setStage(next);
  }
  function showReviewer(id: string, next: Stage = "experiments") {
    setExperimentDraft(null);
    setReviewerFilter(id); setExportReviewer(`reviewer-${id}`); setLegacyHistory(false); explicitAssemblyScope.current = false; setQuery(""); setSelectedId(""); setStage(next);
  }
  function selectVersionScope(scope: string) {
    setLegacyHistory(scope === "legacy");
    if (scope !== "legacy") { setExportReviewer(scope); explicitAssemblyScope.current = true; }
    setSnapshotId("");
  }
  function saveVersion() {
    if (!canSaveVersion) return;
    try { const next = saveResponseSnapshot(project, undefined, toOpenReviewMarkdown(assembled), exportReviewer); updateProject(next); setSnapshotId(next.snapshots?.at(-1)?.id || ""); setNotice(t("已保存回复版本", "Response version saved")); }
    catch (error) { setNotice(error instanceof Error ? error.message : t("无法保存版本", "Could not save version")); }
  }
  function openReviews() { setStage("reviews"); }
  function setReviewDraft(next: Reviewer[] | ((current: Reviewer[]) => Reviewer[])) {
    updateProject(p => ({ ...p, reviews: typeof next === "function" ? next(p.reviews) : next }));
  }
  function normalizedReviews(reviews: Reviewer[]) {
    return reviews.map((review, index) => ({ ...review, name: review.name.trim() || `Reviewer ${index + 1}` }));
  }
  function addReviewer() {
    if (busy || readOnly) return;
    if (reviewDraft.length >= 100) { setNotice(t("每个项目最多保存 100 位审稿人。", "A project can contain up to 100 reviewers.")); return; }
    setReviewDraft(current => [...current, { id: makeId(), name: `Reviewer ${current.length + 1}`, text: "" }]);
  }
  async function copy(text: string) { try { await navigator.clipboard.writeText(toOpenReviewMarkdown(text)); setNotice(t("已复制 OpenReview 源码", "OpenReview source copied")); } catch { setNotice(t("复制失败，请下载 Markdown 文件。", "Copy failed. Download the Markdown file instead.")); } }
  async function restore(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]; event.target.value = ""; if (!file || busy || hostedRef.current) return;
    const ownerId = user.id;
    setBusy("restore");
    try {
      if (file.size > 60 * 1024 * 1024) throw Error(t("备份文件不能超过 60 MB。", "Backup must be under 60 MB."));
      const raw = await file.text();
      if (raw.length > 20_000_000) throw Error(t("备份内容超过工作区大小限制。", "Backup exceeds the workspace size limit."));
      const incoming = validateWorkspace(JSON.parse(raw));
      // Import as separate projects so the active drafts are never replaced.
      const imported = incoming.projects.map(p => ({ ...p, id: makeId() }));
      if (!mounted.current || activeUserId.current !== ownerId) return;
      const current = latestWorkspace.current;
      if (current.projects.length + imported.length > 100) throw Error(t("导入后项目总数不能超过 100。", "The imported workspace would exceed 100 projects."));
      const proposed = validateWorkspace({ ...current, projects: [...current.projects, ...imported], activeProjectId: imported[0].id });
      setWorkspace(proposed); setInPaper(true); paperUrl(proposed.activeProjectId); setStage("overview");
      setNotice(t("备份已导入为独立项目", "Backup imported as separate projects"));
    } catch (error) { setNotice(error instanceof Error ? error.message : t("备份格式无效", "Invalid backup")); }
    finally { if (mounted.current) setBusy(""); }
  }
  async function importReviews(event: ChangeEvent<HTMLInputElement>) {
    const files = [...(event.target.files || [])]; event.target.value = "";
    if (!files.length || busy || readOnly) return;
    const ownerId = user.id;
    setBusy("import");
    try {
      const reviews: Reviewer[] = [];
      for (const file of files) {
        if (file.size > 256 * 1024) throw Error(t("单个审稿文件不能超过 256 KB。", "Each review file must be under 256 KB."));
        const text = await file.text();
        reviews.push(...(file.name.toLowerCase().endsWith(".json") ? importReviewFile(JSON.parse(text)) : [{ id: makeId(), name: file.name.replace(/\.(txt|md)$/i, ""), text }]));
      }
      if (!mounted.current || activeUserId.current !== ownerId || hostedRef.current?.readOnly) return;
      const current = latestWorkspace.current;
      const target = current.projects.find(p => p.id === project.id);
      if (!target) return;
      const onlyReview = target.reviews.length === 1 ? target.reviews[0] : undefined;
      const pristineReviewer = onlyReview?.name === "Reviewer 1" && !hasReviewContent(onlyReview) && !target.issues.some(issue => issue.reviewerId === onlyReview.id);
      const existing = pristineReviewer ? [] : target.reviews;
      if (existing.length + reviews.length > 100) throw Error(t("导入后审稿人总数不能超过 100。", "The import would exceed 100 reviewers."));
      const proposedReviews = normalizedReviews([...existing, ...reviews.map(review => ({ ...review, id: makeId() }))]);
      validateWorkspace({ ...current, projects: current.projects.map(p => p.id === project.id ? { ...p, reviews: proposedReviews } : p) });
      setReviewDraft(proposedReviews);
    } catch (error) { setNotice(error instanceof Error ? error.message : t("文件格式无效", "Invalid file")); }
    finally { if (mounted.current) setBusy(""); }
  }
  function mergeIssues(reviews: Reviewer[], incoming: { issues: Issue[]; experiments: Experiment[] }, projectId: string) {
    if (hostedRef.current?.readOnly) return;
    const target = latestWorkspace.current.projects.find(p => p.id === projectId);
    if (!target) return;
    const added = incoming.issues.filter(issue => !target.issues.some(old => old.reviewerId === issue.reviewerId && old.quote.trim() === issue.quote.trim()));
    if (target.issues.length + added.length > 2000 || target.experiments.length + incoming.experiments.length > 2000)
      throw Error(t("项目最多支持 2,000 条问题及实验，请拆分为多个项目。", "A project supports up to 2,000 concerns and experiments. Split larger reviews into separate projects."));
    updateProject(p => {
      const added = incoming.issues.filter(issue => !p.issues.some(old => old.reviewerId === issue.reviewerId && old.quote.trim() === issue.quote.trim()));
      return { ...p, reviews, issues: [...p.issues, ...added], experiments: [...p.experiments, ...incoming.experiments.filter(e => added.some(i => i.id === e.issueId))] };
    }, projectId);
    // Re-analysis may recreate a deleted concern with a new ID.
    setRemoved(current => current?.projectId === projectId ? null : current);
    if (latestWorkspace.current.activeProjectId === projectId && (hosted || paperFromUrl() === projectId)) {
      setStage("reviews"); setReviewerFilter("all"); setQuery("");
    }
  }
  async function analyze(useAI: boolean) {
    if (readOnly) return;
    const allReviews = normalizedReviews(reviewDraft);
    const reviews = allReviews.filter(review => review.text.trim());
    if (!reviews.length) { setNotice(t("请先粘贴至少一份审稿意见。", "Paste at least one review first.")); return; }
    if (useAI && reviews.length > 20) { setNotice(t("AI 每次最多分析 20 位审稿人，请使用本地拆解处理更多审稿意见。", "AI analysis supports up to 20 reviewers at a time. Use local breakdown for larger sets.")); return; }
    if (useAI && (reviews.some(review => review.text.length > 30000 || review.name.length > 200) || reviews.reduce((sum, review) => sum + review.text.length, 0) > 120000)) {
      setNotice(t("AI 输入限制：每份审稿 30,000 字符、姓名 200 字符、审稿总计 120,000 字符。原文可保留并使用本地拆解。", "AI limits are 30,000 characters per review, 200 per reviewer name, and 120,000 total review characters. You can preserve the source and use local breakdown.")); return;
    }
    const projectId = project.id; setBusy("analyze");
    try {
      let incoming: { issues: Issue[]; experiments: Experiment[] };
      if (useAI) {
        const result = await localRequest("/api/rebuttal/analyze", { method: "POST", body: JSON.stringify({ reviews: reviews.map(({ id, name, text }) => ({ id, name, text })), context: project.context, language: locale }), signal: AbortSignal.timeout(95000) }) as { issues: AnalyzedIssue[] };
        const analyzed = result.issues.map(({ experiment, ...item }) => ({
          issue: { ...item, id: makeId(), status: "todo" as const, outline: "", response: "" }, experiment,
        }));
        incoming = { issues: analyzed.map(item => item.issue), experiments: analyzed.flatMap(({ issue, experiment }) => experiment ? [{ ...experiment, id: makeId(), issueId: issue.id, owner: "", status: "planned" as const, result: "" }] : []) };
      } else incoming = splitReviews(reviews, locale);
      if (!mounted.current || readOnly || hostedRef.current?.readOnly) return;
      mergeIssues(allReviews, incoming, projectId);
      setNotice(t(`已拆解 ${incoming.issues.length} 条问题`, `${incoming.issues.length} concerns extracted`));
    } catch (error) { setNotice(error instanceof Error ? error.message : t("拆解失败", "Analysis failed")); }
    finally { if (mounted.current) setBusy(""); }
  }
  function addIssue() {
    if (busy || readOnly) return;
    if (project.issues.length >= 2000) { setNotice(t("已达到 2,000 条问题上限。", "This project has reached the 2,000 concern limit.")); return; }
    const reviewerId = reviewerFilter === "all" ? activeReviewer?.id : reviewerFilter;
    if (!reviewerId) { openReviews(); return; }
    const issue: Issue = { id: makeId(), reviewerId, title: t("新问题", "New concern"), quote: "", category: "clarification", priority: "medium", status: "todo", suggestion: "", outline: "", response: "" };
    updateProject(p => ({ ...p, issues: [...p.issues, issue] })); setSelectedId(issue.id); setQuery(""); setStage("issues");
  }
  function startExperiment(issue?: Issue) {
    if (busy || readOnly) return;
    const reviewerId = issue?.reviewerId || activeReviewer?.id;
    if (!reviewerId) { openReviews(); return; }
    if (project.experiments.length >= 2000) { setNotice(t("已达到 2,000 项实验上限。", "This project has reached the 2,000 experiment limit.")); return; }
    setExperimentDraft({ issueId: issue?.id || experimentIssues[0]?.id || "", title: "", newIssueTitle: "" });
    setReviewerFilter(reviewerId); setStage("experiments");
  }
  function createExperiment() {
    if (busy || readOnly || !experimentDraft || !activeReviewer || !experimentDraft.title.trim()) return;
    if (project.experiments.length >= 2000) { setNotice(t("已达到 2,000 项实验上限。", "This project has reached the 2,000 experiment limit.")); return; }
    let issue = experimentIssues.find(item => item.id === experimentDraft.issueId);
    if (experimentDraft.issueId && !issue) return;
    const isNewIssue = !issue;
    if (isNewIssue) {
      if (!experimentDraft.newIssueTitle.trim()) return;
      if (project.issues.length >= 2000) { setNotice(t("已达到 2,000 条问题上限。", "This project has reached the 2,000 concern limit.")); return; }
      issue = { id: makeId(), reviewerId: activeReviewer.id, title: experimentDraft.newIssueTitle.trim(), quote: "", category: "experiment", priority: "medium", status: "todo", suggestion: "", outline: "", response: "" };
    }
    const linkedIssue = issue!;
    const experiment: Experiment = { id: makeId(), issueId: linkedIssue.id, title: experimentDraft.title.trim(), protocol: "", metric: "", owner: "", status: "planned", result: "" };
    updateProject(p => ({ ...p, issues: isNewIssue ? [...p.issues, linkedIssue] : p.issues, experiments: [...p.experiments, experiment] }));
    setExperimentDraft(null);
    requestAnimationFrame(() => document.getElementById(`experiment-${experiment.id}`)?.scrollIntoView({ block: "center", behavior: "smooth" }));
  }
  function deleteContent(kind: "issue" | "experiment", id: string) {
    if (busy || readOnly) return;
    const result = kind === "issue" ? removeConcern(project, id) : removeExperiment(project, id);
    if (!result.removed) return;
    setRemoved({ ...result.removed, ownerId: user.id }); setSuggestion(null);
    updateProject(result.project);
  }
  function undoRemoval() {
    if (busy || readOnly || !removed || removed.ownerId !== user.id || removed.projectId !== project.id) return;
    const restored = restoreRemovedContent(project, removed);
    if (restored === project) { setNotice(t("当前内容无法恢复此条删除。", "This deletion cannot be restored in the current state.")); return; }
    updateProject(restored); setRemoved(null);
  }
  async function assist(action: "draft" | "polish" | "shorten" | "translate") {
    if (readOnly) return;
    if (!selected) return;
    if (!selected.title.trim() || !selected.quote.trim()) { setNotice(t("请先在「调整拆解」中填写问题标题与原文。", "Add the concern title and source quote under Adjust breakdown first.")); return; }
    if (!ai.configured) { if (hosted) setNotice(t("AI 服务尚未启用。", "AI service is not enabled.")); else setSettings(true); return; }
    setBusy(action);
    const snapshot = { ownerId: user.id, projectId: project.id, issueId: selected.id, base: selected.response, source: assistanceSource(project, selected) };
    try {
      const result = await localRequest("/api/rebuttal/assist", { method: "POST", body: JSON.stringify({ action, text: action === "draft" ? selected.outline : selected.response, context: project.context, issue: { title: selected.title, quote: selected.quote }, evidence: selectedExperiments.filter(e => e.status === "done" && e.result.trim()).map(e => `${e.title}\n${e.result}`).join("\n\n"), language: "en" }), signal: AbortSignal.timeout(95000) }) as { text: string };
      if (mounted.current && activeUserId.current === snapshot.ownerId && !hostedRef.current?.readOnly) setSuggestion({ ...snapshot, text: result.text });
    } catch (error) { setNotice(error instanceof Error ? error.message : t("生成失败", "Generation failed")); }
    finally { if (mounted.current) setBusy(""); }
  }

  async function summarizeReviewers(reviewerId?: string) {
    if (busy || readOnly) return;
    if (!ai.configured) { setNotice(t("AI 服务尚未启用。", "AI service is not enabled.")); if (!hosted) setSettings(true); return; }
    const reviews = normalizedReviews(project.reviews).filter(review => review.text.trim() && (!reviewerId || review.id === reviewerId));
    if (!reviews.length) { openReviews(); return; }
    if (reviews.length > 20 || reviews.some(review => review.text.length > 30000 || review.name.length > 200) || reviews.reduce((sum, review) => sum + review.text.length, 0) > 120000) {
      setNotice(t("每次最多总结 20 位 reviewer，每份审稿不超过 30,000 字符。可按 reviewer 单独生成。", "Summarize up to 20 reviewers, with 30,000 characters per review. You can generate each reviewer separately.")); return;
    }
    const projectId = project.id, ownerId = user.id;
    setBusy(reviewerId ? `summary:${reviewerId}` : "summary");
    try {
      const result = await localRequest("/api/rebuttal/summarize", {
        method: "POST", body: JSON.stringify({ reviews: reviews.map(({ id, name, text }) => ({ id, name, text })), context: project.context }), signal: AbortSignal.timeout(95000),
      }) as { summaries: { reviewerId: string; zh: string; en: string }[] };
      if (!mounted.current || activeUserId.current !== ownerId || hostedRef.current?.readOnly) return;
      const current = latestWorkspace.current;
      const target = current.projects.find(p => p.id === projectId);
      if (!target) return;
      let applied = 0;
      const updated = { ...target, reviews: target.reviews.map(review => {
        const original = reviews.find(item => item.id === review.id);
        const summary = result.summaries.find(item => item.reviewerId === review.id);
        if (!original || !summary || review.text !== original.text) return review;
        applied++;
        return { ...review, aiSummary: { zh: summary.zh, en: summary.en } };
      }), updatedAt: new Date().toISOString() };
      if (!applied) { setNotice(t("审稿原文已变化，请重新生成总结。", "The source reviews changed. Generate the summaries again.")); return; }
      setWorkspace(validateWorkspace({ ...current, projects: current.projects.map(p => p.id === projectId ? updated : p) }));
      setNotice(t("中英文总结已保存", "Chinese and English summaries saved"));
    } catch (error) { setNotice(error instanceof Error ? error.message : t("总结失败，原内容已保留。", "Summary generation failed. Existing content is preserved.")); }
    finally { if (mounted.current) setBusy(""); }
  }

  const userControl = <details className="rst-user-menu"><summary><UserRound size={16} />{user.name}</summary><div><label>{t("本地用户", "Local user")}<select disabled={!!busy || readOnly} value={user.id} onChange={e => switchUser(e.target.value)}>{directory.users.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label><form onSubmit={e => { e.preventDefault(); addUser(); }}><input aria-label={t("新用户名称", "New user name")} maxLength={80} placeholder={t("新用户名称", "New user name")} value={newUserName} onChange={e => setNewUserName(e.target.value)} /><button className="rst-button" disabled={!!busy || !newUserName.trim()}>{t("添加", "Add")}</button></form></div></details>;

  return <main className={`rst-shell${readOnly ? " rst-read-only" : ""}`}>
    <input hidden ref={backupInput} type="file" accept=".json" onChange={restore} />
    {initial.userError && <div className="rst-alert" role="alert">{t("无法读取用户列表，原记录已保留。", "Could not read the user list. Original records are preserved.")}</div>}
    {saveError && <div className="rst-alert" role="alert">{t("保存失败，请导出备份。", "Could not save. Export a backup.")}</div>}
    {!inPaper ? <RebuttalLibrary key={user.id} workspace={workspace} onOpen={enterPaper} onCreate={(title, venue) => addProject(false, title, venue)} onArchive={(id, archived) => { if (!busy) updateProject({ archived }, id); }} onImport={() => { if (!busy) backupInput.current?.click(); }} onExport={() => download("paperleet-rebuttal-backup.json", JSON.stringify(workspace, null, 2), "application/json")} onSample={() => addProject(true)} userControl={userControl} /> : <>
    <div className="rst-paper-navigation"><button className="rst-back" disabled={!!busy} onClick={backToLibrary}><ArrowLeft size={17} />{t("我的论文", "My papers")}</button>{hosted && <button className="rst-text-button" onClick={hosted.onRefresh}>{t("刷新", "Refresh")}</button>}</div>
    <div className={`rst-paper-heading${hosted ? " rst-hosted-heading" : ""}`}>
      <div className="rst-paper-identity"><h1>{project.title || t("未命名论文", "Untitled paper")}</h1><span>{project.venue}</span></div>
      <span className="rst-save"><span className={(hosted ? hosted.saveStatus === "error" || hosted.saveStatus === "conflict" : saveError) ? "rst-error-dot" : "rst-local-dot"} />{hosted ? readOnly ? t("仅查看", "View only") : hosted.saveStatus === "saved" ? t("已保存", "Saved") : hosted.saveStatus === "saving" ? t("正在保存…", "Saving…") : t("尚未保存", "Unsaved") : saveError ? t("尚未保存", "Unsaved") : saved ? t("已保存", "Saved") : t("正在保存…", "Saving…")}</span>
      <div className="rst-paper-actions"><button className="rst-button" disabled={!!busy} onClick={openReviews}><Upload size={15} />{t("审稿原文", "Reviews")}</button>{hosted && <button className="rst-button rst-share-button" onClick={hosted.onShare}>{hosted.role === "owner" ? t("邀请", "Invite") : t("成员", "Members")}</button>}<button className="rst-icon-button" title={t("论文设置", "Paper settings")} aria-label={t("论文设置", "Paper settings")} onClick={() => setSettings(true)}><Settings2 size={17} /></button></div>
    </div>
    <div className="rst-stage-bar"><nav aria-label={t("论文工作区", "Paper workspace")}>
      {([["overview", t("Reviewer 总览", "Reviewers"), ListChecks], ["experiments", t("实验", "Experiments"), FlaskConical], ["issues", t("逐条回复", "Responses"), MessageSquare], ["export", t("拼接回复", "Assemble"), FileText], ["history", t("历史版本", "Versions"), History]] as const).map(([key, label, Icon]) => <button key={key} disabled={!!busy} aria-current={stage === key ? "page" : undefined} className={stage === key ? "active" : ""} onClick={() => showStage(key)}><Icon size={16} />{label}</button>)}
    </nav></div>
    {(stage === "experiments" || stage === "issues") && meaningfulReviews.length > 0 && <div className="rst-reviewer-tabs" aria-label={t("选择 Reviewer", "Choose reviewer")}>{meaningfulReviews.map(reviewer => <button key={reviewer.id} className={activeReviewer?.id === reviewer.id ? "selected" : ""} onClick={() => showReviewer(reviewer.id, stage)}>{reviewer.name}</button>)}</div>}
    {stage === "overview" && <section className="rst-reviewer-overview">
      <div className="rst-section-heading"><h2>{t("Reviewer 总览", "Reviewers")}</h2>{meaningfulReviews.length > 0 && <div className="rst-summary-actions">
        <button className="rst-button" disabled={!!busy || readOnly} onClick={() => summarizeReviewers()}>{busy.startsWith("summary") ? <Loader2 size={15} className="rst-spin" /> : <Sparkles size={15} />}{busy.startsWith("summary") ? t("正在总结…", "Summarizing…") : t("AI 总结", "AI summaries")}</button>
        <button className="rst-button rst-primary" disabled={!!busy || readOnly} onClick={openReviews}><Plus size={15} />{t("导入审稿", "Import reviews")}</button>
      </div>}</div>
      {meaningfulReviews.length > 0 ? <div className="rst-reviewer-grid">{meaningfulReviews.map(reviewer => {
        const concerns = project.issues.filter(i => i.reviewerId === reviewer.id);
        const experiments = project.experiments.filter(e => concerns.some(i => i.id === e.issueId));
        const overview = reviewerOverview(reviewer, concerns);
        const summaryText = reviewer.aiSummary?.[locale] || overview.summary;
        const expanded = expandedSummaries.includes(reviewer.id);
        return <article className="rst-reviewer-card" key={reviewer.id}>
          <header><h2>{reviewer.name}</h2>{overview.rating && <span title={overview.rating}>{t("评分", "Rating")} {overview.rating.split(/[:：]/)[0]}</span>}</header>
          <div className="rst-summary-label"><span>{reviewer.aiSummary ? t("AI 总结", "AI summary") : t("原文摘要", "Source summary")}</span><button className="rst-icon-button" disabled={!!busy || readOnly || !reviewer.text.trim()} aria-label={t(`更新 ${reviewer.name} 的双语总结`, `Update bilingual summary for ${reviewer.name}`)} title={t("生成中英文总结", "Generate Chinese and English summaries")} onClick={() => summarizeReviewers(reviewer.id)}>{busy === `summary:${reviewer.id}` ? <Loader2 size={14} className="rst-spin" /> : <Sparkles size={14} />}</button></div>
          <p className={`rst-reviewer-summary${expanded ? " expanded" : ""}`} lang={reviewer.aiSummary ? locale : undefined}>{summaryText || t("尚未拆解", "Not analyzed yet")}</p>
          {summaryText.length > 160 && <button className="rst-summary-expand" aria-expanded={expanded} onClick={() => setExpandedSummaries(current => expanded ? current.filter(id => id !== reviewer.id) : [...current, reviewer.id])}>{expanded ? t("收起", "Show less") : t("展开", "Read more")}</button>}
          {!reviewer.aiSummary && concerns.length > 0 && !concerns.some(i => overview.summary.includes(i.title)) && <ul>{concerns.slice(0, 3).map(issue => <li key={issue.id}>{issue.title}</li>)}</ul>}
          <div className="rst-reviewer-counts"><span>{concerns.filter(i => i.status === "ready").length} / {concerns.length} {t("条回复", "responses")}</span><span>{experiments.filter(e => e.status === "done").length} / {experiments.length} {t("项实验", "experiments")}</span></div>
          <footer><button className="rst-button" disabled={!!busy} onClick={() => showReviewer(reviewer.id, "experiments")}><FlaskConical size={14} />{t("实验", "Experiments")}</button><button className="rst-button rst-primary" disabled={!!busy} onClick={() => showReviewer(reviewer.id, concerns.length ? "issues" : "reviews")}>{concerns.length ? t("逐条回复", "Write responses") : t("拆解审稿", "Analyze review")}<ArrowRight size={14} /></button></footer>
        </article>;
      })}</div> : <div className="rst-simple-empty"><MessageSquare size={30} /><h3>{t("导入这篇论文的审稿意见", "Add this paper's reviews")}</h3><button className="rst-button" onClick={openReviews}>{t("导入审稿", "Import reviews")}</button></div>}
    </section>}
    {stage === "history" && <section className="rst-version-view">
      <div className="rst-section-heading"><h2>{t("历史版本", "Response versions")}</h2><button className="rst-button" disabled={!canSaveVersion} onClick={saveVersion}><Plus size={15} />{draftCloud ? t("保存共享版本", "Save shared version") : t("保存当前版本", "Save current version")}</button></div>
      <label className="rst-version-scope"><span>{t("版本范围", "Version scope")}</span><select aria-label={t("版本范围", "Version scope")} value={historyScope} onChange={e => selectVersionScope(e.target.value)}>{historyScopes.map(group => <option value={group.scope} key={group.scope}>{group.name}</option>)}</select></label>
      {!versionsAvailable && <p className="rst-assembly-status" role="status">{t("历史版本正在更新，请稍后刷新。", "Versions are updating. Refresh shortly.")}</p>}
      {historyVersions.length ? <div className="rst-version-layout"><div className="rst-version-list">{[...historyVersions].reverse().map(item => <button key={item.id} className={snapshot?.id === item.id ? "selected" : ""} onClick={() => setSnapshotId(item.id)}><strong>{versionLabel(item.label)}</strong><span>{new Date(item.createdAt).toLocaleString(locale === "zh" ? "zh-CN" : "en-US")}</span></button>)}</div>
        {snapshot && <section className="rst-version-content"><header><h3>{historyName} · {versionLabel(snapshot.label)}</h3><button className="rst-button" onClick={() => download("rebuttal-version.md", toOpenReviewMarkdown(snapshot.markdown), "text/markdown")}><ArrowDownToLine size={14} />Markdown</button></header><Preview text={snapshot.markdown} /></section>}
      </div> : <div className="rst-simple-empty"><History size={28} /><p>{t("暂无保存版本", "No saved versions")}</p></div>}
    </section>}


    {stage === "reviews" && <>
      <div className="rst-review-layout">
        <section className="rst-review-stage">
          <header><h2>{t("审稿意见", "Reviews")}</h2>{!hosted && <button className="rst-text-button" disabled={!!busy || readOnly} onClick={() => addProject(true)}>{t("加载示例", "Load example")}</button>}</header>
          <div className="rst-review-inputs">{reviewDraft.map((review, index) => <div key={review.id} className="rst-review-input"><div><span>R{index + 1}</span><input maxLength={300} disabled={!!busy || readOnly} aria-label={t("审稿人名称", "Reviewer name")} value={review.name} onChange={e => setReviewDraft(current => current.map(r => r.id === review.id ? { ...r, name: e.target.value } : r))} /></div><textarea maxLength={200000} disabled={!!busy || readOnly} aria-label={t(`审稿意见 ${index + 1}`, `Review ${index + 1}`)} rows={8} placeholder={t("粘贴这位审稿人的完整意见…", "Paste this reviewer's full review…")} value={review.text} onChange={e => setReviewDraft(current => current.map(r => r.id === review.id ? { ...r, text: e.target.value, summary: undefined, strengths: undefined, rating: undefined, confidence: undefined, aiSummary: undefined } : r))} /></div>)}</div>
          <div className="rst-import-add"><button disabled={!!busy || readOnly} onClick={addReviewer}><Plus size={15} />{t("添加审稿人", "Add reviewer")}</button><button disabled={!!busy || readOnly} onClick={() => reviewInput.current?.click()}><Upload size={15} />{t("导入文件", "Import file")}</button><input hidden ref={reviewInput} type="file" multiple accept=".txt,.md,.json" onChange={importReviews} /></div>
          <footer><button disabled={!!busy || readOnly || !reviewDraft.some(r => r.text.trim())} className={project.issues.length ? "rst-button" : "rst-button rst-primary"} title={ai.configured ? t("使用已配置的 AI", "Use configured AI") : t("按段落和编号拆解", "Split by paragraphs and numbered items")} onClick={() => analyze(ai.configured)}>{busy ? <Loader2 size={15} className="rst-spin" /> : <ListChecks size={15} />}{busy ? t("正在拆解…", "Analyzing…") : t("拆解为问题", "Extract concerns")}</button>{ai.configured && <button className="rst-button" disabled={!!busy || readOnly || !reviewDraft.some(r => r.text.trim())} onClick={() => analyze(false)}>{t("本地拆解", "Local breakdown")}</button>}</footer>
        </section>
        <section className="rst-breakdown-panel"><header><h2>{t("问题清单", "Concerns")}</h2><span>{project.issues.length}</span></header>
          {project.issues.length ? <div className="rst-breakdown-list">{project.issues.map((issue, index) => <article className="rst-breakdown-item" key={issue.id}><header><span>{index + 1}. {project.reviews.find(r => r.id === issue.reviewerId)?.name}</span><span>{issue.category === "experiment" ? t("需要实验", "Experiment") : t("直接回复", "Direct reply")}</span></header><input disabled={readOnly} maxLength={2000} aria-label={t(`问题 ${index + 1}`, `Concern ${index + 1}`)} value={issue.title} onChange={e => updateIssue(issue.id, { title: e.target.value })} /><details><summary>{t("查看原文", "Source quote")}</summary><p>{issue.quote}</p></details></article>)}</div> : <div className="rst-simple-empty"><ListChecks size={26} /><p>{t("暂无问题", "No concerns yet")}</p></div>}
        </section>
      </div>
      <div className="rst-flow-footer"><span>{project.issues.length ? t(`${project.issues.length} 条问题 · ${project.experiments.length} 项实验`, `${project.issues.length} concerns · ${project.experiments.length} experiments`) : ""}</span><button className="rst-button rst-primary" disabled={!!busy || !project.issues.length} onClick={() => setStage("overview")}>{t("查看 Reviewer 总览", "Reviewers overview")}<ArrowRight size={15} /></button></div>
    </>}

    {stage === "issues" && <div className="rst-workbench">
      <aside className="rst-sidebar"><div className="rst-sidebar-title"><h2>{t("问题清单", "Concerns")}</h2><button className="rst-text-button rst-inline-action" disabled={!!busy || readOnly} aria-label={t("添加问题", "Add concern")} onClick={addIssue}><Plus size={15} />{t("添加", "Add")}</button></div>
        <div className="rst-search"><Search size={15} /><input maxLength={2000} aria-label={t("搜索问题", "Search concerns")} placeholder={t("搜索问题…", "Search concerns…")} value={query} onChange={e => setQuery(e.target.value)} /></div>
        <div className="rst-issue-list">{issues.map(issue => <button className={`rst-issue ${selected?.id === issue.id ? "selected" : ""}`} key={issue.id} onClick={() => { setSelectedId(issue.id); }}><div><span className="rst-reviewer">{project.reviews.find(r => r.id === issue.reviewerId)?.name}</span><span className={`rst-priority ${issue.priority}`}>{issue.priority === "high" ? t("优先", "High") : issue.category === "experiment" ? t("实验", "Experiment") : issue.category === "writing" ? t("文字", "Writing") : t("澄清", "Clarify")}</span></div><h3>{issue.title}</h3><footer><span className={`rst-status-dot ${issue.status}`} />{issue.status === "ready" ? t("回复就绪", "Ready") : issue.response || issue.status === "drafting" ? t("撰写中", "In progress") : t("待回复", "To do")}<ChevronRight size={13} /></footer></button>)}</div>
        {!issues.length && project.issues.length > 0 && <p className="rst-no-results">{t("没有匹配的问题", "No matching concerns")}</p>}
        <button className="rst-sidebar-import" onClick={openReviews}><Upload size={15} />{t("管理审稿原文", "Manage source reviews")}</button>
      </aside>
      {selected ? <section className="rst-editor-pane" key={selected.id}>
        <div className="rst-editor-header"><div className="rst-eyebrow">{project.reviews.find(r => r.id === selected.reviewerId)?.name} / {String(project.issues.indexOf(selected) + 1).padStart(2, "0")}</div><div className="rst-row-actions"><select disabled={readOnly} aria-label={t("回复状态", "Response status")} value={selected.status} onChange={e => { if (e.target.value === "ready" && !selected.response.trim()) { setNotice(t("请先填写回复。", "Write a response first.")); return; } updateIssue(selected.id, { status: e.target.value as Issue["status"] }); }}><option value="todo">{t("待回复", "To do")}</option><option value="drafting">{t("撰写中", "Drafting")}</option><option value="ready">{t("回复就绪", "Ready")}</option></select><button className="rst-text-button rst-inline-action rst-delete" disabled={!!busy || readOnly} title={t("删除问题及其关联实验", "Delete concern and linked experiments")} aria-label={t("删除问题", "Delete concern")} onClick={() => deleteContent("issue", selected.id)}><Trash2 size={14} />{t("删除", "Delete")}</button></div></div>
        <input disabled={readOnly} maxLength={2000} className="rst-issue-title" aria-label={t("问题标题", "Concern title")} value={selected.title} onChange={e => updateIssue(selected.id, { title: e.target.value })} />
        <div className="rst-quote"><Quote size={16} /><div><span>{t("审稿原文", "REVIEWER’S COMMENT")}</span><p>{selected.quote || t("展开下方「调整拆解」填写原文。", "Open Adjust breakdown below to add the source quote.")}</p></div></div>
        <details className="rst-adjust"><summary>{t("调整拆解", "Adjust breakdown")}</summary><div className="rst-adjust-fields">
          <Field label={t("审稿人", "Reviewer")}><select disabled={readOnly} value={selected.reviewerId} onChange={e => updateIssue(selected.id, { reviewerId: e.target.value })}>{project.reviews.map(r => <option key={r.id} value={r.id}>{r.name}</option>)}</select></Field>
          <Field label={t("问题类型", "Concern type")}><select disabled={readOnly} value={selected.category} onChange={e => updateIssue(selected.id, { category: e.target.value as Issue["category"] })}><option value="experiment">{t("实验", "Experiment")}</option><option value="clarification">{t("澄清", "Clarification")}</option><option value="writing">{t("文字修改", "Writing")}</option></select></Field>
          <Field label={t("优先级", "Priority")}><select disabled={readOnly} value={selected.priority} onChange={e => updateIssue(selected.id, { priority: e.target.value as Issue["priority"] })}><option value="high">{t("高", "High")}</option><option value="medium">{t("中", "Medium")}</option><option value="low">{t("低", "Low")}</option></select></Field>
        </div><Field label={t("引用的审稿原文", "Quoted source text")}><textarea disabled={readOnly} rows={3} maxLength={200000} value={selected.quote} onChange={e => updateIssue(selected.id, { quote: e.target.value })} /></Field></details>
        <div className="rst-outline-grid"><Field label={t("回复思路", "RESPONSE OUTLINE")}><textarea disabled={readOnly} maxLength={30000} rows={4} placeholder={t("写下回复要点…", "Response notes…")} value={selected.outline} onChange={e => updateIssue(selected.id, { outline: e.target.value })} /></Field><div className="rst-evidence"><div><span>{t("关联实验", "LINKED EXPERIMENTS")}</span><button className="rst-text-button rst-inline-action" disabled={!!busy || readOnly} aria-label={t("添加关联实验", "Add linked experiment")} onClick={() => startExperiment(selected)}><Plus size={14} />{t("添加", "Add")}</button></div>{selectedExperiments.length ? selectedExperiments.map(e => <button key={e.id} onClick={() => setStage("experiments")}><FlaskConical size={14} /><span>{e.title}<small>{e.status === "done" ? t("已完成", "Complete") : t("待完成", "Pending")}</small></span><ChevronRight size={14} /></button>) : <p>{t("暂无关联实验", "No linked experiments")}</p>}</div></div>
        <div className="rst-response-heading"><h3>{t("回复草稿", "Response draft")}</h3></div>
        <RebuttalEditor key={`${project.id}:${selected.id}`} value={selected.response} locale={locale} readOnly={readOnly}
          onChange={response => updateIssue(selected.id, { response, status: "drafting" })}
          actions={!readOnly && <div className="rst-ai-tools">{([['polish', t("润色", "Polish")], ['shorten', t("压缩", "Shorten")], ['translate', t("译为英文", "English")]] as const).map(([action, label]) => <button key={action} disabled={!!busy || !selected.response.trim()} onClick={() => assist(action)}>{busy === action ? <Loader2 size={12} className="rst-spin" /> : <Sparkles size={12} />}{label}</button>)}</div>}
          footer={<div className="rst-editor-bottom"><span>{countWords(selected.response)} {t("词 / 中文字", "words / CJK characters")} · {selected.response.length} {t("字符", "characters")}</span>{!readOnly && <button disabled={!!busy || !selected.outline.trim()} onClick={() => assist("draft")}><Sparkles size={14} />{busy === "draft" ? t("正在起草…", "Drafting…") : t("从思路起草", "Draft from outline")}</button>}</div>}
        />
        {suggestion?.ownerId === user.id && suggestion.projectId === project.id && suggestion.issueId === selected.id && <div className="rst-ai-proposal"><div><h3>{t("AI 修改建议", "AI suggestion")}</h3><button className="rst-icon-button" aria-label={t("关闭建议", "Dismiss suggestion")} onClick={() => setSuggestion(null)}><X size={16} /></button></div><Preview text={suggestion.text} /><button className="rst-button rst-primary" disabled={readOnly || suggestionChanged} onClick={() => { if (readOnly || suggestionChanged) return; updateIssue(selected.id, { response: suggestion.text, status: "drafting" }); setSuggestion(null); }}><Check size={14} />{suggestionChanged ? t("内容已变化，请重新生成", "Content changed; generate again") : t("应用到草稿", "Apply to draft")}</button></div>}
        <div className="rst-editor-footer"><button className="rst-button" disabled={!selected.response.trim()} onClick={() => copy(selected.response)}><Copy size={14} />{t("复制此条回复", "Copy response")}</button><button className="rst-button rst-primary" disabled={readOnly || !selected.response.trim()} onClick={() => { updateIssue(selected.id, { status: "ready" }); const next = project.issues.find(i => i.reviewerId === selected.reviewerId && i.id !== selected.id && i.status !== "ready"); if (next) { setQuery(""); setSelectedId(next.id); } else { setExportReviewer(`reviewer-${selected.reviewerId}`); setStage("export"); } }}><Check size={15} />{t("完成并继续", "Complete & continue")}<ArrowRight size={14} /></button></div>
      </section> : <section className="rst-empty"><MessageSquare size={30} /><h2>{t("暂无问题", "No concerns yet")}</h2><button className="rst-button rst-primary" disabled={!!busy || readOnly} onClick={addIssue}><Plus size={15} />{t("添加问题", "Add concern")}</button><button className="rst-text-button" onClick={openReviews}>{t("拆解审稿原文", "Analyze source reviews")}</button></section>}

    </div>}

    {stage === "experiments" && <section className="rst-experiments">
      <div className="rst-section-heading"><h2>{activeReviewer?.name} · {t("实验", "Experiments")}</h2><div className="rst-row-actions">
        <button className="rst-button" disabled={!!busy || readOnly || !!experimentDraft} onClick={() => startExperiment()}><Plus size={15} />{t("添加实验", "Add experiment")}</button>
        <button className="rst-button rst-primary" onClick={() => showStage("issues")}>{t("去写回复", "Write responses")}<ArrowRight size={15} /></button>
      </div></div>
      {experimentDraft && <form className="rst-experiment-create" onSubmit={e => { e.preventDefault(); createExperiment(); }}>
        <h3>{t("添加实验", "Add experiment")}</h3>
        <Field label={t("实验名称", "Experiment title")}><input autoFocus required disabled={readOnly} maxLength={2000} value={experimentDraft.title} onChange={e => setExperimentDraft({ ...experimentDraft, title: e.target.value })} /></Field>
        <Field label={t("关联问题", "Linked concern")}><select disabled={readOnly} value={experimentDraft.issueId} onChange={e => setExperimentDraft({ ...experimentDraft, issueId: e.target.value })}>
          {experimentIssues.map(issue => <option key={issue.id} value={issue.id}>{issue.title}</option>)}
          <option value="">{t("新建问题…", "New concern…")}</option>
        </select></Field>
        {!experimentDraft.issueId && <Field label={t("新问题标题", "New concern title")}><input required disabled={readOnly} maxLength={2000} value={experimentDraft.newIssueTitle} onChange={e => setExperimentDraft({ ...experimentDraft, newIssueTitle: e.target.value })} /></Field>}
        <div className="rst-row-actions"><button type="button" className="rst-button" onClick={() => setExperimentDraft(null)}>{t("取消", "Cancel")}</button><button type="submit" className="rst-button rst-primary" disabled={!!busy || readOnly || !experimentDraft.title.trim() || (!experimentDraft.issueId && !experimentDraft.newIssueTitle.trim())}><Plus size={15} />{t("添加", "Add")}</button></div>
      </form>}
      {!visibleExperiments.length ? <div className="rst-simple-empty"><FlaskConical size={30} /><h3>{t("暂无补充实验", "No experiments planned")}</h3></div> : <div className="rst-experiment-grid">{visibleExperiments.map((experiment, index) => {
        const issue = project.issues.find(i => i.id === experiment.issueId);
        return <article id={`experiment-${experiment.id}`} className={`rst-experiment-card ${experiment.status}`} key={experiment.id}>
          <div className="rst-experiment-top"><span>E{String(index + 1).padStart(2, "0")}</span><div className="rst-row-actions">
            <select disabled={readOnly} aria-label={t("实验状态", "Experiment status")} value={experiment.status} onChange={e => updateExperiment(experiment.id, { status: e.target.value as Experiment["status"] })}><option value="planned">{t("待开始", "Planned")}</option><option value="running">{t("进行中", "Running")}</option><option value="done">{t("已完成", "Complete")}</option></select>
            <button className="rst-text-button rst-inline-action rst-delete" disabled={!!busy || readOnly} aria-label={t(`删除实验：${experiment.title}`, `Delete experiment: ${experiment.title}`)} onClick={() => deleteContent("experiment", experiment.id)}><Trash2 size={14} />{t("删除", "Delete")}</button>
          </div></div>
          <input disabled={readOnly} maxLength={2000} className="rst-experiment-title" aria-label={t("实验名称", "Experiment title")} value={experiment.title} onChange={e => updateExperiment(experiment.id, { title: e.target.value })} />
          <button className="rst-experiment-link" onClick={() => { setSelectedId(experiment.issueId); setReviewerFilter(issue?.reviewerId || "all"); setQuery(""); setStage("issues"); }}><MessageSquare size={12} />{issue?.title}<ChevronRight size={12} /></button>
          <Field label={t("实验设计与对照", "PROTOCOL & BASELINES")}><textarea disabled={readOnly} maxLength={200000} rows={4} value={experiment.protocol} placeholder={t("数据、对照方法、变量、随机种子…", "Dataset, baselines, variables, seeds…")} onChange={e => updateExperiment(experiment.id, { protocol: e.target.value })} /></Field>
          <div className="rst-experiment-fields"><Field label={t("衡量指标", "METRICS")}><input disabled={readOnly} maxLength={10000} value={experiment.metric} placeholder="Accuracy / latency / …" onChange={e => updateExperiment(experiment.id, { metric: e.target.value })} /></Field><Field label={t("负责人", "OWNER")}><input disabled={readOnly} maxLength={300} value={experiment.owner} placeholder={t("待分配", "Unassigned")} onChange={e => updateExperiment(experiment.id, { owner: e.target.value })} /></Field></div>
          <Field label={t("实验结果与证据", "RESULTS & EVIDENCE")}><textarea disabled={readOnly} maxLength={200000} rows={4} value={experiment.result} placeholder={t("填写实验结果…", "Experiment results…")} onChange={e => updateExperiment(experiment.id, { result: e.target.value })} /></Field>
          {experiment.status === "done" && !experiment.result.trim() && <p className="rst-inline-warning">{t("请填写实验结果", "Add experiment results")}</p>}
        </article>;
      })}</div>}
    </section>}

    {stage === "export" && <section className="rst-export">
      <aside><h2>{t("拼接回复", "Assemble responses")}</h2>
        <Field label={t("导出范围", "INCLUDE")}><select value={exportReviewer} onChange={e => { setExportReviewer(e.target.value); explicitAssemblyScope.current = true; setLegacyHistory(false); }}><option value="all">{t("所有审稿人", "All reviewers")}</option>{project.reviews.map(r => <option value={`reviewer-${r.id}`} key={r.id}>{r.name}</option>)}</select></Field>
        <Field label={t("开场说明", "OPENING")}><textarea disabled={draftSharedReadOnly} maxLength={200000} rows={4} value={project.opening} placeholder="We thank the reviewers for their thoughtful feedback." onChange={e => updateProject({ opening: e.target.value })} /></Field>
        <Field label={t("结尾说明", "CLOSING")}><textarea disabled={draftSharedReadOnly} maxLength={200000} rows={3} value={project.closing} placeholder={t("总结澄清和修改…", "Summarize clarifications and revisions…")} onChange={e => updateProject({ closing: e.target.value })} /></Field>
        <Field label={t("自定义词数目标（0 = 不限）", "WORD TARGET (0 = UNLIMITED)")}><input disabled={draftSharedReadOnly} type="number" min="0" max="100000" step="1" value={project.wordLimit} onChange={e => updateProject({ wordLimit: Math.floor(Math.max(0, Math.min(100000, Number(e.target.value) || 0))) })} /></Field>
        <div className="rst-export-meter"><strong className={project.wordLimit > 0 && words > project.wordLimit ? "over" : ""}>{words}<small>{project.wordLimit > 0 ? ` / ${project.wordLimit}` : ""} {t("词 / 中文字", "words / CJK characters")}</small></strong><span>{assembled.length} {t("字符（含 Markdown）", "characters including Markdown")}</span></div>
      </aside>
      <div className="rst-export-document">
        <div className="rst-export-toolbar"><div className="rst-assembly-status" role="status">{!draftAvailable ? t("个人草稿服务正在更新，请稍后刷新。", "Personal drafts are updating. Refresh shortly.") : draft.state.status === "loading" ? t("正在读取个人草稿…", "Loading personal draft…") : draft.state.status === "saving" ? t("个人草稿保存中…", "Saving personal draft…") : draft.state.status === "error" || draft.state.status === "conflict" ? t("个人草稿未保存", "Personal draft unsaved") : draft.state.source === null ? t("自动拼接", "Auto assembled") : draftCloud ? t("个人草稿已保存", "Personal draft saved") : t("个人草稿已保存到本机", "Personal draft saved locally")}</div><div>
          <button className="rst-button" disabled={!draftAvailable || !draft.state.ready || draft.state.status === "loading" || draft.state.blocked || draft.state.source === null} onClick={() => { if (draft.state.source !== generated && !window.confirm(t("重新拼接将替换当前个人改写，是否继续？", "Reassemble and replace your personal edits?"))) return; draft.store.update(null); }}>{t("重新拼接", "Reassemble")}</button>
          <button className="rst-button" disabled={!canSaveVersion} onClick={saveVersion}><History size={14} />{draftCloud ? t("保存共享版本", "Save shared version") : t("保存当前版本", "Save current version")}</button>
          <button className="rst-button" disabled={!assembled.trim()} onClick={() => download("openreview-response.md", toOpenReviewMarkdown(assembled), "text/markdown")}><ArrowDownToLine size={14} />.md</button>
        </div></div>
        {!versionsAvailable && <p className="rst-assembly-status rst-version-rollout" role="status">{t("历史版本正在更新，请稍后刷新。", "Versions are updating. Refresh shortly.")}</p>}
        {(draft.state.error || draft.state.recoveryError) && <div className="rst-assembly-error" role="alert"><p>{assemblyErrorText(draft.state, locale)}</p><div>
          {!draft.state.blocked && draft.state.status !== "conflict" && <button className="rst-button" onClick={() => void (draft.state.ready ? draft.store.flush() : draft.store.load())}>{t("重试保存", "Retry save")}</button>}
          {draftCloud && <button className="rst-button" disabled={draft.state.status === "saving" || draft.state.status === "loading"} onClick={() => { if (draft.state.dirty && !window.confirm(t("加载已保存版本将替换当前个人改写，是否继续？", "Load the saved draft and replace your personal edits?"))) return; void draft.store.load(true); }}>{t("加载已保存版本", "Load saved draft")}</button>}
        </div></div>}
        {pending > 0 && <div className="rst-export-warning"><ClipboardList size={16} />{t(`${pending} 条问题未回复`, `${pending} unanswered concerns`)}</div>}
        <RebuttalEditor key={`${hosted ? accountUser?.id : user.id}:${project.id}:${exportReviewer}`} assembly value={assembled} onChange={value => { draft.store.update(value); }} readOnly={!draftAvailable || !draft.state.ready || draft.state.status === "loading" || draft.state.blocked} locale={locale} />
      </div>
    </section>}

    </>}
    {removed && inPaper && removed.ownerId === user.id && removed.projectId === project.id && <div className="rst-undo" role="status"><span>{removed.issue ? t("已删除问题及关联实验", "Concern and linked experiments deleted") : t("已删除实验", "Experiment deleted")}</span><button disabled={!!busy || readOnly} onClick={undoRemoval}>{t("撤销", "Undo")}</button><button aria-label={t("关闭撤销提示", "Dismiss undo notice")} onClick={() => setRemoved(null)}><X size={14} /></button></div>}

    {notice && <div className="rst-toast" role="status">{notice}<button aria-label={t("关闭通知", "Dismiss notification")} onClick={() => setNotice("")}><X size={14} /></button></div>}

    {settings && <div className="rst-overlay"><section className="rst-modal rst-settings" role="dialog" aria-modal="true" aria-labelledby="rst-settings-title"><header><h2 id="rst-settings-title">{t("项目设置", "Project settings")}</h2><button className="rst-icon-button" aria-label={t("关闭", "Close")} onClick={() => setSettings(false)}><X size={20} /></button></header><Field label={t("论文标题", "Paper title")}><input disabled={readOnly} maxLength={2000} value={project.title} onChange={e => updateProject({ title: e.target.value })} /></Field><Field label={t("会议 / 期刊", "Venue")}><input disabled={readOnly} maxLength={300} placeholder="ICLR / ICML / NeurIPS / …" value={project.venue} onChange={e => updateProject({ venue: e.target.value })} /></Field><Field label={t("论文背景与核心贡献", "Paper context and contributions")}><textarea disabled={readOnly} maxLength={30000} rows={5} value={project.context} placeholder={t("提供摘要、方法概要与已有实验，帮助 AI 理解论文。", "Provide an abstract, method summary, and existing experiments for AI context.")} onChange={e => updateProject({ context: e.target.value })} /></Field>{hosted ? <div className="rst-ai-config"><h3><Sparkles size={16} />{ai.configured ? t("AI 已启用", "AI enabled") : t("AI 服务尚未启用", "AI service is not enabled")}</h3></div> : <details className="rst-ai-config"><summary>{t("AI 设置", "AI settings")}</summary><h3><Sparkles size={16} />{ai.configured ? `${t("已配置", "Configured")}: ${ai.model}` : t("连接模型服务", "Connect a model service")}</h3><p>{t("在启动本地服务的终端配置兼容 Chat Completions 的接口，然后重启 make dev。密钥只保留在服务端；点击 AI 功能时会发送相关原文与背景。", "Configure a Chat Completions compatible endpoint in the terminal that starts the local service, then restart make dev. Keys stay on the server. AI actions send the relevant reviews and context to that provider.")}</p><pre>{"export PAPERLEET_REBUTTAL_API_URL=\"https://your-provider/v1/chat/completions\"\nexport PAPERLEET_REBUTTAL_MODEL=\"your-model\"\nexport PAPERLEET_REBUTTAL_API_KEY=\"your-key\"\nmake dev"}</pre></details>}<a className="rst-attribution" href="https://github.com/runtsang/RebuttalStudio" target="_blank" rel="noreferrer">RebuttalStudio ↗</a><footer><button className="rst-button rst-primary" onClick={() => setSettings(false)}><Check size={15} />{t("完成", "Done")}</button></footer></section></div>}
  </main>;
}
