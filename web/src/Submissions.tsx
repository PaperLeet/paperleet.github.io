import AppLink from './AppLink';
import { PublishingRules } from './PublishingRules';
import { currentQuotaMessage } from './publicationQuota';
import { apiFetch } from "./apiClient";
import { useEffect, useRef, useState } from "react";
import {
  ArrowLeft,
  ArrowRight,
  BookOpen,
  CheckCircle2,
  ChevronRight,
  Code2,
  ExternalLink,
  FlaskConical,
  LoaderCircle,
  Plus,
  RefreshCw,
  ShieldCheck,
  XCircle,
} from "lucide-react";
import { useI18n } from "./i18n";
import { useAccount } from "./social";
import { signInHref } from "./accountNavigation";
import { Highlight } from "./SyntaxHighlight";
import { Prose } from "./RichText";
import { ArtifactCharts } from "./Visuals";
import { tabDraftStorage } from "./communityDrafts";
import { clearSubmittedProposalDraft, emptyProposalDraft, proposalDraftOwner, readProposalDraft, writeProposalDraft, type ProposalDraft } from "./proposalDrafts";
import PaperPool from './PaperPool';
import { hostedEdition } from './deployment';
import './lc-paper.css';

type Candidate = {
  title: string;
  title_en: string;
  difficulty: string;
  summary: string;
  readme: string;
  readme_en: string;
  starter: string;
  solution: string;
  notes: string;
  notes_en: string;
  functions: { name: string; signature: string }[];
  evidence: { claim: string; source_url: string; locator: string }[];
  tests: { name: string; layer: string; code: string }[];
  mutants: { name: string; bug: string; code: string }[];
};
type Submission = {
  id: string;
  title: string;
  paper_url: string;
  code_url?: string;
  excerpt?: string;
  focus?: string;
  status: string;
  created_at: string;
  updated_at: string;
  error?: string;
  package_hash?: string;
  candidate?: Candidate;
  validation?: {
    passed: boolean;
    status: string;
    package_hash: string;
    checks: { name: string; passed: boolean; detail: string }[];
    output?: string;
    runs?: Record<
      string,
      {
        exit_code: number;
        error?: string;
        output?: string;
        tests: {
          id: string;
          layer: string;
          passed: boolean;
          message?: string;
          why?: string;
          artifacts?: Record<string, unknown>;
        }[];
      }
    >;
  };
  reviews: {
    decision: string;
    reviewer: string;
    notes: string;
    created_at?: string;
  }[];
  published_problem?: { id: string; number: number };
};
type Capabilities = {
  mode?: "local" | "hosted";
  can_submit?: boolean;
  can_generate?: boolean;
  can_review?: boolean;
  reason?: string;
  generation: {
    ready: boolean;
    provider: string;
    model: string;
    reason?: string;
  };
  sandbox: { ready: boolean; image: string; reason?: string };
  human_review_required: boolean;
};
type Selection = { id?: string };

// Normalize the API's UTC spelling without dropping its microsecond precision.
function submissionVersion(value: string) {
  const match = /^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,6}))?(?:Z|\+00:00)$/.exec(value);
  return match ? `${match[1]}.${(match[2] || "").padEnd(6, "0")}` : undefined;
}

const EMPTY_CHECKS = {
  paper_fidelity: false,
  scope_clarity: false,
  test_independence: false,
  license_checked: false,
};
const busyStatus = (s?: string) => s === "generating" || s === "validating";
const safeUrl = (s: string) => (/^https?:\/\//i.test(s) ? s : undefined);
async function api<T>(path: string, body?: unknown): Promise<T> {
  const response = await apiFetch(
    `/api/submissions${path}`,
    body === undefined
      ? { cache: "no-store" }
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
  );
  const text = await response.text();
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    throw Error(
      "Submission service is unavailable. Please refresh after the service is running.",
    );
  }
  if (!response.ok)
    throw Error(currentQuotaMessage(data) || data.error || `Request failed (${response.status})`);
  return data;
}
export default function Submissions({onPublished}: {onPublished: () => void}) {
  const {health} = useAccount();
  if (hostedEdition || health?.mode === 'hosted') return <PaperPool />;
  return <LegacySubmissions onPublished={onPublished} />;
}

export function LegacySubmissions({onPublished}: {onPublished: () => void}) {
  const {user, health} = useAccount();
  const owner = proposalDraftOwner(health?.mode, user?.id);
  return <SubmissionWorkspace key={owner || "unresolved"} draftOwner={owner} onPublished={onPublished} />;
}

function SubmissionWorkspace({onPublished, draftOwner}: {onPublished: () => void; draftOwner?: string}) {
  const { t, locale } = useI18n();
  const { user, health, capabilities: accountCapabilities } = useAccount();

  const [loadedDraft] = useState(() => readProposalDraft(tabDraftStorage(), draftOwner));
  const [draft, setDraft] = useState<ProposalDraft>(() => loadedDraft.record?.draft || emptyProposalDraft());
  const draftSource = useRef(draft);
  const storedRevision = useRef(loadedDraft.record?.revision);
  const [draftSaved, setDraftSaved] = useState(loadedDraft.available);
  const [draftRestored, setDraftRestored] = useState(!!loadedDraft.record);
  const [draftUncleared, setDraftUncleared] = useState(false);
  const {title, paperUrl, codeUrl, excerpt, focus} = draft;

  const [caps, setCaps] = useState<Capabilities>(),
    [items, setItems] = useState<Submission[]>([]),
    [selected, setSelected] = useState<Submission>(),
    [view, setView] = useState("new"),
    [error, setError] = useState(""),
    [loading, setLoading] = useState(true),
    [busy, setBusy] = useState(false),
    [candidateTab, setCandidateTab] = useState("problem"),
    [reviewer, setReviewer] = useState(""),
    [reviewNotes, setReviewNotes] = useState(""),
    [decision, setDecision] = useState("approve"),
    [checks, setChecks] = useState(EMPTY_CHECKS),
    [notice, setNotice] = useState(""),
    [selectionKey, setSelectionKey] = useState(0);
  const mounted = useRef(true);
  const selection = useRef<Selection>({});
  const working = useRef(false);
  const draftRevision = useRef(0);
  const listRequest = useRef(0);
  const itemsRevision = useRef(0);
  const latestItems = useRef(new Map<string, Submission>());
  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);
  const current = (token: Selection) => mounted.current && selection.current === token;
  function trackSelection(token: Selection) {
    selection.current = token;
    setSelectionKey((key) => key + 1);
  }
  function editDraft(change: Partial<ProposalDraft>) {
    draftRevision.current += 1;
    const next = {...draftSource.current, ...change};
    draftSource.current = next;
    setDraft(next);
    const saved = writeProposalDraft(tabDraftStorage(), draftOwner, next);
    storedRevision.current = saved.revision;
    setDraftSaved(saved.saved);
    setDraftRestored(false);
    setDraftUncleared(false);
  }
  function newSubmission() {
    trackSelection({});
    setSelected(undefined);
    setView("new");
    setError("");
    setNotice("");
  }
  const hosted = health?.mode === "hosted" || caps?.mode === "hosted";
  const canSubmit = !!draftOwner && (caps?.can_submit ?? health?.mode === "local");
  const canGenerate = caps?.can_generate ?? health?.mode === "local";
  const canReview = caps?.can_review ?? health?.mode === "local";
  const managedNotice = t(
    "内测版接收论文提案；生成、验证和发布由维护者在离线流程中完成。",
    "The beta accepts paper proposals. Generation, validation, and publication are managed by maintainers offline.",
  );
  const labels: Record<string, string> = {
    submitted: t("已提交", "Submitted"),
    generating: t("生成中", "Generating"),
    generation_failed: t("生成失败", "Generation failed"),
    validation_pending: t("待验证", "Validation pending"),
    validating: t("验证中", "Validating"),
    validation_failed: t("验证失败", "Validation failed"),
    needs_review: t("待人工评审", "Needs human review"),
    changes_requested: t("需要修改", "Changes requested"),
    rejected: t("已拒绝", "Rejected"),
    published: t("已进入题库", "Published"),
  };
  // A saved row and the currently viewed proposal have separate lifetimes.
  function rememberItem(item: Submission) {
    if (!mounted.current) return item;
    const known = latestItems.current.get(item.id);
    const version = submissionVersion(item.updated_at);
    const knownVersion = known && submissionVersion(known.updated_at);
    if (known && version && knownVersion && knownVersion > version) return known;
    latestItems.current.set(item.id, item);
    itemsRevision.current += 1;
    setItems((prev) =>
      [item, ...prev.filter((p) => p.id !== item.id)].sort((a, b) =>
        b.created_at.localeCompare(a.created_at),
      ),
    );
    return item;
  }
  function showMutation(item: Submission) {
    const latest = rememberItem(item);
    if (selection.current.id !== item.id) return;
    // A user may leave and return while the operation is pending. Its result
    // belongs to that proposal, while older detail reads are now superseded.
    trackSelection({id:item.id});
    setSelected(latest);
  }
  async function refresh() {
    const request = ++listRequest.current;
    const revision = itemsRevision.current;
    const active = () => mounted.current && request === listRequest.current;
    setLoading(true);
    setError("");
    try {
      const capabilities = await api<Capabilities>("/capabilities");
      if (!active()) return;
      setCaps(capabilities);
      if (capabilities.can_submit === false) {
        setItems([]);
        return;
      }
      const list = await api<{ submissions: Submission[] }>("");
      // An older list must not erase a proposal saved while it was loading.
      if (active() && itemsRevision.current === revision) setItems(list.submissions);
    } catch (e) {
      if (active()) setError(e instanceof Error ? e.message : t("无法加载提交列表", "Unable to load submissions"));
    } finally {
      if (active()) setLoading(false);
    }
  }
  useEffect(() => { void refresh(); }, []);
  useEffect(() => {
    if (!selected || !busyStatus(selected.status)) return;
    const token = selection.current;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    const active = () => !cancelled && current(token);
    async function poll() {
      try {
        const item = await api<Submission>(`/${token.id}`);
        if (!active()) return;
        setSelected(rememberItem(item));
      } catch (e) {
        if (active()) setError(e instanceof Error ? e.message : t("无法加载提交详情", "Unable to load submission"));
      }
      // Wait after the response; slow requests must not overlap each other.
      if (active()) timer = setTimeout(() => { void poll(); }, 2000);
    }
    timer = setTimeout(() => { void poll(); }, 2000);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [selected?.id, selected?.status, selectionKey]);
  useEffect(() => {
    setChecks({ ...EMPTY_CHECKS });
    setReviewNotes("");
    setDecision("approve");
  }, [selected?.id, selected?.package_hash]);
  async function choose(item: Submission) {
    const token: Selection = {id:item.id};
    trackSelection(token);
    setView("detail");
    setSelected(latestItems.current.get(item.id) ?? item);
    setError("");
    setNotice("");
    setCandidateTab("problem");
    try {
      const detail = await api<Submission>(`/${item.id}`);
      if (!current(token)) return;
      setSelected(rememberItem(detail));
    } catch (e) {
      if (current(token)) setError(e instanceof Error ? e.message : t("无法加载提交详情", "Unable to load submission"));
    }
  }
  async function create(event: React.FormEvent) {
    event.preventDefault();
    if (!canSubmit || loading || working.current) return;
    working.current = true;
    const revision = draftRevision.current;
    let token = selection.current;
    const submitted = {...draftSource.current};
    const submittedStorageRevision = storedRevision.current;
    const payload = {title:submitted.title, paper_url:submitted.paperUrl, code_url:submitted.codeUrl, excerpt:submitted.excerpt, focus:submitted.focus};
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const item = await api<Submission>("", payload);
      if (!mounted.current) return;
      rememberItem(item);
      const unchangedDraft = draftRevision.current === revision;
      const cleared = clearSubmittedProposalDraft(tabDraftStorage(), draftOwner, submittedStorageRevision, submitted);
      if (unchangedDraft) {
        const empty = emptyProposalDraft();
        draftSource.current = empty;
        storedRevision.current = undefined;
        setDraft(empty);
        setDraftRestored(false);
        setDraftSaved(cleared === "cleared");
        setDraftUncleared(cleared !== "cleared");
      }
      if (current(token) && unchangedDraft) {
        token = {id:item.id};
        trackSelection(token);
        setSelected(item);
        setView("detail");
      } else {
        setNotice(item.title);
      }
      if (canGenerate && caps?.generation.ready) {
        const generated = await api<Submission>(`/${item.id}/generate`, {});
        if (!mounted.current) return;
        showMutation(generated);
      }
    } catch (e) {
      if (current(token)) setError(e instanceof Error ? e.message : t("提交失败", "Submission failed"));
    } finally {
      working.current = false;
      if (mounted.current) setBusy(false);
    }
  }
  async function act(action: "generate" | "validate" | "review") {
    if (!selected || working.current || (action === "review" ? !canReview : !canGenerate)) return;
    working.current = true;
    // Invalidate a detail/poll request that started before this mutation.
    const token: Selection = {id:selected.id};
    trackSelection(token);
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const body = action === "review"
        ? {decision, reviewer:reviewer.trim(), notes:reviewNotes.trim(), package_hash:selected.package_hash, checks}
        : {};
      const item = await api<Submission>(`/${selected.id}/${action}`, body);
      if (!mounted.current) return;
      showMutation(item);
      if (item.status === "published") onPublished();
    } catch (e) {
      if (current(token)) setError(e instanceof Error ? e.message : t("操作失败", "Action failed"));
    } finally {
      working.current = false;
      if (mounted.current) setBusy(false);
    }
  }
  const candidate = selected?.candidate,
    running = busy || busyStatus(selected?.status);
  const validated =
    selected?.validation?.passed === true &&
    selected.validation.package_hash === selected.package_hash;
  const canApprove = validated && Object.values(checks).every(Boolean);
  const reviewable =
    canReview &&
    !!candidate &&
    !running &&
    selected?.status !== "published" &&
    selected?.status !== "rejected";
  return (
    <main className="submissions-page">
      <div className="submission-heading">
        <div>
          <AppLink className="back-link" href="#">
            <ArrowLeft size={15} />
            {t("返回题库", "Back to problems")}
          </AppLink>
          <h1>{t("提交论文", "Submit paper")}</h1>
        </div>
        <button
          className="button subtle"
          onClick={newSubmission}
        >
          <Plus size={16} />
          {t("新提交", "New submission")}
        </button>
      </div>
      <div className="submission-pipeline">
        {[
          t("提交来源", "Submit sources"),
          t("生成代码与测试", "Generate code & tests"),
          t("隔离验证", "Sandbox validation"),
          t("人工评审入库", "Human review & publish"),
        ].map((label, i) => (
          <div key={i}>
            <span>{i + 1}</span>
            <b>{label}</b>
            {i < 3 && <ChevronRight size={14} />}
          </div>
        ))}
      </div>
      {error && (
        <div className="submission-error" role="alert">
          <XCircle size={18} />
          <div>{error}</div>
        </div>
      )}
      {notice && <p className="submission-muted" role="status">{t(`「${notice}」已保存，可从提交记录打开；当前页面和新编辑的内容已保留。`, `“${notice}” was saved and is available in the submission list. Your current view and newer edits were kept.`)}</p>}
      <div className="submission-columns">
        <aside className="submission-list">
          <div className="aside-title">
            <h3>
              {hosted && user?.role !== "admin"
                ? t("我的提案", "My proposals")
                : t("提交记录", "Submissions")}
            </h3>
            <button
              className="icon-button"
              aria-label={t(
                "刷新提交和服务状态",
                "Refresh submissions and service status",
              )}
              onClick={refresh}
              disabled={loading}
            >
              <RefreshCw size={14} />
            </button>
          </div>
          {loading && !items.length ? (
            <p className="submission-muted">{t("加载中…", "Loading…")}</p>
          ) : !items.length ? (
            <p className="submission-muted">
              {t(
                "还没有提交。添加第一篇论文。",
                "No submissions yet. Add your first paper.",
              )}
            </p>
          ) : (
            items.map((item) => (
              <button
                key={item.id}
                className={`submission-list-item ${view === "detail" && selected?.id === item.id ? "selected" : ""}`}
                onClick={() => void choose(item)}
              >
                <strong>{item.title}</strong>
                <span className={`submission-status ${item.status}`}>
                  {labels[item.status] || item.status}
                </span>
                <small>
                  {new Date(item.created_at).toLocaleDateString(
                    locale === "zh" ? "zh-CN" : "en-US",
                  )}
                </small>
              </button>
            ))
          )}
          {caps && (
            <div className="capabilities">
              <h4>{t("服务状态", "Service status")}</h4>
              <div>
                <i
                  className={caps.generation.ready ? "green-dot" : "yellow-dot"}
                />
                {t("模型生成", "Model generation")}
                <b>
                  {!canGenerate
                    ? t("由维护者处理", "Maintainer managed")
                    : caps.generation.ready
                      ? t("就绪", "Ready")
                      : t("未配置", "Not configured")}
                </b>
              </div>
              <div>
                <i
                  className={caps.sandbox.ready ? "green-dot" : "yellow-dot"}
                />
                {t("隔离验证", "Sandbox")}
                <b>
                  {caps.sandbox.ready
                    ? t("就绪", "Ready")
                    : t("不可用", "Unavailable")}
                </b>
              </div>
              <details>
                <summary>{t("配置详情", "Configuration details")}</summary>
                <p>{caps.generation.model}</p>
                {!caps.generation.ready && canGenerate && (
                  <p>
                    {caps.generation.reason ||
                      t(
                        "请配置模型服务后重试生成。",
                        "Configure the model service, then retry generation.",
                      )}
                  </p>
                )}
                {!caps.sandbox.ready && canGenerate && (
                  <p>
                    {caps.sandbox.reason ||
                      t(
                        "请启动验证沙箱后重试。",
                        "Start the validation sandbox and retry.",
                      )}
                  </p>
                )}
              </details>
            </div>
          )}
        </aside>
        <div className="submission-main">
          {view === "new" ? (
            <section className="submission-card">
              <h2>{t("论文来源", "Paper sources")}</h2>
              <p className="submission-muted">
                {hosted
                  ? t(
                      "分享论文来源和你希望练习的核心模块，维护者将审核提案。",
                      "Share the paper sources and the core modules you would like to practice. Maintainers will review your proposal.",
                    )
                  : t(
                      "模型会归纳方法、接口约定及证据位置，生成初始代码、参考解、测试和错误变体。",
                      "The model extracts the method, interface conventions, and evidence, then drafts starter code, a reference, tests, and faulty variants.",
                    )}
              </p>
              {hosted && <p className="submission-muted">{managedNotice}</p>}
              {!canSubmit && (
                <p className="submission-muted">
                  {loading ? (
                    t("正在检查提交权限…", "Checking submission access…")
                  ) : (
                    <AppLink className="button subtle" href={signInHref("submit")}>
                      {accountCapabilities?.auth_mode === "test"
                        ? t(
                            "内测账号登录后可提交提案",
                            "Sign in with your test account to submit",
                          )
                        : t(
                            "受邀用户登录后可提交提案",
                            "Sign in with an invited account to submit",
                          )}
                    </AppLink>
                  )}
                </p>
              )}
              {canSubmit && (
                <form className="submission-form" onSubmit={create}>
                  <p className="submission-muted" role="status">{draftUncleared
                    ? t("提交已保存；标签页中仍保留一份草稿，刷新后请核对提交记录。", "Your submission was saved, but a draft remains in this tab. Check the submission list after refreshing.")
                    : !draftSaved
                      ? t("此标签页暂时无法保存草稿；离开或刷新前，请先复制内容。", "This tab cannot save the draft right now. Copy your text before leaving or refreshing.")
                      : draftRestored
                        ? t("已恢复此标签页保存的草稿，继续编辑会自动保存。", "The draft saved in this tab was restored. Further edits are saved automatically.")
                        : t("草稿按账号保存在此标签页，刷新或返回此页可恢复。", "Drafts stay in this tab, separately for each account. Refreshing or returning to this page restores them.")}</p>
                  <label>
                    {t("论文标题", "Paper title")}
                    <input
                      required
                      maxLength={240}
                      value={title}
                      onChange={(e) => editDraft({title: e.target.value})}
                      placeholder={t("输入论文标题", "Enter the paper title")}
                    />
                  </label>
                  <label>
                    {t("论文链接", "Paper URL")}
                    <input
                      type="url"
                      required
                      value={paperUrl}
                      maxLength={2048}
                      onChange={(e) => editDraft({paperUrl: e.target.value})}
                      placeholder="https://arxiv.org/abs/…"
                    />
                  </label>
                  <label>
                    {t("官方代码链接（可选）", "Official code URL (optional)")}
                    <input
                      type="url"
                      value={codeUrl}
                      maxLength={2048}
                      onChange={(e) => editDraft({codeUrl: e.target.value})}
                      placeholder="https://github.com/…"
                    />
                  </label>
                  <label>
                    {t("练习重点（可选）", "Practice focus (optional)")}
                    <textarea
                      rows={3}
                      maxLength={4000}
                      value={focus}
                      onChange={(e) => editDraft({focus: e.target.value})}
                      placeholder={t(
                        "例如：只提炼核心 loss，并验证梯度、mask 和归一化。",
                        "For example: extract the core loss and verify gradients, masks, and normalization.",
                      )}
                    />
                  </label>
                  <details>
                    <summary>
                      {t(
                        "补充论文摘录（可选）",
                        "Add a paper excerpt (optional)",
                      )}
                    </summary>
                    <label>
                      <span className="sr-only">
                        {t("论文摘录", "Paper excerpt")}
                      </span>
                      <textarea
                        rows={7}
                        maxLength={60000}
                        value={excerpt}
                        onChange={(e) => editDraft({excerpt: e.target.value})}
                        placeholder={t(
                          "粘贴核心公式、伪代码和相关段落，帮助模型定位。",
                          "Paste key formulas, pseudocode, or relevant passages.",
                        )}
                      />
                    </label>
                  </details>
                  {caps?.mode === "hosted" && <PublishingRules action="paper" />}
                  <div className="review-required">
                    <ShieldCheck size={18} />
                    <p>
                      {t(
                        "生成内容先进入候选区。只有验证通过并由人工评审确认后，才会加入题库。",
                        "Generated content stays in the candidate queue until validation passes and a human reviewer approves it.",
                      )}
                    </p>
                  </div>
                  {caps && !caps.generation.ready && canGenerate && (
                    <p className="submission-muted">
                      {t(
                        "当前可以保存提交；配置模型服务后，可从提交详情继续生成。",
                        "Save a submission now and generate it from its detail page once the model service is configured.",
                      )}
                    </p>
                  )}
                  <button
                    className="button"
                    type="submit"
                    disabled={busy || loading || !canSubmit}
                  >
                    {busy ? (
                      <LoaderCircle size={16} className="spinning" />
                    ) : (
                      <ArrowRight size={16} />
                    )}
                    {canGenerate && caps?.generation.ready
                      ? t("提交并生成候选题", "Submit & generate candidate")
                      : t("保存论文提交", "Save paper submission")}
                  </button>
                </form>
              )}
            </section>
          ) : selected ? (
            <>
              <section className="submission-card">
                <div className="submission-detail-heading">
                  <h2>{selected.title}</h2>
                  <span className={`submission-status ${selected.status}`}>
                    {labels[selected.status] || selected.status}
                  </span>
                </div>
                <div className="submission-sources">
                  <AppLink
                    href={safeUrl(selected.paper_url)}
                    target="_blank"
                    rel="noreferrer"
                  >
                    <BookOpen size={14} />
                    {t("论文", "Paper")}
                    <ExternalLink size={12} />
                  </AppLink>
                  {selected.code_url && (
                    <AppLink
                      href={safeUrl(selected.code_url)}
                      target="_blank"
                      rel="noreferrer"
                    >
                      <Code2 size={14} />
                      {t("官方代码", "Official code")}
                      <ExternalLink size={12} />
                    </AppLink>
                  )}
                </div>
                {selected.focus && (
                  <p className="submission-muted">{selected.focus}</p>
                )}
                {selected.error && (
                  <div className="submission-error" role="alert">
                    {selected.error}
                  </div>
                )}
                {selected.status === "published" &&
                selected.published_problem ? (
                  <AppLink
                    className="button"
                    href={`#/${selected.published_problem.number}`}
                  >
                    <CheckCircle2 size={16} />
                    {t("打开题目", "Open problem")}{" "}
                    {selected.published_problem.number}
                    <ArrowRight size={15} />
                  </AppLink>
                ) : (
                  <div className="submission-actions">
                    {canGenerate && selected.status !== "rejected" && (
                      <button
                        className="button subtle"
                        disabled={running || !caps?.generation.ready}
                        onClick={() => void act("generate")}
                      >
                        {selected.status === "generating" ? (
                          <LoaderCircle size={15} className="spinning" />
                        ) : (
                          <Code2 size={15} />
                        )}
                        {candidate
                          ? t("重新生成候选题", "Regenerate candidate")
                          : t("生成代码与测试", "Generate code & tests")}
                      </button>
                    )}
                    {canGenerate &&
                      candidate &&
                      selected.status !== "rejected" && (
                        <button
                          className="button"
                          disabled={running || !caps?.sandbox.ready}
                          onClick={() => void act("validate")}
                        >
                          {selected.status === "validating" ? (
                            <LoaderCircle size={15} className="spinning" />
                          ) : (
                            <FlaskConical size={15} />
                          )}
                          {t("运行验证", "Run validation")}
                        </button>
                      )}
                  </div>
                )}
                {busyStatus(selected.status) && (
                  <p className="submission-muted" role="status">
                    {t(
                      "正在处理，页面会自动更新。",
                      "Processing. This page will update automatically.",
                    )}
                  </p>
                )}
                {!candidate && !busyStatus(selected.status) && (
                  <p className="submission-muted">
                    {!canGenerate
                      ? managedNotice
                      : caps?.generation.ready
                        ? t(
                            "生成后可查看题面、参考代码、测试和来源证据。",
                            "Generate to inspect the problem, reference code, tests, and source evidence.",
                          )
                        : t(
                            "提交已保存。模型服务配置完成后，刷新服务状态并继续生成。",
                            "Submission saved. Configure the model service, refresh its status, and continue generation.",
                          )}
                  </p>
                )}
              </section>
              {candidate && (
                <section className="submission-card candidate-card">
                  <div className="aside-title">
                    <h2>
                      {locale === "en"
                        ? candidate.title_en || candidate.title
                        : candidate.title}
                    </h2>
                    <span>
                      {candidate.difficulty === "E"
                        ? "Easy"
                        : candidate.difficulty === "M"
                          ? "Medium"
                          : "Hard"}
                    </span>
                  </div>
                  <p className="submission-muted">{candidate.summary}</p>
                  <div className="pane-tabs candidate-tabs">
                    {[
                      ["problem", t("题面", "Problem")],
                      ["code", t("代码", "Code")],
                      ["tests", t("测试", "Tests")],
                      ["mutants", t("错误变体", "Mutants")],
                      ["evidence", t("来源证据", "Evidence")],
                    ].map(([key, label]) => (
                      <button
                        key={key}
                        className={candidateTab === key ? "selected" : ""}
                        onClick={() => setCandidateTab(key)}
                      >
                        {label}
                      </button>
                    ))}
                  </div>
                  {candidateTab === "problem" ? (
                    <Prose
                      content={
                        locale === "en"
                          ? candidate.readme_en || candidate.readme
                          : candidate.readme
                      }
                    />
                  ) : candidateTab === "code" ? (
                    <div className="candidate-files">
                      <details open>
                        <summary>starter.py</summary>
                        <Highlight code={candidate.starter} />
                      </details>
                      <details>
                        <summary>solution.py</summary>
                        <Highlight code={candidate.solution} />
                      </details>
                      <details>
                        <summary>
                          {t("坑点说明", "Implementation notes")}
                        </summary>
                        <Prose
                          content={
                            locale === "en"
                              ? candidate.notes_en || candidate.notes
                              : candidate.notes
                          }
                        />
                      </details>
                    </div>
                  ) : candidateTab === "tests" ? (
                    <div className="candidate-files">
                      {candidate.tests.map((test, i) => (
                        <details key={i}>
                          <summary>
                            <b>{test.layer}</b> {test.name}
                          </summary>
                          <Highlight code={test.code} />
                        </details>
                      ))}
                    </div>
                  ) : candidateTab === "mutants" ? (
                    <div className="candidate-files">
                      {candidate.mutants.map((mutant, i) => (
                        <details key={i}>
                          <summary>{mutant.name}</summary>
                          <p>{mutant.bug}</p>
                          <Highlight code={mutant.code} />
                        </details>
                      ))}
                    </div>
                  ) : (
                    <div className="candidate-evidence">
                      {candidate.evidence.map((source, i) => (
                        <article key={i}>
                          <b>{source.claim}</b>
                          <p>{source.locator}</p>
                          <AppLink
                            href={safeUrl(source.source_url)}
                            target="_blank"
                            rel="noreferrer"
                          >
                            {source.source_url}
                            <ExternalLink size={12} />
                          </AppLink>
                        </article>
                      ))}
                      {!candidate.evidence.length && (
                        <p>
                          {t(
                            "尚未提供来源证据。",
                            "No source evidence was provided.",
                          )}
                        </p>
                      )}
                    </div>
                  )}
                </section>
              )}
              {selected.validation && (
                <section className="submission-card">
                  <div className="aside-title">
                    <h2>{t("验证结果", "Validation results")}</h2>
                    <span
                      className={
                        validated ? "validation-pass" : "validation-fail"
                      }
                    >
                      {validated
                        ? t("通过", "Passed")
                        : t("未通过", "Not passed")}
                    </span>
                  </div>
                  <p className="submission-muted">
                    {t(
                      "验证结果绑定当前候选版本。重新生成后，需要重新验证和评审。",
                      "Results are bound to this candidate version. Regenerating requires new validation and review.",
                    )}
                  </p>
                  <div className="validation-checks">
                    {selected.validation.checks.map((check, i) => (
                      <div key={i}>
                        {check.passed ? (
                          <CheckCircle2 size={17} />
                        ) : (
                          <XCircle size={17} />
                        )}
                        <div>
                          <b>{check.name}</b>
                          <p>{check.detail}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                  {selected.validation.runs && (
                    <div className="validation-runs">
                      {Object.entries(selected.validation.runs).map(
                        ([name, result]) => (
                          <details key={name} className="validation-output">
                            <summary>
                              {name} ·{" "}
                              {result.tests.filter((row) => row.passed).length}{" "}
                              / {result.tests.length} {t("项通过", "passed")}
                            </summary>
                            {result.error && <pre>{result.error}</pre>}
                            {result.tests.map((row, i) => (
                              <details
                                key={i}
                                className="test-detail"
                                open={!row.passed}
                              >
                                <summary>
                                  {row.passed ? (
                                    <CheckCircle2 size={14} />
                                  ) : (
                                    <XCircle size={14} />
                                  )}
                                  <span>{row.id}</span>
                                  <b>{row.layer}</b>
                                </summary>
                                {row.why && <p>{row.why}</p>}
                                {row.message && <pre>{row.message}</pre>}
                                {row.artifacts && <ArtifactCharts data={row.artifacts} referenceRun={name.startsWith("reference")} implementationLabel={name} />}
                              </details>
                            ))}
                            {result.output && <pre>{result.output}</pre>}
                          </details>
                        ),
                      )}
                    </div>
                  )}
                  {selected.validation.output && (
                    <details className="validation-output">
                      <summary>
                        {t("查看验证日志", "View validation output")}
                      </summary>
                      <pre>{selected.validation.output}</pre>
                    </details>
                  )}
                </section>
              )}
              {candidate && (
                <section className="submission-card">
                  <div className="aside-title">
                    <h2>{t("人工评审", "Human evaluation")}</h2>
                    <ShieldCheck size={20} />
                  </div>
                  <p className="submission-muted">
                    {canReview
                      ? t(
                          "请逐项查看论文证据、实现和测试。模型不会替你勾选这些检查。",
                          "Inspect the source evidence, implementation, and tests. These checks must be completed by a human reviewer.",
                        )
                      : managedNotice}
                  </p>
                  {!validated && selected.status !== "published" && (
                    <p className="review-blocked">
                      {t(
                        "候选题须先通过隔离验证，才能批准入库。",
                        "This candidate must pass sandbox validation before approval.",
                      )}
                    </p>
                  )}
                  {reviewable && (
                    <form
                      className="submission-form review-form"
                      onSubmit={(e) => {
                        e.preventDefault();
                        void act("review");
                      }}
                    >
                      <div className="review-checklist">
                        {(
                          [
                            [
                              "paper_fidelity",
                              t(
                                "论文依据准确，公式和约定可追溯",
                                "Paper fidelity: formulas and conventions are supported by the sources",
                              ),
                            ],
                            [
                              "scope_clarity",
                              t(
                                "题目范围和函数接口清楚，适合独立练习",
                                "Clear scope: the interface supports a focused, self-contained exercise",
                              ),
                            ],
                            [
                              "test_independence",
                              t(
                                "测试有独立依据，能捕获错误实现",
                                "Independent tests: assertions have their own basis and catch faulty implementations",
                              ),
                            ],
                            [
                              "license_checked",
                              t(
                                "已核对论文及代码来源和使用许可",
                                "Source and license review: paper and code provenance and usage are checked",
                              ),
                            ],
                          ] as [keyof typeof EMPTY_CHECKS, string][]
                        ).map(([key, label]) => (
                          <label key={key}>
                            <input
                              type="checkbox"
                              checked={checks[key]}
                              onChange={(e) =>
                                setChecks((prev) => ({
                                  ...prev,
                                  [key]: e.target.checked,
                                }))
                              }
                            />
                            {label}
                          </label>
                        ))}
                      </div>
                      <label>
                        {t("评审人", "Reviewer")}
                        <input
                          required
                          maxLength={120}
                          value={reviewer}
                          onChange={(e) => setReviewer(e.target.value)}
                          placeholder={t(
                            "你的姓名或标识",
                            "Your name or identifier",
                          )}
                        />
                      </label>
                      <label>
                        {t("评审结论与依据", "Review notes and rationale")}
                        <textarea
                          required
                          rows={4}
                          maxLength={8000}
                          value={reviewNotes}
                          onChange={(e) => setReviewNotes(e.target.value)}
                          placeholder={t(
                            "说明已核对的证据、测试覆盖或需要修改的内容。",
                            "Describe the evidence checked, test coverage, or changes required.",
                          )}
                        />
                      </label>
                      <label>
                        {t("处理方式", "Decision")}
                        <select
                          value={decision}
                          onChange={(e) => setDecision(e.target.value)}
                        >
                          <option value="approve">
                            {t("批准并加入题库", "Approve and publish")}
                          </option>
                          <option value="changes_requested">
                            {t("退回修改", "Request changes")}
                          </option>
                          <option value="reject">
                            {t("拒绝入库", "Reject")}
                          </option>
                        </select>
                      </label>
                      <button
                        className="button"
                        disabled={
                          running ||
                          (decision === "approve" && !canApprove) ||
                          !reviewer.trim() ||
                          !reviewNotes.trim()
                        }
                        type="submit"
                      >
                        <ShieldCheck size={16} />
                        {decision === "approve"
                          ? t("批准并加入题库", "Approve & publish")
                          : t("提交评审结论", "Submit review")}
                      </button>
                    </form>
                  )}
                  {selected.reviews?.length > 0 && (
                    <div className="review-history">
                      {selected.reviews.map((review, i) => (
                        <article key={i}>
                          <b>{review.reviewer}</b>
                          <span>
                            {review.decision === "approve"
                              ? t("批准", "Approved")
                              : review.decision === "reject"
                                ? t("拒绝", "Rejected")
                                : t("退回修改", "Changes requested")}
                          </span>
                          <p>{review.notes}</p>
                        </article>
                      ))}
                    </div>
                  )}
                </section>
              )}
            </>
          ) : null}
        </div>
      </div>
    </main>
  );
}
