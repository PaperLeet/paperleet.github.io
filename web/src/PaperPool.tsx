import { useEffect, useRef, useState } from 'react';
import { ArrowLeft, ArrowUpRight, BookOpen, CheckCircle2, Code2, FlaskConical, LoaderCircle, MessageSquare, Play, RefreshCw, Send, ThumbsUp } from 'lucide-react';
import { useI18n } from './i18n';
import { Avatar, errorMessage, profileHref, useAccount } from './social';
import { signInHref } from './accountNavigation';
import AffiliationBadge from './AffiliationBadge';
import { Prose } from './RichText';
import { PublishingRules } from './PublishingRules';
import { tabDraftStorage } from './communityDrafts';
import { clearSubmittedProposalDraft, emptyProposalDraft, readProposalDraft, writeProposalDraft, type ProposalDraft } from './proposalDrafts';
import { deferredFeature } from './deferredFeature';
import { EditorFailure, EditorLoading } from './EditorFallback';
import { arxivPaperUrl, poolApi, poolHref, poolRoute } from './paperPoolApi';
import type { PaperPoolCapabilities, PaperPoolVote, PoolAttempt, PoolAttemptResponse, PoolDuplicate, PoolList, PoolProposal, PoolSubmission } from './paperPoolTypes';
import './paper-pool.css';

const TrialEditor = deferredFeature(() => import('./PythonEditor'), ['试做编辑器', 'trial editor'], {
  loading: props => <EditorLoading {...props} />,
  failure: (props, retry) => <EditorFailure {...props} retry={retry} />,
});
const processing = (status: string) => ['queued', 'submitted', 'suitability', 'screening', 'generation', 'generating', 'validation', 'validating'].includes(status);
const reviewable = (status: string) => ['pool', 'ready'].includes(status);

export default function PaperPool() {
  const {user} = useAccount();
  return <PoolWorkspace key={user?.id || 'guest'} owner={user?.id} />;
}

function PoolWorkspace({owner}: {owner?: string}) {
  const {t, locale} = useI18n();
  const [route, setRoute] = useState(() => poolRoute(location.hash));
  const [data, setData] = useState<PoolList>();
  const [error, setError] = useState('');
  const [revision, setRevision] = useState(0);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(0), [loadedPage, setLoadedPage] = useState(0);
  const listWorking = useRef(false);
  useEffect(() => {
    const navigate = () => setRoute(poolRoute(location.hash));
    window.addEventListener('hashchange', navigate);
    return () => window.removeEventListener('hashchange', navigate);
  }, []);
  useEffect(() => {
    let active = true;
    const controller = new AbortController();
    listWorking.current = true; setLoading(true); setError('');
    void poolApi<PoolList>(page ? `?limit=20&offset=${page * 20}` : '', undefined, controller.signal).then(value => {
      if (!Array.isArray(value.items) || !value.capabilities) throw Error(t('候选题服务暂不可用。', 'The paper pool service is unavailable.'));
      if (active) {setData(value); setLoadedPage(page); if (page && value.total !== undefined && page * 20 >= value.total) setPage(Math.max(0, Math.ceil(value.total / 20) - 1));}
    }).catch(e => {if (active) setError(errorMessage(e));}).finally(() => {if (active) {listWorking.current = false; setLoading(false);}});
    return () => {active = false; controller.abort();};
  }, [revision, page]);
  const refresh = () => {listWorking.current = true; setLoading(true); setRevision(value => value + 1);};
  function changePage(next: number) {if (listWorking.current) return; listWorking.current = true; setLoading(true); setPage(next);}
  const items = loadedPage === page ? data?.items || [] : [];
  const total = data?.total ?? data?.items.length ?? 0;
  return <main className="social-page paper-pool-page">
    <header className="pool-page-header">
      <h1>{route.id || route.pool ? t('候选题', 'Problem pool') : t('提交论文', 'Submit paper')}</h1>
      <nav className="pool-tabs" aria-label={t('论文投稿导航', 'Paper submission navigation')}>
        <a href="#/submit" aria-current={!route.id && !route.pool ? 'page' : undefined}><Send size={15} />{t('提交论文', 'Submit paper')}</a>
        <a href={poolHref()} aria-current={route.id || route.pool ? 'page' : undefined}><FlaskConical size={16} />{t('候选题', 'Problem pool')}</a>
      </nav>
    </header>
    {route.id ? <PoolDetail key={route.id} id={route.id} owner={owner} capabilities={data?.capabilities} onChanged={refresh} /> : <>
      {error && <div className="pool-error" role="alert">{error}<button className="text-button" onClick={refresh}>{t('重试', 'Retry')}</button></div>}
      {!route.pool && <PoolSubmissionForm owner={owner} capabilities={data?.capabilities} onSaved={refresh} />}
      <section className="pool-list-section" aria-labelledby="pool-list-title">
        <div className="pool-section-heading"><h2 id="pool-list-title">{route.pool ? t('公开投稿与试做', 'Public proposals & trials') : t('最近的候选题', 'Recent candidates')}</h2>
          <button className="icon-button" type="button" onClick={refresh} disabled={loading} aria-label={t('刷新候选题', 'Refresh pool')}><RefreshCw size={16} /></button></div>
        {loading && (!data || loadedPage !== page) ? <div className="social-loading" role="status"><span className="spinner" />{t('加载中…', 'Loading…')}</div> : !items.length && !error ? <div className="pool-empty"><BookOpen size={25} /><p>{t('暂时没有候选题。', 'No candidates yet.')}</p></div> : <div className="pool-card-list">{items.map(item => <a className="pool-card" href={poolHref(item.id)} key={item.id}>
          <div className="pool-card-main"><PoolStatus status={item.status} /><h3>{item.problem ? locale === 'en' ? item.problem.title_en || item.problem.title : item.problem.title : `arXiv:${item.arxiv_id}`}</h3>
            <div className="pool-card-meta"><span>arXiv:{item.arxiv_id}</span>{item.version && <span>v{item.version}</span>}{item.author && <span>{item.author.display_name}</span>}</div></div>
          <div className="pool-card-votes"><span><ThumbsUp size={14} />{item.review_summary.approve}</span><span><MessageSquare size={14} />{item.review_summary.needs_changes}</span><ArrowUpRight size={17} /></div>
        </a>)}</div>}
        {(total > 20 || page > 0) && <nav className="pool-pagination" aria-label={t('候选题分页', 'Candidate pages')}><button className="button subtle" disabled={loading || page === 0} onClick={() => changePage(page - 1)}>{t('上一页', 'Previous page')}</button><span aria-live="polite">{loadedPage === page && items.length > 0 ? `${page * 20 + 1}–${page * 20 + items.length} / ${total}` : t(`第 ${page + 1} 页`, `Page ${page + 1}`)}</span><button className="button subtle" disabled={loading || (page + 1) * 20 >= total} onClick={() => changePage(page + 1)}>{t('下一页', 'Next page')}</button></nav>}
      </section>
    </>}
  </main>;
}

function PoolSubmissionForm({owner, capabilities, onSaved}: {owner?: string; capabilities?: PaperPoolCapabilities; onSaved: () => void}) {
  const {t, locale} = useI18n();
  const draftOwner = owner ? `account:${owner}` : undefined;
  const [loaded] = useState(() => readProposalDraft(tabDraftStorage(), draftOwner));
  const [draft, setDraft] = useState(() => loaded.record?.draft || emptyProposalDraft());
  const draftRef = useRef(draft), draftRevision = useRef(0), storedRevision = useRef(loaded.record?.revision);
  const [stored, setStored] = useState(loaded.available);
  const [busy, setBusy] = useState(false), working = useRef(false);
  const [error, setError] = useState(''), [duplicate, setDuplicate] = useState<PoolDuplicate>();
  const [saved, setSaved] = useState<PoolProposal>();
  const mounted = useRef(true);
  useEffect(() => {mounted.current = true; return () => {mounted.current = false;};}, []);
  function edit(patch: Partial<ProposalDraft>) {
    const next = {...draftRef.current, ...patch};
    draftRevision.current += 1; draftRef.current = next; setDraft(next);
    const result = writeProposalDraft(tabDraftStorage(), draftOwner, next);
    storedRevision.current = result.revision; setStored(result.saved);
    setError(''); setDuplicate(undefined); setSaved(undefined);
  }
  async function submit() {
    if (!owner || !capabilities?.can_submit || working.current) return;
    const arxivUrl = arxivPaperUrl(draftRef.current.paperUrl);
    if (!arxivUrl) {setError(t('请输入有效的 arXiv 论文或 PDF 链接。', 'Enter a valid arXiv abstract or PDF link.')); return;}
    const submitted = draftRef.current, revision = draftRevision.current, storedAtSubmit = storedRevision.current;
    working.current = true; setBusy(true); setError(''); setDuplicate(undefined); setSaved(undefined);
    try {
      const result = await poolApi<PoolSubmission>('', {arxiv_url: arxivUrl, prompt: submitted.focus.trim(), expected_user_id: owner});
      if (!mounted.current) return;
      if (!result.proposal && !result.duplicate) throw Error(t('服务未返回投稿结果，请刷新后检查。', 'The service did not return a proposal. Refresh to check its status.'));
      setDuplicate(result.duplicate); setSaved(result.duplicate ? undefined : result.proposal || undefined); onSaved();
      if (!result.duplicate && result.proposal && draftRevision.current === revision) {
        const cleared = clearSubmittedProposalDraft(tabDraftStorage(), draftOwner, storedAtSubmit, submitted);
        if (cleared !== 'kept') {
          const next = emptyProposalDraft(); draftRef.current = next; setDraft(next); storedRevision.current = undefined; setStored(cleared === 'cleared');
        }
      }
    } catch (e) {if (mounted.current) setError(errorMessage(e));}
    finally {working.current = false; if (mounted.current) setBusy(false);}
  }
  return <section className="pool-submit-grid">
    <form className="social-panel social-form pool-submission-form" onSubmit={event => {event.preventDefault(); void submit();}}>
      <label>{t('arXiv 链接', 'arXiv link')}<input type="url" required maxLength={512} placeholder="https://arxiv.org/abs/…" autoComplete="off" value={draft.paperUrl} onChange={event => edit({paperUrl: event.target.value})} disabled={!owner} /></label>
      <label>{t('出题方向（可选）', 'Guidance for the exercise (optional)')}<textarea rows={4} maxLength={4000} value={draft.focus} disabled={!owner} onChange={event => edit({focus: event.target.value})} placeholder={t('例如：围绕核心损失函数出一道 NumPy 练习，重点测试数值稳定性。', 'For example: a NumPy exercise on the core loss, with tests for numerical stability.')} /></label>
      <small className="pool-muted">{t('论文链接与出题方向会公开展示，请勿填写私人信息。', 'Paper links and learning preferences are public. Do not include private information.')}</small>
      {error && <p className="pool-error" role="alert">{error}</p>}
      {duplicate && <div className="pool-feedback" role="status"><strong>{t('这篇论文已经有了', 'This paper is already here')}</strong>{duplicate.kind === 'proposal' ? <a href={poolHref(duplicate.proposal.id)}>{t('查看已有投稿', 'View existing proposal')}<ArrowUpRight size={14} /></a> : duplicate.problems.map(problem => <a key={problem.id} href={`#/${problem.id}`}>{locale === 'en' ? problem.title_en || problem.title : problem.title}<ArrowUpRight size={14} /></a>)}<small>{t('重复投稿不会消耗每日额度。', 'Duplicates do not use your daily proposal quota.')}</small></div>}
      {saved && <div className="pool-feedback" role="status"><strong>{t('论文已提交', 'Paper submitted')}</strong><a href={poolHref(saved.id)}>{t('查看处理进度', 'View progress')}<ArrowUpRight size={14} /></a></div>}
      {!owner ? <a className="button" href={signInHref('submit')}>{t('登录后投稿', 'Sign in to submit')}</a> : <button className="button" type="submit" disabled={busy || !capabilities?.can_submit}>{busy ? <LoaderCircle size={15} className="spin" /> : <Send size={15} />}{busy ? t('正在提交…', 'Submitting…') : t('提交论文', 'Submit paper')}</button>}
      {owner && capabilities && !capabilities.can_submit && <p className="pool-muted">{capabilities.reason || t('请先验证学校或公司的邮箱。', 'Verify your school or company email to submit.')}</p>}
      {owner && !!(draft.paperUrl || draft.focus) && <small role="status" className="pool-muted">{stored ? t('草稿已保存在此浏览器标签页。', 'Draft saved in this browser tab.') : t('草稿暂时无法保存，请保留此页。', 'Draft storage is unavailable. Keep this page open.')}</small>}
      <PublishingRules action="paper" />
    </form>
    <aside className="pool-process social-panel">
      <ol><li><span>1</span><div><strong>{t('管理员初审与适用性检查', 'Intake review & suitability')}</strong><p>{t('投稿先由管理员审核，通过后进入生成与测试流程。', 'A maintainer reviews the submission before generation and testing.')}</p></div></li>
        <li><span>2</span><div><strong>{t('生成并验证题目', 'Generate & validate')}</strong><p>{t('AI 生成题面、返回值要求、初始代码和测试，通过独立验证后进入试做池。', 'AI prepares the statement, return contract, starter and tests. Validation comes before community trials.')}</p></div></li>
        <li><span>3</span><div><strong>{t('社区试做与评审', 'Community trial & review')}</strong><p>{t('至少 5 人评审、80% 建议入库，开放满 48 小时后交维护者确认。', 'After 48 hours, at least 5 reviews with 80% approval qualify for maintainer review.')}</p></div></li></ol>
      {capabilities && (!capabilities.automation_enabled || !capabilities.generation.ready) && <p className="pool-service-notice" role="status">{t('自动出题服务尚未就绪。你仍可以投稿，论文会保留并等待维护者启动生成。', 'Automatic generation is not ready. You can still submit; proposals are retained until a maintainer enables generation.')}</p>}
    </aside>
  </section>;
}

function PoolStatus({status}: {status: string}) {
  const {t} = useI18n();
  const labels: Record<string, [string, string]> = {
    pending_review: ['待管理员初审','Awaiting intake review'], queued: ['排队中', 'Queued'], submitted: ['已提交', 'Submitted'], suitability: ['适用性检查', 'Checking suitability'], screening: ['适用性检查', 'Checking suitability'],
    generation: ['生成中', 'Generating'], generating: ['生成中', 'Generating'], validation: ['验证中', 'Validating'], validating: ['验证中', 'Validating'],
    pool: ['公开试做', 'Open for trials'], changes: ['需要修改', 'Needs changes'], needs_changes: ['需要修改', 'Needs changes'], changes_requested: ['需要修改', 'Needs changes'],
    ready: ['社区评审通过', 'Community review complete'], rejected: ['不适合出题', 'Not suitable'], published: ['已入库', 'Published'],
    blocked: ['等待服务就绪', 'Waiting for service'], config_blocked: ['等待服务就绪', 'Waiting for service'], failed: ['处理失败', 'Processing failed'],
  };
  return <span className={`pool-status ${processing(status) ? 'processing' : ''}`}>{processing(status) && <LoaderCircle size={12} className="spin" />}{labels[status] ? t(...labels[status]) : status}</span>;
}

function PoolDetail({id, owner, capabilities, onChanged}: {id: string; owner?: string; capabilities?: PaperPoolCapabilities; onChanged: () => void}) {
  const {t, locale} = useI18n();
  const [proposal, setProposal] = useState<PoolProposal>();
  const [error, setError] = useState(''), [revision, setRevision] = useState(0);
  const [loading, setLoading] = useState(true);
  const latest = useRef<PoolProposal | undefined>(undefined);
  function remember(next: PoolProposal) {
    if (next.id !== id) return;
    const known = latest.current;
    if (known && (next.version || 0) < (known.version || 0)) return;
    if (known && next.updated_at < known.updated_at) return;
    latest.current = next; setProposal(next);
  }
  useEffect(() => {
    let active = true;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function load() {
      try {
        const next = await poolApi<PoolProposal>(`/${id}`, undefined, controller.signal);
        if (!active) return;
        if (!next.id || !next.viewer || !next.review_summary) throw Error(t('无法加载候选题。', 'Could not load this candidate.'));
        remember(next); setError('');
        if (processing(next.status)) timer = setTimeout(() => void load(), 5000);
      } catch (e) {if (active) setError(errorMessage(e));}
      finally {if (active) setLoading(false);}
    }
    setLoading(true); void load();
    return () => {active = false; controller.abort(); clearTimeout(timer);};
  }, [id, revision]);
  const title = proposal?.problem ? locale === 'en' ? proposal.problem.title_en || proposal.problem.title : proposal.problem.title : proposal ? `arXiv:${proposal.arxiv_id}` : t('候选题', 'Candidate');
  const paperUrl = proposal && arxivPaperUrl(proposal.arxiv_url);
  return <section className="pool-detail">
    <div className="pool-detail-top"><a className="social-back" href={poolHref()}><ArrowLeft size={14} />{t('返回候选题', 'Back to pool')}</a><button className="icon-button" onClick={() => setRevision(value => value + 1)} disabled={loading} aria-label={t('刷新当前候选题', 'Refresh candidate')}><RefreshCw size={15} /></button></div>
    {error && <p className="pool-error" role="alert">{error}</p>}
    {!proposal && loading && <div className="social-loading" role="status"><span className="spinner" />{t('加载中…', 'Loading…')}</div>}
    {proposal && <>
      <header className="pool-detail-heading"><div><PoolStatus status={proposal.status} /><h2>{title}</h2><div className="pool-card-meta">{paperUrl && <a href={paperUrl} target="_blank" rel="noreferrer">arXiv:{proposal.arxiv_id}<ArrowUpRight size={13} /></a>}{proposal.version && <span>{t('版本', 'Version')} {proposal.version}</span>}</div></div>
        {proposal.author && <div className="pool-author"><a href={profileHref(proposal.author)}><Avatar user={proposal.author} size={30} /><span>{proposal.author.display_name}</span></a><AffiliationBadge affiliation={proposal.author.affiliation} institution={proposal.author.institution} /></div>}
      </header>
      {proposal.status==='pending_review'&&<p className="pool-service-notice" role="status">{t('投稿已保留，等待管理员初审；通过后进入生成流程。','Your submission is awaiting intake review. Approval starts the generation process.')}</p>}
      {proposal.maintainer_review&&proposal.version&&<p className="pool-muted">{t('当前候选版本审核：','Current candidate review: ')}{proposal.maintainer_review.approved?t('已通过管理员审核','Approved by a maintainer'):t('等待管理员审核','Awaiting maintainer review')}</p>}
      {proposal.maintainer_review?.approved&&proposal.maintainer_review.reason&&<p className="pool-service-notice">{proposal.maintainer_review.reason}</p>}
      {proposal.reason && <p className="pool-service-notice" role="status">{proposal.reason}</p>}
      {processing(proposal.status) && capabilities && (!capabilities.automation_enabled || !capabilities.generation.ready) && <p className="pool-service-notice" role="status">{t('自动出题服务尚未就绪，投稿已保留，等待维护者启动生成。', 'Automatic generation is not ready. Your proposal is retained until a maintainer enables generation.')}</p>}
      {processing(proposal.status) && <p className="pool-muted" role="status">{t('处理进度会自动更新。你可以稍后回来查看。', 'Progress updates automatically. You can return later.')}</p>}
      {proposal.prompt && <details className="pool-guidance"><summary>{t('投稿人的出题方向', 'Author guidance')}</summary><p>{proposal.prompt}</p></details>}
      {proposal.problem && proposal.version ? <CandidateTrial key={`${id}:${proposal.version}`} proposal={proposal} owner={owner} capabilities={capabilities} onChanged={next => {remember(next); onChanged();}} /> : !processing(proposal.status) && <div className="pool-empty"><FlaskConical size={25} /><p>{t('题目通过生成与验证后，会在这里开放试做。', 'Trials open here after generation and validation.')}</p></div>}
      {owner && proposal.viewer.can_manage && proposal.status !== 'published' && <PoolModeration key={`moderate:${id}:${proposal.version}`} proposal={proposal} owner={owner} onChanged={next => {remember(next); setRevision(value => value + 1); onChanged();}} />}
    </>}
  </section>;
}

function PoolModeration(_props: {proposal: PoolProposal; owner: string; onChanged: (proposal: PoolProposal) => void}) {
  const {t}=useI18n();
  return <section className="pool-moderation"><h3>{t('维护者审核','Maintainer review')}</h3><p>{t('在管理员工作台核对当前版本、填写反馈并提交审核决定。','Review the current version and submit reasoned feedback in the administrator workspace.')}</p><a className="button subtle" href="#/community/admin/papers">{t('打开论文审核','Open paper review')}</a></section>;
}

function CandidateTrial({proposal, owner, capabilities, onChanged}: {proposal: PoolProposal; owner?: string; capabilities?: PaperPoolCapabilities; onChanged: (proposal: PoolProposal) => void}) {
  const {t, locale} = useI18n();
  const problem = proposal.problem!;
  const draftKey = `paperleet.pool.trial:${owner || 'guest'}:${proposal.id}:v${proposal.version}`;
  const attemptKey = `${draftKey}:attempt`;
  const [restoredAttempt] = useState<{attempt?: PoolAttempt; source?: string}>(() => {
    try {
      const saved = JSON.parse(sessionStorage.getItem(attemptKey) || 'null');
      if (saved?.attempt?.version === proposal.version && /^[a-f0-9]{32}$/.test(saved.attempt.id) && ['queued', 'running'].includes(saved.attempt.status) && typeof saved.source === 'string') return saved;
    } catch { /* The server is always the authority for restored trials. */ }
    return {};
  });
  const [code, setCode] = useState(() => {try {return sessionStorage.getItem(draftKey) ?? problem.starter_code;} catch {return problem.starter_code;}});
  const [draftStored, setDraftStored] = useState(true);
  const [theme, setTheme] = useState<'dark' | 'light'>(() => document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light');
  const [attempt, setAttempt] = useState<PoolAttempt | undefined>(restoredAttempt.attempt);
  const [attemptedCode, setAttemptedCode] = useState(restoredAttempt.source || '');
  const [pollRevision, setPollRevision] = useState(0);
  const [pollError, setPollError] = useState('');
  const [busy, setBusy] = useState<'trial' | 'vote'>(), working = useRef(false);
  const [error, setError] = useState(''), [notice, setNotice] = useState('');
  const currentVote = typeof proposal.viewer.vote === 'string' ? proposal.viewer.vote : proposal.viewer.vote?.vote;
  const [vote, setVote] = useState<PaperPoolVote>(currentVote || 'approve');
  const [reason, setReason] = useState(typeof proposal.viewer.vote === 'object' ? proposal.viewer.vote?.reason || '' : currentVote ? proposal.viewer.reason || '' : '');
  const mounted = useRef(true);
  const changed = useRef(onChanged); changed.current = onChanged;
  const pendingAttempt = attempt && ['queued', 'running'].includes(attempt.status);
  useEffect(() => {mounted.current = true; return () => {mounted.current = false;};}, []);
  useEffect(() => {
    if (!attempt || !pendingAttempt || !owner) return;
    let active = true, timer: ReturnType<typeof setTimeout>;
    const controller = new AbortController();
    setPollError('');
    async function poll() {
      try {
        const result = await poolApi<PoolAttemptResponse>(`/${proposal.id}/attempts/${attempt!.id}`, undefined, controller.signal);
        if (!active) return;
        validateAttempt(result, attempt!.id);
        if (result.attempt.status === 'queued' || result.attempt.status === 'running') {
          setAttempt(result.attempt);
          timer = setTimeout(() => void poll(), 2000);
        } else {
          if (result.attempt.status === 'succeeded' && !result.attempt.report) throw Error(t('测试结果缺失，请重试获取。', 'The test report is missing. Retry loading the trial.'));
          setAttempt(result.attempt);
          try {sessionStorage.removeItem(attemptKey);} catch { /* Reopening still verifies with the API. */ }
          changed.current(result.proposal);
        }
      } catch (e) {if (active) setPollError(errorMessage(e));}
    }
    void poll();
    return () => {active = false; clearTimeout(timer); controller.abort();};
  }, [attempt?.id, attempt?.status, pollRevision]);
  useEffect(() => {
    const observer = new MutationObserver(() => setTheme(document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light'));
    observer.observe(document.documentElement, {attributes: true, attributeFilter: ['data-theme']});
    return () => observer.disconnect();
  }, []);
  function editCode(value: string) {
    setCode(value);
    try {sessionStorage.setItem(draftKey, value); setDraftStored(true);} catch {setDraftStored(false);}
  }
  function validateAttempt(result: PoolAttemptResponse, expectedId?: string) {
    if (result.attempt.version !== proposal.version || result.proposal.id !== proposal.id || (expectedId && result.attempt.id !== expectedId) || !/^[a-f0-9]{32}$/.test(result.attempt.id) || !['queued', 'running', 'succeeded', 'failed'].includes(result.attempt.status) || (result.attempt.report && !Array.isArray(result.attempt.report.tests))) throw Error(t('试做结果与当前版本不一致，请刷新候选题。', 'The trial result does not match this version. Refresh the candidate.'));
  }
  async function runTrial() {
    if (!owner || working.current || pendingAttempt || !reviewable(proposal.status) || capabilities?.attempts?.ready !== true) return;
    working.current = true; setBusy('trial'); setError(''); setNotice('');
    const submitted = code;
    try {
      const result = await poolApi<PoolAttemptResponse>(`/${proposal.id}/attempts`, {version: proposal.version, code: submitted, expected_user_id: owner});
      if (!mounted.current) return;
      validateAttempt(result);
      try {sessionStorage.setItem(attemptKey, JSON.stringify({attempt: result.attempt, source: submitted}));} catch { /* The current page retains the pending trial. */ }
      setAttempt(result.attempt); setAttemptedCode(submitted); onChanged(result.proposal);
    } catch (e) {if (mounted.current) setError(errorMessage(e));}
    finally {working.current = false; if (mounted.current) setBusy(undefined);}
  }
  async function review() {
    if (!owner || !proposal.viewer.can_review || working.current) return;
    if (vote === 'needs_changes' && !reason.trim()) {setError(t('请说明需要修改的问题，最好附上失败用例。', 'Describe what needs changing, ideally with a failing example.')); return;}
    working.current = true; setBusy('vote'); setError(''); setNotice('');
    try {
      const result = await poolApi<PoolProposal>(`/${proposal.id}/votes`, {version: proposal.version, vote, reason: reason.trim(), expected_user_id: owner});
      if (!mounted.current) return;
      if (result.id !== proposal.id) throw Error(t('无法确认评审结果，请刷新。', 'Could not confirm your review. Please refresh.'));
      onChanged(result); setNotice(t('评审已保存。', 'Review saved.'));
    } catch (e) {if (mounted.current) setError(errorMessage(e));}
    finally {working.current = false; if (mounted.current) setBusy(undefined);}
  }
  const summary = proposal.review_summary;
  return <>
    <div className="pool-trial-grid">
      <section className="social-panel pool-statement" aria-label={t('候选题题面', 'Candidate statement')}><Prose content={locale === 'en' ? problem.statement_en || problem.statement : problem.statement} />
        {!!problem.functions?.length && <section className="pool-contract"><h3>{t('函数与返回值', 'Functions & return values')}</h3>{problem.functions.map(fn => <div key={fn.name}><code>{fn.signature}</code>{fn.returns && <p><strong>return</strong> {fn.returns}</p>}</div>)}</section>}
      </section>
      <section className="social-panel pool-code-panel" aria-label={t('试做候选题', 'Try candidate')}><div className="pool-code-toolbar"><span><Code2 size={16} />Python</span><button className="button" onClick={() => void runTrial()} disabled={!!busy || !!pendingAttempt || !owner || !reviewable(proposal.status) || capabilities?.attempts?.ready !== true}>{busy === 'trial' || pendingAttempt ? <LoaderCircle size={14} className="spin" /> : <Play size={14} />}{pendingAttempt ? attempt.status === 'queued' ? t('排队中…', 'Queued…') : t('测试中…', 'Testing…') : busy === 'trial' ? t('提交试做…', 'Starting trial…') : t('运行试做测试', 'Run trial tests')}</button></div>
        <div className="pool-code-editor"><TrialEditor value={code} readOnly={false} font={13} theme={theme} onChange={editCode} label={t('试做代码', 'Trial code')} /></div>
        <div className="pool-code-note"><span>{t('候选题测试在隔离环境运行，不计入正式题库成绩。', 'Candidate tests run in an isolated environment and do not count toward catalog progress.')}</span>{!draftStored && <span role="status">{t('草稿暂时无法保存，请保留此页。', 'Draft storage is unavailable. Keep this page open.')}</span>}</div>
        {!owner && <p className="pool-muted"><a href={signInHref(`submit?proposal=${proposal.id}`)}>{t('登录后运行试做测试', 'Sign in to run trial tests')}</a></p>}
        {owner && capabilities?.attempts?.ready !== true && <p className="pool-service-notice">{capabilities?.attempts?.reason || t('试做服务尚未就绪，暂时不能运行测试。', 'The trial service is not ready. Tests are unavailable for now.')}</p>}
        {pollError && <div className="pool-service-notice" role="alert">{pollError}<button className="text-button" onClick={() => setPollRevision(value => value + 1)}>{t('重试获取试做状态', 'Retry trial status')}</button></div>}
        {attempt?.status === 'failed' && <p className="pool-service-notice" role="alert">{attempt.error || t('试做服务未能完成测试，请稍后重试。', 'The trial service could not complete the tests. Please try again.')}</p>}
        {attempt?.report && <section className="pool-trial-result" aria-label={t('试做结果', 'Trial result')} role="status"><h3>{attempt.report.passed ? <CheckCircle2 size={17} /> : <FlaskConical size={17} />}{attempt.report.passed ? t('测试通过', 'Tests passed') : t('测试未通过', 'Tests did not pass')}</h3>{attemptedCode !== code && <p className="pool-muted">{t('结果对应上次运行的代码。', 'This result is for the last code you ran.')}</p>}{attempt.report.error && <p>{attempt.report.error}</p>}
          <ul>{attempt.report.tests.map((test, index) => <li key={`${test.id}:${index}`}><span>{test.passed ? '✓' : '×'} {test.id}</span>{test.message && <pre>{test.message}</pre>}</li>)}</ul></section>}
      </section>
    </div>
    <section className="social-panel pool-review"><div className="pool-section-heading"><h3>{t('评审当前版本', 'Review this version')}</h3><div className="pool-card-votes"><span><ThumbsUp size={15} />{summary.approve}</span><span><MessageSquare size={15} />{summary.needs_changes}</span></div></div>
      <p className="pool-muted">{t(`至少 ${summary.minimum_reviews} 人评审、${Math.round(summary.approval_ratio * 100)}% 建议入库，并开放满 ${summary.review_hours} 小时。已确认的问题需先修复，最终由维护者确认入库。`, `At least ${summary.minimum_reviews} reviews, ${Math.round(summary.approval_ratio * 100)}% approval and ${summary.review_hours} hours open. Confirmed issues must be fixed before final maintainer approval.`)}</p>
      {error && <p className="pool-error" role="alert">{error}</p>}{notice && <p className="pool-feedback" role="status">{notice}</p>}
      {!proposal.viewer.can_review && <p className="pool-service-notice">{proposal.viewer.eligibility_reason || (!owner ? t('验证机构邮箱并试做当前版本后，即可参与评审。', 'Verify your organization email and try this version to review it.') : t('请先试做当前版本。投稿人不能评审自己的题目。', 'Try the current version first. Authors cannot review their own candidates.'))}</p>}
      <form className="social-form pool-review-form" onSubmit={event => {event.preventDefault(); void review();}}><fieldset disabled={!proposal.viewer.can_review || !!busy}><legend className="sr-only">{t('评审意见', 'Review decision')}</legend><label><input type="radio" name="pool-review" value="approve" checked={vote === 'approve'} onChange={() => setVote('approve')} /><ThumbsUp size={15} />{t('建议入库', 'Recommend inclusion')}</label><label><input type="radio" name="pool-review" value="needs_changes" checked={vote === 'needs_changes'} onChange={() => setVote('needs_changes')} /><MessageSquare size={15} />{t('需要修改', 'Needs changes')}</label></fieldset>
        <label>{t('反馈与失败用例', 'Feedback & failing examples')}<textarea rows={3} maxLength={4000} required={vote === 'needs_changes'} value={reason} disabled={!proposal.viewer.can_review || !!busy} onChange={event => setReason(event.target.value)} placeholder={t('题面、返回值、参考答案或测试哪里需要改进？', 'What should change in the statement, return contract, reference answer or tests?')} /></label>
        <button className="button" type="submit" disabled={!proposal.viewer.can_review || !!busy}>{busy === 'vote' && <LoaderCircle size={14} className="spin" />}{currentVote ? t('更新评审', 'Update review') : t('提交评审', 'Submit review')}</button>
      </form>
      {!!proposal.reviews?.length && <div className="pool-public-reviews">{proposal.reviews.map(review => <article key={review.author.id}>
        <header><a href={profileHref(review.author)}><Avatar user={review.author} size={26} />{review.author.display_name}</a><span>{review.vote === 'approve' ? t('建议入库', 'Recommends inclusion') : t('需要修改', 'Requests changes')}</span></header>
        <AffiliationBadge affiliation={review.author.affiliation} institution={review.author.institution} />
        {review.reason && <p>{review.reason}</p>}
      </article>)}</div>}
    </section>
  </>;
}
