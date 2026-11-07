import { useEffect, useRef, useState } from 'react';
import type { ChangeEvent, FormEvent } from 'react';
import { Archive, ArrowLeft, ArrowRight, Download, FileText, Loader2, Plus, RefreshCw, Search, Upload, Users, X } from 'lucide-react';
import AppLink from './AppLink';
import { appLocation, publicHref } from './appLocation';
import { signInHref } from './accountNavigation';
import { errorMessage, socialApi, useAccount } from './social';
import { useI18n } from './i18n';
import { deferredFeature } from './deferredFeature';
import { makeId, newProject, validateWorkspace } from './rebuttalModel';
import type { RebuttalProject } from './rebuttalModel';
import {
  downloadRebuttalProject, readRebuttalPaper, readRebuttalPapers, readRebuttalRoute,
  rebuttalHref, rebuttalRequestId, useRebuttalCloud, validateCloudProject,
} from './rebuttalCloud';
import type { RebuttalPaperRow, RebuttalRole } from './rebuttalCloud';
import './rebuttal.css';
import './hostedRebuttal.css';

const RebuttalStudio = deferredFeature(() => import('./RebuttalStudio'), ['回复工作台', 'Rebuttal workspace']);

type Translator = (zh: string, en: string) => string;
const roleName = (role: RebuttalRole, t: Translator) => role === 'owner' ? t('所有者', 'Owner') : role === 'editor' ? t('可编辑', 'Editor') : t('只读', 'Viewer');

export default function HostedRebuttal({ route = 'rebuttal' }: { route?: string }) {
  const { user, loading, error, refresh } = useAccount();
  const { t } = useI18n();
  const view = readRebuttalRoute(route);
  if (!view) return <main className="rst-shell hreb-shell"><h1>{t('找不到这篇论文', 'Paper not found')}</h1><AppLink href={rebuttalHref()}>{t('返回我的论文', 'Back to my papers')}</AppLink></main>;
  if (loading) return <main className="rst-shell hreb-shell"><p role="status">{t('正在确认账号…', 'Checking your account…')}</p></main>;
  if (!user) return <main className="rst-shell hreb-shell hreb-access"><FileText size={28} /><h1>Rebuttal Studio</h1>
    {error ? <><p role="alert">{error}</p><button className="rst-button" onClick={() => void refresh()}>{t('重试', 'Retry')}</button></> : <p>{t('登录后管理论文，与合作者一起准备回复。', 'Sign in to manage papers and prepare responses with collaborators.')}</p>}
    <AppLink className="rst-button rst-primary" href={signInHref(rebuttalHref(view.paperId))}>{t('登录', 'Sign in')}</AppLink>
  </main>;
  return view.paperId ? <PaperWorkspace key={`${user.id}:${view.paperId}`} owner={user.id} paperId={view.paperId} /> : <PaperLibrary key={user.id} owner={user.id} />;
}

function PaperLibrary({ owner }: { owner: string }) {
  const { t, locale } = useI18n();
  const [papers, setPapers] = useState<RebuttalPaperRow[]>([]), [loading, setLoading] = useState(true), [error, setError] = useState('');
  const [filter, setFilter] = useState<'mine' | 'shared' | 'archived'>('mine'), [query, setQuery] = useState('');
  const [creating, setCreating] = useState(false), [title, setTitle] = useState(''), [venue, setVenue] = useState('');
  const [imports, setImports] = useState<RebuttalProject[]>([]), [busy, setBusy] = useState(false), [notice, setNotice] = useState('');
  const [, redraw] = useState(0), picker = useRef<HTMLInputElement>(null), mounted = useRef(true), controllers = useRef(new Set<AbortController>()), reads = useRef(0);
  const creation = useRef<{ projects: RebuttalProject[]; index: number; requestId: string; created: string[] } | undefined>(undefined);
  const archive = useRef<{ id: string; archived: boolean; body?: { project: RebuttalProject; expected_revision: number; request_id: string; expected_user_id: string } } | undefined>(undefined);
  const working = useRef(false);
  const pending = !!creation.current || !!archive.current;
  const request = async (path: string, body?: unknown) => {
    if (!mounted.current) throw Error('Account changed');
    const controller = new AbortController(); controllers.current.add(controller);
    const timeout = setTimeout(() => controller.abort(), 20000);
    try { return await socialApi<unknown>(path, body, controller.signal); }
    finally { clearTimeout(timeout); controllers.current.delete(controller); }
  };
  async function refreshList() {
    const epoch = ++reads.current; setLoading(true); setError('');
    try { const rows = readRebuttalPapers(await request('rebuttal/papers')); if (mounted.current && epoch === reads.current) setPapers(rows); }
    catch (reason) { if (mounted.current && epoch === reads.current) setError(errorMessage(reason)); }
    finally { if (mounted.current && epoch === reads.current) setLoading(false); }
  }
  useEffect(() => {
    mounted.current = true; void refreshList();
    return () => { mounted.current = false; controllers.current.forEach(controller => controller.abort()); controllers.current.clear(); };
  }, [owner]);
  async function runCreation(projects?: RebuttalProject[]) {
    if (working.current || archive.current || !mounted.current) return;
    if (!creation.current) {
      if (!projects?.length) return;
      try { projects = projects.map(project => validateCloudProject(project)); }
      catch (reason) { setError(errorMessage(reason)); return; }
      creation.current = { projects: structuredClone(projects), index: 0, requestId: rebuttalRequestId(), created: [] };
    }
    working.current = true; setBusy(true); setError('');
    try {
      const batch = creation.current;
      while (batch.index < batch.projects.length && mounted.current) {
        const document = readRebuttalPaper(await request('rebuttal/papers', { project: batch.projects[batch.index], request_id: batch.requestId, expected_user_id: owner }));
        if (!mounted.current) return;
        batch.created.push(document.project.id); batch.index++; batch.requestId = rebuttalRequestId();
      }
      if (!mounted.current) return;
      creation.current = undefined; setCreating(false); setImports([]); setTitle(''); setVenue('');
      if (batch.created.length === 1) appLocation.hash = rebuttalHref(batch.created[0]);
      else { setNotice(t(`已导入 ${batch.created.length} 篇论文`, `${batch.created.length} papers imported`)); await refreshList(); }
    } catch (reason) {
      if (mounted.current) {
        if (reason && typeof reason === 'object' && 'status' in reason &&
          Number(reason.status) >= 400 && Number(reason.status) < 500 && ![408, 429].includes(Number(reason.status))) {
          if (creation.current && creation.current.index > 0) setImports(creation.current.projects.slice(creation.current.index));
          creation.current = undefined;
        }
        setError(errorMessage(reason));
      }
    }
    finally { working.current = false; if (mounted.current) { setBusy(false); redraw(value => value + 1); } }
  }
  async function archivePaper(row?: RebuttalPaperRow) {
    if (working.current || creation.current || !mounted.current) return;
    if (!archive.current) { if (!row || row.role !== 'owner') return; archive.current = { id: row.id, archived: !row.archived }; }
    working.current = true; setBusy(true); setError('');
    try {
      const command = archive.current;
      if (!command.body) {
        const document = readRebuttalPaper(await request(`rebuttal/papers/${command.id}`), command.id);
        if (!mounted.current) return;
        if (document.role !== 'owner') throw Error(t('只有所有者可以归档论文', 'Only the owner can archive this paper'));
        command.body = { project: { ...document.project, archived: command.archived }, expected_revision: document.revision, request_id: rebuttalRequestId(), expected_user_id: owner };
      }
      readRebuttalPaper(await request(`rebuttal/papers/${command.id}`, command.body), command.id);
      if (!mounted.current) return;
      archive.current = undefined; await refreshList();
    } catch (reason) {
      if (mounted.current) {
        if (reason && typeof reason === 'object' && 'status' in reason && reason.status === 409) archive.current = undefined;
        setError(errorMessage(reason));
      }
    } finally { working.current = false; if (mounted.current) { setBusy(false); redraw(value => value + 1); } }
  }
  async function readBackup(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]; event.target.value = ''; if (!file || busy || pending) return;
    setError('');
    try {
      if (file.size > 20_000_000) throw Error(t('备份文件过大，请按论文分别导出。', 'This backup is too large. Export papers separately.'));
      const source = await file.text(); if (!mounted.current) return;
      const workspace = validateWorkspace(JSON.parse(source));
      setImports(workspace.projects.map(project => validateCloudProject({ ...project, id: makeId() }))); setCreating(false);
    } catch (reason) { if (mounted.current) setError(errorMessage(reason)); }
  }
  function createPaper(event: FormEvent) {
    event.preventDefault(); if (!title.trim()) return;
    void runCreation([{ ...newProject(), title: title.trim(), venue: venue.trim() }]);
  }
  const visible = papers.filter(paper => filter === 'archived' ? paper.archived : !paper.archived && (filter === 'mine' ? paper.role === 'owner' : paper.role !== 'owner'))
    .filter(paper => `${paper.title} ${paper.venue} ${paper.owner.display_name} ${paper.owner.username}`.toLocaleLowerCase().includes(query.trim().toLocaleLowerCase()))
    .sort((a, b) => Date.parse(b.updatedAt) - Date.parse(a.updatedAt));
  return <main className="rst-shell hreb-shell">
    <header className="hreb-heading"><h1>{t('我的论文', 'My papers')}</h1><div>
      <button className="rst-button" disabled={busy || pending} onClick={() => picker.current?.click()}><Upload size={15} />{t('导入本地备份', 'Import local backup')}</button>
      <button className="rst-button rst-primary" disabled={busy || pending} onClick={() => { setCreating(true); setImports([]); }}><Plus size={15} />{t('新建论文', 'New paper')}</button>
    </div></header>
    <input ref={picker} hidden type="file" accept=".json" onChange={readBackup} />
    {error && <div className="hreb-error" role="alert"><p>{error}</p>{pending ? <button className="rst-button" disabled={busy} onClick={() => void (creation.current ? runCreation() : archivePaper())}>{t('重试原操作', 'Retry original operation')}</button> : <button className="rst-button" disabled={loading || busy} onClick={() => void refreshList()}>{t('刷新论文列表', 'Refresh papers')}</button>}</div>}
    {notice && <p className="hreb-notice" role="status">{notice}</p>}
    {creating && <form className="hreb-create" onSubmit={createPaper}>
      <label><span>{t('论文标题', 'Paper title')}</span><input autoFocus required maxLength={2000} disabled={busy || pending} value={title} onChange={event => setTitle(event.target.value)} /></label>
      <label><span>{t('会议 / 期刊', 'Venue')}</span><input maxLength={300} disabled={busy || pending} value={venue} onChange={event => setVenue(event.target.value)} /></label>
      <div><button type="button" className="rst-button" disabled={busy} onClick={() => setCreating(false)}>{t('收起', 'Close')}</button><button className="rst-button rst-primary" disabled={busy || pending || !title.trim()}>{t('创建并打开', 'Create & open')}</button></div>
    </form>}
    {imports.length > 0 && <section className="hreb-import"><h2>{t('导入到当前账号', 'Import into this account')}</h2><ul>{imports.map(project => <li key={project.id}>{project.title || t('未命名论文', 'Untitled paper')}</li>)}</ul><div><button className="rst-button" disabled={busy || pending} onClick={() => setImports([])}>{t('取消', 'Cancel')}</button><button className="rst-button rst-primary" disabled={busy || pending} onClick={() => void runCreation(imports)}><Upload size={14} />{t(`导入 ${imports.length} 篇论文`, `Import ${imports.length} papers`)}</button></div></section>}
    <div className="hreb-toolbar"><nav aria-label={t('论文范围', 'Paper scope')}>{([['mine', t('我的论文', 'My papers')], ['shared', t('共享论文', 'Shared with me')], ['archived', t('归档', 'Archived')]] as const).map(([key, label]) => <button key={key} aria-pressed={filter === key} onClick={() => setFilter(key)}>{label}</button>)}</nav>
      <label className="hreb-search"><Search size={15} /><input aria-label={t('搜索论文', 'Search papers')} value={query} onChange={event => setQuery(event.target.value)} placeholder={t('搜索论文…', 'Search papers…')} /></label>
      <button className="rst-icon-button" disabled={loading || busy || pending} title={t('刷新', 'Refresh')} aria-label={t('刷新论文列表', 'Refresh paper library')} onClick={() => void refreshList()}><RefreshCw size={16} /></button>
    </div>
    {loading && <p role="status" className="hreb-loading"><Loader2 size={16} className="rst-spin" />{t('正在读取论文…', 'Loading papers…')}</p>}
    {!loading && !visible.length && !error && <div className="hreb-empty"><FileText size={27} /><h2>{query ? t('没有匹配的论文', 'No matching papers') : filter === 'shared' ? t('暂无共享论文', 'No shared papers') : filter === 'archived' ? t('暂无归档论文', 'No archived papers') : t('还没有论文', 'No papers yet')}</h2></div>}
    <div className="hreb-papers">{visible.map(paper => <article key={paper.id}>
      <button className="hreb-paper-open" disabled={busy || pending} onClick={() => { appLocation.hash = rebuttalHref(paper.id); }}>
        <span><strong>{paper.title || t('未命名论文', 'Untitled paper')}</strong><span className="hreb-paper-meta">{paper.venue && <span>{paper.venue}</span>}<span>{paper.reviewerCount} reviewers</span>{paper.role !== 'owner' && <span>{paper.owner.display_name} · {roleName(paper.role, t)}</span>}</span></span>
        <span className="hreb-progress">{paper.readyCount} / {paper.issueCount} {t('条回复', 'responses')}</span><time dateTime={paper.updatedAt}>{new Date(paper.updatedAt).toLocaleDateString(locale === 'zh' ? 'zh-CN' : 'en-US')}</time><ArrowRight size={16} />
      </button>
      {paper.role === 'owner' && <button className="rst-icon-button" disabled={busy || pending} onClick={() => void archivePaper(paper)} title={paper.archived ? t('恢复论文', 'Restore paper') : t('归档论文', 'Archive paper')} aria-label={`${paper.archived ? t('恢复', 'Restore') : t('归档', 'Archive')} ${paper.title}`}><Archive size={16} /></button>}
    </article>)}</div>
  </main>;
}

function PaperWorkspace({ owner, paperId }: { owner: string; paperId: string }) {
  const { t } = useI18n();
  const { state, store } = useRebuttalCloud(owner, paperId);
  const [sharing, setSharing] = useState(false), [leaving, setLeaving] = useState(false);
  const mounted = useRef(true);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
  async function back() {
    setLeaving(true); const saved = await store.flush();
    if (mounted.current) { setLeaving(false); if (saved || !store.hasPending()) appLocation.hash = rebuttalHref(); }
  }
  useEffect(() => {
    const followLink = (event: MouseEvent) => {
      if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey || !store.hasPending()) return;
      const anchor = (event.target as Element | null)?.closest<HTMLAnchorElement>('a[href]');
      if (!anchor || anchor.target === '_blank' || anchor.hasAttribute('download')) return;
      const target = new URL(anchor.href, location.href); if (target.href === location.href) return;
      event.preventDefault(); event.stopPropagation();
      void store.flush().then(saved => { if (saved && mounted.current) location.assign(target.href); });
    };
    window.addEventListener('click', followLink, true);
    return () => window.removeEventListener('click', followLink, true);
  }, [store]);
  if (!state.project || !state.document) return <main className="rst-shell hreb-shell"><button className="rst-button" onClick={() => { appLocation.hash = rebuttalHref(); }}><ArrowLeft size={15} />{t('我的论文', 'My papers')}</button>
    {state.loading ? <p role="status" className="hreb-loading">{t('正在读取论文…', 'Loading paper…')}</p> : <div className="hreb-error" role="alert"><p>{state.error || t('无法读取论文', 'Could not load this paper')}</p><button className="rst-button" onClick={() => void store.load()}>{t('重试', 'Retry')}</button></div>}
  </main>;
  const conflict = state.status === 'conflict';
  return <div className="hreb-workspace">
    {(state.error || state.recoveryUnavailable) && <section className="rst-shell hreb-banner"><div className="hreb-error" role="alert"><p>{conflict ? t('服务器版本已变化，当前草稿已保留。加载服务器版本会替换当前未保存修改。', 'The server version changed. Your draft is preserved. Loading the server version will replace your unsaved changes.') : state.error || t('浏览器暂时无法备份草稿，请下载副本。', 'Browser draft recovery is unavailable. Download a copy.')}</p><div>
      <button className="rst-button" onClick={() => downloadRebuttalProject(state.project!)}><Download size={14} />{t('下载当前草稿', 'Download draft')}</button>
      {!conflict && !state.blocked && <button className="rst-button" disabled={state.status === 'saving' || state.memberBusy} onClick={() => void (store.hasPendingMember() ? store.changeMember() : store.flush())}>{t('重试保存', 'Retry save')}</button>}
      <button className="rst-button" disabled={state.status === 'saving' || state.memberBusy} onClick={() => void store.refresh(true)}><RefreshCw size={14} />{t('加载服务器版本', 'Load server version')}</button>
    </div></div></section>}
    <RebuttalStudio key={`${paperId}:${state.reloadVersion}`} hosted={{ project: state.project, readOnly: state.document.role === 'viewer' || state.blocked || state.loading || state.memberBusy || store.hasPendingMember() || leaving,
      saveStatus: state.status, role: state.document.role, onChange: project => { store.update(project); }, onBack: () => { void back(); },
      onShare: () => setSharing(true), onRefresh: () => { void store.flush().then(saved => { if (saved) void store.refresh(); }); } }} />
    {sharing && <MembersDialog state={state} store={store} onClose={() => setSharing(false)} />}
  </div>;
}

function MembersDialog({ state, store, onClose }: { state: ReturnType<typeof useRebuttalCloud>['state']; store: ReturnType<typeof useRebuttalCloud>['store']; onClose: () => void }) {
  const { t } = useI18n();
  const [username, setUsername] = useState(''), [role, setRole] = useState<'editor' | 'viewer'>('editor'), [copied, setCopied] = useState(false);
  const dialog = useRef<HTMLElement>(null);
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    dialog.current?.querySelector<HTMLElement>('button,input,select')?.focus();
    const keydown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !store.getSnapshot().memberBusy) onClose();
      if (event.key !== 'Tab') return;
      const items = [...(dialog.current?.querySelectorAll<HTMLElement>('button:not(:disabled),input:not(:disabled),select:not(:disabled),a[href]') || [])];
      const first = items[0], last = items.at(-1);
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
    };
    document.addEventListener('keydown', keydown);
    return () => { document.removeEventListener('keydown', keydown); if (previous?.isConnected) previous.focus(); };
  }, [store]);
  const paper = state.document!;
  const canManage = paper.role === 'owner' && !state.blocked;
  const disabled = state.memberBusy || state.status === 'conflict' || store.hasPendingMember();
  async function invite(event: FormEvent) {
    event.preventDefault(); const exact = username.trim().replace(/^@/, '');
    if (!/^[a-z0-9_]{3,24}$/.test(exact) || disabled) return;
    if (await store.changeMember({ username: exact, role })) setUsername('');
  }
  return <div className="hreb-overlay"><section ref={dialog} className="hreb-members" role="dialog" aria-modal="true" aria-labelledby="hreb-members-title">
    <header><h2 id="hreb-members-title"><Users size={18} />{t('论文成员', 'Paper members')}</h2><button className="rst-icon-button" disabled={state.memberBusy} aria-label={t('关闭成员窗口', 'Close members')} onClick={onClose}><X size={18} /></button></header>
    {state.error && <p className="hreb-error" role="alert">{state.error}</p>}
    {canManage && <form className="hreb-invite" onSubmit={invite}><label><span>{t('精确用户名', 'Exact username')}</span><input autoComplete="off" spellCheck={false} maxLength={25} disabled={disabled} value={username} onChange={event => setUsername(event.target.value)} placeholder="username" /></label><label><span>{t('权限', 'Permission')}</span><select value={role} disabled={disabled} onChange={event => setRole(event.target.value as 'editor' | 'viewer')}><option value="editor">{t('可编辑', 'Editor')}</option><option value="viewer">{t('只读', 'Viewer')}</option></select></label><button className="rst-button rst-primary" disabled={disabled || !/^[a-z0-9_]{3,24}$/.test(username.trim().replace(/^@/, ''))}>{state.memberBusy ? t('正在保存…', 'Saving…') : t('添加成员', 'Add member')}</button></form>}
    <ul className="hreb-member-list">{paper.members.map(member => <li key={member.user.id}><span><strong>{member.user.display_name}</strong><span>@{member.user.username}</span></span>
      {canManage && member.role !== 'owner' ? <div><select aria-label={`${member.user.username} ${t('权限', 'permission')}`} disabled={disabled} value={member.role} onChange={event => void store.changeMember({ username: member.user.username, role: event.target.value as 'editor' | 'viewer' })}><option value="editor">{t('可编辑', 'Editor')}</option><option value="viewer">{t('只读', 'Viewer')}</option></select><button className="hreb-remove" disabled={disabled} onClick={() => void store.changeMember({ action: 'remove', user_id: member.user.id })}>{t('移除', 'Remove')}</button></div> : <span>{roleName(member.role, t)}</span>}
    </li>)}</ul>
    <footer>{store.hasPendingMember() && <button className="rst-button" disabled={state.memberBusy || state.status === 'conflict'} onClick={() => void store.changeMember()}>{t('重试原操作', 'Retry original operation')}</button>}<button className="rst-button" onClick={async () => { try { await navigator.clipboard.writeText(new URL(publicHref(rebuttalHref(paper.project.id)) || rebuttalHref(paper.project.id), location.href).href); setCopied(true); } catch { setCopied(false); } }}>{copied ? t('已复制', 'Copied') : t('复制论文链接', 'Copy paper link')}</button></footer>
  </section></div>;
}
