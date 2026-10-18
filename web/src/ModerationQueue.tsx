import {useEffect,useRef,useState} from 'react';
import {Flag,RefreshCw,ShieldCheck} from 'lucide-react';
import {DateLabel,errorMessage,SocialError,socialApi,useAccount} from './social';
import {useI18n} from './i18n';
import {signInHref} from './accountNavigation';
import {reportReasons,reportRequestId,type ReviewCase,type ReviewCommand,type ReviewPost,type ReviewRow} from './communityReports';
import CommunityMarkdown from './CommunityMarkdown';
import {deferredFeature} from './deferredFeature';
import './community-reports.css';
import './moderation.css';

type Queue={owner_id:string;items:ReviewRow[];total:number;state:string;minimum:number;limit:number;offset:number};
const SnapshotComparison=deferredFeature(()=>import('./ModerationSnapshotComparison'),['内容对比','content comparison']);
function SnapshotChanges({before,after}:{before:ReviewPost;after:ReviewPost}){
 const {t}=useI18n(),[visited,setVisited]=useState(false);
 return <details className="moderation-change-disclosure" onToggle={event=>{if(event.currentTarget.open)setVisited(true);}}><summary>{t('对比当前内容','Compare with current content')}<small>{t('标题、正文与分类','Title, body and category')}</small></summary>{visited&&<SnapshotComparison before={before} after={after}/>}</details>;
}
export default function ModerationQueue({embedded=false,refreshKey=0}:{embedded?:boolean;refreshKey?:number}){
 const {user,loading,capabilities}=useAccount(),{t}=useI18n();
 if(loading)return <main className="moderation-page"><p role="status">{t('正在确认审核权限…','Checking review access…')}</p></main>;
 if(!user)return <main className="moderation-page"><h1>{t('社区举报审核','Community reports')}</h1><a href={signInHref('#/community/moderation')}>{t('登录管理员账号','Sign in as an administrator')}</a></main>;
 if(user.role!=='admin'||user.status&&user.status!=='active')return <main className="moderation-page"><h1>{t('需要管理员权限','Administrator access required')}</h1></main>;
 if(capabilities?.community_reports!==true)return <main className="moderation-page"><p>{t('此服务暂不支持举报审核。','Report review is unavailable on this service.')}</p></main>;
 return <QueueView key={user.id} owner={user.id} embedded={embedded} refreshKey={refreshKey}/>;
}
function QueueView({owner,embedded=false,refreshKey=0}:{owner:string;embedded?:boolean;refreshKey?:number}){
 const {t}=useI18n(),[state,setState]=useState('pending'),[minimum,setMinimum]=useState(1),[offset,setOffset]=useState(0),[selected,setSelected]=useState('');
 const [data,setData]=useState<Queue>(),[loading,setLoading]=useState(false),[error,setError]=useState(''),sequence=useRef(0),mounted=useRef(true);
 useEffect(()=>{mounted.current=true;return()=>{mounted.current=false;sequence.current++;};},[]);
 async function load(){const request=++sequence.current;setLoading(true);setError('');try{
  const value=await socialApi<Queue>(`community/moderation?state=${state}&minimum=${minimum}&limit=20&offset=${offset}`,undefined,AbortSignal.timeout(20_000));
  if(value.owner_id!==owner||value.state!==state||value.minimum!==minimum||value.offset!==offset||value.limit!==20||!Array.isArray(value.items)||!Number.isSafeInteger(value.total)||value.total<0||value.items.some(row=>!row.post_id||!Number.isSafeInteger(row.revision)||row.revision<1||!Number.isSafeInteger(row.total_reports)||!Number.isSafeInteger(row.pending_reports)||row.pending_reports<0||row.total_reports<row.pending_reports||!['pending','dismissed','removed'].includes(row.state)||row.post&&row.post.id!==row.post_id))throw Error('The review queue could not be confirmed.');
  if(mounted.current&&request===sequence.current)setData(value);
 }catch(e){if(mounted.current&&request===sequence.current)setError(errorMessage(e));}finally{if(mounted.current&&request===sequence.current)setLoading(false);}}
 useEffect(()=>{void load();},[state,minimum,offset,refreshKey]);
 const Root=embedded?"section":"main";
 return <Root className="moderation-page">{embedded?<p className="admin-intro">{t('按待审举报人数优先排列。每个账号对同一帖子只计一票，举报不会自动删除帖子。','Prioritized by pending reporters. Each account counts once per post. Reports do not automatically remove content.')}</p>:<header className="moderation-heading"><div><span><ShieldCheck size={17}/>{t('管理员工作台','Administrator workspace')}</span><h1>{t('社区举报审核','Community reports')}</h1><p>{t('按待审举报人数优先排列。每个账号对同一帖子只计一票，举报不会自动删除帖子。','Prioritized by pending reporters. Each account counts once per post. Reports do not automatically remove content.')}</p></div><a href="#/community">{t('返回社区','Back to community')}</a></header>}
  <div className="moderation-filters"><label>{t('审核状态','Review status')}<select value={state} onChange={event=>{setState(event.target.value);setOffset(0);setSelected('');}}><option value="pending">{t('待审核','Pending')}</option><option value="resolved">{t('已处理','Resolved')}</option><option value="all">{t('全部','All')}</option></select></label><label>{t('累计举报至少','At least this many reports')}<select value={minimum} onChange={event=>{setMinimum(Number(event.target.value));setOffset(0);setSelected('');}}>{[1,3,5,10].map(n=><option key={n} value={n}>{n}</option>)}</select></label><button className="button subtle" type="button" disabled={loading} onClick={()=>void load()}><RefreshCw size={15}/>{t('刷新队列','Refresh queue')}</button></div>
  {loading&&<p role="status">{t('正在读取举报…','Loading reports…')}</p>}{error&&<p role="alert">{error}</p>}
  {data&&<><p className="moderation-count">{t(`共 ${data.total} 个帖子`,`${data.total} posts`)}</p><div className="moderation-layout"><section className="moderation-list" aria-label={t('举报帖子','Reported posts')}>
   {!data.items.length&&<p>{t('当前筛选下没有举报帖子。','No reported posts match these filters.')}</p>}
   {data.items.map(row=><button type="button" key={row.post_id} disabled={loading||!!error} aria-pressed={selected===row.post_id} onClick={()=>setSelected(row.post_id)}><span className="moderation-report-count"><Flag size={14}/>{t(`待审 ${row.pending_reports} · 累计 ${row.total_reports}`,`${row.pending_reports} pending · ${row.total_reports} total`)}</span><b>{row.post?.title||row.title}</b><small>{row.state==='pending'?t('待审核','Pending'):row.state==='removed'?t('已删除','Removed'):t('已保留','Kept')} · <DateLabel value={row.last_report_at}/></small>{!row.post&&<small>{t('帖子已不存在，举报快照仍保留','Post unavailable; report snapshots retained')}</small>}</button>)}
   <nav aria-label={t('审核队列翻页','Review queue pages')}><button type="button" className="text-button" disabled={loading||!offset} onClick={()=>{setOffset(value=>Math.max(0,value-20));setSelected('');}}>{t('上一页','Previous')}</button><span>{Math.floor(offset/20)+1}</span><button type="button" className="text-button" disabled={loading||offset+20>=data.total} onClick={()=>{setOffset(value=>value+20);setSelected('');}}>{t('下一页','Next')}</button></nav>
  </section>{selected?<ReviewCaseView key={owner+':'+selected} owner={owner} postId={selected} refreshKey={refreshKey} onReviewed={()=>void load()}/>:<section className="moderation-case"><p>{t('选择一篇帖子查看内容、举报原因和处理记录。','Choose a post to read its content, reports and review history.')}</p></section>}</div></>}
 </Root>;
}
function ReviewCaseView({owner,postId,onReviewed,refreshKey}:{owner:string;postId:string;onReviewed:()=>void;refreshKey:number}){
 const {t}=useI18n(),[data,setData]=useState<ReviewCase>(),[offset,setOffset]=useState(0),[loading,setLoading]=useState(false),[error,setError]=useState(''),[notice,setNotice]=useState('');
 const [note,setNote]=useState(''),[action,setAction]=useState<'dismiss'|'remove'>('dismiss'),[consent,setConsent]=useState(false),[command,setCommand]=useState<ReviewCommand>(),[busy,setBusy]=useState(false),[recover,setRecover]=useState(false);
 const mounted=useRef(true),sequence=useRef(0),working=useRef(false),heading=useRef<HTMLHeadingElement>(null),acceptedVersion=useRef(''),path=`community/moderation/${encodeURIComponent(postId)}`;
 useEffect(()=>{mounted.current=true;return()=>{mounted.current=false;sequence.current++;};},[]);
 async function load(clear=false){const request=++sequence.current;setLoading(true);setError('');try{
  const value=await socialApi<ReviewCase>(`${path}?offset=${offset}`,undefined,AbortSignal.timeout(20_000));
  if(value.owner_id!==owner||value.post_id!==postId||value.offset!==offset||value.limit!==5||!Number.isSafeInteger(value.revision)||value.revision<1||!['pending','dismissed','removed'].includes(value.state)||value.post&&(value.post.id!==postId||typeof value.post.body!=='string'||!/^[a-f0-9]{64}$/.test(value.post.edit_version||''))||!Number.isSafeInteger(value.total_reports)||!Number.isSafeInteger(value.pending_reports)||value.pending_reports<0||value.total_reports<value.pending_reports||!Array.isArray(value.reports)||value.reports.length>5||value.reports.some(row=>row.snapshot?.id!==postId||typeof row.snapshot?.body!=='string'||!/^[a-f0-9]{64}$/.test(row.snapshot?.edit_version||'')||!row.id)||!Array.isArray(value.reviews))throw Error('The reviewed content could not be confirmed.');
  if(mounted.current&&request===sequence.current){const version=value.revision+':'+(value.post?.edit_version||'');if(acceptedVersion.current&&acceptedVersion.current!==version){setConsent(false);if(value.state==='pending')setNotice(t('内容或举报有更新，请重新检查；删除确认已重置。','Content or reports changed. Review them again; removal confirmation has been reset.'));}acceptedVersion.current=version;setData(value);if(clear){setCommand(undefined);setRecover(false);setConsent(false);}}
 }catch(e){if(mounted.current&&request===sequence.current)setError(errorMessage(e));}finally{if(mounted.current&&request===sequence.current)setLoading(false);}}
 useEffect(()=>{if(!working.current&&!command)void load();},[offset,refreshKey]);
 async function review(previous?:ReviewCommand){
  if(!data||loading||working.current||!previous&&(!!command||data.state!=='pending'||!note.trim()||action==='remove'&&!consent))return;
  const sent=previous||{expected_user_id:owner,request_id:reportRequestId(),expected_revision:data.revision,expected_content_version:data.post?.edit_version||null,action,note:note.trim()};
  working.current=true;setBusy(true);setError('');setNotice('');setCommand(sent);setRecover(false);
  try{const value=await socialApi<{owner_id:string;post_id:string;request_id:string;review_id:string;action:string;revision:number}>(path,sent,AbortSignal.timeout(20_000));
   if(value.owner_id!==owner||value.post_id!==postId||value.request_id!==sent.request_id||value.action!==sent.action||value.revision!==sent.expected_revision+1||!/^[a-f0-9]{32}$/.test(value.review_id))throw Error('The review decision is unconfirmed. Retry the same decision.');
   if(!mounted.current)return;setCommand(undefined);setConsent(false);setNotice(t('处理已记录，下方显示最新状态。','The decision is recorded. The latest state is shown below.'));void load();onReviewed();
  }catch(e){if(mounted.current){setError(errorMessage(e));setRecover(e instanceof SocialError&&[400,403,404,409].includes(e.status));}}
  finally{working.current=false;if(mounted.current)setBusy(false);}
 }
 const reasonTitle=(reason:string)=>{const label=reportReasons.find(row=>row[0]===reason);return label?t(label[1],label[2]):reason;};
 const focus=()=>heading.current?.focus({preventScroll:true});
 return <section className="moderation-case"><header><h2 ref={heading} tabIndex={-1}>{t('审核内容','Review content')}</h2><button type="button" className="text-button" disabled={loading||busy||!!command} onClick={()=>void load()}>{t('刷新内容','Refresh content')}</button></header>
  {loading&&<p role="status">{t('正在读取审核内容…','Loading review content…')}</p>}{notice&&<p role="status">{notice}</p>}
  {error&&<div role="alert"><p>{error}</p><button type="button" className="text-button" disabled={busy||loading} onClick={()=>{focus();if(command&&!recover)void review(command);else void load(!!command);}}>{command&&!recover?t('重试原审核决定','Retry original decision'):t('载入最新内容','Load latest content')}</button></div>}
  {data&&<><p>{t(`待审 ${data.pending_reports} 人 · 累计 ${data.total_reports} 人`,`${data.pending_reports} pending reporters · ${data.total_reports} total`)}</p>
   {data.post?<><h3>{data.post.title}</h3><a href={`#/community/${postId}`} target="_blank" rel="noopener noreferrer">{t('在新标签打开原帖','Open original post in a new tab')} <span aria-hidden="true">↗</span></a><p>{data.post.tags?.map(tag=>t(tag.zh,tag.en)).join(" · ")}</p><CommunityMarkdown body={data.post.body}/></>:<p>{t('帖子已不存在，可根据快照结案。','The post is unavailable. You can close this case using its snapshots.')}</p>}
   <h3>{t('举报记录与内容快照','Reports and content snapshots')}</h3><p>{t('举报人的身份不在此处展示。快照保留举报时的公开内容。','Reporter identities are not shown. Snapshots preserve the public content at reporting time.')}</p>
   {data.reports.map(report=><article key={report.id} className="moderation-report"><header><b>{reasonTitle(report.reason)}</b><DateLabel value={report.created_at}/><span>{report.pending?t('待审','Pending'):t('已审','Reviewed')}</span></header>{report.details&&<p className="moderation-report-note">{report.details}</p>}{data.post&&data.post.edit_version!==report.snapshot.edit_version&&<SnapshotChanges before={report.snapshot} after={data.post}/>}<details><summary>{t('查看举报时快照','View reported snapshot')}{data.post?.edit_version!==report.snapshot.edit_version&&<span> · {t('与当前内容不同','Different from current content')}</span>}</summary><h4>{report.snapshot.title}</h4><p>{report.snapshot.tags?.map(tag=>t(tag.zh,tag.en)).join(" · ")}</p><CommunityMarkdown body={report.snapshot.body}/></details></article>)}
   <nav aria-label={t('举报记录翻页','Report pages')}><button type="button" className="text-button" disabled={busy||loading||!!command||!offset} onClick={()=>setOffset(value=>Math.max(0,value-5))}>{t('上一页记录','Previous reports')}</button><span>{offset+1}–{Math.min(offset+5,data.total_reports)} / {data.total_reports}</span><button type="button" className="text-button" disabled={busy||loading||!!command||offset+5>=data.total_reports} onClick={()=>setOffset(value=>value+5)}>{t('下一页记录','Next reports')}</button></nav>
   <form onSubmit={event=>{event.preventDefault();void review();}}><h3>{data.state==='pending'?t('处理全部待审举报','Resolve all pending reports'):t('审核说明草稿','Review note draft')}</h3>
    <label>{t('审核说明（必填）','Review note (required)')}<textarea required rows={3} maxLength={1000} value={note} onChange={event=>setNote(event.target.value)}/></label>
    {data.state==='pending'?<><label>{t('处理方式','Decision')}<select value={action} onChange={event=>{setAction(event.target.value as 'dismiss'|'remove');setConsent(false);}}><option value="dismiss">{t('保留帖子，结案','Keep post and resolve')}</option><option value="remove">{t('删除帖子，结案','Remove post and resolve')}</option></select></label>{action==='remove'&&<label className="moderation-consent"><input type="checkbox" checked={consent} onChange={event=>setConsent(event.target.checked)}/><span>{t('确认删除帖子及其评论；举报快照和审核记录会保留。','Confirm removal of the post and its replies. Report snapshots and review history are retained.')}</span></label>}<button className="button" type="submit" disabled={busy||loading||!!error||!!command||!note.trim()||action==='remove'&&!consent}>{busy?t('正在确认…','Confirming…'):action==='remove'?t('确认删除并结案','Confirm removal and resolve'):t('保留帖子并结案','Keep post and resolve')}</button></>:<><p role="status">{data.state==='removed'?t('此帖已删除并结案。','This post was removed and the case resolved.'):t('此帖已保留并结案。新举报会重新进入待审队列。','This post was kept and the case resolved. New reporters reopen the case.')}</p><p>{t('后续输入仅保留在本页，不会改写已提交的审核说明。','Later input is kept on this page and does not change the submitted review note.')}</p></>}
   </form>
   <h3>{t('最近处理记录','Recent review history')}</h3>{data.reviews.length?data.reviews.map(row=><article className="moderation-report" key={row.id}><b>{row.action==='remove'?t('删除','Removed'):t('保留','Kept')}</b> · <DateLabel value={row.created_at}/><p>{row.note}</p></article>):<p>{t('暂无处理记录。','No review history yet.')}</p>}
  </>}
 </section>;
}
