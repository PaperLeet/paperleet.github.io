import ReviewUserMessage from './ReviewUserMessage';
import ModerationUserMessage from './ModerationUserMessage';
import {validReviewUserMessage} from './reviewUserMessageModel';
import ReviewDeliveryHint from './ReviewDeliveryHint';
import AppLink from './AppLink';
import {useEffect,useRef,useState} from 'react';
import {Flag,RefreshCw,ShieldCheck} from 'lucide-react';
import {DateLabel,errorMessage,SocialError,socialApi,useAccount} from './social';
import {useI18n} from './i18n';
import {signInHref} from './accountNavigation';
import {reportReasons,reportRequestId,type ReviewCase,type ReviewCommand,type ReviewPost} from './communityReports';
import CommunityMarkdown from './CommunityMarkdown';
import {deferredFeature} from './deferredFeature';
import './community-reports.css';
import './moderation.css';
import type {AdminView} from './adminNavigation';
import {keepReadingTriggerVisible} from './readingViewport';
import {confirmedModerationQueue,type ModerationQueueData as Queue} from './moderationQueueModel';

type Navigation={view:AdminView;onChange:(view:AdminView)=>void};
const SnapshotComparison=deferredFeature(()=>import('./ModerationSnapshotComparison'),['内容对比','content comparison']);
function SnapshotChanges({before,after}:{before:ReviewPost;after:ReviewPost}){
 const {t}=useI18n(),[visited,setVisited]=useState(false);
 return <details className="moderation-change-disclosure" onToggle={event=>{if(event.currentTarget.open)setVisited(true);}}><summary>{t('对比当前内容','Compare with current content')}<small>{t('标题、正文与分类','Title, body and category')}</small></summary>{visited&&<SnapshotComparison before={before} after={after}/>}</details>;
}
export default function ModerationQueue({embedded=false,refreshKey=0,initialPostId,navigation}:{embedded?:boolean;refreshKey?:number;initialPostId?:string;navigation?:Navigation}){
 const {user,loading,capabilities}=useAccount(),{t}=useI18n(),Root=embedded?'section':'main';
 if(loading)return <Root className="moderation-page"><p role="status">{t('正在确认审核权限…','Checking review access…')}</p></Root>;
 if(!user)return <Root className="moderation-page"><h1>{t('社区举报审核','Community reports')}</h1><AppLink href={signInHref('#/community/moderation')}>{t('登录管理员账号','Sign in as an administrator')}</AppLink></Root>;
 if(user.role!=='admin'||user.status&&user.status!=='active')return <Root className="moderation-page"><h1>{t('需要管理员权限','Administrator access required')}</h1></Root>;
 if(capabilities?.community_reports!==true)return <Root className="moderation-page"><p>{t('此服务暂不支持举报审核。','Report review is unavailable on this service.')}</p></Root>;
 return <QueueView key={user.id} owner={user.id} embedded={embedded} refreshKey={refreshKey} initialPostId={initialPostId} navigation={navigation}/>;
}
function QueueView({owner,embedded=false,refreshKey=0,initialPostId='',navigation}:{owner:string;embedded?:boolean;refreshKey?:number;initialPostId?:string;navigation?:Navigation}){
 const {t}=useI18n(),[localState,setState]=useState('pending'),[localMinimum,setMinimum]=useState(1),[localOffset,setOffset]=useState(0),[localSelected,setSelected]=useState(initialPostId);
 const state=navigation?.view.state||localState,minimum=Number(navigation?.view.minimum||localMinimum),offset=navigation?(navigation.view.page-1)*20:localOffset,selected=navigation?navigation.view.item||'':localSelected;
 const scope=JSON.stringify([owner,state,minimum,offset]),[snapshot,setSnapshot]=useState<{scope:string;data:Queue}>(),[request,setRequest]=useState<{scope:string;loading:boolean;error:string}>();
 const data=snapshot?.scope===scope?snapshot.data:undefined,loading=request?.scope!==scope||request.loading,error=request?.scope===scope?request.error:'';
 const sequence=useRef(0),mounted=useRef(true),transport=useRef<{controller:AbortController;timer:ReturnType<typeof setTimeout>}|undefined>(undefined),queueHeading=useRef<HTMLParagraphElement>(null);
 const abort=()=>{if(transport.current){transport.current.controller.abort();clearTimeout(transport.current.timer);transport.current=undefined;}};
 useEffect(()=>{mounted.current=true;return()=>{mounted.current=false;sequence.current++;abort();};},[]);
 async function load(){abort();const ticket=++sequence.current,controller=new AbortController(),timer=setTimeout(()=>{controller.abort();if(mounted.current&&ticket===sequence.current){sequence.current++;setRequest({scope,loading:false,error:t('读取举报超时，请重试。','Report queue request timed out. Retry loading.')});}},20_000);transport.current={controller,timer};setRequest({scope,loading:true,error:''});try{
  const value=confirmedModerationQueue(await socialApi<Queue>(`community/moderation?state=${state}&minimum=${minimum}&limit=20&offset=${offset}`,undefined,controller.signal),owner,state,minimum,offset);
  if(mounted.current&&ticket===sequence.current&&!controller.signal.aborted)setSnapshot({scope,data:value});
 }catch(e){if(mounted.current&&ticket===sequence.current)setRequest({scope,loading:false,error:errorMessage(e)});}
 finally{clearTimeout(timer);if(mounted.current&&ticket===sequence.current)setRequest(previous=>previous?.scope===scope?{...previous,loading:false}:previous);}}
 useEffect(()=>{void load();return()=>{sequence.current++;abort();};},[scope,refreshKey]);
 useEffect(()=>{if(!navigation)setSelected(initialPostId);},[initialPostId]);
 function filter(update:{state?:string;minimum?:number}){if(navigation)navigation.onChange({...navigation.view,state:update.state??navigation.view.state,minimum:update.minimum===undefined?navigation.view.minimum:String(update.minimum),page:1,item:undefined});else{if(update.state!==undefined)setState(update.state);if(update.minimum!==undefined)setMinimum(update.minimum);setOffset(0);setSelected('');}}
 function select(item:string){if(navigation)navigation.onChange({...navigation.view,item:item||undefined});else setSelected(item);}
 function page(next:number){if(!Number.isSafeInteger(next)||next<0||next>10000||next===offset)return;if(queueHeading.current){queueHeading.current.focus({preventScroll:true});keepReadingTriggerVisible(queueHeading.current);}if(navigation)navigation.onChange({...navigation.view,page:next/20+1,item:undefined});else{setOffset(next);setSelected('');}}
 const lastOffset=data?Math.min(10000,Math.max(0,Math.ceil(data.total/20)-1)*20):0;
 const Root=embedded?"section":"main";
 return <Root className="moderation-page">{embedded?<p className="admin-intro">{t('按待审举报人数优先排列。每个账号对同一帖子只计一票，举报不会自动删除帖子。','Prioritized by pending reporters. Each account counts once per post. Reports do not automatically remove content.')}</p>:<header className="moderation-heading"><div><span><ShieldCheck size={17}/>{t('管理员工作台','Administrator workspace')}</span><h1>{t('社区举报审核','Community reports')}</h1><p>{t('按待审举报人数优先排列。每个账号对同一帖子只计一票，举报不会自动删除帖子。','Prioritized by pending reporters. Each account counts once per post. Reports do not automatically remove content.')}</p></div><AppLink href="#/community">{t('返回社区','Back to community')}</AppLink></header>}
  <div className="moderation-filters"><label>{t('审核状态','Review status')}<select value={state} onChange={event=>{filter({state:event.target.value});}}><option value="pending">{t('待审核','Pending')}</option><option value="resolved">{t('已处理','Resolved')}</option><option value="all">{t('全部','All')}</option></select></label><label>{t('累计举报至少','At least this many reports')}<select value={minimum} onChange={event=>{filter({minimum:Number(event.target.value)});}}>{[1,3,5,10].map(n=><option key={n} value={n}>{n}</option>)}</select></label><button className="button subtle" type="button" disabled={loading} onClick={()=>void load()}><RefreshCw size={15}/>{t('刷新队列','Refresh queue')}</button></div>
  {loading&&<p role="status">{t('正在读取举报…','Loading reports…')}</p>}{error&&<p role="alert">{error} <button type="button" className="text-button" disabled={loading} onClick={()=>void load()}>{t('重试加载队列','Retry queue loading')}</button></p>}
  <p className="moderation-count" ref={queueHeading} tabIndex={-1} role="status">{data?t(`共 ${data.total} 个帖子 · 第 ${offset/20+1} 页`,`${data.total} posts · Page ${offset/20+1}`):error?t('举报队列暂不可用','Report queue unavailable'):t('正在确认当前队列…','Checking the current queue…')}</p>{selected&&navigation&&<div className="moderation-return"><button type="button" className="text-button" onClick={()=>{queueHeading.current?.focus({preventScroll:true});select('');}}>{t('返回本页队列','Return to this queue page')}</button><span>{t('已保留审核状态、举报人数与页码。','Review status, reporter threshold and page are kept.')}</span></div>}{data&&<><div className="moderation-layout"><section className="moderation-list" aria-label={t('举报帖子','Reported posts')}>
   {!data.items.length&&<div className="moderation-empty"><p>{offset?t('这一页已没有举报帖子，筛选仍保留。','This page is empty. Your filters are kept.'):t('当前筛选下没有举报帖子。','No reported posts match these filters.')}</p>{offset>0&&<button type="button" className="text-button" disabled={loading||!!error} onClick={()=>page(lastOffset)}>{t('返回最后一个可查看页','Go to the last available page')}</button>}</div>}
   {data.items.map(row=><button type="button" key={row.post_id} disabled={loading||!!error} aria-pressed={selected===row.post_id} onClick={()=>select(row.post_id)}><span className="moderation-report-count"><Flag size={14}/>{t(`待审 ${row.pending_reports} · 累计 ${row.total_reports}`,`${row.pending_reports} pending · ${row.total_reports} total`)}</span><b>{row.post?.title||row.title}</b><small>{row.state==='pending'?t('待审核','Pending'):row.state==='removed'?t('已删除','Removed'):!row.post?t('已结案，内容不可用','Resolved; content unavailable'):t('已保留','Kept')} · <DateLabel value={row.last_report_at}/></small>{!row.post&&<small>{t('帖子已不存在，举报快照仍保留','Post unavailable; report snapshots retained')}</small>}</button>)}
   <nav aria-label={t('审核队列翻页','Review queue pages')}><button type="button" className="text-button" disabled={loading||!!error||!offset} onClick={()=>page(Math.max(0,offset-20))}>{t('上一页','Previous')}</button><span>{Math.floor(offset/20)+1}</span><button type="button" className="text-button" disabled={loading||!!error||offset>=10000||offset+20>=data.total} onClick={()=>page(offset+20)}>{t('下一页','Next')}</button></nav>{offset>=10000&&data.total>10020&&<p role="status">{t('已到可查看范围的最后一页，请调整筛选缩小队列。','You reached the last available page. Narrow the filters to browse further.')}</p>}
  </section>{selected?<ReviewCaseView key={owner+':'+selected} owner={owner} postId={selected} refreshKey={refreshKey} onReviewed={()=>void load()}/>:<section className="moderation-case"><p>{t('选择一篇帖子查看内容、举报原因和处理记录。','Choose a post to read its content, reports and review history.')}</p></section>}</div></>}
 </Root>;
}
function ReviewCaseView({owner,postId,onReviewed,refreshKey}:{owner:string;postId:string;onReviewed:()=>void;refreshKey:number}){
 const {capabilities}=useAccount(),messagesAvailable=capabilities?.moderation_user_messages===true;
 const [userMessage,setUserMessage]=useState('');
 const {t}=useI18n(),[data,setData]=useState<ReviewCase>(),[offset,setOffset]=useState(0),[loading,setLoading]=useState(false),[error,setError]=useState(''),[notice,setNotice]=useState('');
 const [note,setNote]=useState(''),[action,setAction]=useState<'dismiss'|'remove'>('dismiss'),[consent,setConsent]=useState(false),[command,setCommand]=useState<ReviewCommand>(),[busy,setBusy]=useState(false),[recover,setRecover]=useState(false);
 const mounted=useRef(true),sequence=useRef(0),working=useRef(false),heading=useRef<HTMLHeadingElement>(null),acceptedVersion=useRef(''),path=`community/moderation/${encodeURIComponent(postId)}`;
 useEffect(()=>{mounted.current=true;return()=>{mounted.current=false;sequence.current++;};},[]);
 async function load(clear=false){const request=++sequence.current;setLoading(true);setError('');try{
  const value=await socialApi<ReviewCase>(`${path}?offset=${offset}`,undefined,AbortSignal.timeout(20_000));
  if(value.owner_id!==owner||value.post_id!==postId||value.offset!==offset||value.limit!==5||!Number.isSafeInteger(value.revision)||value.revision<1||!['pending','dismissed','removed'].includes(value.state)||value.post&&(value.post.id!==postId||typeof value.post.body!=='string'||!/^[a-f0-9]{64}$/.test(value.post.edit_version||''))||!Number.isSafeInteger(value.total_reports)||!Number.isSafeInteger(value.pending_reports)||value.pending_reports<0||value.total_reports<value.pending_reports||!Array.isArray(value.reports)||value.reports.length>5||value.reports.some(row=>row.snapshot?.id!==postId||typeof row.snapshot?.body!=='string'||!/^[a-f0-9]{64}$/.test(row.snapshot?.edit_version||'')||!row.id)||!Array.isArray(value.reviews)||value.reviews.some(row=>row.user_message!==undefined&&!validReviewUserMessage(row.user_message)))throw Error('The reviewed content could not be confirmed.');
  if(mounted.current&&request===sequence.current){const version=value.revision+':'+(value.post?.edit_version||'');if(acceptedVersion.current&&acceptedVersion.current!==version){setConsent(false);if(value.state==='pending')setNotice(t('内容或举报有更新，请重新检查；删除确认已重置。','Content or reports changed. Review them again; removal confirmation has been reset.'));}acceptedVersion.current=version;setData(value);if(clear){setCommand(undefined);setRecover(false);setConsent(false);}}
 }catch(e){if(mounted.current&&request===sequence.current)setError(errorMessage(e));}finally{if(mounted.current&&request===sequence.current)setLoading(false);}}
 useEffect(()=>{if(!working.current&&!command)void load();},[offset,refreshKey]);
 async function review(previous?:ReviewCommand){
  if(!data||loading||working.current||!previous&&(!!command||data.state!=='pending'||!note.trim()||messagesAvailable&&(!validReviewUserMessage(userMessage)||action==='remove'&&!userMessage.trim())||action==='remove'&&!consent))return;
  const sent=previous||{expected_user_id:owner,request_id:reportRequestId(),expected_revision:data.revision,expected_content_version:data.post?.edit_version||null,action,note:note.trim(),...(messagesAvailable?{user_message:userMessage.trim()}:{})};
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
  {command?.user_message!==undefined&&<section className="moderation-user-message-preview" aria-label={t('待确认操作的用户说明','Unconfirmed user explanation')}><b>{t('原操作的用户说明','Explanation captured in the original operation')}</b><small>{command.action==='remove'?t('原操作：移除帖子','Original decision: remove post'):t('原操作：不执行移除，结案','Original decision: resolve without removal')} · {command.action==='remove'&&command.expected_content_version!==null?t('作者与本批待审举报人','Author and pending reporters'):t('本批待审举报人','Pending reporters')}</small><p>{command.user_message||t('未附用户说明','No user explanation')}</p><small>{t('重试沿用这段内容；后续编辑仍未提交。','Retry uses this captured text. Later edits remain unsent.')}</small></section>}
  {data&&<><p>{t(`待审 ${data.pending_reports} 人 · 累计 ${data.total_reports} 人`,`${data.pending_reports} pending reporters · ${data.total_reports} total`)}</p>
   {data.post?<><h3>{data.post.title}</h3><AppLink href={`#/community/${postId}`} target="_blank" rel="noopener noreferrer">{t('在新标签打开原帖','Open original post in a new tab')} <span aria-hidden="true">↗</span></AppLink><p>{data.post.tags?.map(tag=>t(tag.zh,tag.en)).join(" · ")}</p><CommunityMarkdown body={data.post.body}/></>:<p>{t('帖子已不存在，可根据快照结案。','The post is unavailable. You can close this case using its snapshots.')}</p>}
   <h3>{t('举报记录与内容快照','Reports and content snapshots')}</h3><p>{t('举报人的身份不在此处展示。快照保留举报时的公开内容。','Reporter identities are not shown. Snapshots preserve the public content at reporting time.')}</p>
   {data.reports.map(report=><article key={report.id} className="moderation-report"><header><b>{reasonTitle(report.reason)}</b><DateLabel value={report.created_at}/><span>{report.pending?t('待审','Pending'):t('已审','Reviewed')}</span></header>{report.details&&<p className="moderation-report-note">{report.details}</p>}{data.post&&data.post.edit_version!==report.snapshot.edit_version&&<SnapshotChanges before={report.snapshot} after={data.post}/>}<details><summary>{t('查看举报时快照','View reported snapshot')}{data.post?.edit_version!==report.snapshot.edit_version&&<span> · {t('与当前内容不同','Different from current content')}</span>}</summary><h4>{report.snapshot.title}</h4><p>{report.snapshot.tags?.map(tag=>t(tag.zh,tag.en)).join(" · ")}</p><CommunityMarkdown body={report.snapshot.body}/></details></article>)}
   <nav aria-label={t('举报记录翻页','Report pages')}><button type="button" className="text-button" disabled={busy||loading||!!command||!offset} onClick={()=>setOffset(value=>Math.max(0,value-5))}>{t('上一页记录','Previous reports')}</button><span>{offset+1}–{Math.min(offset+5,data.total_reports)} / {data.total_reports}</span><button type="button" className="text-button" disabled={busy||loading||!!command||offset+5>=data.total_reports} onClick={()=>setOffset(value=>value+5)}>{t('下一页记录','Next reports')}</button></nav>
   <form onSubmit={event=>{event.preventDefault();void review();}}><h3>{data.state==='pending'?t('处理全部待审举报','Resolve all pending reports'):t('审核说明草稿','Review note draft')}</h3>
    <ReviewDeliveryHint/>
    {data.state==='pending'&&<label>{t('处理方式','Decision')}<select value={action} onChange={event=>{setAction(event.target.value as 'dismiss'|'remove');setConsent(false);}}><option value="dismiss">{t('保留帖子，结案','Keep post and resolve')}</option><option value="remove">{t('删除帖子，结案','Remove post and resolve')}</option></select></label>}
    <label>{messagesAvailable?t('内部审核备注（必填，仅管理员可见）','Internal review note (required, administrators only)'):t('审核说明（必填）','Review note (required)')}<textarea required rows={3} maxLength={1000} value={note} onChange={event=>setNote(event.target.value)}/></label>
    {messagesAvailable&&<ModerationUserMessage value={userMessage} onChange={value=>{setUserMessage(value);setConsent(false);}} remove={action==='remove'} postAvailable={!!data.post} draftOnly={!!command||data.state!=='pending'}/>}
    {data.state==='pending'?<>{action==='remove'&&<label className="moderation-consent"><input type="checkbox" checked={consent} onChange={event=>setConsent(event.target.checked)}/><span>{messagesAvailable?t('确认删除帖子及评论，并发送上述用户说明。举报快照和审核记录会保留。','Confirm removal and delivery of the explanation above. Report snapshots and review history are retained.'):t('确认删除帖子及其评论；举报快照和审核记录会保留。','Confirm removal of the post and its replies. Report snapshots and review history are retained.')}</span></label>}<button className="button" type="submit" disabled={busy||loading||!!error||!!command||!note.trim()||messagesAvailable&&(!validReviewUserMessage(userMessage)||action==='remove'&&!userMessage.trim())||action==='remove'&&!consent}>{busy?t('正在确认…','Confirming…'):action==='remove'?t('确认删除并结案','Confirm removal and resolve'):t('保留帖子并结案','Keep post and resolve')}</button></>:<><p role="status">{data.state==='removed'?t('此帖已删除并结案。','This post was removed and the case resolved.'):!data.post?t('此案已处理，原帖已不可用；举报快照与记录仍保留。','This case is resolved; the original post is unavailable. Snapshots and records are retained.'):t('此帖已保留并结案。新举报会重新进入待审队列。','This post was kept and the case resolved. New reporters reopen the case.')}</p><p>{t('后续输入仅保留在本页，不会改写已提交的审核说明。','Later input is kept on this page and does not change the submitted review note.')}</p></>}
   </form>
   <h3>{t('最近处理记录','Recent review history')}</h3>{data.reviews.length?data.reviews.map(row=><article className="moderation-report" key={row.id}><b>{row.action==='remove'?t('删除','Removed'):t('保留','Kept')}</b> · <DateLabel value={row.created_at}/><small>{t('内部备注','Internal note')}</small><p>{row.note}</p><ReviewUserMessage message={row.user_message} label={t('给用户的说明','Message delivered to users')}/></article>):<p>{t('暂无处理记录。','No review history yet.')}</p>}
  </>}
 </section>;
}
