import {appLocation} from './appLocation';
import AppLink from './AppLink';
import {useLayoutEffect,useRef,useState} from 'react';
import {ArrowRight,BookOpen,Code2,FilePenLine,RefreshCw,Trash2} from 'lucide-react';
import {useI18n} from './i18n';
import {useAccount} from './social';
import {hostedAppHref} from './deployment';
import {signInHref} from './accountNavigation';
import {communityPostHref} from './communityNavigation';
import {DraftPostComposer} from './Community';
import InterviewComposer from './InterviewComposer';
import CommunitySearch from './CommunitySearch';
import {postPreview} from './postExcerpt';
import {topicTitle,type ProblemTopic} from './communityTopics';
import {interviewLabel} from './interviewModels';
import {useWritingDisclosure} from './useWritingDisclosure';
import {keepReadingTriggerVisible} from './readingViewport';
import {discardInventoryDraft,matchesInventoryDraft,readDraftInventory,type DraftEntry,type DraftKind,type InventoryStorage} from './draftInventoryModel';
import './draft-center.css';

const tabStorage=()=>{try{return window.sessionStorage as InventoryStorage;}catch{return;}};
type Props={route:string;topics:ProblemTopic[];catalogLoading?:boolean;catalogError?:string;onRetryCatalog?:()=>void};
export default function DraftCenter(props:Props){
 const {user,loading}=useAccount(),{t}=useI18n();
 if(props.route!=='drafts')return <main className="social-page draft-center"><h1>{t('草稿链接无效','Invalid draft link')}</h1><AppLink className="button" href="#/drafts">{t('打开创作草稿','Open writing drafts')}</AppLink></main>;
 if(loading)return <main className="social-page draft-center"><p role="status">{t('正在确认账号…','Checking your account…')}</p></main>;
 if(!user)return <main className="social-page draft-center"><FilePenLine size={30} aria-hidden="true"/><h1>{t('登录后继续创作','Sign in to keep writing')}</h1><p>{t('帖子、题解和面经草稿按账号保存在当前标签页。','Post, solution and interview drafts stay in this tab, separately for each account.')}</p><AppLink className="button" href={hostedAppHref(signInHref('drafts'))}>{t('登录查看草稿','Sign in to view drafts')}</AppLink></main>;
 return <OwnedDrafts key={user.id} {...props} owner={user.id}/>;
}

function OwnedDrafts({owner,topics,catalogLoading,catalogError,onRetryCatalog}:Props&{owner:string}){
 const {t,locale}=useI18n(),{user,loading,capabilities}=useAccount(),actor=useRef(user?.id);actor.current=user?.id;
 const root=useRef<HTMLElement>(null),heading=useRef<HTMLHeadingElement>(null),controls=useRef<HTMLDivElement>(null);
 const [inventory,setInventory]=useState(()=>readDraftInventory(tabStorage(),owner)),[kind,setKind]=useState<DraftKind|'all'>('all'),[query,setQuery]=useState(''),[limit,setLimit]=useState(20);
 const [editing,setEditing]=useState<DraftEntry>(),[confirm,setConfirm]=useState<DraftEntry>(),[notice,setNotice]=useState('');
 const writing=useWritingDisclosure(owner+':draft-center');
 const focus=useRef<{href:string;opener:HTMLElement;target?:string;cancel?:boolean;discard?:boolean}>(undefined),confirmCancel=useRef<HTMLButtonElement>(null);
 const here=()=>!loading&&actor.current===owner&&!!root.current?.isConnected&&appLocation.hash==='#/drafts';
 const refresh=()=>{if(here()){setInventory(readDraftInventory(tabStorage(),owner));setLimit(20);}};
 const label=(value:DraftKind)=>value==='solution'?t('题解','Solution'):value==='interview'?t('面经','Interview'):t('帖子','Post');
 const searched=inventory.rows.filter(row=>matchesInventoryDraft(row,query));
 const filtered=searched.filter(row=>kind==='all'||row.kind===kind),shown=filtered.slice(0,limit);
 const context=(row:DraftEntry)=>{
  if(row.kind==='interview')return row.interview?interviewLabel('companies',row.interview.interview.company,locale==='en'):t('面试复盘','Interview experience');
  const id=row.scope.replace(/:solutions$/,'');if(id==='general')return t('社区','Community');
  const topic=topics.find(topic=>topic.id===id);return topic?topicTitle(topic,locale==='en'):id;
 };
 useLayoutEffect(()=>{
  const intent=focus.current;if(!intent)return;focus.current=undefined;
  const active=document.activeElement;
  if(!here()||location.href!==intent.href||active!==intent.opener&&!(active===document.body&&!intent.opener.isConnected))return;
  const row=intent.target?[...root.current!.querySelectorAll<HTMLElement>('[data-draft-id]')].find(node=>node.dataset.draftId===intent.target):undefined;
  const target=intent.cancel?confirmCancel.current:row?.querySelector<HTMLElement>(intent.discard?'[data-draft-discard]':'[data-draft-resume]')||heading.current;
  target?.focus({preventScroll:true});if(target)keepReadingTriggerVisible(target,100);
 },[confirm,inventory,limit,editing]);
 function closeEditor(){
  if(!here())return;
  const active=document.activeElement;
  if(editing&&active instanceof HTMLElement&&writing.panelProps.ref.current?.contains(active))focus.current={href:location.href,opener:active,target:editing.id};
  writing.close();setEditing(undefined);refresh();
 }
 function remove(row:DraftEntry,trigger:HTMLElement){
  if(!here()||editing)return;
  const index=filtered.findIndex(value=>value.id===row.id),target=filtered[index+1]?.id||filtered[index-1]?.id;
  const result=discardInventoryDraft(tabStorage(),owner,row);
  focus.current={opener:trigger,href:location.href,target};setConfirm(undefined);refresh();
  setNotice(result==='removed'?t('草稿已丢弃。','Draft discarded.'):result==='changed'?t('草稿已更新，新内容已保留。请重新核对后操作。','The draft changed. Your newer work is kept; review it before trying again.'):result==='pending'?t('发布状态待确认，原草稿和请求已保留。请进入编辑器继续确认。','Publication is unconfirmed. The draft and request are kept; continue in the editor.'):t('草稿暂时无法清除，内容已保留。请重试读取。','The draft could not be removed. Your work is kept; retry reading drafts.'));
 }
 const editingTopicId=editing?.kind!=='interview'?editing?.scope.replace(/:solutions$/,''):undefined;
 const editingTopic=editingTopicId&&editingTopicId!=='general'?(topics.find(topic=>topic.id===editingTopicId)||{id:editingTopicId,title:editingTopicId,retired:true}):undefined;
 const incompleteTopic=!!editingTopic?.retired;
 return <main ref={root} className="social-page draft-center">
  <header className="draft-center-heading"><div><span className="heading-kicker"><FilePenLine size={17} aria-hidden="true"/>{t('继续创作','KEEP WRITING')}</span><h1 ref={heading} tabIndex={-1}>{t('创作草稿','Writing drafts')}</h1><p>{t('继续完善帖子、题解和面经。这里显示当前账号在本标签页保存的草稿。','Pick up a post, solution or interview. These drafts belong to your current account in this tab.')}</p></div><button type="button" className="button subtle" onClick={()=>{refresh();setNotice('');}}><RefreshCw size={15} aria-hidden="true"/>{t('刷新草稿','Refresh drafts')}</button></header>
  <div className="draft-center-toolbar" ref={controls} tabIndex={-1}>
   <div className="draft-center-kinds" role="group" aria-label={t('按创作类型筛选','Filter by writing type')}>{(['all','post','solution','interview'] as const).map(value=><button key={value} type="button" aria-pressed={kind===value} onClick={()=>{setKind(value);setLimit(20);setConfirm(undefined);}}>{value==='all'?t('全部','All'):label(value)} <small>{value==='all'?searched.length:searched.filter(row=>row.kind===value).length}</small></button>)}</div>
   <CommunitySearch applied={query} onApply={value=>{setQuery(value);setLimit(20);setConfirm(undefined);}} label={t('搜索当前标签页的创作草稿','Search writing drafts in this tab')} placeholder={t('搜索标题、正文或题目 ID…','Search title, text or module ID…')}/>
  </div>
  <p className="draft-center-count" role="status">{inventory.failed||inventory.limited||inventory.unreadable?t(`已读取 ${inventory.rows.length} 份草稿`,`Read ${inventory.rows.length} drafts`):t(`共 ${inventory.rows.length} 份草稿`,`${inventory.rows.length} drafts`)} · {t(`显示 ${shown.length} / ${filtered.length} 份`,`Showing ${shown.length} of ${filtered.length}`)}</p>
  {inventory.failed&&<p className="draft-center-warning" role="alert">{t('部分草稿暂时无法读取，原数据已保留。可以重试读取；已找回的页面暂存内容仍可继续编辑。','Some drafts could not be read. Stored data is kept. Retry reading; recovered in-page work can still be edited.')}</p>}
  {inventory.limited&&<p className="draft-center-warning" role="status">{t('草稿记录较多，当前仅显示已读取的部分。原数据均保留。','There are many draft records. This view contains the entries read so far; all stored data is kept.')}</p>}
  {!!inventory.unreadable&&<p className="draft-center-warning" role="status">{t('部分草稿记录的格式暂不可读取，原数据已保留。','Some draft records use an unreadable format. Their original data is kept.')}</p>}
  {inventory.rows.some(row=>!row.durable)&&<p className="draft-center-warning" role="status">{t('标记为「页面暂存」的内容尚未完整写入本标签页，可以先继续编辑并确认保存状态。','Work marked “Held in this page” is not fully saved in this tab. Continue editing and check its save status.')}</p>}
  {notice&&<p className="draft-center-notice" role="status">{notice}</p>}
  {editing&&writing.expanded&&<div className="draft-center-editor">
   {incompleteTopic&&<p className="draft-center-warning" role="status">{catalogLoading?t('正在加载题目信息，草稿仍可编辑。','Loading module details. You can keep editing the draft.'):t('题目信息暂不可用，草稿仍可编辑；恢复题库后可发布新内容。','Module details are unavailable. Keep editing; new publication is available when the catalog recovers.')} {catalogError&&onRetryCatalog&&<button type="button" className="text-button" onClick={onRetryCatalog}>{t('重试题库','Retry catalog')}</button>}</p>}
   {editing.kind==='interview'?<InterviewComposer key={owner+':'+editing.id} owner={owner} company={editing.interview?.interview.company||''} problems={topics.filter(topic=>!topic.retired&&!!topic.number).map(topic=>({id:topic.id,number:topic.number!,title:topic.title,...(topic.title_en?{translations:{en:{title:topic.title_en,readme:'',notes:''}}}:{})}))} panelProps={writing.panelProps} onClose={closeEditor} onPublished={post=>{if(here())appLocation.hash=communityPostHref(post.id);}}/>:<DraftPostComposer key={owner+':'+editing.id} owner={owner} scope={editing.scope} topic={editingTopic} topics={topics} category={editing.post?.category||String(editing.pending?.receipt.payload.category||'papers')} solution={editing.kind==='solution'} papersAvailable={capabilities?.community_papers===true} publicationBlocked={incompleteTopic} returnTo={editing.kind==='solution'?'#/community?solutions=1':'#/community'} panelProps={writing.panelProps} onClose={closeEditor} onDraftChange={()=>{}}/>}
  </div>}
  <div className="draft-center-list">
   {shown.map(row=><article key={row.id} className="draft-center-card" data-draft-id={row.id}>
    <div className="draft-center-card-meta"><span>{row.kind==='interview'?<BookOpen size={14} aria-hidden="true"/>:row.kind==='solution'?<Code2 size={14} aria-hidden="true"/>:<FilePenLine size={14} aria-hidden="true"/>}{label(row.kind)}</span><span>{context(row)}</span>{row.pending&&<span className="draft-center-pending">{t('发布待确认','Publication unconfirmed')}</span>}{!row.durable&&<span>{t('页面暂存','Held in this page')}</span>}{(row.post?.anonymous||row.interview?.anonymous)&&<span>{t('匿名发布','Anonymous publication')}</span>}</div>
    <h2>{row.title.trim()||t('未命名草稿','Untitled draft')}</h2><p className="draft-center-excerpt">{postPreview(row.body,180,locale==='en'?'[formula]':'[公式]').text||t('已保留选择和设置，继续补充正文。','Your selections are kept. Continue with the text.')}</p>
    <footer><button type="button" data-draft-resume="" className="button subtle" aria-expanded={editing?.id===row.id&&writing.expanded} aria-controls={editing?.id===row.id&&writing.expanded?writing.panelProps.id:undefined} disabled={!!editing||!!confirm} onClick={event=>{if(!here())return;setEditing(row);writing.triggerProps.onClick(event);setNotice('');}}>{row.pending?t('继续确认发布','Continue publication'):t('继续编辑','Continue writing')}<ArrowRight size={14} aria-hidden="true"/></button><button type="button" data-draft-discard="" className="text-button draft-center-discard" disabled={!!editing||!!confirm||!!row.pending||inventory.failed} onClick={event=>{if(!here())return;focus.current={href:location.href,opener:event.currentTarget,cancel:true};setConfirm(row);}}><Trash2 size={14} aria-hidden="true"/>{t('丢弃','Discard')}</button></footer>
    {confirm?.id===row.id&&<div className="draft-center-confirm" role="alertdialog" aria-label={t('确认丢弃草稿','Confirm discarding draft')}><p>{t('丢弃这份未发布的草稿？这会移除当前标签页保存的内容。','Discard this unpublished draft? This removes the copy saved in this tab.')}</p><button type="button" className="button subtle" onClick={event=>remove(confirm,event.currentTarget)}>{t('确认丢弃','Discard draft')}</button><button ref={confirmCancel} type="button" className="text-button" onClick={event=>{focus.current={opener:event.currentTarget,href:location.href,target:row.id,discard:true};setConfirm(undefined);}}>{t('保留草稿','Keep draft')}</button></div>}
   </article>)}
  </div>
  {!shown.length&&!inventory.failed&&<section className="draft-center-empty"><FilePenLine size={30} aria-hidden="true"/><h2>{query||kind!=='all'?t('没有匹配的草稿','No matching drafts'):t('从一个想法开始','Start with an idea')}</h2><p>{query||kind!=='all'?t('试试其他关键词，或显示全部类型。','Try another search or show all writing types.'):t('在社区写帖子、分享题解或记录面经，未发布的内容可以在这里继续。','Start a community post, solution or interview. Return here to continue unpublished work.')}</p>{query||kind!=='all'?<button type="button" className="button subtle" onClick={()=>{controls.current?.focus({preventScroll:true});setQuery('');setKind('all');setLimit(20);}}>{t('显示全部草稿','Show all drafts')}</button>:<div><AppLink className="button subtle" href="#/community">{t('去社区写一篇','Write in the community')}</AppLink><AppLink className="button subtle" href="#/interviews">{t('分享面经','Share an interview')}</AppLink></div>}</section>}
  {filtered.length>shown.length&&<button type="button" className="button subtle draft-center-more" onClick={event=>{focus.current={opener:event.currentTarget,href:location.href,target:filtered[shown.length].id};setLimit(value=>value+20);}}>{t('再显示 20 份','Show 20 more')}</button>}
 </main>;
}
