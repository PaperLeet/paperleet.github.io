import {appLocation} from './appLocation';
import AppLink from './AppLink';
import {useRef,useState} from 'react';
import {ArrowDown,ArrowLeft,ArrowRight,ArrowUp,Check,FolderPlus,Pencil,Plus,Trash2,X} from 'lucide-react';
import {useI18n} from './i18n';
import {useAccount} from './social';
import {useStudyLists} from './useStudyLists';
import {CreateStudyList,StudyListFeedback} from './StudyListControls';
import {usePracticeProgress} from './PracticeProgressContext';
import {favoriteHref,readFavoriteRoute,withLibraryContext} from './libraryNavigation';
import {signInHref} from './accountNavigation';
import {hostedAppHref} from './deployment';
import type {Problem} from './types';
import {STUDY_LIST_ITEM_LIMIT,type StudyList} from './studyListStore';
import './study-lists.css';

type Props={route?:string;compact?:boolean;onFilters?:(href:string)=>void;problems:Problem[];catalogReady:boolean;catalogError?:string;onRetryCatalog:()=>void};
export default function StudyLists(props:Props){
 const {user,loading}=useAccount();
 return <Content key={loading?'checking':user?.id||'guest'} {...props}/>;
}
function useListProgress(){
 const progress=usePracticeProgress(),{user,loading,health}=useAccount();
 const expected=loading?'unresolved':health?.mode==='local'?'local':health?.mode==='hosted'?user?'user:'+user.id:'anonymous':'unresolved';
 const ready=expected!=='unresolved'&&progress?.owner===expected&&progress.ready&&!progress.failed;
 return {ready,read:(id:string)=>ready?progress?.read(id):undefined,progress};
}
function Content({route='favorites',compact=false,onFilters,problems,catalogReady,catalogError,onRetryCatalog}:Props){
 const {t}=useI18n(),{user,loading,capabilities,state,store}=useStudyLists(),heading=useRef<HTMLHeadingElement>(null),view=readFavoriteRoute(route),selected=state.snapshot?.lists.find(row=>row.id===view?.list);
 const Wrapper=compact?'section':'main',Title=compact?'h2':'h1';
 const back=<AppLink className="back-link" href="#/favorites"><ArrowLeft size={15}/>{t('全部收藏','All saved problems')}</AppLink>;
 const focus=()=>heading.current?.focus({preventScroll:true});
 if(compact&&(!user||loading||capabilities?.study_lists!==true))return null;
 return <Wrapper className={`study-lists${compact?' study-lists-compact':' study-lists-page'}`}>
  {!compact&&back}<header className="study-lists-heading"><div><Title ref={heading} tabIndex={-1}>{selected?.title||t('私人题单','Private study lists')}</Title><p>{t('按目标整理练习，仅自己可见并同步到账号。','Organize practice by goal. Private and synced to your account.')}</p></div>{!compact&&<AppLink href="#">{t('浏览题库','Browse papers')}<ArrowRight size={14}/></AppLink>}</header>
  {loading?<p role="status">{t('正在确认账号…','Checking your account…')}</p>:!user?<AppLink className="button" href={hostedAppHref(signInHref(favoriteHref(view||{query:'',state:'all'})))}>{t('登录查看题单','Sign in to view study lists')}</AppLink>:capabilities?.study_lists!==true?<p role="status">{t('此服务暂未开放私人题单，仍可使用原有收藏。','Private study lists are unavailable on this service. Saved problems remain available.')}</p>:<>
   <StudyListFeedback state={state} store={store} focus={focus}/>
   {!state.snapshot?<p role="status">{state.loading?t('正在读取题单…','Loading study lists…'):t('题单尚未读取。','Study lists have not loaded.')}</p>:<>
    <nav className="study-list-nav" aria-label={t('我的题单','Your study lists')}><AppLink href="#/favorites" aria-current={!view?.list?'page':undefined}>{t('全部收藏','All saved problems')}</AppLink>{state.snapshot.lists.map(row=><AppLink key={row.id} href={favoriteHref({query:'',state:'all',list:row.id})} aria-current={row.id===view?.list?'page':undefined}>{row.title}<small>{row.items.length}</small></AppLink>)}</nav>
    {view?.list?selected?<ListDetail key={selected.id} list={selected} route={route} onFilters={onFilters} problems={problems} catalogReady={catalogReady} catalogError={catalogError} onRetryCatalog={onRetryCatalog} state={state} store={store} onDeleteFocus={focus}/>:<div className="study-list-empty"><p>{t('此题单不存在，或不属于当前账号。','This list does not exist or does not belong to this account.')}</p>{back}</div>:<>
     {!state.snapshot.lists.length&&<p>{t('可以按“面试准备”“需要复习”等目标建立题单，再从题目页或题单中添加练习。','Create lists for goals such as interview preparation or review, then add exercises here or from a problem.')}</p>}
     <CreateStudyList state={state} store={store}/>
    </>}
   </>}
  </>}
 </Wrapper>;
}
type DetailProps=Props&{list:StudyList;state:ReturnType<typeof useStudyLists>['state'];store:ReturnType<typeof useStudyLists>['store'];onDeleteFocus:()=>void};
function ListDetail({list,route='favorites',onFilters,problems,catalogReady,catalogError,onRetryCatalog,state,store,onDeleteFocus}:DetailProps){
 const {t,locale}=useI18n(),view=readFavoriteRoute(route)!,progress=useListProgress(),toolbar=useRef<HTMLDivElement>(null);
 const [visitedRenaming,setVisitedRenaming]=useState(false),renameForm=useRef<HTMLFormElement>(null),addingPanel=useRef<HTMLElement>(null),[visitedAdding,setVisitedAdding]=useState(false);
 const [renaming,setRenaming]=useState(false),[name,setName]=useState(list.title),[deleting,setDeleting]=useState(false),[adding,setAdding]=useState(false),[search,setSearch]=useState(''),[chosen,setChosen]=useState<string[]>([]);
 const blocked=!!state.pending||state.loading||!!state.error,byId=new Map(problems.map(problem=>[problem.id,problem])),ids=list.items.map(item=>item.problem_id),title=(p:Problem)=>locale==='en'?p.translations?.en?.title||p.title:p.title;
 const matches=(p:Problem,q:string)=>[p.id,p.number,p.title,p.translations?.en?.title].join(' ').toLocaleLowerCase().includes(q.trim().toLocaleLowerCase());
 const available=ids.flatMap(id=>byId.has(id)?[byId.get(id)!]:[]),ready=catalogReady&&!catalogError&&progress.ready&&available.every(p=>progress.read(p.id)),done=ready?available.filter(p=>progress.read(p.id)?.state==='done').length:undefined;
 const shown=ids.filter(id=>{const p=byId.get(id);return (!view.query.trim()||(p?matches(p,view.query):id.includes(view.query.trim())))&&(view.state==='all'||!!p&&ready&&progress.read(id)?.state===view.state);});
 const candidates=shown.flatMap(id=>byId.has(id)?[byId.get(id)!]:[]),next=ready?(candidates.find(p=>progress.read(p.id)?.state==='started')||candidates.find(p=>progress.read(p.id)?.state==='todo')||candidates[0]):undefined;
 const returnTo=favoriteHref(view),reordering=!view.query&&view.state==='all';
 function change(patch:Partial<typeof view>){const href=favoriteHref({...view,...patch});if(onFilters)onFilters(href);else appLocation.hash=href.slice(1);}
 function beforeRemoval(event:React.MouseEvent<HTMLButtonElement>){if(document.activeElement===event.currentTarget)toolbar.current?.focus({preventScroll:true});}
 return <section className="study-list-detail" aria-label={t('题单内容','Study-list contents')}>
  <div ref={toolbar} className="study-list-toolbar" role="group" tabIndex={-1} aria-label={t('题单操作','Study-list actions')}><span>{t(`${ids.length} 道题 · ${available.length} 道可用`,`${ids.length} problems · ${available.length} available`)}{available.length>0&&done!==undefined?` · ${t(`已通过 ${done}/${available.length} 道可用题`,`Solved ${done}/${available.length} available`)}`:''}</span><div><button type="button" className="text-button" onClick={event=>{if(adding&&addingPanel.current?.contains(document.activeElement))event.currentTarget.focus({preventScroll:true});setVisitedAdding(true);setAdding(value=>!value);}} aria-expanded={adding}><Plus size={15}/>{t('添加题目','Add problems')}</button><button type="button" className="text-button" onClick={event=>{if(renaming&&renameForm.current?.contains(document.activeElement))event.currentTarget.focus({preventScroll:true});if(!visitedRenaming){setName(list.title);setVisitedRenaming(true);}setRenaming(value=>!value);}} aria-expanded={renaming}><Pencil size={14}/>{t('重命名','Rename')}</button><button type="button" className="text-button" disabled={blocked} onClick={()=>setDeleting(true)}><Trash2 size={14}/>{t('删除题单','Delete list')}</button></div></div>
  {visitedRenaming&&<form ref={renameForm} hidden={!renaming} className="study-list-create" onSubmit={event=>{event.preventDefault();if(blocked||!name.trim()||Array.from(name.trim()).length>60)return;void store?.mutate({action:'rename',list_id:list.id,title:name.trim()});}}><label><span>{t('题单名称','Study-list name')}</span><input value={name} onChange={event=>setName(event.target.value)} required maxLength={120}/></label><button type="submit" className="button subtle" disabled={blocked||!name.trim()||Array.from(name.trim()).length>60}>{t('保存名称','Save name')}</button></form>}
  {deleting&&<div className="study-list-delete" role="group" aria-label={t('确认删除题单','Confirm list deletion')}><p>{t(`删除“${list.title}”？只删除此题单，原有收藏、代码和解题记录会保留。`,`Delete “${list.title}”? Saved problems, code and practice history are kept.`)}</p><button type="button" className="button subtle" disabled={blocked} onClick={event=>{if(document.activeElement===event.currentTarget)onDeleteFocus();void store?.mutate({action:'delete',list_id:list.id});}}>{t('确认删除题单','Confirm deletion')}</button><button type="button" className="text-button" disabled={blocked} onClick={()=>setDeleting(false)}>{t('取消','Cancel')}</button></div>}
  {catalogError&&<div className="study-list-feedback" role="status"><p>{t('题库暂不可用，题单内容仍会保留。','The catalog is unavailable. Your list is kept.')}</p><button type="button" className="text-button" onClick={onRetryCatalog}>{t('重试题库','Retry catalog')}</button></div>}
  {visitedAdding&&<section ref={addingPanel} hidden={!adding} className="study-list-add" aria-label={t('选择要加入的题目','Choose problems to add')}><label>{t('搜索题库','Search catalog')}<input type="search" value={search} onChange={event=>setSearch(event.target.value)} maxLength={200}/></label><div className="study-list-options">{catalogReady&&!catalogError?problems.filter(p=>matches(p,search)).map(p=><label key={p.id}><input type="checkbox" disabled={ids.includes(p.id)||!chosen.includes(p.id)&&ids.length+chosen.filter(id=>!ids.includes(id)).length>=STUDY_LIST_ITEM_LIMIT} checked={chosen.includes(p.id)||ids.includes(p.id)} onChange={event=>setChosen(current=>event.target.checked?[...current,p.id]:current.filter(id=>id!==p.id))}/><span>{p.number}. {title(p)}{ids.includes(p.id)&&<small>{t('已在题单中','Already in this list')}</small>}</span></label>):<p role="status">{t('等待题库加载…','Waiting for the catalog…')}</p>}</div><p>{t(`每个题单最多 ${STUDY_LIST_ITEM_LIMIT} 道题，已有题目不会重复添加。`,`Each list can hold ${STUDY_LIST_ITEM_LIMIT} problems. Existing problems are not added twice.`)}</p><div className="study-list-add-actions"><button type="button" className="button" disabled={blocked||!catalogReady||!!catalogError||!chosen.length||ids.length+chosen.filter(id=>!ids.includes(id)).length>STUDY_LIST_ITEM_LIMIT||chosen.some(id=>!byId.has(id))} onClick={()=>{const selected=[...chosen];void store?.mutate({action:'add',list_id:list.id,problem_ids:selected}).then(ok=>{if(ok)setChosen(current=>current.filter(id=>!selected.includes(id)));});}}>{t(`加入 ${chosen.length} 道题`,`Add ${chosen.length} problems`)}</button><button type="button" className="text-button" onClick={()=>setChosen([])}>{t('清空选择','Clear selection')}</button></div></section>}
  <div className="study-list-filters"><label>{t('搜索题单','Search this list')}<input type="search" value={view.query} maxLength={200} onChange={event=>change({query:event.target.value})}/></label><label>{t('学习状态','Practice status')}<select value={view.state} onChange={event=>change({state:event.target.value})}><option value="all">{t('全部状态','All statuses')}</option>{(['todo','started','done']as const).map((value,index)=><option key={value} value={value} disabled={!ready}>{[t('未开始','Not started'),t('进行中','In progress'),t('已通过','Solved')][index]}</option>)}</select></label>{next&&<AppLink className="button" href={withLibraryContext(next.number,returnTo)}>{progress.read(next.id)?.state==='done'?t('复习此范围','Review this view'):progress.read(next.id)?.state==='started'?t('继续练习','Continue practice'):t('开始练习','Start practice')}<ArrowRight size={15}/></AppLink>}</div>
  {!reordering&&<p className="study-list-hint">{t('清除搜索和状态筛选后可调整完整题单的顺序。','Clear search and status filters to reorder the full list.')}</p>}
  {view.state!=='all'&&!ready?<p role="status">{t('正在核对学习进度，暂不显示筛选结果。','Checking practice progress before showing filtered results.')}</p>:<ol className="study-list-items">{shown.map(id=>{const p=byId.get(id),index=ids.indexOf(id);return <li key={id}><div>{p?<AppLink href={withLibraryContext(p.number,returnTo)}>{p.number}. {title(p)}<ArrowRight size={14}/></AppLink>:<strong>{id}</strong>}<small>{p?ready?progress.read(id)?.state==='done'?t('已通过','Solved'):progress.read(id)?.state==='started'?t('进行中','In progress'):t('未开始','Not started'):t('进度尚未就绪','Progress is not ready'):!catalogReady||catalogError?t('题目详情暂不可用','Problem details are unavailable'):t('此题已不在当前题库，可从题单移除','This problem left the catalog; you can remove it')}</small></div><div className="study-list-row-actions">{[-1,1].map(direction=><button key={direction} type="button" className="icon-button" aria-label={direction<0?t(`上移 ${p?title(p):id}`,`Move ${p?title(p):id} up`):t(`下移 ${p?title(p):id}`,`Move ${p?title(p):id} down`)} disabled={blocked||!reordering||index+direction<0||index+direction>=ids.length} onClick={()=>{const reordered=[...ids];[reordered[index],reordered[index+direction]]=[reordered[index+direction],reordered[index]];void store?.mutate({action:'reorder',list_id:list.id,problem_ids:reordered});}}>{direction<0?<ArrowUp size={15}/>:<ArrowDown size={15}/>}</button>)}<button type="button" className="icon-button" aria-label={t(`从题单移除 ${p?title(p):id}`,`Remove ${p?title(p):id} from this list`)} disabled={blocked} onClick={event=>{beforeRemoval(event);void store?.mutate({action:'remove',list_id:list.id,problem_id:id});}}><X size={15}/></button></div></li>;})}</ol>}
  {!shown.length&&(view.state==='all'||ready)&&<p>{ids.length?t('没有匹配的题目。可调整搜索或状态。','No matching problems. Adjust search or status.'):t('题单还是空的。添加想练的题目开始准备。','This list is empty. Add problems to get started.')}</p>}
 </section>;
}

export function ProblemStudyListPicker({problemId}:{problemId:string}){
 const {t}=useI18n(),{state,store}=useStudyLists(),heading=useRef<HTMLHeadingElement>(null),blocked=!!state.pending||state.loading||!!state.error;
 return <><h3 ref={heading} tabIndex={-1}>{t('加入私人题单','Add to a private study list')}</h3><StudyListFeedback state={state} store={store} focus={()=>heading.current?.focus({preventScroll:true})}/>{!state.snapshot?<p role="status">{state.loading?t('正在读取题单…','Loading study lists…'):t('题单尚未读取。','Study lists have not loaded.')}</p>:<><div className="study-list-picker-options">{state.snapshot.lists.map(list=>{const active=list.items.some(item=>item.problem_id===problemId);return <button type="button" key={list.id} aria-pressed={active} disabled={blocked} onClick={()=>void store?.mutate(active?{action:'remove',list_id:list.id,problem_id:problemId}:{action:'add',list_id:list.id,problem_ids:[problemId]})}><span>{list.title}</span><small>{list.items.length}</small>{active?<Check size={16}/>:<Plus size={16}/>}</button>;})}</div><CreateStudyList state={state} store={store} problemId={problemId}/><AppLink href="#/favorites">{t('管理私人题单','Manage private study lists')}<ArrowRight size={14}/></AppLink></>}</>;
}

export function StudyListPracticeNavigation({problemId,returnTo,problems}:{problemId:string;returnTo:string;problems:Problem[]}){
 const {user,loading}=useAccount();return <PracticeNavigation key={loading?'checking':user?.id||'guest'} problemId={problemId} returnTo={returnTo} problems={problems}/>;
}
function PracticeNavigation({problemId,returnTo,problems}:{problemId:string;returnTo:string;problems:Problem[]}){
 const {t}=useI18n(),{user,loading,capabilities,state,store}=useStudyLists(),view=readFavoriteRoute(returnTo),list=state.snapshot?.lists.find(row=>row.id===view?.list),byId=new Map(problems.map(p=>[p.id,p])),index=list?.items.findIndex(item=>item.problem_id===problemId)??-1;
 const before=index>=0?list?.items.slice(0,index).reverse().find(item=>byId.has(item.problem_id)):undefined,after=index>=0?list?.items.slice(index+1).find(item=>byId.has(item.problem_id)):undefined;
 return <nav className="study-list-practice-nav" aria-label={t('私人题单导航','Private study-list navigation')}>
  {loading?<span>{t('正在确认账号…','Checking account…')}</span>:!user||capabilities?.study_lists!==true?<AppLink href={returnTo}>{t('返回私人题单','Return to your study list')}</AppLink>:<><StudyListFeedback state={state} store={store}/>{list&&index>=0?<><AppLink href={returnTo} title={list.title}><FolderPlus size={14}/><span>{list.title}</span></AppLink><span>{index+1} / {list.items.length}</span><small>{t('按完整题单顺序','Full list order')}</small>{before&&<AppLink href={withLibraryContext(byId.get(before.problem_id)!.number,returnTo)} aria-label={t('上一道可用题目','Previous available problem')}><ArrowLeft size={15}/></AppLink>}{after&&<AppLink href={withLibraryContext(byId.get(after.problem_id)!.number,returnTo)} aria-label={t('下一道可用题目','Next available problem')}><ArrowRight size={15}/></AppLink>}</>:state.loading?<span>{t('正在读取题单…','Loading study list…')}</span>:<AppLink href={returnTo}>{t('此题单或题目已变化，返回查看','The list or membership changed. Return to review it.')}</AppLink>}</>}
 </nav>;
}
