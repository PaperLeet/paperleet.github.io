import {appLocation, publicHref} from './appLocation';
import AppLink from './AppLink';
import {useEffect,useId,useMemo,useRef,useState,useSyncExternalStore} from 'react';
import {ArrowLeft,ArrowRight,Bookmark,CheckCircle2,RefreshCw,Search} from 'lucide-react';
import {useI18n} from './i18n';
import {SocialError,SocialNotice,socialApi,useAccount} from './social';
import {hostedAppAvailable,hostedAppHref} from './deployment';
import {signInHref} from './accountNavigation';
import {favoriteHref,readFavoriteRoute,withLibraryContext,type FavoriteView,type LibraryPageSize} from './libraryNavigation';
import {status,type Problem,type Saved} from './types';
import {createFavoriteStore,type FavoriteState} from './problemFavoriteStore';
import {useReadingPosition} from './useReadingPosition';
import ReadingReturnNotice from './ReadingReturnNotice';
import StudyListButton from './StudyListButton';
import {deferredFeature} from './deferredFeature';
import './problem-favorites.css';
const StudyLists=deferredFeature<Parameters<typeof import('./StudyLists').default>[0]>(()=>import('./StudyLists'),['私人题单','private study lists']);

type PageControls=import('./FavoritePagination').Props;
function PageFallback({page,size,total,listId,compact,canAct,onPage}:PageControls){
 const {t}=useI18n(),pages=Math.max(1,Math.ceil(total/size));
 return <nav className="favorite-pages-fallback" aria-label={compact?t('收藏快捷翻页','Quick saved problem pages'):t('收藏分页','Saved problem pages')}><button type="button" className="text-button" aria-controls={listId} disabled={page<=1||page>pages} onClick={()=>{if(canAct())onPage(page-1);}}>{t('上一页','Previous page')}</button><span>{page} / {pages}</span><button type="button" className="text-button" aria-controls={listId} disabled={page>=pages} onClick={()=>{if(canAct())onPage(page+1);}}>{t('下一页','Next page')}</button></nav>;
}
function PageFailure({retry,...props}:PageControls&{retry:()=>void}){const {t}=useI18n();return <><PageFallback {...props}/><button type="button" className="text-button favorite-pages-retry" onClick={()=>{if(!props.canAct())return;document.getElementById(props.listId)?.focus({preventScroll:true});retry();}}>{t('重试分页控件','Retry page controls')}</button></>;}
const FavoritePages=deferredFeature(()=>import('./FavoritePagination'),['收藏分页','saved problem pages'],{loading:props=><PageFallback {...props}/>,failure:(props,retry)=><PageFailure {...props} retry={retry}/>});

const changedKey='paperleet.problem-favorites.changed';
const stores=new Map<string,ReturnType<typeof createFavoriteStore>>();
const empty:FavoriteState={loading:false,error:'',pending:{}};
const watchers=new Map<ReturnType<typeof createFavoriteStore>,{count:number;stop:()=>void}>();
function watchFavorites(store:ReturnType<typeof createFavoriteStore>,owner:string){
  let watcher=watchers.get(store);
  if(!watcher){
    const refresh=()=>void store.refresh();
    const storage=(event:StorageEvent)=>{if(event.key===changedKey){try{if(JSON.parse(event.newValue||'{}').owner===owner)refresh();}catch{/* Ignore malformed invalidations. */}}};
    window.addEventListener('focus',refresh);window.addEventListener('storage',storage);
    watcher={count:0,stop:()=>{window.removeEventListener('focus',refresh);window.removeEventListener('storage',storage);}};
    watchers.set(store,watcher);
  }
  watcher.count++;
  return()=>{if(--watcher.count===0){watcher.stop();watchers.delete(store);}};
}

function useFavorites(refreshOnMount=false){
  const account=useAccount();
  const owner=!account.loading&&account.capabilities?.problem_bookmarks&&account.user?.id||'';
  const store=useMemo(()=>{
    if(!owner)return;
    if(!stores.has(owner)){
      if(stores.size>=3)stores.delete(stores.keys().next().value!);
      stores.set(owner,createFavoriteStore(owner,async(body,signal)=>{
        try{return await socialApi('account/problem-bookmarks',body,signal);}
        catch(error){if(error instanceof SocialError&&(error.status===401||error.code==='account_changed')){try{await account.refresh();}catch{/* Explicit recovery remains available. */}}throw error;}
      },()=>{
        try{localStorage.setItem(changedKey,JSON.stringify({owner,at:Date.now(),nonce:Math.random()}));}catch{/* Sync resumes on focus when browser storage is unavailable. */}
      }));
    }
    return stores.get(owner)!;
  },[owner]);
  const state=useSyncExternalStore(store?.subscribe||(()=>()=>{}),store?.getSnapshot||(()=>empty));
  useEffect(()=>{if(store&&(refreshOnMount||!store.getSnapshot().snapshot&&!store.getSnapshot().error))void store.refresh();},[store,refreshOnMount]);
  useEffect(()=>store?watchFavorites(store,owner):undefined,[store,owner]);
  return {...account,state,store};
}
type ButtonProps={problemId:string;returnTo?:string;label?:string;canSave?:boolean;problemName?:string;showStudyLists?:boolean;canAct?:()=>boolean};
export function ProblemFavoriteButton(props:ButtonProps){
  const {user}=useAccount();
  return hostedAppAvailable?<FavoriteControl key={(user?.id||'guest')+':'+props.problemId} {...props}/>:null;
}
function FavoriteControl({problemId,returnTo='favorites',label,canSave=true,problemName,showStudyLists=true,canAct}:ButtonProps){
  const {t}=useI18n(),{user,loading,capabilities,state,store,refresh}=useFavorites();
  const [checking,setChecking]=useState(false),[authError,setAuthError]=useState(''),[recovering,setRecovering]=useState(false);
  const actionButton=useRef<HTMLButtonElement>(null),recoveryFocus=useRef<{opener:Element|null;href:string}|undefined>(undefined);
  const active=!!state.snapshot?.items.some(item=>item.problem_id===problemId),pending=state.pending[problemId];
  useEffect(()=>{
    if(checking||loading||!recoveryFocus.current)return;
    const intent=recoveryFocus.current;recoveryFocus.current=undefined;
    if(location.href===intent.href&&(document.activeElement===intent.opener||document.activeElement===document.body&&!intent.opener?.isConnected))actionButton.current?.focus({preventScroll:true});
  },[checking,loading]);
  async function confirmAccount(){
    recoveryFocus.current={opener:document.activeElement,href:location.href};
    setChecking(true);setAuthError('');
    try{await refresh();}catch(error){setAuthError(error instanceof Error?error.message:String(error));}finally{setChecking(false);}
  }
  async function retryLoad(){setRecovering(true);try{await store?.refresh();}finally{setRecovering(false);}}
  const describe=(value:string)=>problemName?value+t('：',': ')+problemName:value;
  if(!loading&&capabilities&&!capabilities.problem_bookmarks)return null;
  if(!loading&&!user)return <AppLink className="problem-favorite-button" href={hostedAppHref(signInHref(returnTo))} onClick={event=>{if(canAct?.()===false||event.currentTarget.closest('[hidden],[aria-hidden="true"]'))event.preventDefault();}} aria-label={describe(t('登录后收藏题目','Sign in to save this problem'))} title={describe(t('登录后收藏题目','Sign in to save this problem'))}><Bookmark size={16}/>{label&&<span>{label}</span>}</AppLink>;
  const reloading=!!state.error||recovering,unavailable=!reloading&&!canSave&&!active&&!pending;
  const action=reloading?t('重试收藏加载','Retry saved problems'):unavailable?t('已移除，题目暂不可用','Removed; this problem is currently unavailable'):pending?.error?pending.active?t('重试收藏题目','Retry saving problem'):t('重试取消收藏','Retry removing saved problem'):active?t('取消收藏题目','Remove saved problem'):t('收藏题目','Save problem');
  return <div className="problem-favorite-control">
    <button ref={actionButton} data-favorite-control={problemId} type="button" className="problem-favorite-button" aria-pressed={reloading?undefined:active} aria-disabled={unavailable||undefined} aria-busy={!!pending?.busy||recovering||undefined} aria-label={describe(action)} title={state.error||pending?.error||describe(action)} disabled={loading||checking||recovering||state.loading&&!state.snapshot||!!pending?.busy||!state.snapshot&&!state.error} onClick={event=>{if(unavailable||canAct?.()===false||event.currentTarget.closest('[hidden],[aria-hidden="true"]'))return;if(reloading)void retryLoad();else void store?.set(problemId,!active);}}>
      {reloading?<RefreshCw size={16}/>:<Bookmark size={16} fill={active?'currentColor':'none'}/>}
      {label&&<span>{recovering?t('重试中…','Retrying…'):reloading?t('重试加载','Retry loading'):pending?.busy?t('保存中…','Saving…'):pending?.error?pending.active?t('重试收藏','Retry save'):t('重试取消','Retry removal'):label}</span>}
      {pending?.error&&!reloading&&<span className="favorite-retry-dot" aria-hidden="true">!</span>}
    </button>
    {pending?.error&&!label&&<span className="sr-only" role="status">{t('收藏操作未确认，请重试。','Bookmark unconfirmed. Please retry.')}</span>}
    {pending?.error&&label&&<span className="favorite-feedback" role="status">{t('收藏操作未确认，请重试。','Bookmark unconfirmed. Please retry.')}
      {pending.accountError&&<><button type="button" data-favorite-control={problemId} className="text-button" disabled={checking} onClick={()=>void confirmAccount()}>{t('刷新账号','Refresh account')}</button><AppLink href={hostedAppHref(signInHref(returnTo))}>{t('重新登录','Sign in again')}</AppLink>{authError}</>}
    </span>}
    {showStudyLists&&<StudyListButton problemId={problemId} canSave={canSave}/>}
  </div>;
}

function FavoriteProblemRow({id,problem,isSaved,returnTo,saved,progressReady,catalogError,resumed}:{id:string;problem?:Problem;isSaved:boolean;returnTo:string;saved?:Saved;progressReady:boolean;catalogError?:string;resumed:boolean}){
  const {t,locale}=useI18n();
  const label=isSaved?t('已收藏','Saved'):problem?t('重新收藏','Save again'):t('已移除','Removed');
  return <article className="favorite-problem-row" data-reading-post={id} data-reading-resumed={resumed||undefined} data-removed={!isSaved||undefined}>
    <div className="favorite-problem-main">
      {problem?<>
        <AppLink data-reading-link="practice" data-reading-practice={id} href={withLibraryContext(problem.id,returnTo)}><span className="favorite-problem-number">{problem.number}.</span><strong>{locale==='en'?problem.translations?.en?.title||problem.title:problem.title}</strong><ArrowRight size={15}/></AppLink>
        <div className="favorite-problem-meta"><span className={'difficulty '+problem.difficulty}>{problem.difficulty==='E'?t('简单','Easy'):problem.difficulty==='H'?t('困难','Hard'):t('中等','Medium')}</span><span>{id}</span>{progressReady&&status(problem,saved)==='done'&&<span className="favorite-solved"><CheckCircle2 size={13}/>{t('已通过','Solved')}</span>}</div>
      </>:<><strong>{id}</strong><small>{catalogError?t('题目详情暂时不可用，可重试加载。','Problem details unavailable. Retry the catalog.'):t('题目已不在当前题库，仍可取消收藏。','This problem has left the catalog. You can still remove it.')}</small></>}
      {!isSaved&&<small className="favorite-removed-note" role="status">{problem?t('已取消收藏，可再次收藏。','Removed. Save again to undo.'):t('已取消收藏。题目暂不可用，无法重新收藏。','Removed. This unavailable problem cannot currently be saved again.')}</small>}
    </div>
    <ProblemFavoriteButton problemId={id} canSave={!!problem} returnTo={returnTo} label={label}/>
  </article>;
}
type PageProps={route?:string;onFilters?:(href:string,paging?:boolean)=>void;problems:Problem[];saved:Record<string,Saved>;progressReady:boolean;catalogReady:boolean;catalogError?:string;onRetryCatalog:()=>void};
export function ProblemFavoritesPage(props:PageProps){
  const {t}=useI18n();
  if(!hostedAppAvailable)return <main className="problem-favorites-page"><AppLink className="back-link" href="#"><ArrowLeft size={15}/>{t('返回题库','Back to papers')}</AppLink><div className="favorites-empty"><Bookmark size={30}/><h1>{t('当前版本暂无题目收藏','Saved problems are unavailable here')}</h1><p>{t('可返回题库继续练习。','Return to papers to keep practicing.')}</p></div></main>;
  if(readFavoriteRoute(props.route||'favorites')?.list)return <StudyLists route={props.route} onFilters={props.onFilters} problems={props.problems} catalogReady={props.catalogReady} catalogError={props.catalogError} onRetryCatalog={props.onRetryCatalog}/>;
  return <AccountProblemFavoritesPage {...props}/>;
}
function AccountProblemFavoritesPage({route='favorites',onFilters,problems,saved,progressReady,catalogReady,catalogError,onRetryCatalog}:PageProps){
  const {t}=useI18n(),{user,loading,capabilities,state,store}=useFavorites(true);
  const filterControl=useRef<HTMLSelectElement>(null),searchControl=useRef<HTMLInputElement>(null),resultsHeading=useRef<HTMLParagraphElement>(null),listId=useId();
  const actionFocus=useRef<{owner:string;href:string;id?:string;opener:HTMLElement}|undefined>(undefined);
  useEffect(()=>{
    const cancel=()=>{actionFocus.current=undefined;};
    for(const name of ['pointerdown','keydown','wheel','touchmove','focusin'])window.addEventListener(name,cancel,{passive:true});
    return()=>{for(const name of ['pointerdown','keydown','wheel','touchmove','focusin'])window.removeEventListener(name,cancel);};
  },[]);

  const view=readFavoriteRoute(route);
  const [query,setQuery]=useState(()=>view?.query||''),[filter,setFilter]=useState(()=>view?.state||'all'),[page,setPage]=useState(()=>view?.page||1),[size,setSize]=useState<LibraryPageSize>(()=>view?.page_size||20);
  useEffect(()=>{setQuery(view?.query||'');setFilter(view?.state||'all');setPage(view?.page||1);setSize(view?.page_size||20);},[view?.query,view?.state,view?.page,view?.page_size]);
  const returnTo=favoriteHref({query,state:filter,...(page>1?{page}:{}),...(size!==20?{page_size:size}:{})});
  const live=useRef({owner:user?.id,href:returnTo});live.current={owner:user?.id,href:returnTo};
  function changeView(next:FavoriteView,opener?:HTMLElement,paging=false){
    if(!currentView())return;
    const clean=Array.from(next.query.replace(/[\u0000-\u001f\u007f]/g,' ')).slice(0,200).join(''),nextPage=next.page||1,nextSize=next.page_size||size;
    const href=favoriteHref({...next,query:clean,page:nextPage,page_size:nextSize});
    if(opener)filterControl.current?.focus({preventScroll:true});
    setQuery(clean);setFilter(next.state);setPage(nextPage);setSize(nextSize);
    if(onFilters)onFilters(href,paging);else try{window.history[paging?'pushState':'replaceState'](window.history.state,'',publicHref(href));}catch{/* Explicit links retain the selected filters. */}
  }
  const [rows,setRows]=useState<{owner:string;ids:string[]}>({owner:'',ids:[]});
  async function refreshRows(){if(!user||!store)return;await store.refresh();const fresh=store.getSnapshot();if(live.current.owner===user.id&&!fresh.error&&fresh.snapshot?.owner_id===user.id)setRows({owner:user.id,ids:fresh.snapshot.items.map(item=>item.problem_id)});}
  // Keep removed rows in place so undo never jumps under the pointer or keyboard.
  useEffect(()=>{
    const snapshot=state.snapshot;if(!snapshot)return;
    setRows(previous=>({owner:snapshot.owner_id,ids:previous.owner===snapshot.owner_id?[...previous.ids,...snapshot.items.map(item=>item.problem_id).filter(id=>!previous.ids.includes(id))]:snapshot.items.map(item=>item.problem_id)}));
  },[state.snapshot]);
  const entries=new Map(problems.map(problem=>[problem.id,problem]));
  const ids=rows.owner===user?.id?rows.ids:[];
  const active=new Set(state.snapshot?.items.map(item=>item.problem_id));
  const shown=ids.filter(id=>{const problem=entries.get(id);return (!query.trim()||[id,problem?.number,problem?.title,problem?.translations?.en?.title].filter(value=>value!==undefined).join(' ').toLocaleLowerCase().includes(query.trim().toLocaleLowerCase()))&&(filter==='all'||!!problem&&progressReady&&status(problem,saved[id])===filter);});
  const reading=useReadingPosition(loading?undefined:user?'account:'+user.id:'guest',returnTo,!!view&&!loading&&!!user&&!!capabilities?.problem_bookmarks&&!!state.snapshot&&!state.loading&&!state.error&&catalogReady&&!catalogError&&(filter==='all'||progressReady)&&rows.owner===user?.id);
  const pages=Math.max(1,Math.ceil(shown.length/size)),offset=(page-1)*size,outOfRange=page>pages,visible=shown.slice(offset,offset+size),removed=shown.some(id=>!active.has(id));
  function currentView(){const now=readFavoriteRoute(appLocation.hash);return !!user&&!loading&&live.current.owner===user.id&&live.current.href===returnTo&&state.snapshot?.owner_id===user.id&&!!capabilities?.problem_bookmarks&&!!reading.container.current?.isConnected&&!reading.container.current.closest('[hidden],[aria-hidden="true"]')&&!!now&&favoriteHref(now)===returnTo;}
  function paginate(next:number,nextSize:LibraryPageSize=size){
    if(!currentView()||filter!=='all'&&!progressReady||!catalogReady&&!catalogError||![20,50,100].includes(nextSize)||!Number.isSafeInteger(next)||next<1||next>Math.max(1,Math.ceil(shown.length/nextSize))||next===page&&nextSize===size)return;
    resultsHeading.current?.focus({preventScroll:true});resultsHeading.current?.scrollIntoView({block:'start',behavior:'instant'});
    changeView({query,state:filter,page:next,page_size:nextSize},undefined,true);
  }
  const controls={page,size,total:shown.length,listId,canAct:currentView,onPage:(next:number)=>paginate(next),onSize:(next:LibraryPageSize)=>paginate(Math.min(Math.max(1,Math.ceil(shown.length/next)),Math.floor(offset/next)+1),next)};
  useEffect(()=>{
    const intent=actionFocus.current;
    if(!intent||loading||state.loading||!state.snapshot)return;
    if(intent.owner!==user?.id||intent.href!==location.href){actionFocus.current=undefined;return;}
    const target=intent.id?[...reading.container.current?.querySelectorAll<HTMLButtonElement>('.problem-favorite-button[data-favorite-control]')||[]].find(node=>node.dataset.favoriteControl===intent.id):reading.container.current?.querySelector<HTMLButtonElement>('[data-favorite-refresh]');
    if(!target||target.disabled)return;
    actionFocus.current=undefined;
    if(!intent.opener.isConnected&&document.activeElement===document.body)target.focus({preventScroll:true});
  },[loading,state.loading,state.snapshot,state.pending,rows,user?.id]);
  if(!view)return <main className="problem-favorites-page"><SocialNotice error={t('收藏筛选链接无效。','This saved-problem filter is invalid.')}/><AppLink className="button subtle" href="#/favorites">{t('查看全部收藏','View all saved problems')}</AppLink></main>;
  return <main className="problem-favorites-page" ref={reading.container} onClickCapture={event=>{
    const action=event.target instanceof window.Element?event.target.closest<HTMLElement>('[data-favorite-control],[data-favorite-retry]'):null;
    if(action&&user)actionFocus.current={owner:user.id,href:location.href,id:action.dataset.favoriteControl,opener:action};
  }}>
    <AppLink className="back-link" href="#"><ArrowLeft size={15}/>{t('返回题库','Back to papers')}</AppLink>
    <header className="favorites-heading"><div><h1>{t('收藏题目','Saved problems')}</h1><p>{t('把想练习和回顾的题留在这里，随时继续。','Keep problems to practice or revisit, then pick up where you left off.')}</p></div>{user&&<span className="favorites-private">{t('仅自己可见 · 同步到账号','Private · Synced to your account')}</span>}</header>
    <ReadingReturnNotice kind="problem" notice={reading.notice} onRestore={reading.restore}/>
    {loading?<p role="status">{t('正在确认账号…','Checking your account…')}</p>:!user?<div className="favorites-empty"><Bookmark size={30}/><h2>{t('登录后保存你的练习清单','Sign in to keep your practice list')}</h2><p>{t('收藏会同步到账号，可在不同设备上继续练习。','Your saved problems sync across devices.')}</p><AppLink className="button" href={hostedAppHref(signInHref(returnTo))}>{t('登录账号','Sign in')}</AppLink></div>:!capabilities?.problem_bookmarks?<SocialNotice error={t('题目收藏暂未开放，请稍后重试。','Saved problems are currently unavailable. Please try again later.')}/>:<>
      {capabilities.study_lists===true&&<StudyLists compact route={route} onFilters={onFilters} problems={problems} catalogReady={catalogReady} catalogError={catalogError} onRetryCatalog={onRetryCatalog}/>}
      {state.error&&<div className="favorites-error"><SocialNotice error={state.error}/><button type="button" className="button subtle" data-favorite-retry="" disabled={state.loading} onClick={()=>void store?.refresh()}>{t('重试加载','Retry loading')}</button></div>}
      {catalogError&&<div className="favorites-error"><SocialNotice error={catalogError}/><button type="button" className="button subtle" onClick={onRetryCatalog}>{t('重试题库加载','Retry catalog')}</button></div>}
      {!state.snapshot?<p role="status">{state.loading?t('正在加载收藏…','Loading saved problems…'):t('尚未读取收藏。','Saved problems have not loaded.')}</p>:<>
        <div className="favorites-toolbar"><label><Search size={16}/><span className="sr-only">{t('搜索收藏题目','Search saved problems')}</span><input ref={searchControl} type="search" value={query} maxLength={200} autoComplete="off" onKeyDown={event=>{if(event.key==='Escape'&&!event.repeat&&!event.ctrlKey&&!event.metaKey&&!event.altKey&&!event.shiftKey&&!event.nativeEvent.isComposing&&event.nativeEvent.keyCode!==229&&query){event.preventDefault();event.stopPropagation();changeView({query:'',state:filter});}}} onChange={event=>changeView({query:event.target.value,state:filter})} placeholder={t('搜索题目名称或编号','Search problems or IDs')}/></label><select ref={filterControl} aria-label={t('学习状态','Practice status')} value={filter} onChange={event=>changeView({query,state:event.target.value})}><option value="all">{t('全部状态','All statuses')}</option><option value="todo" disabled={!progressReady}>{t('未开始','Not started')}</option><option value="started" disabled={!progressReady}>{t('练习中','In progress')}</option><option value="done" disabled={!progressReady}>{t('已通过','Solved')}</option></select><span role="status">{active.size} {t('道收藏题目',active.size===1?'saved problem':'saved problems')}</span><button type="button" className="icon-button" data-favorite-refresh="" aria-label={t('刷新收藏','Refresh saved problems')} disabled={state.loading} onClick={()=>void refreshRows()}><RefreshCw size={15}/></button></div>
        {filter!=='all'&&!progressReady?<div className="favorites-empty"><p role="status">{t('学习进度尚不可用，可以先查看全部收藏。','Practice status is not available yet. You can view all saved problems.')}</p><button className="button subtle" onClick={event=>changeView({query,state:'all'},event.currentTarget)}>{t('查看全部状态','View all statuses')}</button></div>:!catalogReady&&!catalogError?<p role="status">{t('正在加载题目详情…','Loading problem details…')}</p>:<>
          <div className="favorites-result-bar"><p ref={resultsHeading} id={listId+'-summary'} className="favorites-result-summary" tabIndex={-1} role="status" aria-atomic="true">{outOfRange?t(`第 ${page} 页暂不可用 · 共 ${pages} 页`,`Page ${page} is unavailable · ${pages} pages`):shown.length?t(`当前列表：第 ${offset+1}–${Math.min(page*size,shown.length)} 项，共 ${shown.length} 项`,`Showing ${offset+1}–${Math.min(page*size,shown.length)} of ${shown.length} ${shown.length===1?'problem':'problems'}`):t('当前列表暂无题目','No problems in this view')}</p>{shown.length>size&&!outOfRange&&<FavoritePages {...controls} compact/>}</div>
          {removed&&<p id={listId+'-removed'} className="favorites-retained-note">{t('列表保留本次取消收藏的题目，方便重新收藏；刷新收藏后会移除这些行。','Problems removed during this visit stay here for undo. Refresh saved problems to remove those rows.')}</p>}
          <div id={listId} className="favorites-list" data-reading-list="" tabIndex={-1} role="region" aria-label={t('收藏题目','Saved problems')} aria-describedby={removed?listId+'-removed':undefined}>
            {outOfRange?<div className="favorites-empty favorites-page-recovery"><h2>{t('这一页暂无题目','This page has no problems')}</h2><p>{t('收藏数量或筛选结果已变化，可选择有效页码继续，当前搜索和状态会保留。','Saved problems or filters have changed. Choose an available page while keeping your search and status.')}</p><div><button type="button" className="button subtle" onClick={()=>paginate(1)}>{t('返回第一页','Return to first page')}</button>{pages>1&&<button type="button" className="text-button" onClick={()=>paginate(pages)}>{t('打开最后一页','Open last page')}</button>}</div></div>:<>
              {visible.map(id=><FavoriteProblemRow key={id} id={id} problem={entries.get(id)} isSaved={active.has(id)} returnTo={returnTo} saved={saved[id]} progressReady={progressReady} catalogError={catalogError} resumed={reading.resumedPost===id}/>)}
              {!shown.length&&<div className="favorites-empty"><Bookmark size={30}/><h2>{ids.length?t('没有匹配的题目','No matching problems'):t('从一道想练的题开始','Save your next problem')}</h2><p>{ids.length?t('试试其他关键词或学习状态。','Try another search or practice status.'):t('在题目列表或题目页点击收藏图标，就能在这里继续练习。','Save a problem from the exercise list or workspace to keep it here for your next practice session.')}</p>{ids.length>0&&(query||filter!=='all')&&<button type="button" className="button subtle" onClick={()=>{if(!currentView())return;searchControl.current?.focus({preventScroll:true});changeView({query:'',state:'all'});}}>{t('清除搜索和状态筛选','Clear search and status')}</button>}<AppLink className="button subtle" href="#">{t('浏览题库','Explore papers')}<ArrowRight size={14}/></AppLink></div>}
            </>}
          </div>{shown.length>0&&!outOfRange&&<FavoritePages {...controls}/>}
        </>}
      </>}
    </>}
  </main>;
}
