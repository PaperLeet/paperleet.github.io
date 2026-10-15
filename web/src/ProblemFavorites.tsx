import {useEffect,useMemo,useState,useSyncExternalStore} from 'react';
import {ArrowLeft,ArrowRight,Bookmark,CheckCircle2,RefreshCw,Search} from 'lucide-react';
import {useI18n} from './i18n';
import {SocialError,SocialNotice,socialApi,useAccount} from './social';
import {hostedAppHref} from './deployment';
import {signInHref} from './accountNavigation';
import {favoriteHref,readFavoriteRoute,withLibraryContext} from './libraryNavigation';
import {status,type Problem,type Saved} from './types';
import {createFavoriteStore,type FavoriteState} from './problemFavoriteStore';
import './problem-favorites.css';

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
type ButtonProps={problemId:string;returnTo?:string;label?:string};
export function ProblemFavoriteButton(props:ButtonProps){
  const {user}=useAccount();
  return <FavoriteControl key={(user?.id||'guest')+':'+props.problemId} {...props}/>;
}
function FavoriteControl({problemId,returnTo='favorites',label}:ButtonProps){
  const {t}=useI18n(),{user,loading,capabilities,state,store,refresh}=useFavorites();
  const [checking,setChecking]=useState(false),[authError,setAuthError]=useState('');
  const active=!!state.snapshot?.items.some(item=>item.problem_id===problemId),pending=state.pending[problemId];
  if(!loading&&capabilities&&!capabilities.problem_bookmarks)return null;
  if(!loading&&!user)return <a className="problem-favorite-button" href={hostedAppHref(signInHref(returnTo))} aria-label={t('登录后收藏题目','Sign in to save this problem')} title={t('登录后收藏题目','Sign in to save this problem')}><Bookmark size={16}/>{label&&<span>{label}</span>}</a>;
  const action=pending?.error?t('重试原收藏操作','Retry bookmark action'):active?t('取消收藏题目','Remove saved problem'):t('收藏题目','Save problem');
  return <span className="problem-favorite-control">
    <button type="button" className="problem-favorite-button" aria-pressed={active} aria-label={action} title={pending?.error||action} disabled={loading||checking||state.loading&&!state.snapshot||!!pending?.busy||!state.snapshot||!!state.error} onClick={()=>void store?.set(problemId,!active)}><Bookmark size={16} fill={active?'currentColor':'none'}/>{label&&<span>{pending?.busy?t('保存中…','Saving…'):label}</span>}{pending?.error&&<span className="favorite-retry-dot" aria-hidden="true">!</span>}</button>
    {state.error&&<button type="button" className="favorite-retry text-button" onClick={()=>void store?.refresh()} disabled={state.loading} title={state.error}>{t('重试收藏加载','Retry saved problems')}</button>}
    {pending?.error&&!label&&<span className="sr-only" role="status">{t("收藏操作未确认，请重试。","Bookmark unconfirmed. Please retry.")}</span>}
    {pending?.error&&label&&<span className="favorite-feedback" role="status">{t('收藏操作未确认，请重试。','Bookmark unconfirmed. Please retry.')}<button type="button" className="text-button" disabled={checking} onClick={async()=>{setChecking(true);setAuthError('');try{await refresh();}catch(e){setAuthError(e instanceof Error?e.message:String(e));}finally{setChecking(false);}}}>{t('刷新账号','Refresh account')}</button><a href={hostedAppHref(signInHref(returnTo))}>{t('重新登录','Sign in again')}</a>{authError}</span>}
  </span>;
}
export function ProblemFavoritesPage({route='favorites',problems,saved,progressReady,catalogReady,catalogError,onRetryCatalog}:{route?:string;problems:Problem[];saved:Record<string,Saved>;progressReady:boolean;catalogReady:boolean;catalogError?:string;onRetryCatalog:()=>void}){
  const {t,locale}=useI18n(),{user,loading,capabilities,state,store}=useFavorites(true);
  const view=readFavoriteRoute(route);
  const [query,setQuery]=useState(()=>view?.query||''),[filter,setFilter]=useState(()=>view?.state||'all');
  useEffect(()=>{setQuery(view?.query||'');setFilter(view?.state||'all');},[view?.query,view?.state]);
  const returnTo=favoriteHref({query,state:filter});
  function changeView(next:{query:string;state:string}){setQuery(next.query);setFilter(next.state);try{history.replaceState(history.state,'',favoriteHref(next));}catch{/* Controls stay usable when history changes are unavailable. */}}
  const [rows,setRows]=useState<{owner:string;ids:string[]}>({owner:'',ids:[]});
  // Keep removed rows in place so undo never jumps under the pointer or keyboard.
  useEffect(()=>{
    const snapshot=state.snapshot;if(!snapshot)return;
    setRows(previous=>({owner:snapshot.owner_id,ids:previous.owner===snapshot.owner_id?[...previous.ids,...snapshot.items.map(item=>item.problem_id).filter(id=>!previous.ids.includes(id))]:snapshot.items.map(item=>item.problem_id)}));
  },[state.snapshot]);
  const entries=new Map(problems.map(problem=>[problem.id,problem]));
  const ids=rows.owner===user?.id?rows.ids:[];
  const active=new Set(state.snapshot?.items.map(item=>item.problem_id));
  const shown=ids.filter(id=>{const problem=entries.get(id);return (!query.trim()||(id+' '+problem?.title+' '+problem?.translations?.en?.title).toLocaleLowerCase().includes(query.trim().toLocaleLowerCase()))&&(filter==='all'||!!problem&&progressReady&&status(problem,saved[id])===filter);});
  if(!view)return <main className="problem-favorites-page"><SocialNotice error={t('收藏筛选链接无效。','This saved-problem filter is invalid.')}/><a className="button subtle" href="#/favorites">{t('查看全部收藏','View all saved problems')}</a></main>;
  return <main className="problem-favorites-page">
    <a className="back-link" href="#"><ArrowLeft size={15}/>{t('返回题库','Back to papers')}</a>
    <header className="favorites-heading"><div><span className="favorites-eyebrow"><Bookmark size={15}/>{t('我的学习清单','My study list')}</span><h1>{t('收藏题目','Saved problems')}</h1><p>{t('把想练习和回顾的题留在这里，随时继续。','Keep problems to practice or revisit, then pick up where you left off.')}</p></div>{user&&<span className="favorites-private">{t('仅自己可见 · 同步到账号','Private · Synced to your account')}</span>}</header>
    {loading?<p role="status">{t('正在确认账号…','Checking your account…')}</p>:!user?<div className="favorites-empty"><Bookmark size={30}/><h2>{t('登录后保存你的练习清单','Sign in to keep your practice list')}</h2><p>{t('收藏会同步到账号，可在不同设备上继续练习。','Your saved problems sync across devices.')}</p><a className="button" href={hostedAppHref(signInHref(returnTo))}>{t('登录账号','Sign in')}</a></div>:!capabilities?.problem_bookmarks?<SocialNotice error={t('题目收藏暂未开放，请稍后重试。','Saved problems are currently unavailable. Please try again later.')}/>:<>
      {state.error&&<div className="favorites-error"><SocialNotice error={state.error}/><button type="button" className="button subtle" disabled={state.loading} onClick={()=>void store?.refresh()}>{t('重试加载','Retry loading')}</button></div>}
      {catalogError&&<div className="favorites-error"><SocialNotice error={catalogError}/><button type="button" className="button subtle" onClick={onRetryCatalog}>{t('重试题库加载','Retry catalog')}</button></div>}
      {!state.snapshot?<p role="status">{state.loading?t('正在加载收藏…','Loading saved problems…'):t('尚未读取收藏。','Saved problems have not loaded.')}</p>:<>
        <div className="favorites-toolbar"><label><Search size={16}/><span className="sr-only">{t('搜索收藏题目','Search saved problems')}</span><input type="search" value={query} maxLength={200} onChange={event=>changeView({query:event.target.value,state:filter})} placeholder={t('搜索题目名称或编号','Search problems or IDs')}/></label><select aria-label={t('学习状态','Practice status')} value={filter} onChange={event=>changeView({query,state:event.target.value})}><option value="all">{t('全部状态','All statuses')}</option><option value="todo" disabled={!progressReady}>{t('未开始','Not started')}</option><option value="started" disabled={!progressReady}>{t('练习中','In progress')}</option><option value="done" disabled={!progressReady}>{t('已通过','Solved')}</option></select><span role="status">{active.size} {t('道收藏题目',active.size===1?'saved problem':'saved problems')}</span><button type="button" className="icon-button" aria-label={t('刷新收藏','Refresh saved problems')} disabled={state.loading} onClick={()=>{setRows({owner:user.id,ids:state.snapshot!.items.map(item=>item.problem_id)});void store?.refresh();}}><RefreshCw size={15}/></button></div>
        {filter!=='all'&&!progressReady?<div className="favorites-empty"><p role="status">{t('学习进度尚不可用，可以先查看全部收藏。','Practice status is not available yet. You can view all saved problems.')}</p><button className="button subtle" onClick={()=>changeView({query,state:'all'})}>{t('查看全部状态','View all statuses')}</button></div>:!catalogReady&&!catalogError?<p role="status">{t('正在加载题目详情…','Loading problem details…')}</p>:<div className="favorites-list">{shown.map(id=>{const problem=entries.get(id),isSaved=active.has(id);return <article className="favorite-problem-row" key={id} data-removed={!isSaved||undefined}>
          <div className="favorite-problem-main">{problem?<><a href={withLibraryContext(problem.number,returnTo)}><span className="favorite-problem-number">{problem.number}.</span><strong>{locale==='en'?problem.translations?.en?.title||problem.title:problem.title}</strong><ArrowRight size={15}/></a><div className="favorite-problem-meta"><span className={'difficulty '+problem.difficulty}>{problem.difficulty==='E'?t('简单','Easy'):problem.difficulty==='H'?t('困难','Hard'):t('中等','Medium')}</span><span>{id}</span>{progressReady&&status(problem,saved[id])==='done'&&<span className="favorite-solved"><CheckCircle2 size={13}/>{t('已通过','Solved')}</span>}</div></>:<><strong>{id}</strong><small>{catalogError?t('题目详情暂时不可用，可重试加载。','Problem details unavailable. Retry the catalog.'):t('题目已不在当前题库，仍可取消收藏。','This problem has left the catalog. You can still remove it.')}</small></>}{!isSaved&&<small className="favorite-removed-note" role="status">{t('已取消收藏，可再次收藏。','Removed. Save again to undo.')}</small>}</div>
          {(problem||isSaved)&&<ProblemFavoriteButton problemId={id} returnTo={returnTo} label={isSaved?t('已收藏','Saved'):t('重新收藏','Save again')}/>}
        </article>;})}{!shown.length&&<div className="favorites-empty"><Bookmark size={30}/><h2>{ids.length?t('没有匹配的题目','No matching problems'):t('从一道想练的题开始','Save your next problem')}</h2><p>{ids.length?t('试试其他关键词或学习状态。','Try another search or practice status.'):t('在题目页点击收藏图标，就能在这里继续练习。','Use the bookmark on a problem to keep it here for your next practice session.')}</p><a className="button subtle" href="#">{t('浏览题库','Explore papers')}<ArrowRight size={14}/></a></div>}</div>}
      </>}
    </>}
  </main>;
}
