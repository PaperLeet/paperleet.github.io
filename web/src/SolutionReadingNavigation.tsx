import {createContext,useCallback,useContext,useEffect,useLayoutEffect,useMemo,useRef,useState,type RefObject} from 'react';
import {ChevronLeft,ChevronRight,X} from 'lucide-react';
import {useI18n} from './i18n';
import {COMMUNITY_PAGE_SIZE,MAX_COMMUNITY_PAGE} from './communityNavigation';
import {captureReadingAnchor,restoreReadingAnchor,readingScrollRoot,revealReadingTarget,type ReadingAnchor} from './readingViewport';
import './solution-reading-navigation.css';

type Item={id:string;body:string;title:string};
type Reveal={allowed:()=>boolean;aborted:()=>void};
type Reader={body:string;summary:HTMLElement;open:(request:Reveal)=>void};
type Meta={enabled:boolean;scope:string;page:number;generation?:number;ready:boolean;failed?:boolean;total:number;items:readonly Item[];root:RefObject<HTMLDivElement|null>;toolbar:RefObject<HTMLDivElement|null>;onPage:(page:number)=>void};
type Request={ticket:number;scope:string;href:string;page:number;sourcePage:number;sourceGeneration?:number;sourceId:string;sourceIds:string[];sourceTotal:number;updated?:boolean;title?:string;direction:-1|1;id?:string;generation?:number;before:number;opener:HTMLElement;phase:'loading'|'available'|'unavailable'|'failed'};
type Context={register:(id:string,reader:Reader)=>()=>void;position:(id:string)=>{index:number;total:number;previous:boolean;next:boolean}|undefined;move:(id:string,direction:-1|1,opener:HTMLElement)=>void};
const ReadingContext=createContext<Context|undefined>(undefined);
export const SolutionReadingScope=ReadingContext.Provider;

/** One explicit reading step, with current filters and no practice/editor writes. */
export function useSolutionReadingNavigation(meta:Meta){
 const live=useRef(meta);live.current=meta;
 const readers=useRef(new Map<string,Reader>()),interactions=useRef(0),sequence=useRef(0),pending=useRef<Request|undefined>(undefined);
 const [notice,setNotice]=useState<Request>(),[visibility,setVisibility]=useState(0);
 const dismissed=useRef<{node?:HTMLElement;anchor?:ReadingAnchor;toolbar:HTMLElement;before:number;href:string;scope:string}|undefined>(undefined);
 const current=()=>live.current;
 const here=(request:Request)=>{const m=current();return m.enabled&&m.scope===request.scope&&location.href===request.href&&!!m.root.current?.isConnected;};
 const allowed=(request:Request)=>here(request)&&!document.hidden&&!current().root.current?.closest('[hidden]')&&sequence.current===request.ticket&&interactions.current===request.before&&(document.activeElement===request.opener||document.activeElement===document.body&&!request.opener.isConnected);
 const validPage=(m:Meta)=>m.ready&&m.enabled&&Number.isSafeInteger(m.total)&&m.total>=0&&Number.isSafeInteger(m.page)&&m.page>=1&&m.page<=MAX_COMMUNITY_PAGE&&m.items.length===Math.max(0,Math.min(COMMUNITY_PAGE_SIZE,m.total-(m.page-1)*COMMUNITY_PAGE_SIZE))&&(m.page-1)*COMMUNITY_PAGE_SIZE+m.items.length<=m.total&&new Set(m.items.map(row=>row.id)).size===m.items.length&&m.items.every(row=>/^[a-f0-9]{32}$/.test(row.id));
 function clear(){sequence.current++;pending.current=undefined;setNotice(undefined);}
 function offer(request:Request,phase:Request['phase']='available'){if(!here(request)){clear();return;}const next={...request,phase};pending.current=next;setNotice(value=>value?.ticket===next.ticket&&value.phase===phase&&value.id===next.id?value:next);}
 function open(request:Request){
  const m=current();if(!here(request)||!validPage(m)||m.page!==request.page)return;
  const sourceIndex=m.items.findIndex(row=>row.id===request.sourceId);
  const item=request.id?m.items.find(row=>row.id===request.id):request.page!==request.sourcePage&&sourceIndex>=0?m.items[sourceIndex+request.direction]:request.direction===1?m.items[0]:m.items.at(-1);
  const reader=item&&readers.current.get(item.id);
  if(!item||!reader||!reader.summary.isConnected){offer(request,'unavailable');return;}
  if(!request.id&&request.page!==request.sourcePage&&request.sourceIds.includes(item.id)){offer(request,'unavailable');return;}
  const bound={...request,id:item.id,title:item.title,generation:m.generation,...(!request.id&&request.page!==request.sourcePage&&sourceIndex<0&&m.total!==request.sourceTotal?{updated:true}:{})};
  if(bound.updated&&!request.id){offer(bound);return;}
  if(!allowed(bound)){offer(bound);return;}
  pending.current=undefined;setNotice(undefined);
  reader.open({allowed:()=>allowed(bound)&&current().ready&&current().page===bound.page&&current().generation===bound.generation&&readers.current.get(item.id)?.body===reader.body,aborted:()=>{if(sequence.current===bound.ticket)offer(bound);}});
 }
 const register=useCallback((id:string,reader:Reader)=>{readers.current.set(id,reader);return()=>{if(readers.current.get(id)===reader)readers.current.delete(id);};},[]);
 const move=useCallback((id:string,direction:-1|1,opener:HTMLElement)=>{
  const m=current();if(document.hidden||!validPage(m)||!m.root.current?.contains(opener)||opener.closest('[hidden]'))return;
  const index=m.items.findIndex(row=>row.id===id),next=index+direction;if(index<0)return;
  const page=next<0?m.page-1:next>=m.items.length?m.page+1:m.page;
  if(page<1||page>MAX_COMMUNITY_PAGE||page>Math.ceil(m.total/COMMUNITY_PAGE_SIZE))return;
  const request:Request={ticket:++sequence.current,scope:m.scope,href:location.href,page,sourcePage:m.page,sourceGeneration:m.generation,sourceId:id,sourceIds:m.items.map(row=>row.id),sourceTotal:m.total,direction,...(page===m.page?{id:m.items[next]?.id}:{}),before:interactions.current,opener,phase:'loading'};
  dismissed.current=undefined;
  if(!here(request))return;
  if(page===m.page){open(request);return;}
  pending.current=request;setNotice(request);m.onPage(page);
 },[]);
 useEffect(()=>{
  if(!meta.enabled)return;
  const intent=()=>{interactions.current++;},visible=()=>setVisibility(value=>value+1),route=()=>{if(pending.current&&pending.current.href!==location.href)clear();};
  for(const name of ['pointerdown','keydown','wheel','touchmove','focusin','input'])window.addEventListener(name,intent,{passive:true,capture:true});
  document.addEventListener('visibilitychange',visible);window.addEventListener('hashchange',route);window.addEventListener('popstate',route);
  return()=>{for(const name of ['pointerdown','keydown','wheel','touchmove','focusin','input'])window.removeEventListener(name,intent,{capture:true});document.removeEventListener('visibilitychange',visible);window.removeEventListener('hashchange',route);window.removeEventListener('popstate',route);sequence.current++;pending.current=undefined;dismissed.current=undefined;};
 },[meta.enabled,meta.scope]);
 useLayoutEffect(()=>{
  const request=pending.current;if(!request)return;
  if(!meta.enabled||meta.scope!==request.scope||location.href!==request.href){clear();return;}
  if(meta.page!==request.page){if(meta.ready&&(meta.page!==request.sourcePage||meta.generation!==request.sourceGeneration))offer(request,'unavailable');return;}
  if(meta.ready&&!validPage(meta)){offer(request,'unavailable');return;}
  if(meta.ready&&request.id&&!meta.items.some(row=>row.id===request.id)){offer(request,'unavailable');return;}
  if(meta.failed&&request.phase==='loading'){offer(request,'failed');return;}
  if(!meta.ready||!['loading','failed'].includes(request.phase))return;
  open(request);
 },[meta.enabled,meta.scope,meta.page,meta.ready,meta.failed,meta.generation,meta.items,visibility,typeof location==='undefined'?'':location.href]);
 useLayoutEffect(()=>{
  const request=dismissed.current;if(!request||notice)return;dismissed.current=undefined;
  if(location.href===request.href&&current().scope===request.scope&&interactions.current===request.before&&document.activeElement===request.toolbar&&request.node&&request.anchor)restoreReadingAnchor(request.node,request.anchor);
 },[notice]);
 const value=useMemo<Context>(()=>({register,move,position:id=>{
  if(!validPage(meta))return;const index=meta.items.findIndex(row=>row.id===id);if(index<0)return;
  const absolute=(meta.page-1)*COMMUNITY_PAGE_SIZE+index;
  return {index:absolute+1,total:meta.total,previous:index>0||meta.page>1,next:index<meta.items.length-1||meta.page<Math.min(MAX_COMMUNITY_PAGE,Math.ceil(meta.total/COMMUNITY_PAGE_SIZE))};
 }}),[meta.enabled,meta.ready,meta.page,meta.total,meta.items,register,move]);
 return {value:meta.enabled&&validPage(meta)?value:undefined,notice:notice&&notice.scope===meta.scope?notice:undefined,
  retry:(opener:HTMLElement)=>{const request=pending.current;if(!request||request.phase!=='failed'||!here(request))return;const next={...request,ticket:++sequence.current,before:interactions.current,opener,phase:'loading' as const};pending.current=next;setNotice(next);},
  resume:(opener:HTMLElement)=>{const request=pending.current;if(!request||!here(request)||!validPage(current())||current().page!==request.page)return;const next={...request,ticket:++sequence.current,before:interactions.current,opener,phase:'loading' as const};open(next);},
  dismiss:(opener:HTMLElement)=>{
   const m=current(),toolbar=m.toolbar.current;if(!toolbar)return;
   const nodes=[...readers.current.values()].map(row=>row.summary),pane=m.root.current&&readingScrollRoot(m.root.current),top=pane?pane.getBoundingClientRect().top+pane.clientTop:0,bottom=top+(pane?pane.clientHeight:window.innerHeight);
   const node=nodes.find(node=>{const box=node.getBoundingClientRect();return !node.closest('[hidden]')&&box.height>0&&box.top>=top&&box.bottom<=bottom;});
   const anchor=node?captureReadingAnchor(node):undefined;if(document.activeElement===opener)toolbar.focus({preventScroll:true});
   dismissed.current={node,anchor,toolbar,before:interactions.current,href:location.href,scope:m.scope};clear();
  },canResume:validPage(meta)&&notice?.page===meta.page&&notice?.phase==='available'&&(!notice.id||meta.items.some(row=>row.id===notice.id))};
}

export function SolutionReadingNotice({navigation}:{navigation:ReturnType<typeof useSolutionReadingNavigation>}){
 const {t}=useI18n(),notice=navigation.notice;if(!notice)return null;
 return <div className="solution-reading-notice" role="status"><span>{notice.phase==='loading'?t('正在载入相邻题解…','Loading the adjacent solution…'):notice.phase==='failed'?t('相邻题解暂时无法加载，请重试当前列表。','The adjacent solution could not load. Retry the current list.'):notice.phase==='unavailable'?t('这篇题解已不在当前结果中，可以继续浏览列表。','This solution is no longer in these results. Continue browsing the list.'):notice.updated?t('题解列表有更新，可以按当前结果继续阅读。','The solution list changed. Continue with the current results.'):t('题解已就绪，可以继续阅读。','The solution is ready when you are.')}{notice.phase==='available'&&notice.title&&<strong className="solution-reading-destination">{notice.title}</strong>}</span><div>{notice.phase==='available'&&<button type="button" disabled={!navigation.canResume} aria-label={notice.title?t('继续阅读：'+notice.title,'Continue reading: '+notice.title):undefined} onClick={event=>navigation.resume(event.currentTarget)}>{notice.updated?t('阅读当前结果','Read current results'):t('继续阅读','Continue reading')}</button>}<button type="button" aria-label={t('关闭题解阅读提示','Dismiss solution reading notice')} onClick={event=>navigation.dismiss(event.currentTarget)}><X size={16} aria-hidden="true"/></button></div></div>;
}

export function useSolutionReadingTarget(id:string,body:string,summary:RefObject<HTMLElement|null>,expand:(body:string)=>void){
 const navigation=useContext(ReadingContext),request=useRef<Reveal|undefined>(undefined),[revision,setRevision]=useState(0),register=navigation?.register;
 useLayoutEffect(()=>{const node=summary.current;if(!register||!node)return;return register(id,{body,summary:node,open:next=>{request.current=next;expand(body);setRevision(value=>value+1);}});},[register,id,body,expand,summary]);
 useLayoutEffect(()=>{const next=request.current;if(!next)return;request.current=undefined;if(!next.allowed()){next.aborted();return;}const node=summary.current;if(node&&!node.closest('[hidden]')){node.focus({preventScroll:true});revealReadingTarget(node);}},[revision,summary]);
 const position=navigation?.position(id);
 return position?{...position,move:(direction:-1|1,opener:HTMLElement)=>navigation!.move(id,direction,opener)}:undefined;
}
export function SolutionReadingSteps({reading}:{reading:ReturnType<typeof useSolutionReadingTarget>}){
 const {t}=useI18n();if(!reading)return null;
 return <div className="solution-reading-steps" role="group" aria-label={t('切换题解','Browse adjacent solutions')}><button type="button" disabled={!reading.previous} aria-label={t('上一篇题解','Previous solution')} onClick={event=>reading.move(-1,event.currentTarget)}><ChevronLeft size={16} aria-hidden="true"/><span>{t('上一篇','Previous')}</span></button><span className="solution-reading-position">{t(`题解 ${reading.index} / ${reading.total}`,`Solution ${reading.index} of ${reading.total}`)}</span><button type="button" disabled={!reading.next} aria-label={t('下一篇题解','Next solution')} onClick={event=>reading.move(1,event.currentTarget)}><span>{t('下一篇','Next')}</span><ChevronRight size={16} aria-hidden="true"/></button></div>;
}
