import {controlIsActive} from './activeControls';
import {createContext,useContext,useEffect,useMemo,useRef,useState,useSyncExternalStore,type ReactNode,type RefObject} from 'react';
import {ChevronDown,RefreshCw,Star,X} from 'lucide-react';
import AppLink from './AppLink';
import {useAccount,errorMessage} from './social';
import {useI18n} from './i18n';
import {signInHref} from './accountNavigation';
import {interviewHref,type InterviewFilters} from './interviewNavigation';
import {interviewCatalog} from './interviewModels';
import {socialApi} from './social';
import {createCompanyTargetStore,MAX_COMPANY_TARGETS,type CompanyTargetsState} from './companyTargetStore';
import {useWorkspaceIntent} from './useWorkspaceIntent';
import {revealReadingTarget} from './readingViewport';
import './company-targets.css';

type TargetStore=ReturnType<typeof createCompanyTargetStore>;
type ContextValue={directoryOnly:boolean;setDirectoryOnly:(only:boolean)=>void;canAct?:()=>boolean;available:boolean;owner:string;state:CompanyTargetsState;store?:TargetStore;heading:RefObject<HTMLElement|null>;open:()=>void};
const empty:CompanyTargetsState={loading:false,error:'',accountError:false,pending:{}};
const Context=createContext<ContextValue|undefined>(undefined),subscribe=()=>()=>{},getEmpty=()=>empty;
const changeKey='paperleet.company-targets.changed';
export function CompanyTargetsProvider({enabled,children,canAct}:{enabled:boolean;children:ReactNode;canAct?:()=>boolean}){
 const heading=useRef<HTMLElement>(null),activity=useRef(canAct);activity.current=canAct;
 const active=()=>activity.current?.()!==false;
 const open=()=>{const node=heading.current;if(active()&&controlIsActive(node)&&node&&node.parentElement?.tagName==='DETAILS'){(node.parentElement as HTMLDetailsElement).open=true;node.focus({preventScroll:true});revealReadingTarget(node,'nearest');}};
 const account=useAccount(),available=enabled&&!account.loading&&account.capabilities?.interview_company_targets===true;
 const owner=available&&account.user&&(!account.user.status||account.user.status==='active')?account.user.id:'';
 const [directoryView,setDirectoryView]=useState({owner:'',only:false});
 const directoryOnly=!!owner&&directoryView.owner===owner&&directoryView.only;
 function setDirectoryOnly(only:boolean){if(active()&&owner)setDirectoryView({owner,only});}
 const store=useMemo(()=>owner?createCompanyTargetStore(owner,(body,signal)=>socialApi('account/interview-companies',body,signal),()=>{
  try{localStorage.setItem(changeKey,JSON.stringify({owner,nonce:crypto.randomUUID()}));}catch{/* Focus refresh remains available without browser storage. */}
 }):undefined,[owner]);
 const state=useSyncExternalStore(store?.subscribe||subscribe,store?.getSnapshot||getEmpty,getEmpty);
 useEffect(()=>{
  if(!store)return;store.start();if(active())void store.refresh();
  const refresh=()=>{if(active())void store.refresh();},storage=(event:StorageEvent)=>{if(event.key===changeKey){try{if(JSON.parse(event.newValue||'{}').owner===owner)refresh();}catch{/* Ignore malformed invalidations. */}}};
  window.addEventListener('focus',refresh);window.addEventListener('storage',storage);
  return()=>{window.removeEventListener('focus',refresh);window.removeEventListener('storage',storage);store.stop();};
 },[store,owner]);
 return <Context.Provider value={{available,owner,state,store,heading,open,canAct:active,directoryOnly,setDirectoryOnly}}>{children}</Context.Provider>;
}
/** Read the existing account store; directory views never start another request. */
export function useCompanyTargetsDirectory(owner:string){
 const context=useContext(Context);
 return context?.available&&!!context.owner&&context.owner===owner?{state:context.state,open:context.open,only:context.directoryOnly,setOnly:context.setDirectoryOnly}:undefined;
}
export function companyTargetHref(filters:InterviewFilters,company:string){return interviewHref({...filters,company,view:'companies',q:'',page:1,queue:'',saved:false});}
export function CompanyTargetButton({company,returnTo,canAct}:{company:string;returnTo:string;canAct?:()=>boolean}){
 const context=useContext(Context),{t}=useI18n();
 const allowed=(trigger:HTMLElement)=>controlIsActive(trigger)&&context?.canAct?.()!==false&&canAct?.()!==false;
 if(!context?.available)return null;
 if(!context.owner)return <AppLink className="company-target-button" onClick={event=>{if(!allowed(event.currentTarget))event.preventDefault();}} href={signInHref(returnTo)} aria-label={t('登录后设为目标公司','Sign in to save a target company')}><Star size={15} aria-hidden="true"/>{t('设为目标','Set target')}</AppLink>;
 const {state,store}=context,active=!!state.snapshot?.items.some(row=>row.company_id===company),pending=state.pending[company];
 const full=!active&&!pending&&(state.snapshot?.items.length||0)>=MAX_COMPANY_TARGETS;
 const status=state.accountError||!!state.error||!state.snapshot,capacity=full||pending?.issue==='capacity'&&!pending.retryReady,manage=status||capacity;
 const name=interviewCatalog.companies.find(row=>row.id===company),title=name?t(name.name,name.name_en):company;
 const label=status?t('查看目标公司状态','View target-company status'):capacity?t('已达上限，管理目标公司','Target limit reached; manage target companies'):pending?.error?pending.active?t('重试设为目标','Retry setting target'):t('重试移除目标','Retry removing target'):active?t('移除目标公司','Remove target company'):t('设为目标公司','Set target company');
 return <button className="company-target-button" data-company-target={company} type="button" aria-pressed={state.snapshot?active:undefined} aria-busy={!!pending?.busy} aria-disabled={pending?.busy||undefined} aria-label={`${label} · ${title}`} title={capacity?t(`最多保存 ${MAX_COMPANY_TARGETS} 家目标公司，点击管理`,`Save up to ${MAX_COMPANY_TARGETS} target companies. Open to manage`):label} onClick={event=>{if(allowed(event.currentTarget)&&!pending?.busy){if(manage)context.open();else void store?.set(company,!active);}}}><Star size={15} aria-hidden="true" fill={active?'currentColor':'none'}/>{pending?.busy?t('保存中…','Saving…'):status?t('查看状态','View status'):capacity?t('管理目标','Manage targets'):pending?.error?t('重试设置','Retry change'):active?t('目标公司','Target company'):t('设为目标','Set target')}</button>;
}
export function CompanyTargetsSummary({filters}:{filters:InterviewFilters}){
 const context=useContext(Context),{t,locale}=useI18n(),account=useAccount(),[checking,setChecking]=useState(false),[accountError,setAccountError]=useState('');
 const intent=useWorkspaceIntent(JSON.stringify([context?.owner,interviewHref(filters)]));
 if(!context?.available||!context.owner)return null;
 const {state,store,heading}=context,items=state.snapshot?.items||[],errors=Object.entries(state.pending).filter(([,row])=>!!row.error);
 const uncertain=errors.some(([,row])=>row.issue!=='capacity');
 const name=(id:string)=>{const row=interviewCatalog.companies.find(row=>row.id===id);return row?locale==='en'?row.name_en:row.name:id;};
 async function refreshAccount(){if(checking)return;setChecking(true);setAccountError('');try{await account.refresh();}catch(error){setAccountError(errorMessage(error));}finally{setChecking(false);}}
 async function remove(company:string){const claim=intent.create();const saved=await store?.set(company,false);if(saved&&claim())heading.current?.focus({preventScroll:true});}
 return <details onClickCapture={event=>{if(!controlIsActive(event.currentTarget)||context.canAct?.()===false){event.preventDefault();event.stopPropagation();}}} className="interview-company-targets"><summary ref={heading}><Star size={16} aria-hidden="true"/><b>{t('我的目标公司','My target companies')}</b><span>{state.snapshot?`${items.length} / ${MAX_COMPANY_TARGETS}`:'—'}</span><ChevronDown className="company-targets-chevron" size={15} aria-hidden="true"/>{(state.error||errors.length>0)&&<small role="status">{state.error?t('读取失败 · 展开重试','Unavailable · expand to retry'):uncertain?t('有设置待确认','Change unconfirmed'):t('添加被拒绝 · 展开管理','Addition declined · expand to manage')}</small>}</summary>
  <div className="company-targets-body"><p>{t('仅自己可见，跟随账号同步。从公司卡片设为目标，再从这里回到准备空间。','Private and synced to your account. Set targets on company cards and return to their preparation spaces here.')}</p>
   {items.length>=MAX_COMPANY_TARGETS&&!state.error&&<p className="company-targets-limit">{t(`已保存 ${MAX_COMPANY_TARGETS} 家目标公司。移除一家公司后即可添加新的目标。`,`You have ${MAX_COMPANY_TARGETS} target companies. Remove one before adding another.`)}</p>}
   {state.loading&&<p role="status">{t('正在读取目标公司…','Loading target companies…')}</p>}
   {state.error&&<div className="company-targets-error" role="status"><span>{t('目标公司暂不可用，请重试。','Target companies are unavailable. Please retry.')}</span><button type="button" className="text-button" aria-disabled={state.loading} onClick={()=>{if(!state.loading)void store?.refresh();}}><RefreshCw size={14}/>{t('重试加载','Retry loading')}</button>{state.accountError&&<button type="button" className="text-button" disabled={checking} onClick={()=>void refreshAccount()}>{t('刷新账号','Refresh account')}</button>}</div>}
   {accountError&&<p role="alert">{accountError}</p>}
   {state.snapshot&&(items.length?<ul>{items.map(row=>{const known=interviewCatalog.companies.some(company=>company.id===row.company_id),pending=state.pending[row.company_id];return <li key={row.company_id}>{known?<AppLink href={companyTargetHref(filters,row.company_id)}>{name(row.company_id)}</AppLink>:<span>{name(row.company_id)} · {t('暂不可用','Unavailable')}</span>}<button type="button" className="icon-button" aria-label={t(`移除目标公司 ${name(row.company_id)}`,`Remove target company ${name(row.company_id)}`)} aria-disabled={!!pending?.busy||!!state.error} aria-busy={!!pending?.busy} onClick={()=>{if(!pending?.busy&&!state.error)void remove(row.company_id);}}><X size={14} aria-hidden="true"/></button></li>;})}</ul>:!state.loading&&!state.error&&<p className="company-targets-empty">{t('还没有目标公司。点击公司旁的星标，保存常用入口。','No target companies yet. Use the star beside a company to save a shortcut.')}</p>)}
   {errors.map(([id,pending])=><section key={id} className={'company-targets-error'+(pending.issue==='capacity'?' company-targets-capacity':'')} data-target-issue={id} role="status"><b>{name(id)} · {pending.issue==='capacity'?t('本次添加未保存','This addition was not saved'):t('设置尚未确认','Change unconfirmed')}</b>{pending.issue==='capacity'&&<p>{pending.retryReady?t('列表已更新，可以重试加入这家公司。','The list is updated. You can retry adding this company.'):t('服务端的目标公司已达上限。刷新列表，或移除一家公司后再试。','Your saved targets reached the limit. Refresh the list or remove a company before retrying.')}</p>}<div>{pending.issue==='capacity'&&!pending.retryReady?<button type="button" className="text-button" disabled={state.loading} onClick={()=>void store?.refresh()}>{t('刷新目标列表','Refresh targets')}</button>:<button type="button" className="text-button" disabled={pending.busy||state.loading||!!state.error} onClick={()=>void store?.set(id,pending.active)}>{t('重试原操作','Retry original action')}</button>}{pending.issue==='capacity'&&<button type="button" className="text-button" onClick={()=>{heading.current?.focus({preventScroll:true});store?.dismiss(id);}}>{t('取消这次添加','Cancel this addition')}</button>}</div></section>)}
  </div>
 </details>;
}
