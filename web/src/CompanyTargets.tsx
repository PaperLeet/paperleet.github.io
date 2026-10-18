import {createContext,useContext,useEffect,useMemo,useRef,useState,useSyncExternalStore,type ReactNode} from 'react';
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
import './company-targets.css';

type TargetStore=ReturnType<typeof createCompanyTargetStore>;
type ContextValue={available:boolean;owner:string;state:CompanyTargetsState;store?:TargetStore};
const empty:CompanyTargetsState={loading:false,error:'',accountError:false,pending:{}};
const Context=createContext<ContextValue|undefined>(undefined),subscribe=()=>()=>{},getEmpty=()=>empty;
const changeKey='paperleet.company-targets.changed';
export function CompanyTargetsProvider({enabled,children}:{enabled:boolean;children:ReactNode}){
 const account=useAccount(),available=enabled&&!account.loading&&account.capabilities?.interview_company_targets===true;
 const owner=available&&account.user&&(!account.user.status||account.user.status==='active')?account.user.id:'';
 const store=useMemo(()=>owner?createCompanyTargetStore(owner,(body,signal)=>socialApi('account/interview-companies',body,signal),()=>{
  try{localStorage.setItem(changeKey,JSON.stringify({owner,nonce:crypto.randomUUID()}));}catch{/* Focus refresh remains available without browser storage. */}
 }):undefined,[owner]);
 const state=useSyncExternalStore(store?.subscribe||subscribe,store?.getSnapshot||getEmpty,getEmpty);
 useEffect(()=>{
  if(!store)return;store.start();void store.refresh();
  const refresh=()=>void store.refresh(),storage=(event:StorageEvent)=>{if(event.key===changeKey){try{if(JSON.parse(event.newValue||'{}').owner===owner)refresh();}catch{/* Ignore malformed invalidations. */}}};
  window.addEventListener('focus',refresh);window.addEventListener('storage',storage);
  return()=>{window.removeEventListener('focus',refresh);window.removeEventListener('storage',storage);store.stop();};
 },[store,owner]);
 return <Context.Provider value={{available,owner,state,store}}>{children}</Context.Provider>;
}
export function companyTargetHref(filters:InterviewFilters,company:string){return interviewHref({...filters,company,view:'companies',q:'',page:1,queue:'',saved:false});}
export function CompanyTargetButton({company,returnTo}:{company:string;returnTo:string}){
 const context=useContext(Context),{t}=useI18n();
 if(!context?.available)return null;
 if(!context.owner)return <AppLink className="company-target-button" href={signInHref(returnTo)} aria-label={t('登录后设为目标公司','Sign in to save a target company')}><Star size={15} aria-hidden="true"/>{t('设为目标','Set target')}</AppLink>;
 const {state,store}=context,active=!!state.snapshot?.items.some(row=>row.company_id===company),pending=state.pending[company];
 const blocked=state.accountError||!!state.error||!state.snapshot||state.loading&&!state.snapshot||!!pending?.busy;
 const full=!active&&!pending&&(state.snapshot?.items.length||0)>=MAX_COMPANY_TARGETS;
 const name=interviewCatalog.companies.find(row=>row.id===company),title=name?t(name.name,name.name_en):company;
 const label=pending?.error?pending.active?t('重试设为目标','Retry setting target'):t('重试移除目标','Retry removing target'):active?t('移除目标公司','Remove target company'):full?t('目标公司已达上限','Target-company limit reached'):t('设为目标公司','Set target company');
 return <button className="company-target-button" data-company-target={company} type="button" aria-pressed={state.snapshot?active:undefined} aria-busy={!!pending?.busy} aria-disabled={blocked||full||undefined} aria-label={`${label} · ${title}`} title={full?t(`最多保存 ${MAX_COMPANY_TARGETS} 家目标公司`,`Save up to ${MAX_COMPANY_TARGETS} target companies`):label} onClick={()=>{if(!blocked&&!full)void store?.set(company,!active);}}><Star size={15} aria-hidden="true" fill={active?'currentColor':'none'}/>{pending?.busy?t('保存中…','Saving…'):pending?.error?t('重试设置','Retry change'):active?t('目标公司','Target company'):t('设为目标','Set target')}</button>;
}
export function CompanyTargetsSummary({filters}:{filters:InterviewFilters}){
 const context=useContext(Context),{t,locale}=useI18n(),account=useAccount(),heading=useRef<HTMLElement>(null),[checking,setChecking]=useState(false),[accountError,setAccountError]=useState('');
 const intent=useWorkspaceIntent(JSON.stringify([context?.owner,interviewHref(filters)]));
 if(!context?.available||!context.owner)return null;
 const {state,store}=context,items=state.snapshot?.items||[],errors=Object.entries(state.pending).filter(([,row])=>!!row.error);
 const name=(id:string)=>{const row=interviewCatalog.companies.find(row=>row.id===id);return row?locale==='en'?row.name_en:row.name:id;};
 async function refreshAccount(){if(checking)return;setChecking(true);setAccountError('');try{await account.refresh();}catch(error){setAccountError(errorMessage(error));}finally{setChecking(false);}}
 async function remove(company:string){const claim=intent.create();const saved=await store?.set(company,false);if(saved&&claim())heading.current?.focus({preventScroll:true});}
 return <details className="interview-company-targets"><summary ref={heading}><Star size={16} aria-hidden="true"/><b>{t('我的目标公司','My target companies')}</b><span>{state.snapshot?`${items.length} / ${MAX_COMPANY_TARGETS}`:'—'}</span><ChevronDown className="company-targets-chevron" size={15} aria-hidden="true"/>{(state.error||errors.length>0)&&<small role="status">{state.error?t('读取失败 · 展开重试','Unavailable · expand to retry'):t('有设置待确认','Change unconfirmed')}</small>}</summary>
  <div className="company-targets-body"><p>{t('仅自己可见，跟随账号同步。从公司卡片设为目标，再从这里回到准备空间。','Private and synced to your account. Set targets on company cards and return to their preparation spaces here.')}</p>
   {state.loading&&<p role="status">{t('正在读取目标公司…','Loading target companies…')}</p>}
   {state.error&&<div className="company-targets-error" role="status"><span>{t('目标公司暂不可用，请重试。','Target companies are unavailable. Please retry.')}</span><button type="button" className="text-button" aria-disabled={state.loading} onClick={()=>{if(!state.loading)void store?.refresh();}}><RefreshCw size={14}/>{t('重试加载','Retry loading')}</button>{state.accountError&&<button type="button" className="text-button" disabled={checking} onClick={()=>void refreshAccount()}>{t('刷新账号','Refresh account')}</button>}</div>}
   {accountError&&<p role="alert">{accountError}</p>}
   {state.snapshot&&(items.length?<ul>{items.map(row=>{const known=interviewCatalog.companies.some(company=>company.id===row.company_id),pending=state.pending[row.company_id];return <li key={row.company_id}>{known?<AppLink href={companyTargetHref(filters,row.company_id)}>{name(row.company_id)}</AppLink>:<span>{name(row.company_id)} · {t('暂不可用','Unavailable')}</span>}<button type="button" className="icon-button" aria-label={t(`移除目标公司 ${name(row.company_id)}`,`Remove target company ${name(row.company_id)}`)} aria-disabled={!!pending?.busy||!!state.error} aria-busy={!!pending?.busy} onClick={()=>{if(!pending?.busy&&!state.error)void remove(row.company_id);}}><X size={14} aria-hidden="true"/></button></li>;})}</ul>:!state.loading&&!state.error&&<p className="company-targets-empty">{t('还没有目标公司。点击公司旁的星标，保存常用入口。','No target companies yet. Use the star beside a company to save a shortcut.')}</p>)}
   {errors.map(([id,pending])=><div key={id} className="company-targets-error" role="status"><span>{name(id)} · {t('设置尚未确认','Change unconfirmed')}</span><button type="button" className="text-button" onClick={()=>void store?.set(id,pending.active)}>{t('重试原操作','Retry original action')}</button></div>)}
  </div>
 </details>;
}
