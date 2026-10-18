import {appLocation} from './appLocation';
import AppLink from './AppLink';
import {useEffect,useRef,useState} from 'react';
import {LayoutDashboard,PanelsTopLeft,Tags,Flag,Users,FileCheck,History,ShieldCheck,ArrowUpRight,RefreshCw} from 'lucide-react';
import {socialApi,useAccount,errorMessage,SocialError} from './social';
import {useI18n} from './i18n';
import {signInHref} from './accountNavigation';
import {deferredFeature} from './deferredFeature';
import {refreshCommunityTaxonomy} from './useCommunityTaxonomy';
import {readAdminCommand,holdAdminCommand,clearAdminCommand,type AdminCommand} from './adminCommands';
import {readAdminRoute,type AdminView} from './adminNavigation';
import {adminHref} from './adminLinks';
import {createAdminDraftStore} from './adminDraftStore';
import {AdminDraftContext,AdminDraftTray} from './AdminDrafts';
import {Overview,Taxonomy,Members,Papers,AuditLog} from './AdminLists';
import AdminHighlights from './AdminHighlights';
import type {Run} from './AdminParts';
import './admin-workspace.css';
const Reports=deferredFeature(()=>import('./ModerationQueue'),['举报审核','report review']);
const sections=[['','概览','Overview',LayoutDashboard],['boards','社区版位','Boards',PanelsTopLeft],['tags','标签','Tags',Tags],['highlights','社区精选','Community picks',ShieldCheck],['reports','举报审核','Reports',Flag],['users','用户管理','Users',Users],['papers','论文审核','Papers',FileCheck],['audit','操作记录','Audit log',History]] as const;
export const adminSection=(route:string)=>readAdminRoute(route)?.section;
export default function AdminWorkspace({route}:{route:string}){
 const {user,capabilities,loading}=useAccount(),{t}=useI18n(),view=readAdminRoute(route);
 if(!view)return <main className="admin-access"><h1>{t('找不到这个管理页面','Administration page not found')}</h1><AppLink href="#/community/admin">{t('返回工作台','Back to administration')}</AppLink></main>;
 if(loading)return <main className="admin-access"><p role="status">{t('正在确认管理员权限…','Checking administrator access…')}</p></main>;
 if(!user)return <main className="admin-access"><h1>{t('管理员工作台','Administration')}</h1><AppLink className="button" href={signInHref(adminHref(view))}>{t('登录管理员账号','Sign in as an administrator')}</AppLink></main>;
 if(user.role!=='admin'||user.status&&user.status!=='active')return <main className="admin-access"><h1>{t('需要管理员权限','Administrator access required')}</h1><AppLink href="#/community">{t('返回社区','Back to community')}</AppLink></main>;
 if(capabilities?.admin_workspace!==true)return <main className="admin-access"><h1>{t('管理服务暂不可用','Administration is unavailable')}</h1><AppLink href="#/community/moderation">{t('前往举报审核','Open report review')}</AppLink></main>;
 return <Workspace key={user.id} owner={user.id} view={view} detailsAvailable={capabilities.admin_review_navigation===true} inboxAvailable={capabilities.admin_review_inbox===true} highlightsAvailable={capabilities.admin_highlights===true} reportNavigation={capabilities.admin_moderation_navigation===true}/>;
}
function Workspace({owner,view,detailsAvailable,inboxAvailable,reportNavigation,highlightsAvailable}:{owner:string;view:AdminView;detailsAvailable:boolean;inboxAvailable:boolean;reportNavigation:boolean;highlightsAvailable:boolean}){
 const {t}=useI18n(),[epoch,setEpoch]=useState(0),[store]=useState(createAdminDraftStore),history=useRef(new Map<string,string>());
 const [command,setCommand]=useState<AdminCommand|undefined>(()=>readAdminCommand(owner)),[busy,setBusy]=useState(false),[error,setError]=useState(''),[notice,setNotice]=useState(''),[rejected,setRejected]=useState(false);
 const working=useRef(false),mounted=useRef(true),held=useRef(command);
 useEffect(()=>{mounted.current=true;return()=>{mounted.current=false;};},[]);
 useEffect(()=>{history.current.set(view.section,adminHref(view));},[adminHref(view)]);
 async function send(sent:AdminCommand){
  if(working.current)return;working.current=true;setBusy(true);setError('');setNotice('');setRejected(false);holdAdminCommand(sent);held.current=sent;setCommand(sent);
  try{
   const value=await socialApi<{owner_id:string;request_id:string;audit_id:string;kind:string;resource_id:string;revision:number}>(sent.path,sent.payload,AbortSignal.timeout(20_000));
   if(value?.owner_id!==owner||value.request_id!==sent.payload.request_id||value.resource_id!==sent.resource||value.kind!==sent.kind||value.revision!==sent.payload.expected_revision+1||!/^[a-f0-9]{32}$/.test(value.audit_id))throw Error('The operation is unconfirmed. Retry the original operation.');
   clearAdminCommand(sent);store.acknowledge(sent.draft);
   if(!mounted.current)return;held.current=undefined;setCommand(undefined);setNotice(t('操作已保存，审核记录已写入。可以刷新核对最新状态。','Saved and recorded in the audit log. Review the latest state below.'));setEpoch(value=>value+1);
   if(sent.kind==='boards'||sent.kind==='tags')void refreshCommunityTaxonomy();
  }catch(e){if(mounted.current){setError(errorMessage(e));setRejected(e instanceof SocialError&&[400,401,403,404,409,422].includes(e.status));}}
  finally{working.current=false;if(mounted.current)setBusy(false);}
 }
 const run:Run=async(path,kind,row,label,reason,values,draft)=>{
  if(held.current||working.current||!reason.trim())return;
  await send({path,kind,resource:row.id,label,...(draft?{draft}:{}),payload:{...values,expected_user_id:owner,expected_revision:row.revision,request_id:crypto.randomUUID().replaceAll('-',''),reason:reason.trim()}});
 };
 const blocked=busy||!!command,title=sections.find(row=>row[0]===view.section)!;
 const navigate=(next:AdminView)=>{appLocation.hash=adminHref(next);};
 const props={owner,epoch,view,run,blocked,onChange:navigate,detailsAvailable,inboxAvailable};
 return <AdminDraftContext.Provider value={{store,view}}><main className="admin-workspace">
  <aside className="admin-sidebar"><AppLink className="admin-brand" href="#/community/admin"><ShieldCheck size={22}/><span>{t('管理员工作台','Administration')}</span></AppLink><nav aria-label={t('管理员导航','Administration navigation')}>{sections.filter(([id])=>id!=='highlights'||highlightsAvailable).map(([id,zh,en,Icon])=><AppLink key={id} href={history.current.get(id)||adminHref({section:id,page:1})} aria-current={view.section===id?'page':undefined}><Icon size={18}/>{t(zh,en)}</AppLink>)}</nav><AppLink className="admin-community" href="#/community">{t('返回社区','Back to community')}<ArrowUpRight size={16}/></AppLink></aside>
  <div className="admin-content"><header className="admin-heading"><div><span>{t('PaperLeet 管理','PaperLeet administration')}</span><h1>{t(title[1],title[2])}</h1></div><button type="button" className="button subtle" disabled={blocked} onClick={()=>setEpoch(value=>value+1)}><RefreshCw size={16}/>{t('刷新','Refresh')}</button></header>
   <AdminDraftTray detailsAvailable={detailsAvailable}/>
   {notice&&<p className="admin-notice" role="status">{notice}</p>}{error&&<div className="admin-error" role="alert"><p>{error}</p></div>}
   {command&&<section className="admin-pending" aria-label={t('待确认的管理操作','Unconfirmed administrator operation')}><b>{command.label} · {command.resource}</b><p>{t('这次操作的内容已固定。网络结果不明确时，重试会使用同一个请求编号。','This operation is captured. Retrying an uncertain result uses the same request ID.')}</p><p>{String(command.payload.reason)}</p>{!rejected?<button type="button" className="button" disabled={busy} onClick={()=>void send(command)}>{busy?t('正在确认…','Confirming…'):t('重试原操作','Retry original operation')}</button>:<button type="button" className="button subtle" disabled={busy} onClick={()=>{clearAdminCommand(command);held.current=undefined;setCommand(undefined);setError('');setEpoch(value=>value+1);}}>{t('载入最新状态，重新核对','Load latest state and review again')}</button>}</section>}
   {view.section===''?<Overview {...props}/>:view.section==='highlights'?<AdminHighlights {...props}/>:view.section==='reports'?(!reportNavigation&&(view.page>1||view.state&&view.state!=='pending'||view.minimum&&view.minimum!=='1')?<section className="admin-stale" role="status"><p>{t('此服务暂不支持保存举报审核位置，当前筛选已保留。','Saved report review navigation is unavailable on this service. Your filters are kept.')}</p><AppLink href={adminHref({section:'reports',page:1,...(inboxAvailable&&view.item?{item:view.item}:{})})}>{t('打开兼容的举报队列','Open the compatible report queue')}</AppLink></section>:<Reports key={reportNavigation?'scoped':'legacy'} embedded refreshKey={epoch} initialPostId={inboxAvailable?view.item:undefined} navigation={reportNavigation?{view,onChange:navigate}:undefined}/>):view.section==='boards'||view.section==='tags'?<Taxonomy key={view.section} {...props}/>:view.section==='users'?<Members {...props}/>:view.section==='papers'?<Papers {...props}/>:<AuditLog {...props}/>}
  </div>
 </main></AdminDraftContext.Provider>;
}
