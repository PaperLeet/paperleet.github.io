import AppLink from './AppLink';
import {createContext,useContext,useRef,useState,useSyncExternalStore,type ReactNode} from 'react';
import {useI18n} from './i18n';
import {adminHref} from './adminLinks';
import type {AdminView} from './adminNavigation';
import {createAdminDraftStore,type AdminDraftIntent,type AdminDraftStore} from './adminDraftStore';
export const AdminDraftContext=createContext<{store:AdminDraftStore;view:AdminView}|undefined>(undefined);
export function useAdminDraft<T extends object>(key:string,initial:T,label:string){
 const context=useContext(AdminDraftContext);if(!context)throw Error('Administrator draft scope is missing');
 const {store,view}=context,latest=useRef(initial);latest.current=initial;
 const entry=useSyncExternalStore(listener=>store.subscribe(key,listener),()=>store.read(key),()=>undefined);
 const value=(entry?.value as T|undefined)||initial;
 return {value,entry,intent:()=>store.intent(key),change:(update:Partial<T>)=>store.update(key,{...(store.read(key)?.value||latest.current),...update},label,view),discard:(intent:AdminDraftIntent)=>store.discard(intent)};
}
export function AdminDraftRow({draftKey,forceOpen=false,onOpenChange,children,...props}:{draftKey:string;forceOpen?:boolean;onOpenChange?:(open:boolean)=>void;children:ReactNode;className?:string;hidden?:boolean}){
 const context=useContext(AdminDraftContext)!,[initialOpen]=useState(()=>context.store.isOpen(draftKey)??!!context.store.read(draftKey));
 return <details {...props} open={forceOpen||initialOpen} onToggle={event=>{if(event.currentTarget.isConnected){context.store.setOpen(draftKey,event.currentTarget.open);onOpenChange?.(event.currentTarget.open);}}}>{children}</details>;
}
export function AdminDraftActions({draft,disabled,onReset}:{draft:ReturnType<typeof useAdminDraft>;disabled:boolean;onReset?:()=>void}){
 const {t}=useI18n(),[confirmation,setConfirmation]=useState<AdminDraftIntent>(),[changed,setChanged]=useState(false);
 if(!draft.entry)return null;
 return <div className="admin-draft-actions"><p role="status">{draft.entry.acknowledged?t('操作已记录，输入保留在本次工作台中便于核对。','Recorded. These inputs remain in this workspace visit for reference.'):t('未提交内容已保留，切换筛选或页签后可以继续。','Unsubmitted input is kept while you browse filters and sections.')}</p>{!confirmation?<button type="button" className="text-button" disabled={disabled} onClick={()=>{setConfirmation(draft.intent());setChanged(false);}}>{t('重置本条草稿','Reset this draft')}</button>:<div><p>{changed?t('草稿刚有更新，请重新确认重置。','The draft changed. Confirm the reset again.'):t('重置这条未提交草稿？已记录的操作和待确认请求不会被撤销。','Reset these inputs? Recorded operations and unconfirmed requests are retained.')}</p><button type="button" disabled={disabled} onClick={()=>{if(draft.discard(confirmation)){onReset?.();setConfirmation(undefined);setChanged(false);}else{setConfirmation(draft.intent());setChanged(true);}}}>{t('确认重置草稿','Confirm draft reset')}</button><button type="button" onClick={()=>setConfirmation(undefined)}>{t('保留草稿','Keep draft')}</button></div>}</div>;
}
export function AdminDraftTray({detailsAvailable}:{detailsAvailable:boolean}){
 const {t}=useI18n(),context=useContext(AdminDraftContext)!,{store}=context,rows=useSyncExternalStore(listener=>store.subscribe('*',listener),store.summary,store.summary);
 if(!rows.length)return null;
 return <details className="admin-draft-tray"><summary>{t(`本次工作台有 ${rows.length} 份未提交草稿`,`${rows.length} unsubmitted drafts in this workspace visit`)}</summary><p>{t('切换页签、筛选和翻页会保留这些输入。离开工作台、刷新页面或重新核验账号时会清除；待确认的操作仍可重试。','Inputs are kept across sections, filters and pages. They clear when you leave, reload or recheck your account. Unconfirmed operations remain retryable.')}</p><ul>{rows.map(({key,entry})=>{const [section,id]=key.split(':'),detail=detailsAvailable||['boards','tags'].includes(section);const view:AdminView={...entry.view,...(section==='papers'&&!detailsAvailable?{review:'all'}:{}),...(detail?id==='$new'?{item:undefined,create:true}:{item:id,create:undefined}:{item:undefined,create:undefined})};return <li key={key}><AppLink href={adminHref(view)}>{entry.label}</AppLink><small>{t('返回继续核对','Continue reviewing')}</small></li>;})}</ul></details>;
}
