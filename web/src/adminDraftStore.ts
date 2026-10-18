import type {AdminView} from './adminNavigation';
import {taxonomyBrowse,type TaxonomyBrowse} from './adminTaxonomyBrowsing';
export type AdminDraftIntent={workspace:string;key:string;stamp:number};
export type AdminDraftEntry={value:object;stamp:number;acknowledged:boolean;label:string;view:AdminView};
/** Private, visit-scoped state. Only an unchanged captured draft can be acknowledged/discarded. */
export function createAdminDraftStore(){
 const workspace=crypto.randomUUID().replaceAll('-',''),entries=new Map<string,AdminDraftEntry>(),open=new Map<string,boolean>(),browsing=new Map<'boards'|'tags',TaxonomyBrowse>(),listeners=new Map<string,Set<()=>void>>();let stamp=0,summary:{key:string;entry:AdminDraftEntry}[]=[];
 const notify=(key:string)=>{for(const listener of listeners.get(key)||[])listener();};
 const summarize=()=>{summary=[...entries].filter(([,entry])=>!entry.acknowledged).map(([key,entry])=>({key,entry}));notify('*');};
 function subscribe(key:string,listener:()=>void){let set=listeners.get(key);if(!set){set=new Set();listeners.set(key,set);}set.add(listener);return()=>{set!.delete(listener);if(!set!.size)listeners.delete(key);};}
 function newTaxonomyIdentifier(kind:'boards'|'tags'){const draft=entries.get(kind+':$new')?.value as {value?:{id?:unknown}}|undefined;return typeof draft?.value?.id==='string'?draft.value.id:undefined;}
 return {workspace,subscribe,read:(key:string)=>entries.get(key),summary:()=>summary,
  update(key:string,value:object,label:string,view:AdminView){entries.set(key,{value,stamp:++stamp,acknowledged:false,label,view:{...view}});notify(key);summarize();},
  intent(key:string):AdminDraftIntent|undefined{const entry=entries.get(key);return entry?{workspace,key,stamp:entry.stamp}:undefined;},
  acknowledge(intent?:AdminDraftIntent){if(!intent||intent.workspace!==workspace)return;const entry=entries.get(intent.key);if(entry?.stamp!==intent.stamp)return;entries.set(intent.key,{...entry,acknowledged:true});notify(intent.key);summarize();},
  discard(intent:AdminDraftIntent){if(intent.workspace!==workspace||entries.get(intent.key)?.stamp!==intent.stamp)return false;entries.delete(intent.key);notify(intent.key);summarize();return true;},
  isOpen:(key:string)=>open.get(key),setOpen:(key:string,value:boolean)=>open.set(key,value),
  newTaxonomyIdentifier,
  taxonomyRetentionSnapshot:(kind:'boards'|'tags')=>JSON.stringify([[...entries.keys()].filter(key=>key.startsWith(kind+':')),newTaxonomyIdentifier(kind)]),
  readTaxonomyBrowse:(kind:'boards'|'tags')=>browsing.get(kind),
  setTaxonomyBrowse:(kind:'boards'|'tags',value:TaxonomyBrowse)=>browsing.set(kind,taxonomyBrowse(value)),
 };
}
export type AdminDraftStore=ReturnType<typeof createAdminDraftStore>;
