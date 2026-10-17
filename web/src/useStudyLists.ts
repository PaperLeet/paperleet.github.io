import {useEffect,useMemo,useSyncExternalStore} from 'react';
import {SocialError,socialApi,useAccount} from './social';
import {createStudyListStore,type StudyListsState} from './studyListStore';
const stores=new Map<string,ReturnType<typeof createStudyListStore>>(),empty:StudyListsState={loading:false,error:''};
const key='paperleet.study-lists.changed';
const watchers=new Map<ReturnType<typeof createStudyListStore>,{count:number;stop:()=>void}>();
function watch(store:ReturnType<typeof createStudyListStore>,owner:string){
 let entry=watchers.get(store);
 if(!entry){
  const focus=()=>void store.refresh(),storage=(event:StorageEvent)=>{if(event.key!==key)return;try{if(JSON.parse(event.newValue||'{}').owner===owner)void store.refresh();}catch{/* Invalidations contain no personal content. */}};
  window.addEventListener('focus',focus);window.addEventListener('storage',storage);entry={count:0,stop:()=>{window.removeEventListener('focus',focus);window.removeEventListener('storage',storage);}};watchers.set(store,entry);
 }
 entry.count++;return()=>{if(--entry.count===0){entry.stop();watchers.delete(store);}};
}
/** Only account IDs are shared for invalidation; list names and contents stay in this owner's store. */
export function useStudyLists(enabled=true){
 const account=useAccount(),owner=enabled&&!account.loading&&account.capabilities?.study_lists===true&&account.user?.id||'';
 const store=useMemo(()=>{
  if(!owner)return;
  if(!stores.has(owner)){
   if(stores.size>=3)stores.delete(stores.keys().next().value!);
   stores.set(owner,createStudyListStore(owner,async(body,signal)=>{try{return await socialApi('account/study-lists',body,signal);}catch(error){if(error instanceof SocialError&&(error.status===401||error.code==='account_changed')){try{await account.refresh();}catch{/* Explicit retry remains available. */}}throw error;}},()=>{try{localStorage.setItem(key,JSON.stringify({owner,at:Date.now(),nonce:Math.random()}));}catch{/* Focus refresh still works without storage. */}}));
  }return stores.get(owner)!;
 },[owner]);
 const state=useSyncExternalStore(store?.subscribe||(()=>()=>{}),store?.getSnapshot||(()=>empty));
 useEffect(()=>{if(!store)return;void store.refresh();return watch(store,owner);},[store,owner]);
 return {...account,store,state};
}
