import {useEffect,useLayoutEffect,useMemo,useRef,useSyncExternalStore} from 'react';
import {Star} from 'lucide-react';
import AppLink from './AppLink';
import {appLocation} from './appLocation';
import {controlIsActive} from './activeControls';
import {useI18n} from './i18n';
import {SocialError,socialApi,useAccount} from './social';
import {hostedAppAvailable,hostedAppHref} from './deployment';
import {signInHref} from './accountNavigation';
import {readLibraryRoute} from './libraryNavigation';
import {paperCollections} from './paperCollections';
import {createCollectionFavoriteStore,type CollectionFavoriteState} from './collectionFavoriteStore';
import './collection-favorites.css';

const changedKey='paperleet.collection-favorites.changed';
const stores=new Map<string,ReturnType<typeof createCollectionFavoriteStore>>();
const empty:CollectionFavoriteState={loading:false,error:'',pending:{}};
const emptySnapshot=()=>empty,emptySubscribe=()=>()=>{};
const watchers=new Map<ReturnType<typeof createCollectionFavoriteStore>,{count:number;stop:()=>void}>();
function watch(store:ReturnType<typeof createCollectionFavoriteStore>,owner:string){
 let watcher=watchers.get(store);
 if(!watcher){
  const refresh=()=>void store.refresh();
  const storage=(event:StorageEvent)=>{if(event.key===changedKey){try{if(JSON.parse(event.newValue||'{}').owner===owner)refresh();}catch{/* Ignore malformed invalidations. */}}};
  window.addEventListener('focus',refresh);window.addEventListener('storage',storage);
  watcher={count:0,stop:()=>{window.removeEventListener('focus',refresh);window.removeEventListener('storage',storage);}};watchers.set(store,watcher);
 }
 watcher.count++;
 return()=>{if(--watcher.count===0){watcher.stop();watchers.delete(store);}};
}

export function useCollectionFavorites(){
 const account=useAccount();
 const owner=hostedAppAvailable&&!account.loading&&account.capabilities?.collection_bookmarks&&account.user?.id||'';
 const store=useMemo(()=>{
  if(!owner)return;
  if(!stores.has(owner)){
   if(stores.size>=3)stores.delete(stores.keys().next().value!);
   stores.set(owner,createCollectionFavoriteStore(owner,async(body,signal)=>{
    try{
     const value=await socialApi<{owner_id?:string}>('account/collection-bookmarks',body,signal);
     if(value?.owner_id&&value.owner_id!==owner)throw new SocialError('Your account changed. Please retry.',409,'account_changed');
     return value;
    }
    catch(error){if(error instanceof SocialError&&(error.status===401||error.code==='account_changed')){try{await account.refresh();}catch{/* The star still offers retry. */}}throw error;}
   },()=>{try{localStorage.setItem(changedKey,JSON.stringify({owner,at:Date.now(),nonce:Math.random()}));}catch{/* Focus refresh still works without storage. */}}));
  }
  return stores.get(owner)!;
 },[owner]);
 const state=useSyncExternalStore(store?.subscribe||emptySubscribe,store?.getSnapshot||emptySnapshot,emptySnapshot);
 useEffect(()=>{if(store)void store.refresh();},[store]);
 useEffect(()=>store?watch(store,owner):undefined,[store,owner]);
 return {...account,owner,state,store};
}

type Controller=ReturnType<typeof useCollectionFavorites>;
type ControlProps={collectionId:string;returnTo:string;controller:Controller;rail?:boolean};
export function CollectionFavoriteControl(props:ControlProps){
 return hostedAppAvailable?<FavoriteControl key={props.controller.owner||'guest'} {...props}/>:null;
}
function FavoriteControl({collectionId,returnTo,controller,rail=false}:ControlProps){
 const {t}=useI18n(),{user,owner,loading,capabilities,state,store}=controller;
 const button=useRef<HTMLButtonElement>(null),intent=useRef<{owner:string;href:string;active:boolean}|undefined>(undefined);
 const active=!!state.snapshot?.items.some(item=>item.collection_id===collectionId),pending=state.pending[collectionId];
 const collection=paperCollections.find(item=>item.id===collectionId),name=collection?t(collection.name,collection.name_en):collectionId;
 const live=useRef(owner);live.current=owner;
 function canAct(element:HTMLElement){
  const route=readLibraryRoute(appLocation.hash);
  return live.current===owner&&controlIsActive(element)&&!!route&&(rail?!route.collectionId:route.collectionId===collectionId);
 }
 useEffect(()=>{
  const cancel=(event:Event)=>{if(event.type!=='focusin'||event.target!==button.current)intent.current=undefined;};
  // A later navigation, focus change or scroll cancels this action's reveal.
  const events=['pointerdown','keydown','wheel','touchmove','focusin','hashchange','popstate'];
  events.forEach(event=>window.addEventListener(event,cancel,{passive:true}));
  return()=>events.forEach(event=>window.removeEventListener(event,cancel));
 },[]);
 useLayoutEffect(()=>{
  const action=intent.current,element=button.current;
  if(!action||pending?.busy)return;
  intent.current=undefined;
  if(pending?.error||action.active!==active||action.owner!==owner||action.href!==location.href||!element||!canAct(element))return;
  if(document.activeElement!==element&&document.activeElement!==document.body)return;
  // Keyed cards retain their button; restore focus only if moving it blurred to body.
  element.focus({preventScroll:true});
  const container=rail?element.closest<HTMLElement>('.collection-rail'):null;
  if(container){
   const bounds=element.getBoundingClientRect(),viewport=container.getBoundingClientRect();
   const delta=bounds.left<viewport.left?bounds.left-viewport.left:bounds.right>viewport.right?bounds.right-viewport.right:0;
   if(delta)container.scrollLeft+=delta;
  }
 },[active,pending,owner]);
 if(!loading&&capabilities&&!capabilities.collection_bookmarks)return null;
 const describe=(label:string)=>`${label}${t('：',': ')}${name}`;
 if(!loading&&!user){
  const label=describe(t('登录后收藏合集','Sign in to save collection'));
  return <AppLink className="collection-favorite-button" href={hostedAppHref(signInHref(returnTo))} aria-label={label} title={label} onClick={event=>{if(!canAct(event.currentTarget))event.preventDefault();}}><Star size={19} aria-hidden="true"/></AppLink>;
 }
 const retry=!!state.error,label=describe(retry?t('重试加载合集收藏','Retry loading saved collections'):pending?.error?pending.active?t('重试收藏合集','Retry saving collection'):t('重试取消收藏合集','Retry removing saved collection'):active?t('取消收藏合集','Remove saved collection'):t('收藏合集','Save collection'));
 const busy=!!pending?.busy||state.loading,disabled=loading||!state.snapshot&&!state.error;
 return <>
  <button ref={button} className="collection-favorite-button" data-collection-favorite={collectionId} type="button" aria-pressed={active} aria-label={label} title={label} aria-busy={busy||undefined} aria-disabled={busy||undefined} disabled={disabled} onClick={event=>{
   if(!canAct(event.currentTarget)||busy)return;
   if(retry){void store?.refresh();return;}
   if(document.activeElement===event.currentTarget)intent.current={owner,href:location.href,active:pending?.active??!active};
   void store?.set(collectionId,!active);
  }}><Star size={19} fill={active?'currentColor':'none'} aria-hidden="true"/>{(state.error||pending?.error)&&<span className="collection-favorite-error" aria-hidden="true">!</span>}</button>
  {(state.error||pending?.error)&&<span className="sr-only" role="status">{t('合集收藏尚未确认，请点击星标重试。','Saved collection unconfirmed. Select the star to retry.')}</span>}
 </>;
}

export type Props={collectionId:string;returnTo:string};
export default function CollectionFavoriteButton(props:Props){
 const controller=useCollectionFavorites();
 return <span className="collection-page-favorite"><CollectionFavoriteControl {...props} controller={controller}/></span>;
}
