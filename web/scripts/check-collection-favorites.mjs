import assert from 'node:assert/strict';
import React,{act} from 'react';
import {renderToString} from 'react-dom/server';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';
const dom=new JSDOM('<!doctype html><div id="root"></div>',{url:'http://paperleet.test/#/',pretendToBeVisual:true});
const values={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,Element:dom.window.Element,localStorage:dom.window.localStorage,sessionStorage:dom.window.sessionStorage,location:dom.window.location,ResizeObserver:class{observe(){}disconnect(){}},IS_REACT_ACT_ENVIRONMENT:true,__collectionFixture:undefined};
const previous=new Map(Object.keys(values).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
for(const [key,value] of Object.entries(values))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
localStorage.setItem('paperleet.locale','en');
function createFixtureServer(environment){return createServer({define:Object.fromEntries(Object.entries(environment).map(([key,value])=>['import.meta.env.'+key,JSON.stringify(value)])),server:{middlewareMode:true,hmr:false,ws:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},plugins:[{name:'collection-account',enforce:'pre',resolveId(source){if(source==='./social')return '\0collection-account';if(source==='./DailyChallenges')return '\0collection-daily';},load(name){if(name==='\0collection-daily')return 'export const DailyProblemRow=()=>null;export default ()=>null;';if(name==='\0collection-account')return "import React from 'react';export const useAccount=()=>globalThis.__collectionFixture.account;export const socialApi=(...args)=>globalThis.__collectionFixture.request(...args);export class SocialError extends Error{constructor(message,status,code){super(message);this.status=status;this.code=code;}};export const SocialNotice=({error})=>React.createElement('p',{role:'alert'},error);";}}]});}
const server=await createFixtureServer({VITE_APP_EDITION:'hosted',VITE_PUBLIC_SITE:'0'});
let root;
try{
 const {createCollectionFavoriteStore,savedCollectionsFirst}=await server.ssrLoadModule('/src/collectionFavoriteStore.ts');
 const snapshot=(revision,ids,owner='unit')=>({owner_id:owner,revision,items:ids.map(collection_id=>({collection_id,created_at:'2026-11-06T00:00:00Z'}))});
 let readResolve,writeResolve,reads=0,fail=false;
 const writes=[];
 const store=createCollectionFavoriteStore('unit',body=>{if(!body){reads++;return new Promise(resolve=>readResolve=resolve);}writes.push(body);if(fail)return Promise.reject(Error('Lost acknowledgement'));return new Promise(resolve=>writeResolve=resolve);});
 const first=store.refresh(),second=store.refresh();assert.equal(reads,1);readResolve(snapshot(0,[]));await Promise.all([first,second]);
 fail=true;await store.set('simclr',true);assert.equal(store.getSnapshot().pending.simclr.active,true);assert.equal(store.getSnapshot().snapshot.items.length,0);
 fail=false;const retry=store.set('simclr',false);assert.equal(writes.at(-1).active,true);writeResolve({...snapshot(1,['simclr']),collection_id:'simclr',active:true});await retry;
 const stale=store.refresh(),removal=store.set('simclr',false);writeResolve({...snapshot(2,[]),collection_id:'simclr',active:false});await removal;readResolve(snapshot(1,['simclr']));await stale;assert.equal(store.getSnapshot().snapshot.revision,2);assert.equal(store.getSnapshot().snapshot.items.length,0);
 for(const value of [snapshot(1,['x'],'other'),snapshot(1,['x','x']),snapshot(-1,[]),snapshot(1,['../invalid'])]){
  const invalid=createCollectionFavoriteStore('unit',async()=>value);await invalid.refresh();assert(invalid.getSnapshot().error);assert(!invalid.getSnapshot().snapshot);
 }
 const raceReplies=[];
 const race=createCollectionFavoriteStore('unit',body=>body?new Promise(resolve=>raceReplies.push({body,resolve})):Promise.resolve(snapshot(0,[])));
 await race.refresh();const a=race.set('a',true),b=race.set('b',true);raceReplies[1].resolve({...snapshot(2,['a','b']),collection_id:'b',active:true});await b;raceReplies[0].resolve({...snapshot(1,['a']),collection_id:'a',active:true});await a;assert.deepEqual(race.getSnapshot().snapshot.items.map(item=>item.collection_id),['a','b']);
 const unconfirmed=createCollectionFavoriteStore('unit',async body=>body?{...snapshot(1,[]),collection_id:'other',active:true}:snapshot(0,[]));await unconfirmed.refresh();await unconfirmed.set('a',true);assert(unconfirmed.getSnapshot().pending.a.error);assert.equal(unconfirmed.getSnapshot().snapshot.revision,0);
 const original=[{id:'a'},{id:'b'},{id:'c'},{id:'d'}];assert.deepEqual(savedCollectionsFirst(original,new Set(['d','b'])).map(item=>item.id),['b','d','a','c']);assert.deepEqual(original.map(item=>item.id),['a','b','c','d']);

 const {LocaleProvider}=await server.ssrLoadModule('/src/i18n.tsx');
 const {paperCollections}=await server.ssrLoadModule('/src/paperCollections.ts');
 const {default:LibraryCollections}=await server.ssrLoadModule('/src/LibraryCollections.tsx');
 const {default:CollectionFavoriteButton}=await server.ssrLoadModule('/src/CollectionFavorites.tsx');
 const {default:CollectionReadingList}=await server.ssrLoadModule('/src/CollectionReadingList.tsx');
 const {signInHref,accountReturnHref}=await server.ssrLoadModule('/src/accountNavigation.ts');
 const {decodeProblemHash}=await server.ssrLoadModule('/src/reports.ts');
 const ids=paperCollections.map(item=>item.id),target=ids.at(-1),firstId=ids[0],third=ids[3];
 const persisted=new Map([['alice',{revision:1,ids:[third]}],['bob',{revision:0,ids:[]}]]);
 let failRead=false,failWrite=false,delayWrite=false,delayRead=false,releaseWrite,releaseRead,wrongOwner=false,refreshCount=0,sessionOwner;
 const account={user:{id:'alice'},loading:false,capabilities:{collection_bookmarks:true},refresh:async()=>{refreshCount++;if(sessionOwner){account.user={id:sessionOwner};root.render(React.createElement(LocaleProvider,null,React.createElement(LibraryCollections,props)));}}};
 const calls=[];
 const update=body=>{const entry=persisted.get(body.expected_user_id);entry.ids=body.active?[...new Set([...entry.ids,body.collection_id])]:entry.ids.filter(id=>id!==body.collection_id);entry.revision++;return {...snapshot(entry.revision,entry.ids,body.expected_user_id),collection_id:body.collection_id,active:body.active};};
 globalThis.__collectionFixture={account,request:async(path,body)=>{
  assert.equal(path,'account/collection-bookmarks');calls.push(body);
  if(!body){if(failRead)throw Error('Unavailable');const owner=sessionOwner||account.user.id,entry=persisted.get(owner),value=snapshot(entry.revision,entry.ids,wrongOwner?'other':owner);if(delayRead)return new Promise(resolve=>releaseRead=()=>resolve(value));return value;}
  assert.equal(body.expected_user_id,account.user.id);if(failWrite)throw Error('Connection lost');if(delayWrite)return new Promise(resolve=>releaseWrite=()=>resolve(update(body)));return update(body);
 }};
 const props={papers:[],problems:[],saved:{},progressReady:true};
 const initialReads=calls.length;
 assert(renderToString(React.createElement(LocaleProvider,null,React.createElement(LibraryCollections,props))).includes('collection-card collection-'));
 assert.equal(calls.length,initialReads,'SSR renders canonical links without fetching account data');
 const {createRoot}=await import('react-dom/client');root=createRoot(document.getElementById('root'));
 const render=async(element=React.createElement(LibraryCollections,props))=>act(async()=>root.render(React.createElement(LocaleProvider,null,element)));
 const star=id=>document.querySelector(`[data-collection-favorite="${id}"]`);
 const order=()=>[...document.querySelectorAll('.collection-card')].map(node=>node.getAttribute('href').split('/').at(-1));
 const expected=saved=>savedCollectionsFirst(paperCollections,new Set(saved)).map(item=>item.id);
 const click=async node=>{assert(node);await act(async()=>{node.focus();node.click();});};
 await render();assert.equal(calls.filter(row=>!row).length,1);assert.deepEqual(order(),expected([third]));
 assert.equal(document.querySelectorAll('.collection-card button,.collection-card a').length,0,'Stars are siblings, never nested interactive elements');
 for(const id of ids)assert.equal(star(id).parentElement.querySelector('.collection-card').getAttribute('href'),'#/collections/'+id);
 const rail=document.querySelector('.collection-rail'),targetButton=star(target),route=location.href;
 rail.scrollLeft=800;rail.getBoundingClientRect=()=>({left:0,right:600});targetButton.getBoundingClientRect=()=>({left:-800,right:-762});
 delayWrite=true;await click(targetButton);assert.equal(targetButton.getAttribute('aria-pressed'),'false');assert.deepEqual(order(),expected([third]));assert.equal(location.href,route);assert.equal(rail.scrollLeft,800);
 await act(async()=>releaseWrite());delayWrite=false;assert.equal(star(target),targetButton);assert.equal(document.activeElement,targetButton);assert.equal(targetButton.getAttribute('aria-pressed'),'true');assert.deepEqual(order(),expected([third,target]));assert.equal(rail.scrollLeft,0,'Only the completed explicit action reveals its moved focused star horizontally');
 assert.deepEqual(Object.keys(JSON.parse(localStorage.getItem('paperleet.collection-favorites.changed'))).sort(),['at','nonce','owner'],'Storage contains only invalidation metadata, never saved collection IDs');
 await click(star(firstId));assert.deepEqual(order(),expected([firstId,third,target]));await click(targetButton);assert.deepEqual(order(),expected([firstId,third]));assert.equal(document.activeElement,targetButton);
 failWrite=true;const before=order();await click(targetButton);assert.equal(targetButton.getAttribute('aria-pressed'),'false');assert.deepEqual(order(),before);assert(targetButton.getAttribute('aria-label').includes('Retry saving'));assert(document.querySelector('[role=status]').textContent.includes('unconfirmed'));
 failWrite=false;await click(targetButton);assert.equal(calls.at(-1).active,true);assert.equal(targetButton.getAttribute('aria-pressed'),'true');
 // Revisit fetches persisted data, rather than using a global browser favorites list.
 await render(null);const beforeRevisit=calls.filter(row=>!row).length;await render();assert.equal(calls.filter(row=>!row).length,beforeRevisit+1);assert.deepEqual(order(),expected(persisted.get('alice').ids));
 const revisitRail=document.querySelector('.collection-rail');revisitRail.scrollLeft=500;
 persisted.get('alice').ids=[target];persisted.get('alice').revision++;
 const beforeOtherTab=calls.length;
 await act(async()=>window.dispatchEvent(new dom.window.StorageEvent('storage',{key:'paperleet.collection-favorites.changed',newValue:JSON.stringify({owner:'bob'})})));assert.equal(calls.length,beforeOtherTab);
 await act(async()=>window.dispatchEvent(new dom.window.StorageEvent('storage',{key:'paperleet.collection-favorites.changed',newValue:JSON.stringify({owner:'alice'})})));assert.deepEqual(order(),expected([target]));assert.equal(revisitRail.scrollLeft,500,'Background updates never move the rail');
 failRead=true;await act(async()=>window.dispatchEvent(new dom.window.Event('focus')));assert(star(target).getAttribute('aria-label').includes('Retry loading'));assert.equal(star(target).getAttribute('aria-pressed'),'true');const beforeRetryWrites=calls.filter(Boolean).length;failRead=false;await click(star(target));assert.equal(calls.filter(Boolean).length,beforeRetryWrites);
 // Leaving the route or moving focus while an acknowledgement is pending must not steal it back.
 delayWrite=true;await click(star(firstId));const newer=document.createElement('button');document.body.append(newer);newer.focus();await act(async()=>releaseWrite());assert.equal(document.activeElement,newer);newer.remove();
 await click(star(third));window.history.replaceState(null,'','#/settings');await act(async()=>releaseWrite());assert.equal(location.hash,'#/settings');const beforeStale=calls.length;await click(star(target));assert.equal(calls.length,beforeStale,'Retained controls cannot mutate after leaving the library');window.history.replaceState(null,'','#/');delayWrite=false;
 // A delayed Alice mutation may finish only in Alice's store, never in Bob's view.
 delayWrite=true;await click(star(firstId));account.user={id:'bob'};await render();assert.deepEqual(order(),ids);await act(async()=>releaseWrite());assert.deepEqual(order(),ids);assert.equal(document.querySelectorAll('[aria-pressed=true]').length,0);delayWrite=false;
 // An old GET is equally isolated, including a cold account with an untrusted owner DTO.
 persisted.set('carol',{revision:1,ids:[target]});delayRead=true;account.user={id:'carol'};await render();const releaseCarol=releaseRead;delayRead=false;account.user={id:'bob'};await render();await act(async()=>releaseCarol());assert.deepEqual(order(),ids);
 persisted.set('wrong',{revision:0,ids:[target]});account.user={id:'wrong'};wrongOwner=true;await render();assert.deepEqual(order(),ids);assert(star(target).getAttribute('aria-label').includes('Retry loading'));wrongOwner=false;await click(star(target));assert.deepEqual(order(),expected([target]));
 // A login changed on another origin can return HTTP 200 for the new owner.
 account.user={id:'bob'};await render();const beforeAccountRefresh=refreshCount;sessionOwner='alice';await act(async()=>window.dispatchEvent(new dom.window.Event('focus')));assert.equal(account.user.id,'alice');assert.equal(refreshCount,beforeAccountRefresh+1);assert.deepEqual(order(),expected(persisted.get('alice').ids),'Owner mismatch refreshes the account before displaying the newly authenticated owner');sessionOwner=undefined;
 account.user=null;await render();assert.deepEqual(order(),ids);assert.equal(document.querySelectorAll('button[data-collection-favorite]').length,0);const guest=document.querySelector('.collection-card-shell .collection-favorite-button');assert.equal(accountReturnHref(decodeProblemHash(guest.getAttribute('href'))),'#/collections/'+firstId);
 const guestCalls=calls.length;await click(guest);await new Promise(resolve=>setTimeout(resolve,0));assert(location.hash.includes('account/continue/collections'));assert.equal(calls.length,guestCalls);
 // Both detail entry points reuse the same accessible action and preserve the collection continuation.
 const filtered='#/collections/'+target+'?q=attention';window.history.replaceState(null,'',filtered);await render(React.createElement(CollectionFavoriteButton,{collectionId:target,returnTo:filtered}));assert.equal(accountReturnHref(decodeProblemHash(document.querySelector('.collection-page-favorite a').getAttribute('href'))),filtered);
 window.history.replaceState(null,'','#/collections/multimodal');await render(React.createElement(CollectionReadingList,{...props,collectionId:'multimodal',route:'#/collections/multimodal'}));assert(document.querySelector('.collection-page-header .collection-favorite-button'));
 account.capabilities.collection_bookmarks=false;window.history.replaceState(null,'','#/');await render();assert(!document.querySelector('.collection-favorite-button'));assert.equal(calls.length,guestCalls);
 for(const edition of [
  {VITE_APP_EDITION:'local',VITE_PUBLIC_SITE:'0',VITE_HOSTED_APP_AVAILABLE:'0'},
  {VITE_APP_EDITION:'local',VITE_PUBLIC_SITE:'1',VITE_HOSTED_APP_AVAILABLE:'1',VITE_HOSTED_APP_URL:'https://app.paperleet.com'},
 ]){
  const editionServer=await createFixtureServer(edition);
  try{
   const {default:View}=await editionServer.ssrLoadModule('/src/CollectionFavorites.tsx'),{LocaleProvider:Locale}=await editionServer.ssrLoadModule('/src/i18n.tsx');
   account.capabilities=undefined;account.user=null;window.history.replaceState(null,'',filtered);const before=calls.length;
   await act(async()=>root.render(React.createElement(Locale,null,React.createElement(View,{collectionId:target,returnTo:filtered}))));
   if(edition.VITE_HOSTED_APP_AVAILABLE==='1')assert.equal(document.querySelector('.collection-favorite-button').getAttribute('href'),'https://app.paperleet.com/'+signInHref(filtered));
   else {assert(!document.querySelector('.collection-favorite-button'));account.user={id:'offline'};account.capabilities={collection_bookmarks:true};await act(async()=>root.render(React.createElement(Locale,null,React.createElement(View,{collectionId:target,returnTo:filtered}))));assert(!document.querySelector('.collection-favorite-button'));}
   assert.equal(calls.length,before);
  }finally{await editionServer.close();}
 }
 assert.equal(refreshCount,2);
 console.log('PASS collection favorites: SSR, independent accessible stars, confirmed stable ordering, save/remove/revisit, fixed-direction retry, strict owner DTOs and monotonic revisions, cross-tab refresh, logout/account switch/stale requests, guest continuation, unsupported editions, keyboard focus and user-only horizontal reveal');
}finally{if(root)await act(async()=>root.unmount());await server.close();dom.window.close();for(const [key,value] of previous){if(value)Object.defineProperty(globalThis,key,value);else delete globalThis[key];}}
