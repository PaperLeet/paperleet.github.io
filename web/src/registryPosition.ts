import {appLocation, routeHash, sameAppUrl} from './appLocation';
import {libraryReturnHref,paperRegistryHref,readPaperRegistryRoute} from './libraryNavigation';

/** Public directory positions only: no account, code, grades or server requests. */
export type RegistryPosition={paper:string;top:number;link?:string;keyboard:boolean;at:number};
export type RegistryPositionNotice={kind:'restored'|'available'|'missing';paper:string;request:number};
const key='paperleet.registry.positions.v1',limit=8,maxAge=6*60*60*1000;
const positions=new Map<string,RegistryPosition>();
let hydrated=false,requestId=0;
const canonical=(href:string)=>{const view=readPaperRegistryRoute(href);return view?paperRegistryHref(view):undefined;};
function valid(href:string,value:unknown):value is RegistryPosition{
 if(!value||typeof value!=='object'||Array.isArray(value))return false;
 const row=value as RegistryPosition;
 return Object.keys(row).every(name=>['paper','top','link','keyboard','at'].includes(name))&&typeof row.paper==='string'&&/^[a-z0-9_-]{1,100}$/.test(row.paper)
  &&Number.isFinite(row.top)&&Math.abs(row.top)<=1e5&&typeof row.keyboard==='boolean'&&Number.isSafeInteger(row.at)&&row.at<=Date.now()&&Date.now()-row.at<=maxAge
  &&(!row.keyboard||typeof row.link==='string')&&(!Object.hasOwn(row,'link')||typeof row.link==='string'&&row.link.startsWith('#/')&&row.link.length<=8192&&libraryReturnHref(row.link)===href);
}
function hydrate(){
 if(hydrated)return;hydrated=true;
 try{
  const raw=sessionStorage.getItem(key);if(!raw||raw.length>65536)return;
  const rows:unknown=JSON.parse(raw);if(!Array.isArray(rows))return;
  for(const item of rows.slice(-limit))if(Array.isArray(item)&&item.length===2&&typeof item[0]==='string'){
   const href=canonical(item[0]);if(href&&valid(href,item[1]))positions.set(href,{...item[1]});
  }
 }catch{/* A denied or corrupt session store must not block browsing. */}
}
export function readRegistryPosition(href:string){
 hydrate();const name=canonical(href),row=name?positions.get(name):undefined;
 return name&&row&&valid(name,row)?{...row}:undefined;
}
export function rememberRegistryPosition(href:string,value:RegistryPosition){
 hydrate();const name=canonical(href);if(!name||!valid(name,value))return;
 positions.delete(name);positions.set(name,{...value});
 while(positions.size>limit)positions.delete(positions.keys().next().value!);
 try{sessionStorage.setItem(key,JSON.stringify([...positions]));}catch{/* The current tab keeps its in-memory positions. */}
}

/** One mounted view owns its listeners and pending frame, including delayed catalogs. */
export function watchRegistryPosition(root:HTMLElement,href:string,restore:boolean,notice:(value:RegistryPositionNotice|undefined)=>void){
 const origin=restore?readRegistryPosition(href):undefined;
 let active=true,ready=false,attempted=!origin,interactions=0,frame:number|undefined,timer:ReturnType<typeof setTimeout>|undefined;
 let anchor:RegistryPosition|undefined=origin,kind:RegistryPositionNotice['kind']|undefined;
 let pending:{request?:number;kind:'restore'|'dismiss';run:FrameRequestCallback}|undefined;
 const mounted=()=>active&&root.isConnected&&root.dataset.registryView===href;
 const current=()=>mounted()&&canonical(appLocation.hash)===href;
 const registryHere=()=>mounted()&&!!canonical(appLocation.hash);
 const rows=()=>Array.from(root.querySelectorAll<HTMLElement>('[data-paper-registry-id]'));
 const card=(paper:string)=>rows().find(row=>row.dataset.paperRegistryId===paper);
 function target(value:RegistryPosition){
  const row=card(value.paper);if(!row)return;
  const link=value.link&&Array.from(row.querySelectorAll<HTMLAnchorElement>('a[href]')).find(node=>routeHash(node.href)===value.link);
  return {row,node:link||(value.link?row.querySelector<HTMLElement>('h2 a')||row:row),focus:link||row.querySelector<HTMLElement>('h2 a')||row};
 }
 function inset(){
  const header=root.closest('.app')?.querySelector<HTMLElement>(':scope > .topbar');if(!header)return 12;
  const style=window.getComputedStyle(header),bounds=header.getBoundingClientRect();
  return ['sticky','fixed'].includes(style.position)&&bounds.height>0&&bounds.top<=0&&Number.isFinite(bounds.bottom)?Math.max(12,bounds.bottom+12):12;
 }
 function viewport(){
  const visual=window.visualViewport,valid=visual&&Number.isFinite(visual.offsetTop)&&Number.isFinite(visual.height)&&visual.height>0;
  const top=valid?Math.max(0,visual.offsetTop):0,bottom=valid?Math.min(window.innerHeight,top+visual.height):window.innerHeight;
  return {top:Math.max(inset(),top+12),bottom};
 }
 function visible(node:HTMLElement){const bounds=node.getBoundingClientRect(),range=viewport();return node.isConnected&&!node.closest('[hidden]')&&bounds.height>0&&bounds.bottom>range.top&&bounds.top<range.bottom;}
 function capture(link?:HTMLAnchorElement,keyboard=false){
  if(!(link?registryHere():current())||!ready)return;
  const preferred=link?.closest<HTMLElement>('[data-paper-registry-id]');
  const retained=anchor&&target(anchor),row=preferred&&root.contains(preferred)?preferred:retained&&visible(retained.node)?retained.row:rows().find(visible);
  if(!row)return;
  const retainedLink=retained?.row===row&&anchor?.link?Array.from(row.querySelectorAll<HTMLAnchorElement>('a[href]')).find(node=>routeHash(node.href)===anchor!.link&&visible(node)):undefined;
  const selected=link||retainedLink,node=selected||row,bounds=node.getBoundingClientRect();
  if(!visible(node))return;
  const hash=selected?routeHash(selected.href):undefined;
  const value:RegistryPosition={paper:row.dataset.paperRegistryId!,top:bounds.top,keyboard:!!selected&&(link?keyboard:!!anchor?.keyboard&&document.activeElement===selected),at:Date.now(),...(hash&&libraryReturnHref(hash)===href?{link:hash}:{})};
  rememberRegistryPosition(href,value);anchor=value;
 }
 function report(next:RegistryPositionNotice['kind'],request=++requestId){
  if(next!=='restored'){pending=undefined;if(frame!==undefined)window.cancelAnimationFrame(frame);frame=undefined;}
  if(mounted()&&origin){kind=next;notice({kind:next,paper:origin.paper,request});}
 }
 function schedule(opener?:HTMLElement){
  if(!origin||!ready||!mounted())return;
  const found=target(origin);if(!found){attempted=true;report('missing');return;}
  const focused=document.activeElement;
  if(!opener&&(!current()||interactions!==0||focused!==document.body&&focused?.isConnected)){attempted=true;report('available');return;}
  attempted=true;if(frame!==undefined)window.cancelAnimationFrame(frame);frame=undefined;
  const before=interactions,url=location.href,request=++requestId;
  pending={kind:'restore',request,run:()=>{
   frame=undefined;
   if(!mounted()||location.href!==url||!ready)return;
   if(document.hidden){attempted=false;report('available');return;}
   const focused=document.activeElement;
   if(interactions!==before||(opener?focused!==opener&&!(focused===document.body&&!opener.isConnected):focused!==document.body&&!!focused?.isConnected)){report('available');return;}
   const found=target(origin);if(!found){report('missing');return;}
   if(!found.node.isConnected||found.node.closest('[hidden]')){report('available');return;}
   const bounds=found.node.getBoundingClientRect(),range=viewport(),top=range.top,bottom=range.bottom-Math.min(Math.max(44,bounds.height),Math.max(44,range.bottom-top))-12;
   if(!Number.isFinite(bounds.top)||bounds.height<=0||top>=range.bottom-24){report('available');return;}
   const desired=Math.max(top,Math.min(origin.top,bottom));
   window.scrollTo({top:Math.max(0,window.scrollY+bounds.top-desired),behavior:'instant'});
   anchor={...origin,top:desired};
   if(origin.keyboard||opener)found.focus.focus({preventScroll:true});
  }};
  report('restored',request);
 }
 const intent=()=>{interactions++;};
 const scroll=()=>{clearTimeout(timer);timer=setTimeout(()=>capture(),120);};
 const pagehide=()=>capture();
 const visibility=()=>{if(!document.hidden&&ready&&!attempted)schedule();};
 const click=(event:MouseEvent)=>{
  if(event.defaultPrevented||event.button!==0||event.ctrlKey||event.metaKey||event.shiftKey||event.altKey||!(event.target instanceof window.Element))return;
  const link=event.target.closest<HTMLAnchorElement>('a[href]');if(!link||!root.contains(link)||link.target&&link.target!=='_self'||link.hasAttribute('download'))return;
  const destination=new URL(link.href,location.href);
  if(!sameAppUrl(destination)||libraryReturnHref(routeHash(destination))!==href)return;
  capture(link,event.detail===0&&document.activeElement===link);
 };
 for(const name of ['wheel','touchmove','pointerdown','keydown','focusin'])window.addEventListener(name,intent,{passive:true});
 window.addEventListener('scroll',scroll,{passive:true});window.addEventListener('pagehide',pagehide);document.addEventListener('visibilitychange',visibility);root.addEventListener('click',click,true);
 return {
  ready(value:boolean){
   ready=value;if(!ready&&(frame!==undefined||pending)){if(pending?.kind==='restore'||kind==='restored')attempted=false;if(frame!==undefined)window.cancelAnimationFrame(frame);frame=undefined;pending=undefined;}
   if(ready&&!document.hidden&&!attempted)schedule();else if(ready&&origin&&kind){if(!target(origin))report('missing');else if(kind==='missing')report('available');}
  },
  // The notice changes row geometry; only its matching DOM commit may start measurement.
  commit(request:number|undefined){
   const next=pending;if(!next||!ready||!mounted()||document.hidden)return;
   const element=root.querySelector<HTMLElement>('.paper-registry-position');
   if(next.kind==='restore'?(request!==next.request||element?.dataset.registryPositionRequest!==String(request)):(request!==undefined||!!element))return;
   pending=undefined;frame=window.requestAnimationFrame(next.run);
  },
  restore(opener:HTMLElement){if(mounted()&&root.contains(opener)&&document.activeElement===opener)schedule(opener);},
  dismiss(opener:HTMLElement,keyboard:boolean){
   if(!mounted()||!root.contains(opener)||document.activeElement!==opener)return;
   const retained=anchor&&target(anchor),row=retained&&visible(retained.node)?retained.row:rows().find(visible);
   const node=row&&(retained?.row===row?retained.node:row),top=node?.getBoundingClientRect().top,before=interactions,url=location.href;
   attempted=true;if(frame!==undefined)window.cancelAnimationFrame(frame);frame=undefined;pending=undefined;kind=undefined;
   if(!node||top===undefined){notice(undefined);return;}
   pending={kind:'dismiss',run:()=>{
    frame=undefined;const focused=document.activeElement;
    if(!mounted()||location.href!==url||!node.isConnected||interactions!==before||focused!==opener&&!(focused===document.body&&!opener.isConnected))return;
    const delta=node.getBoundingClientRect().top-top;
    if(Number.isFinite(delta)&&Math.abs(delta)>.5)window.scrollTo({top:Math.max(0,window.scrollY+delta),behavior:'instant'});
    if(keyboard)(retained&&retained.row===row?retained.focus:row?.querySelector<HTMLElement>('h2 a')||node).focus({preventScroll:true});
   }};
   notice(undefined);
  },
  stop(){active=false;pending=undefined;clearTimeout(timer);if(frame!==undefined)window.cancelAnimationFrame(frame);for(const name of ['wheel','touchmove','pointerdown','keydown','focusin'])window.removeEventListener(name,intent);window.removeEventListener('scroll',scroll);window.removeEventListener('pagehide',pagehide);document.removeEventListener('visibilitychange',visibility);root.removeEventListener('click',click,true);},
 };
}
