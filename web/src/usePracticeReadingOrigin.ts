import {appLocation, routeHash} from './appLocation';
import {useEffect,useRef} from 'react';

export function usePracticeReadingOrigin(owner:string|undefined,problem:{id:string;number:number}|undefined,href:string|undefined,route?:string){
 const id=problem?.id,number=problem?.number,expected=route??(typeof location==='undefined'?'':appLocation.hash),live=useRef({owner,active:true});live.current.owner=owner;
 useEffect(()=>{
  live.current.active=true;
  if(!owner||!href)return()=>{live.current.active=false;};
  let current=true,snapshot:string|undefined;const origin=appLocation.hash,events:{before:string;after:string}[]=[];
  try{const raw=sessionStorage.getItem('paperleet.reading.return.v1');if(raw&&raw.length<=12000)snapshot=raw;}catch{/* An existing in-memory reader already owns its navigation listener. */}
  const stop=()=>window.removeEventListener('hashchange',observe);
  const observe=(event:HashChangeEvent)=>{try{const before=routeHash(event.oldURL),after=routeHash(event.newURL);if(events.length===8||before.length>8192||after.length>8192){snapshot=undefined;events.length=0;stop();return;}events.push({before,after});}catch{snapshot=undefined;stop();}};
  if(snapshot)window.addEventListener('hashchange',observe);
  void import('./practiceReadingOrigin').then(module=>{
   if(!live.current.active||live.current.owner!==owner)return;
   if(current)module.resumePracticeOrigin(owner,id&&number?{id,number}:undefined,href,expected);
   if(snapshot&&events.length)module.replayPracticeOrigin(owner,href,expected,origin,events,snapshot);
  }).catch(()=>{/* Canonical source links remain usable; the reader offers explicit recovery. */}).finally(stop);
  return()=>{current=false;live.current.active=false;queueMicrotask(()=>{if(!live.current.active||live.current.owner!==owner)stop();});};
 },[owner,id,number,href,expected]);
}
