import {useEffect} from 'react';

export function usePracticeReadingOrigin(owner:string|undefined,problem:{id:string;number:number}|undefined,href:string|undefined,route?:string){
 const id=problem?.id,number=problem?.number,expected=route??(typeof location==='undefined'?'':location.hash);
 useEffect(()=>{
  if(!owner||!id||!number||!href)return;
  let current=true;
  void import('./practiceReadingOrigin').then(module=>{if(current)module.resumePracticeOrigin(owner,{id,number},href,expected);}).catch(()=>{/* Canonical source links remain usable; the reader offers explicit recovery. */});
  return()=>{current=false;};
 },[owner,id,number,href,expected]);
}
