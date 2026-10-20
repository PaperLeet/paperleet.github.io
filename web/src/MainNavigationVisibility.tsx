import {useEffect,useRef} from 'react';
import {controlIsActive} from './activeControls';

/** Progressive enhancement for native links; never focus or scroll the document. */
export default function MainNavigationVisibility(){
 const marker=useRef<HTMLSpanElement>(null);
 useEffect(()=>{
  const parent=marker.current?.parentElement;
  if(!parent||parent.tagName!=='NAV'||typeof window.requestAnimationFrame!=='function')return;
  const nav=parent;
  let frame:number|undefined,disposed=false,lastWidth=nav.clientWidth;
  const active=()=>nav.querySelector<HTMLElement>(':scope > [aria-current="page"],:scope > .active');
  function cancel(){if(frame!==undefined){window.cancelAnimationFrame(frame);frame=undefined;}}
  function reveal(target:HTMLElement|null,focused=false){
   cancel();if(!target)return;
   const href=window.location.href;
   frame=window.requestAnimationFrame(()=>{
    frame=undefined;
    if(disposed||window.location.href!==href||!controlIsActive(nav)||target.parentElement!==nav||!controlIsActive(target)||target.matches(':disabled,[aria-disabled="true"]'))return;
    if(focused?document.activeElement!==target:active()!==target)return;
    const width=nav.clientWidth,max=nav.scrollWidth-width;
    if(width<=0||max<=0)return;
    const bounds=nav.getBoundingClientRect(),item=target.getBoundingClientRect(),start=bounds.left+nav.clientLeft+8,end=start+width-16;
    const delta=item.width>end-start||item.left<start?item.left-start:item.right>end?item.right-end:0;
    const next=Math.max(0,Math.min(max,nav.scrollLeft+delta));
    if(next!==nav.scrollLeft)nav.scrollLeft=next;
   });
  }
  function refresh(){const focused=document.activeElement;focused instanceof HTMLElement&&focused.parentElement===nav&&focused.matches('a,button')?reveal(focused,true):reveal(active());}
  function resized(){const width=nav.clientWidth;if(width!==lastWidth){lastWidth=width;refresh();}}
  function focus(event:FocusEvent){const target=event.target;if(target instanceof HTMLElement&&target.parentElement===nav&&target.matches('a,button'))reveal(target,true);}
  // A newer wheel, pointer, key or focus choice retires pending measurements.
  const events=['pointerdown','touchstart','wheel','keydown','focusin'] as const;
  for(const event of events)window.addEventListener(event,cancel,{capture:true,passive:true});
  nav.addEventListener('focusin',focus);
  window.addEventListener('resize',resized,{passive:true});
  const resize=typeof window.ResizeObserver==='undefined'?undefined:new window.ResizeObserver(resized);
  resize?.observe(nav);
  const changes=new window.MutationObserver(refresh);
  changes.observe(nav,{subtree:true,childList:true,attributes:true,attributeFilter:['aria-current','aria-hidden','class','hidden','inert']});
  // A slow enhancement import must preserve an already scrolled/restored row.
  if(nav.scrollLeft===0||nav.contains(document.activeElement))refresh();
  return()=>{disposed=true;cancel();changes.disconnect();resize?.disconnect();nav.removeEventListener('focusin',focus);window.removeEventListener('resize',resized);for(const event of events)window.removeEventListener(event,cancel,true);};
 },[]);
 return <span ref={marker} hidden aria-hidden="true"/>;
}
