import {useEffect,useRef,type ReactNode} from 'react';
import {controlIsActive} from './activeControls';

/** Keep native board controls visible without moving focus or the reading page. */
export default function CommunityBoardNavigation({children,canAct}:{children:ReactNode;canAct:()=>boolean}){
 const root=useRef<HTMLDivElement>(null),latest=useRef(canAct);latest.current=canAct;
 useEffect(()=>{
  const node=root.current;if(!node||typeof window.requestAnimationFrame!=='function')return;
  const row=node;
  let frame:number|undefined,disposed=false,lastWidth=row.clientWidth;
  const buttons=()=>row.querySelector('.board-list');
  const selected=()=>buttons()?.querySelector<HTMLElement>(':scope > button[aria-pressed="true"]')||null;
  const own=(node:HTMLElement)=>node.parentElement===buttons()&&node.matches('button')&&controlIsActive(node)&&!node.matches(':disabled,[aria-disabled="true"]');
  function cancel(){if(frame!==undefined){window.cancelAnimationFrame(frame);frame=undefined;}}
  function reveal(target:HTMLElement|null,focused=false){
   cancel();if(!target)return;const href=window.location.href;
   frame=window.requestAnimationFrame(()=>{
    frame=undefined;
    if(disposed||window.location.href!==href||!latest.current()||!controlIsActive(row)||row.closest('details:not([open]),dialog:not([open])')||!own(target))return;
    if(focused?document.activeElement!==target:selected()!==target)return;
    const width=row.clientWidth,max=row.scrollWidth-width;if(width<=0||max<=0)return;
    const bounds=row.getBoundingClientRect(),item=target.getBoundingClientRect(),start=bounds.left+row.clientLeft+8,end=start+width-16;
    const delta=item.width>end-start||item.left<start?item.left-start:item.right>end?item.right-end:0;
    const next=Math.max(0,Math.min(max,row.scrollLeft+delta));
    if(Number.isFinite(next)&&next!==row.scrollLeft)row.scrollLeft=next;
   });
  }
  function refresh(){const target=document.activeElement;target instanceof HTMLElement&&own(target)?reveal(target,true):reveal(selected());}
  function resized(){const width=row.clientWidth;if(width!==lastWidth){lastWidth=width;refresh();}}
  function focus(event:FocusEvent){const target=event.target;if(target instanceof HTMLElement&&own(target))reveal(target,true);}
  // Yield queued measurements to a newer reading, typing or scrolling choice.
  const events=['pointerdown','touchstart','wheel','keydown','focusin'] as const;
  for(const event of events)window.addEventListener(event,cancel,{capture:true,passive:true});
  row.addEventListener('focusin',focus);window.addEventListener('resize',resized,{passive:true});
  const resize=typeof window.ResizeObserver==='undefined'?undefined:new window.ResizeObserver(resized);resize?.observe(row);
  const changes=new window.MutationObserver(refresh);
  changes.observe(row,{subtree:true,childList:true,characterData:true,attributes:true,attributeFilter:['aria-pressed','aria-hidden','hidden','inert','disabled','aria-disabled']});
  // Retained/native scroll restoration wins over a late enhancement mount.
  if(row.scrollLeft===0||row.contains(document.activeElement))refresh();
  return()=>{disposed=true;cancel();changes.disconnect();resize?.disconnect();row.removeEventListener('focusin',focus);window.removeEventListener('resize',resized);for(const event of events)window.removeEventListener(event,cancel,true);};
 },[]);
 return <div className="community-board-scroll" ref={root}>{children}</div>;
}
