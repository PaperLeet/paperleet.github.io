import {useCallback,useId,useLayoutEffect,useRef,useState,type ReactNode} from 'react';
import {ChevronLeft,ChevronRight} from 'lucide-react';
import {useI18n} from './i18n';
import './community-table.css';

type Edges={tools:boolean;overflow:boolean;before:boolean;after:boolean};
const empty:Edges={tools:false,overflow:false,before:false,after:false};

/** Native table semantics inside a local, keyboard-reachable horizontal viewport. */
export default function CommunityTable({children}:{children?:ReactNode}){
 const {t}=useI18n(),id=useId(),viewport=useRef<HTMLDivElement>(null),table=useRef<HTMLTableElement>(null);
 const [edges,setEdges]=useState<Edges>(empty);
 const measure=useCallback(()=>{
  const node=viewport.current;if(!node)return;
  const max=Math.max(0,node.scrollWidth-node.clientWidth),overflow=node.clientWidth>0&&max>1;
  setEdges(current=>{
   const next={tools:current.tools||overflow,overflow,before:overflow&&node.scrollLeft>1,after:overflow&&node.scrollLeft<max-1};
   return current.tools===next.tools&&current.overflow===next.overflow&&current.before===next.before&&current.after===next.after?current:next;
  });
 },[]);
 useLayoutEffect(()=>{
  const node=viewport.current,content=table.current;if(!node||!content)return;
  let frame:number|undefined,live=true;
  const schedule=()=>{if(live&&frame===undefined)frame=window.requestAnimationFrame(()=>{frame=undefined;if(live)measure();});};
  const observer=typeof ResizeObserver==='undefined'?undefined:new ResizeObserver(schedule);
  observer?.observe(node);observer?.observe(content);window.addEventListener('resize',schedule);measure();
  return()=>{live=false;observer?.disconnect();window.removeEventListener('resize',schedule);if(frame!==undefined)window.cancelAnimationFrame(frame);};
 },[measure]);
 // Content/translation can change even when the viewport itself has not resized.
 useLayoutEffect(measure,[children,measure]);
 function move(direction:-1|1){
  if(direction<0?!edges.before:!edges.after)return;
  const node=viewport.current;if(!node||node.clientWidth<=0)return;
  const max=Math.max(0,node.scrollWidth-node.clientWidth);if(max<=1)return;
  const next=Math.max(0,Math.min(max,node.scrollLeft+direction*Math.max(120,Math.round(node.clientWidth*.75))));
  if(next===node.scrollLeft)return;
  node.scrollTo({left:next,behavior:window.matchMedia?.('(prefers-reduced-motion: reduce)').matches?'instant':'smooth'});
 }
 return <div className="community-table-reader" data-scroll-before={edges.before||undefined} data-scroll-after={edges.after||undefined}>
  {edges.tools&&<div className="community-table-toolbar"><p id={id+'-hint'}>{edges.overflow?t('表格可横向浏览','Scroll to explore the table'):t('全部表格列已显示','All table columns are visible')}{edges.overflow&&<span>{t('也可聚焦表格后使用方向键。','You can also focus the table and use arrow keys.')}</span>}</p><div role="group" aria-label={t('横向浏览表格','Browse table columns')}>
   <button type="button" aria-label={t('查看左侧表格列','Show columns to the left')} aria-controls={id} aria-disabled={!edges.before} onClick={()=>move(-1)}><ChevronLeft size={17} aria-hidden="true"/></button>
   <button type="button" aria-label={t('查看右侧表格列','Show columns to the right')} aria-controls={id} aria-disabled={!edges.after} onClick={()=>move(1)}><ChevronRight size={17} aria-hidden="true"/></button>
  </div></div>}
  <div ref={viewport} id={id} className="community-table-scroll" role={edges.tools?'region':undefined} aria-label={edges.tools?t('表格内容','Table contents'):undefined} aria-describedby={edges.tools?id+'-hint':undefined} tabIndex={edges.tools?0:undefined} onScroll={measure}><table ref={table}>{children}</table></div>
 </div>;
}
