import {useCallback, useEffect, useLayoutEffect, useRef, useState, type RefObject} from 'react';
import {ArrowLeft, ArrowRight} from 'lucide-react';
import {useI18n} from './i18n';
import './reading-tab-controls.css';

type Props = {
  container:RefObject<HTMLElement|null>;
  viewport:RefObject<HTMLElement|null>;
  content:RefObject<HTMLElement|null>;
  controlsId:string;
  selectedKey:string;
};
const initialEdges = {overflow:false, previous:false, next:false};

/** Enhance the existing tab DOM without owning its selection or editor state. */
export default function ReadingTabControls({container,viewport,content,controlsId,selectedKey}:Props) {
  const {t,locale}=useI18n();
  const controls=useRef<HTMLDivElement>(null);
  const last=useRef(initialEdges);
  const [edges,setEdges]=useState(initialEdges);
  const measure=useCallback((reveal=false)=>{
    const bar=container.current,scroll=viewport.current,list=content.current;
    if(!bar?.isConnected||!scroll?.isConnected||!list?.isConnected||!bar.clientWidth||!scroll.clientWidth)return;
    const selected=list.querySelector<HTMLElement>('[role="tab"][aria-selected="true"]');
    if(reveal&&selected){
      const bounds=scroll.getBoundingClientRect(),item=selected.getBoundingClientRect();
      const shift=item.width>bounds.width||item.left<bounds.left?item.left-bounds.left:item.right>bounds.right?item.right-bounds.right:0;
      if(shift)scroll.scrollLeft=Math.max(0,Math.min(scroll.scrollWidth-scroll.clientWidth,scroll.scrollLeft+shift));
    }
    const next={overflow:list.scrollWidth>bar.clientWidth+1,previous:scroll.scrollLeft>1,next:scroll.scrollLeft+scroll.clientWidth<scroll.scrollWidth-1};
    const previous=last.current;
    if(previous.overflow&&!next.overflow&&controls.current?.contains(document.activeElement))selected?.focus({preventScroll:true});
    if(previous.overflow!==next.overflow||previous.previous!==next.previous||previous.next!==next.next){last.current=next;setEdges(next);}
  },[container,viewport,content]);
  useLayoutEffect(()=>{measure(true);},[measure,selectedKey,locale,edges.overflow]);
  useEffect(()=>{
    const bar=container.current,scroll=viewport.current,list=content.current;
    if(!bar||!scroll||!list)return;
    let disposed=false;
    const resize=()=>{if(!disposed)measure(true);},moved=()=>{if(!disposed)measure();};
    scroll.addEventListener('scroll',moved,{passive:true});
    const observer=typeof ResizeObserver==='undefined'?undefined:new ResizeObserver(resize);
    if(observer){observer.observe(bar);observer.observe(scroll);observer.observe(list);}
    else window.addEventListener('resize',resize);
    resize();
    return()=>{disposed=true;observer?.disconnect();scroll.removeEventListener('scroll',moved);if(!observer)window.removeEventListener('resize',resize);};
  },[container,viewport,content,measure]);
  function move(direction:number){
    const scroll=viewport.current;
    if(!scroll||!(direction<0?last.current.previous:last.current.next))return;
    scroll.scrollLeft=Math.max(0,Math.min(scroll.scrollWidth-scroll.clientWidth,scroll.scrollLeft+direction*Math.max(120,scroll.clientWidth*.8)));
    measure();
  }
  if(!edges.overflow)return null;
  return <div ref={controls} className="reading-tab-controls" role="group" aria-label={t('题目选项导航','Problem tab navigation')}>
    <button type="button" aria-controls={controlsId} aria-label={t('向左查看题目选项','Scroll problem tabs left')} title={t('向左查看题目选项','Scroll problem tabs left')} aria-disabled={!edges.previous} onClick={()=>move(-1)}><ArrowLeft size={15} aria-hidden="true"/></button>
    <button type="button" aria-controls={controlsId} aria-label={t('向右查看题目选项','Scroll problem tabs right')} title={t('向右查看题目选项','Scroll problem tabs right')} aria-disabled={!edges.next} onClick={()=>move(1)}><ArrowRight size={15} aria-hidden="true"/></button>
  </div>;
}
