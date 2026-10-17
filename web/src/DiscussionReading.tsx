import {useEffect,useId,useLayoutEffect,useMemo,useRef,useState} from 'react';
import {ChevronDown,ListTree} from 'lucide-react';
import CommunityMarkdown from './CommunityMarkdown';
import {markdownOutline} from './markdownOutline';
import {readingScrollRoot,readingToolbar,readingToolbarInset,revealReadingSection} from './readingViewport';
import {useI18n} from './i18n';
import './discussion-reading.css';

export default function DiscussionReading({body,postId,compact=false,instance=false}:{body:string;postId:string;compact?:boolean;instance?:boolean}){
  const {t}=useI18n(),root=useRef<HTMLDivElement>(null),instanceId=useId();
  const suffix=instance?'-instance-'+instanceId.split('').map(character=>character.charCodeAt(0).toString(16).padStart(4,'0')).join(''):'';
  const scope=`reading-${postId}${suffix}`,sections=useMemo(()=>markdownOutline(body,scope),[body,scope]);
  const [open,setOpen]=useState(()=>!compact&&typeof window!=='undefined'&&!!window.matchMedia?.('(min-width: 1200px)').matches);
  const [active,setActive]=useState(''),[jump,setJump]=useState<{id:string;opener:Element|null;body:string;postId:string;href:string}>();
  const minimum=Math.min(...sections.map(section=>section.depth));
  useEffect(()=>{
    if(!open||sections.length<2)return;
    const headings=sections.map(section=>document.getElementById(section.id)).filter((node):node is HTMLElement=>!!node&&!!root.current?.contains(node));
    let pane=root.current?readingScrollRoot(root.current):null,viewport:HTMLElement|Window=pane||window;
    let frame:number|undefined,live=true;
    const measure=()=>{
      if(!live)return;
      frame=undefined;
      const next=root.current?readingScrollRoot(root.current):null;
      if(next!==pane){
        viewport.removeEventListener('scroll',schedule);if(pane)observer?.unobserve?.(pane);
        pane=next;viewport=pane||window;viewport.addEventListener('scroll',schedule,{passive:true});if(pane)observer?.observe(pane);
      }
      let left=0,right=headings.length-1,index=0;
      const inset=root.current?readingToolbarInset(root.current):24;
      const edge=pane?pane.getBoundingClientRect().top+pane.clientTop+inset:root.current&&readingToolbar(root.current)?inset:110;
      while(left<=right){const middle=Math.floor((left+right)/2);if(headings[middle].getBoundingClientRect().top<=edge){index=middle;left=middle+1;}else right=middle-1;}
      setActive(headings[index]?.id||'');
    };
    const schedule=()=>{if(live&&frame===undefined)frame=window.requestAnimationFrame(measure);};
    const observer=typeof ResizeObserver==='undefined'?undefined:new ResizeObserver(schedule);
    if(root.current)observer?.observe(root.current);if(pane)observer?.observe(pane);
    const toolbar=root.current?readingToolbar(root.current):null;if(toolbar)observer?.observe(toolbar);
    measure();viewport.addEventListener('scroll',schedule,{passive:true});window.addEventListener('resize',schedule);
    return()=>{live=false;viewport.removeEventListener('scroll',schedule);window.removeEventListener('resize',schedule);observer?.disconnect();if(frame!==undefined)window.cancelAnimationFrame(frame);};
  },[sections,open]);
  useLayoutEffect(()=>{
    if(!jump)return;setJump(undefined);
    if(jump.body!==body||jump.postId!==postId||jump.href!==location.href)return;
    if(document.activeElement!==jump.opener&&!(document.activeElement===document.body&&!jump.opener?.isConnected))return;
    const target=document.getElementById(jump.id);
    if(!target||!root.current?.contains(target))return;
    target.focus({preventScroll:true});revealReadingSection(target);setActive(jump.id);
  },[jump,body,postId]);
  return <div ref={root} className={`discussion-reading${sections.length>1?' has-outline':''}${compact?' is-compact':''}`}>
    {sections.length>1&&<details className="discussion-outline" open={open} onToggle={event=>setOpen(event.currentTarget.open)}>
      <summary><ListTree size={16} aria-hidden="true"/><strong>{t('文章目录','On this page')}</strong><small aria-label={t(`${sections.length} 个小节`,`${sections.length} sections`)}>{sections.length}</small><ChevronDown size={14} aria-hidden="true"/></summary>
      {open&&<nav aria-label={t('文章目录','On this page')}><ol>{sections.map(section=><li key={section.id} style={{paddingInlineStart:Math.min(section.depth-minimum,3)*10}}>
        <button type="button" aria-current={active===section.id?'location':undefined} onClick={()=>{
          setJump({id:section.id,opener:document.activeElement,body,postId,href:location.href});
          if(compact||!window.matchMedia?.('(min-width: 1200px)').matches)setOpen(false);
        }}>{section.label}</button>
      </li>)}</ol></nav>}
    </details>}
    <div className="discussion-reading-body"><CommunityMarkdown body={body} headingScope={scope}/></div>
  </div>;
}
