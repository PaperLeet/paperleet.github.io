import {controlIsActive} from './activeControls';
import {useEffect,useId,useLayoutEffect,useMemo,useRef,useState} from 'react';
import {ChevronDown,ListTree,X} from 'lucide-react';
import CommunityMarkdown from './CommunityMarkdown';
import ReadingAppearance from './ReadingAppearance';
import {markdownOutline} from './markdownOutline';
import {keepReadingTriggerVisible,readingScrollRoot,readingToolbar,readingToolbarInset,revealReadingSection} from './readingViewport';
import {useI18n} from './i18n';
import {filterReadingSections,headingFilterDraft} from './readingSectionSearch';
import {postSearchQuery} from './postSearchModel';
import PostSearchMatch from './PostSearchMatch';
import './discussion-reading.css';

export default function DiscussionReading({body,displayBody=body,postId,compact=false,instance=false,canAct}:{body:string;displayBody?:string;postId:string;compact?:boolean;instance?:boolean;canAct?:()=>boolean}){
  const {t}=useI18n(),root=useRef<HTMLDivElement>(null),bodyRoot=useRef<HTMLDivElement>(null),instanceId=useId();
  const suffix=instance?'-instance-'+instanceId.split('').map(character=>character.charCodeAt(0).toString(16).padStart(4,'0')).join(''):'';
  const scope=`reading-${postId}${suffix}`,sections=useMemo(()=>{const source=markdownOutline(body,scope);if(body===displayBody)return source;const display=markdownOutline(displayBody,scope);return source.map((section,index)=>({...section,label:display[index]?.label||section.label}));},[body,displayBody,scope]);
  const [open,setOpen]=useState(()=>!compact&&typeof window!=='undefined'&&!!window.matchMedia?.('(min-width: 1200px)').matches);
  const [active,setActive]=useState(''),[jump,setJump]=useState<{id:string;opener:Element|null;body:string;postId:string;href:string}>();
  const minimum=Math.min(...sections.map(section=>section.depth));
  const summary=useRef<HTMLElement>(null),filterInput=useRef<HTMLInputElement>(null),composing=useRef(false);
  const closing=useRef<{opener:HTMLElement;href:string;body:string;scope:string}|undefined>(undefined);
  const [filter,setFilter]=useState<{body:string;scope:string;draft:string;query:string}>();
  const current=filter?.body===body&&filter.scope===scope?filter:undefined,draft=current?.draft||'',query=current?.query||'';
  const filtered=useMemo(()=>filterReadingSections(sections,query),[sections,query]);
  const searchable=sections.length>=8;
  useLayoutEffect(()=>{composing.current=false;setFilter(undefined);closing.current=undefined;},[body,scope]);
  function changeFilter(value:string,transient=false) {
    const next=transient?value:headingFilterDraft(value);
    setFilter({body,scope,draft:next,query:transient?query:postSearchQuery(next)});
  }
  function clearFilter(){composing.current=false;changeFilter('');filterInput.current?.focus({preventScroll:true});}
  useLayoutEffect(()=>{
    const pending=closing.current;if(open||!pending)return;closing.current=undefined;
    if(pending.body!==body||pending.scope!==scope||pending.href!==location.href)return;
    const active=document.activeElement;
    if(active!==pending.opener&&!(active===document.body&&!pending.opener.isConnected))return;
    if(summary.current){summary.current.focus({preventScroll:true});keepReadingTriggerVisible(summary.current);}
  },[open,body,scope]);
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
  function moveHeading(event:import('react').KeyboardEvent<HTMLElement>){
    if(event.defaultPrevented||composing.current||event.nativeEvent.isComposing||event.nativeEvent.keyCode===229||event.altKey||event.ctrlKey||event.metaKey||event.shiftKey||!['ArrowUp','ArrowDown','Home','End'].includes(event.key))return;
    const nav=event.currentTarget,current=nav.ownerDocument.activeElement;
    if(!controlIsActive(nav)||nav.closest('details:not([open])')||event.target!==current)return;
    const buttons=Array.from(nav.querySelectorAll<HTMLButtonElement>('button')).filter(button=>!button.disabled&&controlIsActive(button)),index=buttons.findIndex(button=>button===current);
    if(index<0)return;
    const next=event.key==='Home'?0:event.key==='End'?buttons.length-1:Math.max(0,Math.min(buttons.length-1,index+(event.key==='ArrowDown'?1:-1))),target=buttons[next];
    event.preventDefault();event.stopPropagation();target.focus({preventScroll:true});
    // Browsing the directory scrolls only its own list; article navigation waits for activation.
    if(!nav.clientHeight||!controlIsActive(nav)||nav.ownerDocument.activeElement!==target)return;
    const edge=nav.getBoundingClientRect().top+nav.clientTop,box=target.getBoundingClientRect();
    const delta=box.top<edge?box.top-edge:box.bottom>edge+nav.clientHeight?box.bottom-edge-nav.clientHeight:0;
    if(delta)nav.scrollTop+=delta;
  }
  return <div ref={root} className={`discussion-reading${sections.length>1?' has-outline':''}${compact?' is-compact':''}`}>
   <div className="discussion-reading-layout">
    {sections.length>1&&<details className="discussion-outline" open={open} onToggle={event=>setOpen(event.currentTarget.open)} onKeyDown={event=>{
      if(!open||event.key!=='Escape'||event.defaultPrevented||event.repeat||event.nativeEvent.isComposing||event.nativeEvent.keyCode===229||event.altKey||event.ctrlKey||event.metaKey||event.shiftKey||composing.current)return;
      const active=document.activeElement;if(!(active instanceof HTMLElement)||!event.currentTarget.contains(active))return;
      event.preventDefault();event.stopPropagation();if(draft){clearFilter();return;}
      closing.current={opener:active,href:location.href,body,scope};setOpen(false);
    }}>
      <summary ref={summary} aria-controls={open?`${instanceId}-outline`:undefined} aria-keyshortcuts={open?'Escape':undefined}><ListTree size={16} aria-hidden="true"/><strong>{t('文章目录','On this page')}</strong><small aria-label={t(`${sections.length} 个小节`,`${sections.length} sections`)}>{sections.length}</small><ChevronDown size={14} aria-hidden="true"/></summary>
      {open&&searchable&&<div className="discussion-outline-tools">
        <form role="search" className="discussion-outline-filter" aria-label={t('筛选文章目录','Filter article headings')} onSubmit={event=>event.preventDefault()}>
          <span className="sr-only" id={`${instanceId}-filter-hint`}>{t('仅筛选小节标题，正文保持完整。最多200个字符；Esc先清空筛选，再收起目录。','Filter headings only; the full article remains visible. Up to 200 characters; Escape clears the filter, then closes the directory.')}</span>
          <input ref={filterInput} type="search" value={draft} maxLength={400} aria-label={t('筛选文章目录','Filter article headings')} placeholder={t('筛选小节标题…','Filter headings…')} aria-describedby={`${instanceId}-filter-hint ${instanceId}-filter-count`}
            onChange={event=>changeFilter(event.target.value,composing.current||(event.nativeEvent as InputEvent).isComposing===true)}
            onCompositionStart={()=>{composing.current=true;}} onCompositionEnd={event=>{composing.current=false;changeFilter(event.currentTarget.value);}}
            onKeyDown={event=>{if(event.key==='Escape'&&!composing.current&&!event.nativeEvent.isComposing&&event.nativeEvent.keyCode!==229&&!event.repeat&&!event.altKey&&!event.ctrlKey&&!event.metaKey&&!event.shiftKey&&draft){event.preventDefault();event.stopPropagation();clearFilter();}}}/>
          {!!draft&&<button type="button" onClick={clearFilter} aria-label={t('清空目录筛选','Clear heading filter')}><X size={15} aria-hidden="true"/></button>}
        </form>
        <p id={`${instanceId}-filter-count`} role="status">{query?t(`${filtered.matched} / ${sections.length} 个匹配小节`,`${filtered.matched} of ${sections.length} headings match`):t(`${sections.length} 个小节`,`${sections.length} headings`)}</p>
      </div>}
      {open&&query&&filtered.rows.length===0&&<div className="discussion-outline-empty"><p>{t('没有匹配的小节标题。','No headings match.')}</p><button type="button" onClick={clearFilter}>{t('查看全部小节','Show all headings')}</button></div>}
      {open&&<span id={`${instanceId}-outline-keys`} className="sr-only">{t('目录中用上下方向键、Home和End移动焦点；Enter跳转到正文。','Use Arrow Up, Arrow Down, Home and End to browse headings; Enter opens the section.')}</span>}
      {open&&<nav id={`${instanceId}-outline`} aria-label={t('文章目录','On this page')} aria-describedby={`${instanceId}-outline-keys`} onKeyDown={moveHeading}><ol>{filtered.rows.map(({section,match})=><li key={section.id} className={query&&!match?'is-context':undefined} style={{paddingInlineStart:Math.min(section.depth-minimum,3)*10}}>
        <button type="button" aria-current={active===section.id?'location':undefined} onClick={()=>{
          setJump({id:section.id,opener:document.activeElement,body,postId,href:location.href});
          if(compact||!window.matchMedia?.('(min-width: 1200px)').matches)setOpen(false);
        }}><PostSearchMatch text={section.label} query={query}/></button>
      </li>)}</ol></nav>}
    </details>}
    <div ref={bodyRoot} id={`${instanceId}-body`} className="discussion-reading-body">{(!compact||displayBody.length>=1200)&&<ReadingAppearance canAct={canAct} target={bodyRoot} controlsId={`${instanceId}-body`} scope={JSON.stringify([scope,body,displayBody])}/>}<CommunityMarkdown body={body} displayBody={displayBody} headingScope={scope}/></div>
   </div>
  </div>;
}
