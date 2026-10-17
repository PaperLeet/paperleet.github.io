import {useId,useLayoutEffect,useMemo,useRef,useState} from 'react';
import {ArrowUp,ArrowUpRight,ChevronDown,Code2} from 'lucide-react';
import {useI18n} from './i18n';
import DiscussionReading from './DiscussionReading';
import {FeatureBoundary} from './deferredFeature';
import {postPreview} from './postExcerpt';
import {PostPreviewContent} from './PostPreview';
import {communityPostHref} from './communityNavigation';
import {keepReadingTriggerVisible} from './readingViewport';
import './solution-reading-card.css';

/** Local disclosure leaves both the practice editor and card replies mounted. */
export default function SolutionReadingCard({id,title,body,returnTo,onCode}:{id:string;title:string;body:string;returnTo?:string;onCode?:()=>void}) {
  const {t,locale}=useI18n(),regionId=useId();
  const [expandedBody,setExpandedBody]=useState<string>();
  const open=expandedBody===body;
  const summary=useRef<HTMLElement>(null);
  const intent=useRef<{opener:HTMLElement;href:string;body:string}|undefined>(undefined);
  const preview=useMemo(()=>postPreview(body,170,locale==='en'?'[formula]':'[公式]'),[body,locale]);
  function collapse(opener:HTMLElement){intent.current={opener,href:location.href,body};setExpandedBody(undefined);}
  useLayoutEffect(()=>{
    const pending=intent.current;if(open||!pending)return;intent.current=undefined;
    if(pending.href!==location.href||pending.body!==body)return;
    const active=document.activeElement;
    if(active!==pending.opener&&!(active===document.body&&!pending.opener.isConnected))return;
    const target=summary.current;if(!target)return;
    target.focus({preventScroll:true});keepReadingTriggerVisible(target);
  },[open,body]);
  return <div className="solution-reading-card">
    <details open={open} onToggle={event=>setExpandedBody(event.currentTarget.open?body:undefined)} onKeyDown={event=>{
      if(!open||event.key!=='Escape'||event.defaultPrevented||event.repeat||event.altKey||event.ctrlKey||event.metaKey||event.shiftKey||event.nativeEvent.isComposing||event.nativeEvent.keyCode===229)return;
      const active=document.activeElement;
      if(!(active instanceof HTMLElement)||!event.currentTarget.contains(active)||active.closest('input,textarea,select,[contenteditable]:not([contenteditable="false"])'))return;
      event.preventDefault();event.stopPropagation();collapse(active);
    }}>
      <summary ref={summary} aria-expanded={open} aria-controls={regionId} aria-keyshortcuts={open?'Escape':undefined}>
        <span className="post-title">{title}</span><ChevronDown size={17} aria-hidden="true"/>
      </summary>
      <div id={regionId} className="solution-reading-content" role="region" aria-label={t('题解全文','Full solution')}>
        {open&&<>
          {onCode&&<div className="solution-reading-toolbar" role="group" aria-label={t('题解阅读操作','Solution reading actions')}>
            <button type="button" className="solution-reading-collapse" onClick={event=>collapse(event.currentTarget)}><ArrowUp size={15} aria-hidden="true"/>{t('收起题解','Collapse solution')}</button>
            <button type="button" className="solution-reading-code" onClick={onCode}><Code2 size={15} aria-hidden="true"/>{t('返回代码','Back to code')}</button>
          </div>}
          <FeatureBoundary fallback={<><p role="status">{t('格式暂时无法显示，以下是原始题解。收起后可重新展开。','Formatting is unavailable. The original solution is below; collapse and reopen to retry.')}</p><pre className="inline-reading-plain">{body}</pre></>}>
            <DiscussionReading body={body} postId={id} compact/>
          </FeatureBoundary>
          {!onCode&&<div className="solution-reading-footer"><button type="button" className="solution-reading-collapse" onClick={event=>collapse(event.currentTarget)}><ArrowUp size={15} aria-hidden="true"/>{t('收起题解','Collapse solution')}</button></div>}
        </>}
      </div>
    </details>
    {!open&&<PostPreviewContent summary={preview} inlinePost/>}
    <a data-reading-link="read" className="solution-reading-page" href={communityPostHref(id,returnTo)}>{t('在独立页面打开','Open full page')}<ArrowUpRight size={14} aria-hidden="true"/></a>
  </div>;
}
