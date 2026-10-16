import {memo,useId,useLayoutEffect,useMemo,useRef,useState} from 'react';
import {ChevronDown,ChevronUp} from 'lucide-react';
import CommunityMarkdown from './CommunityMarkdown';
import {PostPreviewContent} from './PostPreview';
import {postPreview} from './postExcerpt';
import {useI18n} from './i18n';
import {FeatureBoundary} from './deferredFeature';
import './community-card-actions.css';

const InlineCommentBody=memo(function InlineCommentBody({body,variant='reply',limit=280}:{body:string;variant?:'reply'|'post';limit?:number}){
 const {t,locale}=useI18n(),regionId=useId();
 const post=variant==='post',fullLabel=post?t('完整正文','Full post'):t('完整回复','Full reply');
 const collapseLabel=post?t('收起全文','Show less'):t('收起回复','Show less');
 const summary=useMemo(()=>postPreview(body,limit,locale==='en'?'[formula]':'[公式]'),[body,limit,locale]);
 const expandable=summary.truncated||summary.rich||summary.unavailable||!summary.text;
 const [openedBody,setOpenedBody]=useState<string>();
 const open=!!expandable&&openedBody===body;
 const toggle=useRef<HTMLButtonElement>(null);
 const readingIntent=useRef<{opener:HTMLElement;href:string;body:string}|undefined>(undefined);
 useLayoutEffect(()=>{
  const intent=readingIntent.current;if(!intent)return;readingIntent.current=undefined;
  if(intent.body!==body||intent.href!==location.href)return;
  const active=document.activeElement;
  if(active!==intent.opener&&!(active===document.body&&!intent.opener.isConnected))return;
  const control=toggle.current;if(!control)return;
  control.focus({preventScroll:true});
  // Keep the compact control reachable when a long body leaves the document.
  const bounds=control.getBoundingClientRect();
  if(bounds.top<0||bounds.bottom>window.innerHeight)control.scrollIntoView({block:'nearest',behavior:'instant'});
 },[open,body]);
 function change(value:boolean,opener:HTMLElement){
  readingIntent.current={opener,href:location.href,body};
  setOpenedBody(value?body:undefined);
 }
 return <div className={'inline-comment-body'+(post?' inline-post-body':'')+(open?' is-open':!expandable?' is-complete':'')} onKeyDown={event=>{
  if(!open||event.key!=='Escape'||event.defaultPrevented||event.repeat||event.nativeEvent.isComposing||event.nativeEvent.keyCode===229||event.altKey||event.ctrlKey||event.metaKey||event.shiftKey)return;
  const active=document.activeElement;
  if(!(active instanceof HTMLElement)||!event.currentTarget.contains(active)||active.closest('input,textarea,select,[contenteditable]:not([contenteditable="false"])'))return;
  event.preventDefault();event.stopPropagation();change(false,active);
 }}>
  {expandable&&<div className="inline-comment-controls">
   <button ref={toggle} className="inline-comment-toggle" type="button" aria-expanded={open} aria-controls={regionId} aria-keyshortcuts={open?'Escape':undefined} onClick={event=>change(!open,event.currentTarget)}>
    {open?<ChevronUp size={15} aria-hidden="true"/>:<ChevronDown size={15} aria-hidden="true"/>}{open?collapseLabel:post?t('展开全文','Read full post'):t('展开回复','Read more')}
   </button>
   {open&&<><span>{fullLabel}</span><kbd className="inline-reading-keyhint" aria-hidden="true" title={t('Esc 收起当前阅读内容','Press Escape to collapse this reader')}>Esc</kbd></>}
  </div>}
  <div className="inline-comment-content" id={regionId} tabIndex={open?0:-1} role={open?'region':undefined} aria-label={open?fullLabel:undefined} aria-keyshortcuts={open?'Escape':undefined}>
   {open?<FeatureBoundary fallback={<><p className="inline-reading-error" role="status">{t('暂时无法显示格式，以下为原始内容。可收起后重新展开。','Formatting is unavailable. The original text is shown below; collapse and reopen to retry.')}</p><pre className="inline-reading-plain">{body}</pre></>}><CommunityMarkdown body={body}/></FeatureBoundary>:<PostPreviewContent summary={summary} reply={!post} inlinePost={post}/>}
  </div>
  {open&&<button className="inline-comment-collapse" type="button" aria-controls={regionId} aria-expanded onClick={event=>change(false,event.currentTarget)}><ChevronUp size={15} aria-hidden="true"/>{collapseLabel}</button>}
 </div>;
});
export function InlinePostBody({body,limit=210}:{body:string;limit?:number}){
 return <InlineCommentBody body={body} limit={limit} variant="post"/>;
}
export default InlineCommentBody;
