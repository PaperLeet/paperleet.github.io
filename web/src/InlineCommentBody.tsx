import {memo,useId,useLayoutEffect,useMemo,useRef,useState} from 'react';
import {ChevronDown,ChevronUp} from 'lucide-react';
import CommunityMarkdown from './CommunityMarkdown';
import DiscussionReading from './DiscussionReading';
import {PostPreviewContent} from './PostPreview';
import {postPreview} from './postExcerpt';
import {useI18n} from './i18n';
import {FeatureBoundary} from './deferredFeature';
import {controlIsActive} from './activeControls';
import {useWorkspaceIntent} from './useWorkspaceIntent';
import {keepReadingTriggerVisible} from './readingViewport';
import './community-card-actions.css';

type Props={body:string;displayBody?:string;variant?:'reply'|'post';limit?:number;postId?:string;query?:string;canAct?:()=>boolean};
const InlineCommentBody=memo(function InlineCommentBody(props:Props){
 return <InlineReader key={JSON.stringify([props.variant||'reply',props.postId,props.body])} {...props}/>;
});
function InlineReader({body,displayBody=body,variant='reply',limit=280,postId,query,canAct}:Props){
 const {t,locale}=useI18n(),regionId=useId();
 const post=variant==='post',fullLabel=post?t('完整正文','Full post'):t('完整回复','Full reply');
 const collapseLabel=post?t('收起全文','Show less'):t('收起回复','Show less');
 const summary=useMemo(()=>postPreview(displayBody,limit,locale==='en'?'[formula]':'[公式]',query),[displayBody,limit,locale,query]);
 const expandable=summary.truncated||summary.rich||summary.unavailable||!summary.text;
 const [openedBody,setOpenedBody]=useState<string>();
 const open=!!expandable&&openedBody===body;
 const toggle=useRef<HTMLButtonElement>(null),root=useRef<HTMLDivElement>(null),intent=useWorkspaceIntent(displayBody);
 const href=typeof window==='undefined'?'':window.location.href;
 const current=()=>intent.isCurrent()&&controlIsActive(root.current)&&!root.current?.closest('details:not([open]),dialog:not([open])')&&location.href===href&&canAct?.()!==false;
 const readingIntent=useRef<{opener:HTMLElement;href:string;body:string;claim:()=>boolean}|undefined>(undefined);
 useLayoutEffect(()=>{
  const pending=readingIntent.current;if(!pending)return;readingIntent.current=undefined;
  const control=toggle.current,active=document.activeElement;
  if(active!==pending.opener&&!(active===document.body&&!pending.opener.isConnected)){intent.cancel();return;}
  if(pending.body!==body||pending.href!==location.href||!current()||!controlIsActive(control)||!pending.claim()){intent.cancel();return;}
  if(!control)return;
  control.focus({preventScroll:true});
  // Keep the compact control reachable when a long body leaves the document.
  keepReadingTriggerVisible(control);
 },[open,body]);
 function change(value:boolean,opener:HTMLElement){
  if(!current()||!root.current?.contains(opener)||!controlIsActive(opener))return;
  readingIntent.current={opener,href:location.href,body,claim:intent.create()};
  setOpenedBody(value?body:undefined);
 }
 return <div ref={root} className={'inline-comment-body'+(post?' inline-post-body':'')+(open?' is-open':!expandable?' is-complete':'')} onKeyDown={event=>{
  if(!current()||!open||event.key!=='Escape'||event.defaultPrevented||event.repeat||event.nativeEvent.isComposing||event.nativeEvent.keyCode===229||event.altKey||event.ctrlKey||event.metaKey||event.shiftKey)return;
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
   {open?<FeatureBoundary fallback={<><p className="inline-reading-error" role="status">{t('暂时无法显示格式，以下为原始内容。可收起后重新展开。','Formatting is unavailable. The original text is shown below; collapse and reopen to retry.')}</p><pre className="inline-reading-plain">{body}</pre></>}>{post&&postId?<DiscussionReading body={body} displayBody={displayBody} postId={postId} compact instance canAct={current}/>:<CommunityMarkdown body={body} displayBody={displayBody}/>}</FeatureBoundary>:<PostPreviewContent summary={summary} reply={!post} inlinePost={post} query={query}/>}
  </div>
  {open&&<button className="inline-comment-collapse" type="button" aria-controls={regionId} aria-expanded onClick={event=>change(false,event.currentTarget)}><ChevronUp size={15} aria-hidden="true"/>{collapseLabel}</button>}
 </div>;
}
export function InlinePostBody({body,displayBody,limit=210,postId,query,canAct}:{body:string;displayBody?:string;limit?:number;postId?:string;query?:string;canAct?:()=>boolean}){
 return <InlineCommentBody body={body} displayBody={displayBody} limit={limit} postId={postId} variant="post" query={query} canAct={canAct}/>;
}
export default InlineCommentBody;
