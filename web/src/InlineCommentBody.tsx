import {memo,useId,useLayoutEffect,useMemo,useRef,useState} from 'react';
import {ChevronDown,ChevronUp} from 'lucide-react';
import CommunityMarkdown from './CommunityMarkdown';
import {PostPreviewContent} from './PostPreview';
import {postPreview} from './postExcerpt';
import {useI18n} from './i18n';

const InlineCommentBody=memo(function InlineCommentBody({body}:{body:string}){
 const {t,locale}=useI18n(),regionId=useId();
 const summary=useMemo(()=>postPreview(body,280,locale==='en'?'[formula]':'[公式]'),[body,locale]);
 const expandable=summary.truncated||summary.rich||summary.unavailable||!summary.text;
 const [openedBody,setOpenedBody]=useState<string>();
 const open=!!expandable&&openedBody===body;
 const toggle=useRef<HTMLButtonElement>(null);
 const readingIntent=useRef<{opener:HTMLButtonElement;href:string;body:string}|undefined>(undefined);
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
 function change(value:boolean,opener:HTMLButtonElement){
  readingIntent.current={opener,href:location.href,body};
  setOpenedBody(value?body:undefined);
 }
 return <div className={'inline-comment-body'+(open?' is-open':!expandable?' is-complete':'')}>
  {expandable&&<div className="inline-comment-controls">
   <button ref={toggle} className="inline-comment-toggle" type="button" aria-expanded={open} aria-controls={regionId} onClick={event=>change(!open,event.currentTarget)}>
    {open?<ChevronUp size={15} aria-hidden="true"/>:<ChevronDown size={15} aria-hidden="true"/>}{open?t('收起回复','Show less'):t('展开回复','Read more')}
   </button>
   {open&&<span>{t('完整回复','Full reply')}</span>}
  </div>}
  <div className="inline-comment-content" id={regionId} role={open?'region':undefined} aria-label={open?t('完整回复','Full reply'):undefined}>
   {open?<CommunityMarkdown body={body}/>:<PostPreviewContent summary={summary} reply/>}
  </div>
  {open&&<button className="inline-comment-collapse" type="button" aria-controls={regionId} aria-expanded onClick={event=>change(false,event.currentTarget)}><ChevronUp size={15} aria-hidden="true"/>{t('收起回复','Show less')}</button>}
 </div>;
});
export default InlineCommentBody;
