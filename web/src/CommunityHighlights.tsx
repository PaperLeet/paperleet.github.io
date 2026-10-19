import {useEffect,useLayoutEffect,useRef,useState} from 'react';
import {Sparkles,ChevronDown,RefreshCw,ArrowUpRight} from 'lucide-react';
import AppLink from './AppLink';
import CommunityAuthor from './CommunityAuthor';
import HighlightContext from './HighlightContext';
import {socialApi,errorMessage} from './social';
import {useI18n} from './i18n';
import {communityPostHref} from './communityNavigation';
import {postPreview} from './postExcerpt';
import {readHighlights,type HighlightList} from './communityHighlightsModel';
import './community-highlights.css';

export default function CommunityHighlights({category,returnTo,resume,onResumeReady,resumedPost}:{category:string;returnTo:string;resume?:{id:number;post:string};onResumeReady?:(id?:number)=>void;resumedPost?:string}){
 const {t}=useI18n(),[retry,setRetry]=useState(0),[result,setResult]=useState<{category:string;attempt:number;value?:HighlightList;error?:string}>(),[open,setOpen]=useState(!!resume),heading=useRef<HTMLElement>(null),cache=useRef<typeof result>(undefined),manualDisclosure=useRef(false),shouldRead=open||!!resume;
 useLayoutEffect(()=>{if(resume&&!manualDisclosure.current)setOpen(true);},[resume?.id]);
 useEffect(()=>{if(!shouldRead)return;if(cache.current?.category===category&&cache.current.attempt===retry&&cache.current.value){setResult(cache.current);return;}if(resume)onResumeReady?.(undefined);let live=true;const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),12000);setResult(undefined);
  socialApi<unknown>('community/highlights'+(category?'?category='+encodeURIComponent(category):''),undefined,controller.signal).then(value=>{const data=readHighlights(value,category);if(!data)throw Error('recommendation_response_invalid');if(live){cache.current={category,attempt:retry,value:data};setResult(cache.current);}}).catch(e=>{if(live)setResult({category,attempt:retry,error:errorMessage(e)});}).finally(()=>clearTimeout(timer));
  return()=>{live=false;clearTimeout(timer);controller.abort();};
 },[category,retry,shouldRead,resume?.id,onResumeReady]);
 const value=result?.category===category&&result.attempt===retry?result:undefined;
 useEffect(()=>{if(resume&&value?.value)onResumeReady?.(resume.id);},[resume?.id,value,onResumeReady]);
 return <details className="community-highlights" data-reading-region="highlights" data-reading-ready={resume&&value?.value?resume.id:undefined} open={open} onToggle={event=>setOpen(event.currentTarget.open)}><summary ref={heading} onClick={()=>{manualDisclosure.current=true;}}><Sparkles size={18} aria-hidden="true"/><span><b>{t('社区精选','Community picks')}</b><small>{t('管理员选出的题解、讨论与面经','Solutions, discussions and experiences selected by the community team')}</small></span><ChevronDown size={15} aria-hidden="true"/></summary>
  {open&&value?.value&&<div className="community-highlight-tools"><button type="button" className="text-button" onClick={()=>{heading.current?.focus({preventScroll:true});setRetry(n=>n+1);}}><RefreshCw size={14}/>{t('刷新精选','Refresh recommendations')}</button></div>}
  {(open||!!value?.value)&&(!value?<p role="status">{t('正在读取精选…','Loading recommendations…')}</p>:value.error?<p role="alert">{value.error==='recommendation_response_invalid'?t('精选内容暂时无法确认，请重试。','Recommendations could not be confirmed. Retry.'):value.error} <button type="button" className="text-button" onClick={()=>{heading.current?.focus({preventScroll:true});setRetry(n=>n+1);}}><RefreshCw size={14}/>{t('重试精选','Retry recommendations')}</button></p>:!value.value!.items.length?<p>{t('这里暂时没有精选内容。可以继续浏览下面的讨论。','No recommendations yet. Continue browsing the discussions below.')}</p>:<ol>{value.value!.items.map(({slot,post})=><li key={slot} data-reading-post={post.id} data-reading-region="highlights" data-reading-resumed={resumedPost===post.id||undefined}><span className="community-highlight-number" aria-hidden="true">0{slot}</span><div><AppLink data-reading-link="title" className="community-highlight-title" href={communityPostHref(post.id,returnTo)}>{post.title}<ArrowUpRight size={15} aria-hidden="true"/></AppLink><HighlightContext context={post.context}/><p>{postPreview(post.body_excerpt,135,t('[公式]','[formula]')).text}</p><CommunityAuthor user={post.author} compact avatar={false}/></div></li>)}</ol>)}
 </details>;
}
