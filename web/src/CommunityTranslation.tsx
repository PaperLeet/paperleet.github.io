import {useEffect,useRef,useState,type ReactNode} from 'react';
import {useI18n} from './i18n';
import {socialApi,useAccount} from './social';
import CommunityMarkdown from './CommunityMarkdown';
import './community-translation.css';

type Source={id:string;title?:string;body:string;translation_version?:string};
type Result={status:'pending'|'ready'|'unavailable'|'stale';version:string;target:string;title?:string;body?:string};
const cache=new Map<string,{result:Result;at:number}>();
const pause=(signal:AbortSignal)=>new Promise<void>((resolve,reject)=>{
 const abort=()=>{clearTimeout(timer);reject(new DOMException('Cancelled','AbortError'));};
 const timer=setTimeout(()=>{signal.removeEventListener('abort',abort);resolve();},1500);
 signal.addEventListener('abort',abort,{once:true});
});

/** Display-only fields. Callers keep the source object for edits, quotes and search. */
export function useCommunityTranslation(kind:'post'|'reply',source?:Source,disabled=false){
 const {locale}=useI18n(),{capabilities}=useAccount();
 const ref=useRef<HTMLElement>(null);
 const [visible,setVisible]=useState(''),[attempt,retry]=useState(0);
 const [state,setState]=useState<{key:string;result:Result}>();
 const [original,setOriginal]=useState('');
 const version=source?.translation_version;
 const key=JSON.stringify([kind,source?.id,version,locale,source?.title,source?.body]);
 const enabled=capabilities?.community_translation===true&&!disabled&&!!version;
 useEffect(()=>{
  if(!enabled||!ref.current)return;
  if(typeof IntersectionObserver==='undefined'){setVisible(key);return;}
  const observer=new IntersectionObserver(entries=>{if(entries.some(entry=>entry.isIntersecting)){setVisible(key);observer.disconnect();}},{rootMargin:'100px'});
  observer.observe(ref.current);return()=>observer.disconnect();
 },[enabled,key]);
 useEffect(()=>{
  if(!enabled||visible!==key||!source||!version)return;
  const cached=cache.get(key);
  if(cached&&Date.now()-cached.at<300000){setState({key,result:cached.result});return;}
  const controller=new AbortController(),{signal}=controller;
  const timeout=setTimeout(()=>controller.abort(),90000);
  const path=`community/translations/${kind}/${encodeURIComponent(source.id)}/${locale}`;
  setState({key,result:{status:'pending',version,target:locale}});
  void(async()=>{
   try{
    let result=await socialApi<Result>(path,{version},signal);
    while(!signal.aborted&&result.status==='pending'){
     await pause(signal);
     result=await socialApi<Result>(`${path}?version=${encodeURIComponent(version)}`,undefined,signal);
    }
    if(signal.aborted)return;
    if(result.version!==version||result.target!==locale||!['ready','unavailable','stale'].includes(result.status)||(result.status==='ready'&&(typeof result.body!=='string'||typeof result.title!=='string')))throw Error('Invalid translation');
    if(result.status==='ready'){
     cache.delete(key);cache.set(key,{result,at:Date.now()});
     while(cache.size>100)cache.delete(cache.keys().next().value!);
    }
    setState({key,result});
   }catch{
    if(!signal.aborted)setState({key,result:{status:'unavailable',version,target:locale}});
   }finally{clearTimeout(timeout);}
  })();
  // Timeout also clears the loading state; unmounted/changed effects cannot update it.
  const expired=()=>setState({key,result:{status:'unavailable',version,target:locale}});
  signal.addEventListener('abort',expired,{once:true});
  return()=>{signal.removeEventListener('abort',expired);clearTimeout(timeout);controller.abort();};
 },[enabled,visible,key,kind,version,locale,attempt]);
 const current=enabled&&state?.key===key?state.result:undefined;
 const translated=current?.status==='ready'&&(current.body!==source?.body||current.title!==(source?.title||''));
 const showing=!!translated&&original!==key;
 return {ref,title:showing?current!.title!:source?.title||'',body:showing?current!.body!:source?.body||'',
  translated,showing,pending:current?.status==='pending',failed:current?.status==='unavailable',
  toggle:()=>setOriginal(showing?key:''),retry:()=>retry(value=>value+1)};
}
export function TranslationNotice({translation}:{translation:ReturnType<typeof useCommunityTranslation>}){
 const {t}=useI18n();
 if(!translation.translated&&!translation.pending&&!translation.failed)return null;
 return <div className="community-translation" aria-live="polite">
  {translation.translated?<><span>{translation.showing?t('机器翻译','Machine translated'):t('原文','Original')}</span><button type="button" onClick={translation.toggle}>{translation.showing?t('查看原文','View original'):t('查看翻译','View translation')}</button></>:
   translation.pending?<span>{t('正在翻译 · 当前为原文','Translating · Showing original')}</span>:<><span>{t('原文 · 翻译暂不可用','Original · Translation unavailable')}</span><button type="button" onClick={translation.retry}>{t('重试翻译','Retry translation')}</button></>}
 </div>;
}
export function TranslatedReply({reply,children}:{reply:Source;children?:(body:string)=>ReactNode}){
 const translation=useCommunityTranslation('reply',reply);
 return <div ref={translation.ref as React.RefObject<HTMLDivElement|null>} className="translated-reply"><TranslationNotice translation={translation}/>{children?children(translation.body):<CommunityMarkdown body={reply.body} displayBody={translation.body}/>}</div>;
}
