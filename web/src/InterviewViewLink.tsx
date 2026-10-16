import {useEffect,useRef,useState} from 'react';
import {Check,Copy} from 'lucide-react';
import {useI18n} from './i18n';
import {interviewHref,readInterviewRoute,type InterviewFilters} from './interviewNavigation';

/** A shareable view contains applied filters, not the page's query or previous hash. */
export function interviewViewShareUrl(pageUrl:string,href:string):string {
  const page=new URL(pageUrl),filters=readInterviewRoute(href);
  if(!['http:','https:'].includes(page.protocol)||!filters)throw Error('Invalid interview view');
  return page.origin+page.pathname+interviewHref(filters);
}

export default function InterviewViewLink({filters,owner='guest'}:{filters:InterviewFilters;owner?:string}) {
  const href=interviewHref(filters);
  return <ScopedViewLink key={JSON.stringify([owner,href])} href={href} personal={filters.saved||filters.view==='questions'&&filters.study!=='all'}/>;
}

function ScopedViewLink({href,personal}:{href:string;personal:boolean}) {
  const {t}=useI18n();
  const [status,setStatus]=useState<'idle'|'copying'|'copied'|'failed'>('idle');
  const [fallback,setFallback]=useState('');
  const mounted=useRef(false),busy=useRef(false);
  const deadline=useRef<number|undefined>(undefined);
  useEffect(()=>{mounted.current=true;return()=>{mounted.current=false;window.clearTimeout(deadline.current);};},[]);
  async function copy() {
    if(busy.current)return;
    busy.current=true;setStatus('copying');setFallback('');
    let timer:number|undefined;
    try {
      const url=interviewViewShareUrl(location.href,href);
      setFallback(url);
      const write=navigator.clipboard.writeText(url);
      await Promise.race([write,new Promise<void>((_,reject)=>{
        deadline.current=timer=window.setTimeout(()=>reject(Error('Clipboard timed out')),8000);
      })]);
      if(mounted.current)setStatus('copied');
    } catch {if(mounted.current)setStatus('failed');}
    finally {window.clearTimeout(timer);if(deadline.current===timer)deadline.current=undefined;busy.current=false;}
  }
  const personalNote=t('收藏与学习状态按打开者自己的工作区显示。','Saved experiences and learning states use the viewer’s own workspace.');
  return <div className="interview-view-link">
    <button type="button" className="interview-view-copy" aria-disabled={status==='copying'} aria-busy={status==='copying'} onClick={()=>void copy()}>
      {status==='copied'?<Check size={15} aria-hidden="true"/>:<Copy size={15} aria-hidden="true"/>}{t('复制筛选链接','Copy filtered link')}
    </button>
    <span className={status==='copied'?'interview-view-link-message':'sr-only'} role="status">{status==='copying'?t('正在复制筛选链接…','Copying the filtered link…'):status==='copied'?<>{t('筛选链接已复制。','Filtered link copied.')}{personal&&<> {personalNote}</>}</>:status==='failed'&&fallback?t('自动复制不可用，请手动复制下方链接。','Automatic copy is unavailable. Copy the link below manually.'):null}</span>
    {status==='failed'&&(fallback?<label className="interview-view-link-fallback">{t('自动复制不可用，请选择并复制链接。','Automatic copy is unavailable. Select and copy the link.')}{personal&&<span>{personalNote}</span>}<input aria-label={t('可手动复制的筛选链接','Filtered link to copy manually')} readOnly value={fallback} onFocus={event=>event.currentTarget.select()}/></label>:<span className="interview-view-link-message" role="alert">{t('当前地址无法生成分享链接，请从网站打开此页面。','This address cannot create a shareable link. Open this page on the website.')}</span>)}
  </div>;
}
