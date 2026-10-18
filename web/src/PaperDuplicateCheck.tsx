import {useEffect,useState} from 'react';
import {ArrowUpRight} from 'lucide-react';
import {useI18n} from './i18n';
import {arxivPaperUrl,poolApi,poolHref} from './paperPoolApi';
import type {PoolDuplicate} from './paperPoolTypes';

export function PaperDuplicateNotice({duplicate}:{duplicate:PoolDuplicate}){
  const {t,locale}=useI18n();
  return <div className="pool-feedback" role="status"><strong>{duplicate.kind==='proposal'?t('这篇论文已有投稿','This paper already has a proposal'):t('这篇论文已收录','This paper is already published')}</strong>{duplicate.kind==='proposal'?<a href={poolHref(duplicate.proposal.id)}>{t('查看已有投稿','View existing proposal')}<ArrowUpRight size={14}/></a>:duplicate.problems.map(problem=><a key={problem.id} href={`#/${problem.id}`}>{locale==='en'?problem.title_en||problem.title:problem.title}<ArrowUpRight size={14}/></a>)}<small>{t('重复投稿不会消耗每日额度。','Duplicates do not use your daily proposal quota.')}</small></div>;
}

/** Draft changes cancel old lookups; the POST always checks again before accepting. */
export default function PaperDuplicateCheck({value}:{value:string}){
  const {t}=useI18n(),url=arxivPaperUrl(value);
  const [retry,setRetry]=useState(0),[result,setResult]=useState<{url:string;duplicate?:PoolDuplicate|null;error?:boolean}>();
  useEffect(()=>{
    if(!url)return;
    let active=true;const controller=new AbortController();
    const timer=window.setTimeout(()=>{
      void poolApi<{duplicate:PoolDuplicate|null}>(`/check?${new URLSearchParams({arxiv_url:url})}`,undefined,controller.signal).then(data=>{
        if(!Object.hasOwn(data,'duplicate'))throw Error('Invalid duplicate response');
        if(active)setResult({url,duplicate:data.duplicate});
      }).catch(()=>{if(active)setResult({url,error:true});});
    },300);
    return()=>{active=false;window.clearTimeout(timer);controller.abort();};
  },[url,retry]);
  if(!url)return <small className="pool-muted">{t('粘贴论文链接后自动检查是否已收录或有人投稿。','Paste a paper link to check published papers and existing proposals.')}</small>;
  if(result?.url!==url)return <small className="pool-muted" role="status">{t('正在检查重复论文…','Checking for duplicate papers…')}</small>;
  if(result.error)return <small className="pool-muted" role="status">{t('暂时无法预查，提交时仍会查重。','Preview check unavailable. Submission will still check for duplicates.')} <button className="text-button" type="button" onClick={()=>{setResult(undefined);setRetry(value=>value+1);}}>{t('重试','Retry')}</button></small>;
  return result.duplicate?<PaperDuplicateNotice duplicate={result.duplicate}/>:<small className="pool-muted" role="status">{t('暂未发现重复论文，提交时会再次检查。','No duplicate found. We will check again when you submit.')}</small>;
}
