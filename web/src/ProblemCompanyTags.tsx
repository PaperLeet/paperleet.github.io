import {useEffect,useId,useRef,useState} from 'react';
import {ArrowUpRight,ChevronDown,ChevronUp,Code2} from 'lucide-react';
import {useI18n} from './i18n';
import {useAccount} from './social';
import InterviewCompanySources from './InterviewCompanySources';
import {hostedEdition} from './deployment';
import {useInterviewResource} from './interviewApi';
import {readInterviewSummary,type InterviewSummary} from './interviewModels';
import {interviewHref,emptyInterviewFilters} from './interviewNavigation';
import {interviewPracticePreviewScope} from './interviewSourcePreviewModel';
import {deferredFeature} from './deferredFeature';
import {keepReadingTriggerVisible} from './readingViewport';
import './interviews.css';
import './problem-interview-preview.css';

const SourceReports=deferredFeature<Parameters<typeof import('./InterviewSourcePreview').InterviewSourceReports>[0]>(()=>import('./InterviewSourcePreview').then(module=>({default:module.InterviewSourceReports})),['关联面经','linked experiences'],{failure:(_props,retry)=><PreviewFailure retry={retry}/>});
function PreviewFailure({retry}:{retry:()=>void}){const {t}=useI18n();return <p className="problem-interview-status" role="alert">{t('面经预览暂时无法加载，仍可继续练习。','The preview could not load. You can keep practicing.')} <button type="button" className="text-button" onClick={retry}>{t('重试预览','Retry preview')}</button></p>;}
type Props={problemId:string;number?:number;onCode?:()=>void};
export default function ProblemCompanyTags(props:Props){
 const {user,loading}=useAccount(),owner=user?.id||'guest';
 if(loading)return null;
 return <CompanyTags key={JSON.stringify([owner,props.problemId])} {...props} owner={owner} accountLoading={loading}/>;
}
function CompanyTags({problemId,number,onCode,owner,accountLoading}:Props&{owner:string;accountLoading:boolean}){
 const {t}=useI18n(),id=useId(),toggle=useRef<HTMLButtonElement>(null),panel=useRef<HTMLElement>(null),root=useRef<HTMLDivElement>(null);
 const [retry,setRetry]=useState(0),[opening,setOpening]=useState<string>(),[route,setRoute]=useState(()=>location.hash),lastRoute=useRef(location.hash);
 useEffect(()=>{const changed=()=>{if(lastRoute.current===location.hash)return;lastRoute.current=location.hash;setOpening(undefined);setRoute(location.hash);};window.addEventListener('hashchange',changed);window.addEventListener('popstate',changed);return()=>{window.removeEventListener('hashchange',changed);window.removeEventListener('popstate',changed);};},[]);
 const resource=useInterviewResource<InterviewSummary>('community/interviews/summary?days=365&problem_id='+encodeURIComponent(problemId),hostedEdition&&!accountLoading,retry,owner);
 const summary=readInterviewSummary(resource.data),evidence=summary?.questions.find(row=>row.problem_id===problemId);
 const companies=evidence?.companies.filter(company=>company.reports>0&&company.contributors>0);
 const current=interviewPracticePreviewScope(location.hash,problemId,number),expanded=!!opening&&opening===current&&route===location.hash;
 const here=()=>!accountLoading&&!!root.current?.isConnected&&interviewPracticePreviewScope(location.hash,problemId,number)===opening;
 function close(){
  if(!here())return;
  if(panel.current?.contains(document.activeElement)){toggle.current?.focus({preventScroll:true});if(toggle.current)keepReadingTriggerVisible(toggle.current,100);}
  setOpening(undefined);
 }
 if(accountLoading)return null;
 if(resource.error||resource.data&&!summary)return <div className="problem-company-evidence problem-company-retry" role="status"><small>{t('公司面经暂时无法读取，仍可继续练习。','Company experiences are unavailable. You can keep practicing.')}</small><button type="button" onClick={()=>setRetry(value=>value+1)}>{t('重试面经','Retry experiences')}</button></div>;
 if(!companies?.length)return null;
 const sourceHref=interviewHref({...emptyInterviewFilters(),concept:problemId});
 return <div ref={root} className="problem-company-evidence"><small>{t('近一年面经提及','In reports from the past year')}</small><InterviewCompanySources companies={companies} problemId={problemId}/><a href={interviewHref({...emptyInterviewFilters(),view:'questions',concept:problemId})}>{t('查看统计依据','Explore report-based counts')}</a>
  {current&&<button ref={toggle} type="button" className="problem-interview-toggle" aria-expanded={expanded} aria-controls={id} onClick={()=>{if(accountLoading||!root.current?.isConnected||interviewPracticePreviewScope(location.hash,problemId,number)!==current)return;expanded?close():setOpening(current);}}>{expanded?<ChevronUp size={14} aria-hidden="true"/>:<ChevronDown size={14} aria-hidden="true"/>}{expanded?t('收起面经预览','Collapse experience preview'):t('预览关联面经','Preview linked experiences')}</button>}
  <section id={id} ref={panel} hidden={!expanded} className="problem-interview-panel interview-source-preview" role="region" aria-label={t('本题关联面经预览','Linked experience preview for this exercise')} onKeyDown={event=>{
   if(event.key!=='Escape'||event.defaultPrevented||event.repeat||event.nativeEvent.isComposing||event.nativeEvent.keyCode===229||event.ctrlKey||event.metaKey||event.altKey||event.shiftKey)return;
   event.preventDefault();event.stopPropagation();close();
  }}>{expanded&&<>
   <SourceReports concept={problemId} name={t('本题','this exercise')} reports={evidence!.reports} contributors={evidence!.contributors} sourceHref={sourceHref} returnTo={sourceHref} owner={owner} accountLoading={accountLoading} practice={{href:opening!,number}}/>
   <footer className="problem-interview-actions"><a href={sourceHref} target="_blank" rel="noopener noreferrer">{t('新标签页查看全部面经','Read all experiences in a new tab')}<ArrowUpRight size={14} aria-hidden="true"/></a><div><button type="button" className="text-button" onClick={close}>{t('收起预览','Collapse preview')}</button>{onCode&&<button type="button" className="button subtle" onClick={()=>{if(!here())return;close();onCode();}}><Code2 size={14} aria-hidden="true"/>{t('返回代码','Back to code')}</button>}</div></footer>
  </>}</section>
 </div>;
}
