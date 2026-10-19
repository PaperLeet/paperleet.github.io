import {appLocation} from './appLocation';
import AppLink from './AppLink';
import {useEffect,useId,useLayoutEffect,useRef,useState} from 'react';
import {ArrowUpRight,ChevronDown,ChevronUp,Code2,X} from 'lucide-react';
import {useI18n} from './i18n';
import {useAccount} from './social';
import InterviewCompanySources from './InterviewCompanySources';
import {hostedEdition} from './deployment';
import {useInterviewResource} from './interviewApi';
import {interviewCatalog,interviewLabel} from './interviewModels';
import OrganizationLogo from './OrganizationLogo';
import {interviewHref} from './interviewNavigation';
import {interviewPracticePreviewScope} from './interviewSourcePreviewModel';
import {problemInterviewPeriods,problemInterviewFilters,readProblemInterviewEvidence,type ProblemInterviewPeriod} from './problemInterviewEvidence';
import {deferredFeature} from './deferredFeature';
import {keepReadingTriggerVisible} from './readingViewport';
import {useWorkspaceIntent} from './useWorkspaceIntent';
import './interviews.css';
import './problem-interview-preview.css';
import {practiceEvidenceOrigin,type PracticeEvidenceOrigin} from './practiceEvidenceOrigin';

const SourceReports=deferredFeature<Parameters<typeof import('./InterviewSourcePreview').InterviewSourceReports>[0]>(()=>import('./InterviewSourcePreview').then(module=>({default:module.InterviewSourceReports})),['关联面经','linked experiences'],{failure:(_props,retry)=><PreviewFailure retry={retry}/>});
function PreviewFailure({retry}:{retry:()=>void}){const {t}=useI18n();return <p className="problem-interview-status" role="alert">{t('面经预览暂时无法加载，仍可继续练习。','The preview could not load. You can keep practicing.')} <button type="button" className="text-button" onClick={retry}>{t('重试预览','Retry preview')}</button></p>;}
type Props={problemId:string;number?:number;onCode?:()=>void};
export default function ProblemCompanyTags(props:Props){
 const {user,loading}=useAccount(),owner=user?.id||'guest';
 const [route,setRoute]=useState(()=>appLocation.hash);
 useEffect(()=>{const changed=()=>setRoute(appLocation.hash);window.addEventListener('hashchange',changed);window.addEventListener('popstate',changed);return()=>{window.removeEventListener('hashchange',changed);window.removeEventListener('popstate',changed);};},[]);
 const origin=practiceEvidenceOrigin(route,props.problemId,props.number);
 if(loading)return null;
 return <CompanyTags key={JSON.stringify([owner,props.problemId,origin?.company,origin?.days])} {...props} owner={owner} enabled={hostedEdition} route={route} origin={origin}/>;
}
function CompanyTags({problemId,number,onCode,owner,enabled,route,origin:sourceOrigin}:Props&{owner:string;enabled:boolean;route:string;origin?:PracticeEvidenceOrigin}){
 const {t,locale}=useI18n(),id=useId(),hint=useId(),toggle=useRef<HTMLButtonElement>(null),panel=useRef<HTMLElement>(null),root=useRef<HTMLDivElement>(null),periodControl=useRef<HTMLSelectElement>(null);
 const [days,setDays]=useState<ProblemInterviewPeriod>(sourceOrigin?.days||'365'),[company,setCompany]=useState(sourceOrigin?.company||''),[seen,setSeen]=useState(false);
 const [retry,setRetry]=useState(0),[opening,setOpening]=useState<string>(),lastRoute=useRef(route);
 useEffect(()=>{if(lastRoute.current!==route){lastRoute.current=route;setOpening(undefined);}},[route]);
 const resource=useInterviewResource<unknown>('community/interviews/summary?days='+days+'&problem_id='+encodeURIComponent(problemId),enabled,retry,owner);
 const result=readProblemInterviewEvidence(resource.data,problemId,days),companies=result?.companies,evidence=result?.evidence;
 if(!seen&&!!companies?.length)setSeen(true);
 const current=interviewPracticePreviewScope(appLocation.hash,problemId,number),expanded=!!opening&&opening===current&&route===appLocation.hash;
 const origin=useRef<HTMLElement|undefined>(undefined),heading=useRef<HTMLHeadingElement>(null),pendingFocus=useRef<(()=>boolean)|undefined>(undefined);
 const intent=useWorkspaceIntent(JSON.stringify([owner,enabled,current]));
 useLayoutEffect(()=>{
  const claim=pendingFocus.current;if(!claim)return;pendingFocus.current=undefined;
  if(expanded&&heading.current?.isConnected&&!heading.current.closest('[hidden]')&&claim()){heading.current.focus({preventScroll:true});keepReadingTriggerVisible(heading.current,100);}
  else intent.cancel();
 },[opening,company,expanded]);
 const here=()=>!!opening&&!!root.current?.isConnected&&interviewPracticePreviewScope(appLocation.hash,problemId,number)===opening;
 function close(){
  if(!here())return;
  pendingFocus.current=undefined;intent.cancel();
  if(panel.current?.contains(document.activeElement)){
   const opener=origin.current,control=opener?.isConnected&&root.current?.contains(opener)&&!opener.closest('[hidden],details:not([open])')?opener:toggle.current;
   control?.focus({preventScroll:true});if(control)keepReadingTriggerVisible(control,100);
  }
  setOpening(undefined);
 }
 function previewCompany(value:string,opener:HTMLButtonElement){
  if(!current||!root.current?.isConnected||interviewPracticePreviewScope(appLocation.hash,problemId,number)!==current||!companies?.some(row=>row.company===value&&row.reports>0&&row.contributors>0))return;
  if(expanded&&company===value){close();return;}
  origin.current=opener;pendingFocus.current=intent.create();setCompany(value);setOpening(current);
 }
 const filters=problemInterviewFilters(problemId,days),previewFilters=problemInterviewFilters(problemId,days,company),sourceHref=interviewHref(previewFilters);
 const selected=company?companies?.find(row=>row.company===company):evidence;
 const error=resource.error||resource.data&&!result;
 const period=(value:string)=>value==='all'?t('全部时间','All time'):value==='365'?t('近一年','Past year'):t(`近 ${value} 天`,`Past ${value} days`);
 if(!enabled)return null;
 return <div ref={root} className="problem-interview-reference">
  <div className="problem-interview-reference-heading"><strong>{t('面经参考','Interview evidence')}</strong><label>{t('发布时间','Published')}<select ref={periodControl} value={days} aria-describedby={hint} onChange={event=>{const value=event.target.value;if(problemInterviewPeriods.includes(value as ProblemInterviewPeriod))setDays(value as ProblemInterviewPeriod);}}>{problemInterviewPeriods.map(value=><option value={value} key={value}>{period(value)}</option>)}</select></label><small id={hint}>{t('社区自述 · 非官方考频','Community reports · Not official frequency')}</small></div>
  {sourceOrigin&&<div className="problem-interview-origin" role="group" aria-label={t('来源列表的公司与发布时间','Company and publication period from the source list')}><span>{t('面经预览来源：','Experience preview from your list: ')}<b>{sourceOrigin.company?interviewLabel('companies',sourceOrigin.company,locale==='en'):t('全部公司','All companies')}</b> · {period(sourceOrigin.days)}</span>{(company!==sourceOrigin.company||days!==sourceOrigin.days)&&<button type="button" className="text-button" onClick={event=>{
   if(!current||!root.current?.isConnected||interviewPracticePreviewScope(appLocation.hash,problemId,number)!==current)return;
   if(document.activeElement===event.currentTarget)periodControl.current?.focus({preventScroll:true});setCompany(sourceOrigin.company);setDays(sourceOrigin.days);
  }}>{t('恢复公司与时间','Restore company and period')}</button>}</div>}
  {error?<div className="problem-company-evidence problem-company-retry" role="status"><small>{t('公司面经暂时无法读取，仍可继续练习。','Company experiences are unavailable. You can keep practicing.')}</small><button type="button" onClick={event=>{if(document.activeElement===event.currentTarget)periodControl.current?.focus({preventScroll:true});setRetry(value=>value+1);}}>{t('重试面经','Retry experiences')}</button></div>:resource.loading?<p className="problem-interview-status" role="status">{t('正在核对本题的面经依据…','Checking report evidence for this exercise…')}</p>:companies?.length?<div className="problem-company-evidence"><small>{period(days)}{t('面经提及',' report mentions')}</small><InterviewCompanySources companies={companies} problemId={problemId} filters={filters} preview={current?{panelId:id,company,open:expanded,onSelect:previewCompany}:undefined}/><AppLink href={interviewHref({...filters,view:'questions'})}>{t('查看统计依据','Explore report-based counts')}</AppLink></div>:<p className="problem-interview-status" role="status">{t('当前时间范围暂无公开面经提及此题，可调整范围或去社区查看与分享。','No public reports mention this exercise in this period. Change the period or explore and share experiences.')} <AppLink href={sourceHref} target="_blank" rel="noopener noreferrer">{t('查看与分享面经','Explore and share experiences')}<ArrowUpRight size={13} aria-hidden="true"/></AppLink></p>}
  {current&&!!companies?.length&&<p className="problem-company-preview-hint">{t('点击公司标签，在这里预览关联面经。','Select a company tag to preview its linked experiences here.')}</p>}
  {current&&seen&&<button ref={toggle} type="button" className="problem-interview-toggle" aria-expanded={expanded} aria-controls={id} onClick={event=>{if(!root.current?.isConnected||interviewPracticePreviewScope(appLocation.hash,problemId,number)!==current)return;if(expanded)close();else{origin.current=event.currentTarget;setOpening(current);}}}>{expanded?<ChevronUp size={14} aria-hidden="true"/>:<ChevronDown size={14} aria-hidden="true"/>}{expanded?t('收起面经预览','Collapse experience preview'):company?t(`预览 ${interviewLabel('companies',company,locale==='en')} 面经`,`Preview ${interviewLabel('companies',company,locale==='en')} experiences`):t('预览关联面经','Preview linked experiences')}</button>}
  <section id={id} ref={panel} hidden={!expanded} className="problem-interview-panel interview-source-preview" role="region" aria-label={t('本题关联面经预览','Linked experience preview for this exercise')} onKeyDown={event=>{
   if(event.key!=='Escape'||event.defaultPrevented||event.repeat||event.nativeEvent.isComposing||event.nativeEvent.keyCode===229||event.ctrlKey||event.metaKey||event.altKey||event.shiftKey)return;
   event.preventDefault();event.stopPropagation();close();
  }}>{expanded&&<>
   <header className="problem-interview-preview-heading"><div>{company&&<OrganizationLogo id={company} size={28}/>}<div><h3 ref={heading} tabIndex={-1}>{company?t(`${interviewLabel('companies',company,locale==='en')}的关联面经`,`${interviewLabel('companies',company,locale==='en')} linked experiences`):t('本题关联面经','Linked experiences for this exercise')}</h3><p>{!resource.loading&&!error&&selected?t(`${selected.reports} 篇公开面经 · ${selected.contributors} 位分享者`,`${selected.reports} public ${selected.reports===1?'report':'reports'} · ${selected.contributors} ${selected.contributors===1?'contributor':'contributors'}`):period(days)}</p></div></div><button type="button" className="problem-interview-close" aria-label={t('关闭关联面经预览','Close linked experience preview')} onClick={close}><X size={17} aria-hidden="true"/></button></header>
   <div className="problem-interview-preview-filters"><label>{t('来源公司','Source company')}<select value={company} onChange={event=>{const value=event.target.value;if(!value||interviewCatalog.companies.some(row=>row.id===value))setCompany(value);}}><option value="">{t('全部公司','All companies')}</option>{interviewCatalog.companies.map(row=><option value={row.id} key={row.id}>{locale==='en'?row.name_en:row.name}</option>)}</select></label><span>{period(days)} · {t('按面经发布时间','By report publication date')}</span></div>
   {resource.loading?<p className="problem-interview-status" role="status">{t('正在核对当前时间范围…','Checking the selected publication period…')}</p>:error?<p className="problem-interview-status" role="status">{t('面经依据暂不可用。上方可重试，筛选与练习仍保留。','Report evidence is unavailable. Retry above; your filters and practice are kept.')}</p>:selected&&selected.reports>0?<SourceReports key={sourceHref} concept={problemId} name={t('本题','this exercise')} reports={selected.reports} contributors={selected.contributors} sourceHref={sourceHref} returnTo={sourceHref} owner={owner} practice={{href:opening!,number}}/>:<p className="problem-interview-status" role="status">{company?t('这家公司在当前时间范围暂无关联面经，可换公司或调整发布时间。','No linked reports for this company in this period. Choose another company or publication period.'):t('当前时间范围暂无关联面经，可调整发布时间。','No linked reports in this period. Change the publication period.')}</p>}
   <footer className="problem-interview-actions"><AppLink href={sourceHref} target="_blank" rel="noopener noreferrer">{t('新标签页查看全部面经','Read all experiences in a new tab')}<ArrowUpRight size={14} aria-hidden="true"/></AppLink><div><button type="button" className="text-button" onClick={close}>{t('收起预览','Collapse preview')}</button>{onCode&&<button type="button" className="button subtle" onClick={()=>{if(!here())return;close();onCode();}}><Code2 size={14} aria-hidden="true"/>{t('返回代码','Back to code')}</button>}</div></footer>
  </>}</section>
 </div>;
}
