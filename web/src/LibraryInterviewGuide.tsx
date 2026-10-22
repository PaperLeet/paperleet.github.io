import AppLink from './AppLink';
import {useId,useRef,useState} from 'react';
import type {MouseEvent} from 'react';
import {ArrowRight,Building2,CheckCircle2,ChevronDown,Code2} from 'lucide-react';
import {useI18n} from './i18n';
import {useAccount} from './social';
import {hostedEdition,hostedAppAvailable,hostedAppHref} from './deployment';
import {usePracticeProgress} from './PracticeProgressContext';
import {useInterviewResource} from './interviewApi';
import {interviewCatalog,interviewLabel,type InterviewSummary} from './interviewModels';
import {emptyInterviewFilters,interviewHref,type InterviewFilters} from './interviewNavigation';
import type {LibraryCompanyScope} from './libraryCompanyScope';
import {readLibraryRoute,withLibraryContext} from './libraryNavigation';
import {libraryInterviewEvidence} from './libraryInterviewEvidence';
import type {Problem} from './types';
import './library-interview-guide.css';

type Props={problems:Problem[];returnTo:string;scopeReady?:boolean;external?:LibraryCompanyScope;onCompanyFilter?:(company:string)=>void};
export default function LibraryInterviewGuide(props:Props){
 const {user,loading}=useAccount();
 if(!hostedAppAvailable)return null;
 return <Guide key={(loading?'checking':user?.id||'guest')+(props.external?':external':':local')} {...props} accountLoading={loading}/>;
}
function Guide({problems,returnTo,scopeReady=true,accountLoading,external,onCompanyFilter}:Props&{accountLoading:boolean}){
 const {t,locale}=useI18n(),{user,health}=useAccount(),progress=usePracticeProgress(),heading=useId(),hint=useId(),content=useId();
 const [expanded,setExpanded]=useState(()=>!window.matchMedia('(max-width: 1199px)').matches),[visited,setVisited]=useState(expanded),toggle=useRef<HTMLButtonElement>(null),panel=useRef<HTMLDivElement>(null);
 const [localCompany,setCompany]=useState(''),[retry,setRetry]=useState(0),selector=useRef<HTMLSelectElement>(null);
 const filters=readLibraryRoute(returnTo)?.filters,company=external?filters?.company||'':localCompany,days=external?filters?.company_days||'365':'365';
 const resource=useInterviewResource<InterviewSummary>('community/interviews/summary?days=365',hostedEdition&&!accountLoading&&visited&&!external,retry,user?.id||'guest');
 const data=external?(external.data?.scope===external.scope&&external.data.phase==='ready'?external.data.summary:undefined):resource.data;
 const result=libraryInterviewEvidence(data,problems,company,days),all=result?.companies;
 const error=(external?external.data?.scope===external.scope&&['error','unavailable'].includes(external.data.phase):resource.error)||data&&!result?t('面经依据暂时无法确认，仍可继续浏览题库。','Report evidence is unavailable. You can keep browsing the library.'):undefined;
 const expected=health?.mode==='local'?'local':health?.mode==='hosted'?user?'user:'+user.id:'anonymous':'unresolved';
 const ownerMatched=!!progress&&progress.owner===expected&&expected!=='unresolved';
 const states=result?.questions.map(row=>progress?.ready&&ownerMatched?progress.read(row.problem.id):undefined)||[];
 const ready=!!result?.questions.length&&states.every(Boolean),done=ready?states.filter(state=>state?.state==='done').length:undefined;
 const next=ready?result?.questions[states.findIndex(state=>state?.state==='started')>=0?states.findIndex(state=>state?.state==='started'):Math.max(0,states.findIndex(state=>state?.state==='todo'))]:undefined;
 const label=(id:string)=>interviewLabel('companies',id,locale==='en');
 const hub=interviewHref({...emptyInterviewFilters(),view:'questions',company,days:days as InterviewFilters['days']});
 function close(){if(panel.current?.contains(document.activeElement))toggle.current?.focus({preventScroll:true});setExpanded(false);}
 function retryReports(event:MouseEvent<HTMLButtonElement>){if(document.activeElement===event.currentTarget)(selector.current||toggle.current)?.focus({preventScroll:true});external?(external.data?.missing?external.reload():external.retry()):setRetry(value=>value+1);}
 const rows=(items:NonNullable<typeof result>['questions'])=>items.map(row=>{
  const index=result!.questions.indexOf(row),state=states[index]?.state;
  return <li key={row.problem.id} className="library-interview-row"><AppLink className="library-interview-exercise" href={withLibraryContext(row.problem.number,returnTo)}><span aria-hidden="true">{state==='done'?<CheckCircle2 size={16}/>:<Code2 size={16}/>}</span><span><b>{row.problem.number}. {locale==='en'?row.problem.translations?.en?.title||row.problem.title:row.problem.title}</b><small>{state==='done'?t('已通过 · ','Solved · '):state==='started'?t('进行中 · ','In progress · '):''}{t(`${row.contributors} 位分享者 · ${row.reports} 篇面经`,`${row.contributors} ${row.contributors===1?'contributor':'contributors'} · ${row.reports} ${row.reports===1?'report':'reports'}`)}</small></span></AppLink><AppLink className="library-interview-source" href={hostedAppHref(interviewHref({...emptyInterviewFilters(),company,days:days as InterviewFilters['days'],concept:row.problem.id}))} aria-label={t(`查看${row.problem.title}的来源面经`,`Read source experiences for ${row.problem.translations?.en?.title||row.problem.title}`)}>{t('来源','Sources')}<ArrowRight size={13} aria-hidden="true"/></AppLink></li>;
 });
 return <section className="library-interview-guide" aria-labelledby={heading}>
  <header><h2 id={heading}><button ref={toggle} type="button" aria-expanded={expanded} aria-controls={content} onClick={()=>{if(expanded)close();else{setVisited(true);setExpanded(true);}}}><Building2 size={17} aria-hidden="true"/><span>{t('公司面试准备','Interview prep')}</span><ChevronDown size={14} aria-hidden="true"/></button></h2><AppLink href={hostedAppHref(interviewHref({...emptyInterviewFilters(),view:'companies'}))} aria-label={t('浏览公司索引','Browse company directory')} title={t('公司索引','Companies')}><span>{t('公司索引','Companies')}</span><ArrowRight size={16} aria-hidden="true"/></AppLink></header>
  <div id={content} ref={panel} hidden={!expanded} onKeyDown={event=>{
   if(event.key!=='Escape'||event.defaultPrevented||event.repeat||event.nativeEvent.isComposing||event.nativeEvent.keyCode===229||event.ctrlKey||event.metaKey||event.altKey||event.shiftKey)return;
   event.preventDefault();event.stopPropagation();close();
  }}>
  {!hostedEdition?<p>{t('结合社区公开面经，查看公司的关联知识点与练习。','Explore company concepts and practice linked by public community experiences.')} <AppLink href={hostedAppHref(hub)}>{t('查看公司练习','Explore company practice')}</AppLink></p>:<>
  {external?<p className="library-interview-basis"><b>{label(company)}</b> · {t('沿用当前公司与时间筛选','Using the current company and period')}</p>:<label className="library-interview-company"><span>{t('准备公司','Company')}</span><select ref={selector} aria-describedby={hint} value={company} disabled={accountLoading||!scopeReady} onChange={event=>setCompany(event.target.value)}><option value="">{t('全部公司','All companies')}</option>{interviewCatalog.companies.filter(row=>all?.has(row.id)||company===row.id).map(row=><option key={row.id} value={row.id}>{label(row.id)}{all?.has(row.id)?` · ${all.get(row.id)} ${t('题','modules')}`:''}</option>)}</select></label>}
  <p id={hint} className="library-interview-basis">{external?t('仅当前题库筛选范围 · 依据面经发布时间，按分享者数排序。','Limited to this library view, filtered by publication date and ranked by contributors.'):t('近一年公开面经 · 仅当前题库筛选范围，按分享者数排序。提及次数不代表实际面试频率。','Public reports from the past year, limited to this library view and ranked by contributors. Mentions do not establish interview frequency.')}</p>
  {accountLoading||resource.loading||!scopeReady&&!error?<p role="status">{!scopeReady?t('正在核对题库筛选…','Checking library filters…'):t('正在读取面经依据…','Loading report evidence…')}</p>:error?<div role="status"><p>{error}</p><button type="button" className="text-button" onClick={retryReports}>{t('重试面经依据','Retry report evidence')}</button></div>:result&&<>
   <div className="library-interview-summary" role="status"><b>{t(`${result.questions.length} 道关联练习`,`${result.questions.length} linked ${result.questions.length===1?'exercise':'exercises'}`)}</b>{done!==undefined&&<span>{t(`已通过 ${done} / ${result.questions.length}`,`Solved ${done} / ${result.questions.length}`)}</span>}</div>
   {next&&<AppLink className="button library-interview-next" href={withLibraryContext(next.problem.number,returnTo)}>{done===result.questions.length?t('复习关联练习','Review practice'):states.some(state=>state?.state==='started'||state?.state==='done')?t('继续准备','Continue practice'):t('开始准备','Start practice')}<ArrowRight size={15} aria-hidden="true"/></AppLink>}
   {!ready&&result.questions.length>0&&<p role="status">{ownerMatched&&progress?.failed?t('练习进度暂不可用，仍可逐题开始。','Practice progress is unavailable. You can open any exercise.'):t('正在核对练习进度…','Checking your practice progress…')}{ownerMatched&&progress?.failed&&progress.retry&&<button type="button" className="text-button" onClick={event=>{if(document.activeElement===event.currentTarget)selector.current?.focus({preventScroll:true});progress.retry?.();}}>{t('重试进度','Retry progress')}</button>}</p>}
   {result.questions.length?<><ul className="library-interview-list">{rows(result.questions.slice(0,3))}</ul>{result.questions.length>3&&<details className="library-interview-more"><summary>{t(`展开另外 ${result.questions.length-3} 道练习`,`Show ${result.questions.length-3} more exercises`)}<ChevronDown size={14} aria-hidden="true"/></summary><ul className="library-interview-list">{rows(result.questions.slice(3))}</ul></details>}</>:<p>{company?t('当前筛选范围没有这家公司的关联面经，可调整题库筛选或查看完整公司练习。','No linked experiences for this company in this library view. Adjust the library filters or explore all company practice.'):t('当前筛选范围暂无关联面经。欢迎在社区分享真实面试经历。','No linked experiences in this library view yet. Share your interview experience with the community.')}</p>}
  </>}
  {!external&&company&&onCompanyFilter&&<button type="button" className="button subtle" onClick={()=>onCompanyFilter(company)}>{t('用这家公司筛选题库','Filter the library by this company')}</button>}
  <footer><AppLink href={hostedAppHref(hub)}>{t('查看完整面试准备','Explore all interview practice')}<ArrowRight size={14} aria-hidden="true"/></AppLink></footer>
  </>}
  </div>
 </section>;
}
