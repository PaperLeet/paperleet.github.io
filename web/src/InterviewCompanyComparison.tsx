import {useId,useMemo,useRef,useState} from 'react';
import {ArrowLeftRight,ChevronDown,X} from 'lucide-react';
import {useI18n} from './i18n';
import {interviewCatalog,type InterviewSummary} from './interviewModels';
import {interviewHref,type InterviewFilters} from './interviewNavigation';
import {compareInterviewCompanies,comparisonRows,companyComparisonPair,type CompanyComparisonView} from './interviewCompanyComparisonModel';
import OrganizationLogo from './OrganizationLogo';
import type {Problem,Saved} from './types';
import InterviewPracticeGroup,{InterviewGroupToggle,useInterviewPracticeGroup} from './InterviewPracticeGroup';
import './interview-company-comparison.css';

type Props={owner?:string;filters:InterviewFilters;summary?:InterviewSummary;loading:boolean;error?:string;problems?:Problem[];catalogError?:string;onRetryCatalog?:()=>void;saved?:Record<string,Saved>;progressReady?:boolean;progressError?:string;onRetryProgress?:()=>void;onChange:(update:Partial<InterviewFilters>)=>void};
const EMPTY_PROBLEMS:Problem[]=[],EMPTY_SAVED:Record<string,Saved>={};
const PAGE_SIZE=12;

/** Compare the same public summary; the URL restores choices without extra requests or storage. */
export default function InterviewCompanyComparison({owner='guest',filters,summary,loading,error,problems=EMPTY_PROBLEMS,catalogError,onRetryCatalog,saved=EMPTY_SAVED,progressReady=false,progressError,onRetryProgress,onChange}:Props){
 const {t,locale}=useI18n(),id=useId(),heading=useRef<HTMLElement>(null),leftInput=useRef<HTMLSelectElement>(null);
 // Keep an applied local choice mounted until the hash router observes its URL.
 // A later back/shared route then owns the choice, rather than reviving the prior pair.
 const [selection,setSelection]=useState<{scope?:string;pending?:{next?:string};pair:{left:string;right:string};open:boolean}>(()=>({scope:filters.compare,pair:companyComparisonPair(filters.compare),open:!!filters.compare}));
 let current=selection;
 if(selection.scope!==filters.compare){
  current=selection.pending&&selection.pending.next===filters.compare?{scope:filters.compare,pair:selection.pair,open:selection.open}:{scope:filters.compare,pair:companyComparisonPair(filters.compare),open:!!filters.compare};
  setSelection(current);
 }
 const {open,pair}=current;
 const view:CompanyComparisonView=filters.comparison||'shared';
 const scope=JSON.stringify([pair.left,pair.right,filters.role,filters.stage,filters.outcome,filters.days,filters.concept,view]);
 const [shown,setShown]=useState({scope:'',limit:PAGE_SIZE});
 const compare=useMemo(()=>summary&&summary.days===filters.days&&!loading&&!error?compareInterviewCompanies(summary,pair.left,pair.right):undefined,[summary,filters.days,loading,error,pair.left,pair.right]);
 const byId=useMemo(()=>new Map(problems.map(problem=>[problem.id,problem])),[problems]);
 const label=(company:string)=>{const row=interviewCatalog.companies.find(row=>row.id===company);return row?(locale==='en'?row.name_en:row.name):company;};
 const selected=!!pair.left&&!!pair.right&&pair.left!==pair.right;
 const rows=compare?comparisonRows(compare.rows,view):[],limit=shown.scope===scope?shown.limit:PAGE_SIZE;
 const applied=filters.compare===`${pair.left},${pair.right}`;
 const candidates=rows.flatMap(row=>{const problem=byId.get(row.id);return problem?[problem]:[];});
 const group=useInterviewPracticeGroup({filters,source:'reports',selectionScope:JSON.stringify([scope,filters.queue]),problems,candidates,filterPending:!compare||!applied||!problems.length||!!catalogError,pendingReason:!compare||!applied?'sources':'catalog',saved,ready:progressReady&&!progressError,owner});
 const sourceHref=(company:string,concept=filters.concept)=>interviewHref({...filters,company,concept,view:'reports',q:'',saved:false,queue:'',page:1});
 function choose(side:'left'|'right',value:string){
  if(value&&(!interviewCatalog.companies.some(row=>row.id===value)||value===pair[side==='left'?'right':'left']))return;
  applyPair({...pair,[side]:value});
 }
 function applyPair(next:{left:string;right:string}){
  const compare=next.left&&next.right?`${next.left},${next.right}`:undefined;
  setSelection({scope:filters.compare,pending:compare!==filters.compare?{next:compare}:undefined,pair:next,open:true});setShown({scope:'',limit:PAGE_SIZE});
  if(compare!==filters.compare)onChange({compare,comparison:compare?filters.comparison:undefined});
 }
 const counts=(company:string,reports:number,contributors:number,concept=filters.concept)=><a href={sourceHref(company,concept)} aria-label={t(`阅读${label(company)}${concept?'与此知识点相关的':''} ${reports} 篇来源面经，来自 ${contributors} 位分享者`,`Read ${reports} ${label(company)} source ${reports===1?'report':'reports'}${concept?' for this concept':''}, from ${contributors} ${contributors===1?'contributor':'contributors'}`)}>{t(`${reports} 篇面经 · ${contributors} 位分享者`,`${reports} ${reports===1?'report':'reports'} · ${contributors} ${contributors===1?'contributor':'contributors'}`)}</a>;
 return <details className="company-comparison" open={open} onToggle={event=>{
  const expanded=event.currentTarget.open;
  if(!expanded&&event.currentTarget.querySelector('.company-comparison-content')?.contains(document.activeElement))heading.current?.focus({preventScroll:true});
  setSelection(previous=>previous.open===expanded?previous:{...previous,open:expanded});
 }}>
  <summary ref={heading}><ArrowLeftRight size={17} aria-hidden="true"/><span>{t('对比公司知识点','Compare company concepts')}</span><ChevronDown size={16} aria-hidden="true"/></summary>
  {open&&<div className="company-comparison-content">
   <p id={`${id}-help`}>{t('选择两家公司，对照当前岗位、轮次、结果、知识点与时间范围内的公开面经。公司名称搜索不限制对比选择。','Choose two companies to compare public reports within the current role, stage, outcome, concept and period. The company-name search does not limit these choices.')}</p>
   <div className="company-comparison-controls" role="group" aria-describedby={`${id}-help`} aria-label={t('公司对比选择','Company comparison choices')}>
    {(['left','right'] as const).map(side=><label key={side}><span>{side==='left'?t('第一家公司','First company'):t('第二家公司','Second company')}</span><select ref={side==='left'?leftInput:undefined} value={pair[side]} onChange={event=>choose(side,event.target.value)}><option value="">{t('选择公司','Choose a company')}</option>{interviewCatalog.companies.map(company=><option key={company.id} value={company.id} disabled={company.id===pair[side==='left'?'right':'left']}>{locale==='en'?company.name_en:company.name}</option>)}</select></label>)}
    <button type="button" className="company-comparison-swap" disabled={!selected} onClick={()=>{applyPair({left:pair.right,right:pair.left});}}><ArrowLeftRight size={16} aria-hidden="true"/>{t('交换','Swap')}</button>
    <button type="button" className="text-button" disabled={!pair.left&&!pair.right} onClick={()=>{applyPair({left:'',right:''});leftInput.current?.focus({preventScroll:true});}}><X size={15} aria-hidden="true"/>{t('清空对比','Clear comparison')}</button>
   </div>
   {!selected?<p className="company-comparison-status" role="status">{t('选择两家不同的公司后显示对比。','Select two different companies to see their comparison.')}</p>:loading?<p className="company-comparison-status" role="status">{t('正在读取当前范围的面经统计…','Loading report counts for the current filters…')}</p>:!summary||error?<p className="company-comparison-status" role="status">{error?t('当前统计暂不可用，已保留公司选择。可使用页面的重试入口，或切换筛选。','Counts are unavailable. Your company choices are kept. Use the page retry control or change the filters.'):t('暂时没有可供对比的面经统计。已保留选择，请先确认页面的账号与统计状态。','Report counts are not available for comparison yet. Your choices are kept; check the page account and statistics status.')}</p>:!compare?<p className="company-comparison-status" role="status">{t('暂时无法确认公司对比数据，请重试统计。','The company comparison could not be confirmed. Retry the counts.')}</p>:<>
    <div className="company-comparison-totals">{[compare.left,compare.right].map(company=><div key={company.company}><h3><OrganizationLogo id={company.company} size={24}/>{label(company.company)}</h3>{company.reports?counts(company.company,company.reports,company.contributors):<span>{t('当前范围暂无公开面经','No public reports in this view')}</span>}</div>)}</div>
    <div className="company-comparison-views" role="group" aria-label={t('对比知识点范围','Comparison concept scope')}>{(['shared','different','all'] as const).map(value=><button type="button" key={value} aria-pressed={view===value} onClick={()=>{onChange({comparison:value==='shared'?undefined:value});setShown({scope:'',limit:PAGE_SIZE});}}>{({shared:t('共同提及','Mentioned by both'),different:t('仅一方提及','Mentioned by one'),all:t('全部知识点','All concepts')})[value]}<span>{value==='shared'?compare.shared:value==='different'?compare.rows.length-compare.shared:compare.rows.length}</span></button>)}</div>
    <p className="company-comparison-evidence">{t('数量来自同一范围的用户自述，按知识点统计顺序展示；“未见提及”仅表示当前样本没有记录，不代表公司不考，也不构成官方考频。分享者人数不可跨知识点相加。','Counts come from self-reported experiences in the same filtered view, in concept-statistics order. “No mentions” means no record in this sample, not that a company never asks it or an official interview frequency. Contributor counts must not be added across concepts.')}</p>
    {!!catalogError&&<p className="company-comparison-status" role="status">{t('题库名称暂不可用，仍可按知识点标识查看来源面经。','Catalog names are unavailable. Concept identifiers still link to source reports.')}{onRetryCatalog&&<button type="button" className="text-button" onClick={onRetryCatalog}>{t('重试题库','Retry catalog')}</button>}</p>}
    {applied&&<div className="company-comparison-practice"><p>{t('把当前对比范围内的知识点加入练习组，按自己的顺序连续练习。','Add concepts from this comparison to a batch and practice in your own order.')}</p>{progressError&&<p role="status">{t('练习进度暂不可用，仍可编辑练习组；恢复同步后显示完成情况。','Practice progress is unavailable. You can still edit the batch; completion appears after sync recovers.')}{onRetryProgress&&<button type="button" className="text-button" onClick={onRetryProgress}>{t('重试进度','Retry progress')}</button>}</p>}<InterviewPracticeGroup group={group} onRetryCatalog={onRetryCatalog}/></div>}
    {rows.length?<><div className="company-comparison-table" role="region" tabIndex={0} aria-label={t('公司知识点对比表，可横向滚动','Company concept comparison table, horizontally scrollable')}><table><thead><tr><th scope="col">{t('知识点','Concept')}</th><th scope="col">{label(pair.left)}</th><th scope="col">{label(pair.right)}</th></tr></thead><tbody>{rows.slice(0,limit).map(row=>{
     const problem=byId.get(row.id),title=problem?(locale==='en'?problem.translations?.en?.title||problem.title:problem.title):row.id;
     return <tr key={row.id} data-comparison-concept={row.id}><th scope="row"><a href={interviewHref({...filters,view:'questions',company:'',concept:row.id,q:'',saved:false,study:'all',queue:'',page:1})}>{problem&&<small>{problem.number}.</small>} {title}</a>{problem&&applied&&<InterviewGroupToggle problem={problem} name={title} group={group}/>}</th>{(['left','right'] as const).map(side=><td key={side}>{row[side]?counts(pair[side],row[side]!.reports,row[side]!.contributors,row.id):<span className="company-comparison-no-mentions">{t('未见提及','No mentions')}</span>}</td>)}</tr>;
    })}</tbody></table></div><div className="company-comparison-more"><span role="status">{t(`已显示 ${Math.min(limit,rows.length)} / ${rows.length} 个知识点`,`Showing ${Math.min(limit,rows.length)} of ${rows.length} concepts`)}</span>{rows.length>PAGE_SIZE&&<button type="button" onClick={()=>setShown({scope,limit:limit<rows.length?limit+PAGE_SIZE:PAGE_SIZE})}>{limit<rows.length?t(`再显示 ${Math.min(PAGE_SIZE,rows.length-limit)} 个`,`Show ${Math.min(PAGE_SIZE,rows.length-limit)} more`):t(`只显示前 ${PAGE_SIZE} 个`,`Show first ${PAGE_SIZE}`)}</button>}</div></>:<p className="company-comparison-status" role="status">{view==='shared'?t('当前样本没有共同提及的知识点，可切换到全部知识点。','No concepts are mentioned by both in this sample. Try all concepts.'):view==='different'?t('当前关联知识点都被双方提及。','Every linked concept in this sample is mentioned by both.'):t('当前范围尚无这两家公司的关联知识点。','No linked concepts for these companies in the current view.')}</p>}
   </>}
  </div>}
 </details>;
}
