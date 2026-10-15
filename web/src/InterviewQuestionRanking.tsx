import {useState} from 'react';
import {ArrowRight,BookOpen,CheckCircle2,Clock3,RefreshCw} from 'lucide-react';
import {useI18n} from './i18n';
import {interviewLabel,type InterviewSummary} from './interviewModels';
import {interviewHref,type InterviewFilters} from './interviewNavigation';
import {withLibraryContext} from './libraryNavigation';
import PreparationGuide from './PreparationGuide';
import {status,type Problem,type Saved} from './types';
import type {ReactNode} from 'react';
import './interview-ranking.css';

type PracticeFilter='all'|'todo'|'started'|'done';
type Props={filters:InterviewFilters;questions:InterviewSummary['questions'];problems:Problem[];saved:Record<string,Saved>;ready:boolean;error?:string;catalogError?:string;onRetryProgress?:()=>void;onRetryCatalog?:()=>void;onClearSearch:()=>void;emptyAction?:ReactNode};

export default function InterviewQuestionRanking({filters,questions,problems,saved,ready:loaded,error,catalogError,onRetryProgress,onRetryCatalog,onClearSearch,emptyAction}:Props){
 const {t,locale}=useI18n();
 const [choice,setChoice]=useState<PracticeFilter>('all');
 const ready=loaded&&!error,active=ready?choice:'all';
 const byId=new Map(problems.map(problem=>[problem.id,problem]));
 const unavailable=questions.filter(row=>!byId.has(row.problem_id)).length;
 const title=(problem:Problem)=>locale==='en'?problem.translations?.en?.title||problem.title:problem.title;
 const query=filters.q.toLocaleLowerCase();
 const rows=questions.flatMap((evidence,index)=>{
  const problem=byId.get(evidence.problem_id);
  if(query&&!`${evidence.problem_id} ${problem?.title||''} ${problem?.translations?.en?.title||''}`.toLocaleLowerCase().includes(query))return [];
  return [{evidence,problem,rank:index+1,state:ready&&problem?status(problem,saved[problem.id]):undefined}];
 });
 const available=rows.filter(row=>row.problem);
 const counts={all:rows.length,todo:rows.filter(row=>row.state==='todo').length,started:rows.filter(row=>row.state==='started').length,done:rows.filter(row=>row.state==='done').length};
 const shown=rows.filter(row=>active==='all'||row.state===active);
 const resume=ready&&active!=='done'?(shown.filter(row=>row.state==='started').sort((a,b)=>(saved[b.problem!.id]?.updated||'').localeCompare(saved[a.problem!.id]?.updated||''))[0]||shown.find(row=>row.state==='todo'))?.problem:undefined;
 const practice=(problem:Problem)=>withLibraryContext(problem.number,interviewHref(filters));
 const stateLabel=(state:typeof rows[number]['state'])=>state==='done'?t('已通过','Passed'):state==='started'?t('进行中','In progress'):state==='todo'?t('未开始','Not started'):t('进度未就绪','Progress not ready');
 return <div className="interview-question-workbench">
  {unavailable>0&&<p className="interview-progress-status" role="status">{catalogError?t('部分练习信息暂不可用，搜索结果可能不完整；清除搜索后仍可阅读全部来源面经。','Some practice details are unavailable and search may be incomplete. Clear search to read all source reports.'):!problems.length?t('正在加载练习信息，来源面经仍可阅读。','Loading practice details. Source reports remain readable.'):t(`${unavailable} 个知识点暂未匹配到练习，仍保留它们的来源。`,`${unavailable} concepts have no available practice module. Their sources are kept here.`)} {catalogError&&onRetryCatalog&&<button type="button" className="text-button" onClick={onRetryCatalog}><RefreshCw size={12}/>{t('重试题库','Retry catalog')}</button>}</p>}
  {!!rows.length&&<>
   <div className="interview-ranking-toolbar">
    <div><b>{t('把高频变成准备进度','Turn concepts into preparation')}</b><p>{t('筛选你的学习状态；榜单仍按社区分享者人数排序。','Filter your learning state; the ranking keeps its community contributor order.')}</p></div>
    {ready&&available.length>0&&<div className="interview-ranking-completion"><span>{t('已通过','Passed')} <b>{counts.done} / {available.length}</b></span><progress aria-label={t('当前知识点练习进度','Practice progress for these concepts')} value={counts.done} max={available.length}/></div>}
   </div>
   {!ready&&<p className="interview-progress-status" role="status">{error?t('学习进度暂不可用。仍可查看来源、自测和打开练习。','Learning progress is unavailable. Sources, study prompts and practice remain available.'):t('正在读取学习进度，暂不按状态筛选。','Loading learning progress before filtering by state.')} {error&&onRetryProgress&&<button type="button" className="text-button" onClick={onRetryProgress}>{t('重试学习进度','Retry learning progress')}</button>}</p>}
   <div className="interview-ranking-filters" role="group" aria-label={t('按学习状态筛选知识点','Filter concepts by learning state')}>{(['all','todo','started','done'] as const).map(value=><button key={value} type="button" disabled={value!=='all'&&!ready} aria-pressed={active===value} onClick={()=>setChoice(value)}><span>{value==='all'?t('全部知识点','All concepts'):stateLabel(value)}</span><small>{value==='all'||ready?counts[value]:'—'}</small></button>)}</div>
   {resume&&<a className="interview-ranking-resume" href={practice(resume)}><span><Clock3 size={17}/><span><b>{status(resume,saved[resume.id])==='started'?t('继续当前范围的草稿','Resume a draft in this view'):t('开始当前范围的练习','Start practicing in this view')}</b><small>{title(resume)}</small></span></span><ArrowRight size={16}/></a>}
   <p className="interview-ranking-result-count" role="status">{t(`显示 ${shown.length} / ${rows.length} 个知识点`,`Showing ${shown.length} of ${rows.length} concepts`)}{active!=='all'&&<button type="button" className="text-button" onClick={()=>setChoice('all')}>{t('显示全部状态','Show all learning states')}</button>}</p>
  </>}
  {shown.length?<div className="interview-ranking-list">{shown.map(({evidence,problem,rank,state})=>{
   const name=problem?title(problem):evidence.problem_id;
   const sources=interviewHref({...filters,view:'reports',page:1,q:'',concept:evidence.problem_id,saved:false});
   const heading=`interview-ranked-${encodeURIComponent(evidence.problem_id)}`;
   return <article key={evidence.problem_id} className="interview-ranking-row" aria-labelledby={heading}>
    <span className="interview-rank" aria-label={t(`社区排序第 ${rank} 位`,`Community rank ${rank}`)}>{String(rank).padStart(2,'0')}</span>
    <div className="interview-ranking-copy"><div className="interview-ranking-title">{problem?<a id={heading} href={practice(problem)}>{name}</a>:<b id={heading}>{name}</b>}{problem&&<span className={`interview-ranking-state ${state||''}`}>{state==='done'?<CheckCircle2 size={12}/>:state==='started'?<Clock3 size={12}/>:null}{stateLabel(state)}</span>}</div>
     {problem&&<p className="interview-ranking-module-meta">{[({E:t('简单','Easy'),M:t('中等','Medium'),H:t('困难','Hard')} as Record<string,string>)[problem.difficulty],problem.estimated_minutes?`${problem.estimated_minutes} min`:''].filter(Boolean).join(' · ')}</p>}
     <div className="interview-company-tags">{evidence.companies.map(company=><a key={company.company} href={interviewHref({...filters,view:'reports',company:company.company,concept:evidence.problem_id,q:'',page:1,saved:false})} aria-label={t(`阅读${interviewLabel('companies',company.company,false)}关于${name}的 ${company.reports} 篇来源面经`,`Read ${company.reports} ${interviewLabel('companies',company.company,true)} source reports about ${name}`)}>{interviewLabel('companies',company.company,locale==='en')}<small>{company.reports}</small></a>)}</div>
    </div>
    <div className="interview-ranking-actions"><span className="interview-ranking-evidence"><span><b>{evidence.contributors}</b> {t('位分享者','contributors')}</span><a href={sources}>{t(`查看 ${evidence.reports} 篇来源`,`Read ${evidence.reports} source ${evidence.reports===1?'report':'reports'}`)}</a></span>{problem?<a className="button subtle" href={practice(problem)}>{state==='done'?t('复习','Review'):state==='started'?t('继续练习','Resume practice'):state==='todo'?t('开始练习','Start practice'):t('打开练习','Open practice')}<ArrowRight size={13}/></a>:<small>{t('练习暂不可用','Practice unavailable')}</small>}</div>
    {problem&&<PreparationGuide problemId={problem.id} practiceHref={practice(problem)}/>}
   </article>;
  })}</div>:<div className="interview-empty"><BookOpen size={27}/><h3>{rows.length?t('当前状态下没有知识点','No concepts with this learning state'):filters.q&&unavailable?t('练习信息未齐，搜索暂不完整','Search is waiting for practice details'):filters.q?t('没有匹配的知识点','No matching concepts'):t('积累真实样本，再谈高频','Real reports come before frequency')}</h3><p>{rows.length?t('可以切换学习状态，继续阅读来源面经或准备其他知识点。','Switch learning states to read the source experiences or prepare another concept.'):filters.q&&unavailable?t('清除搜索可阅读已汇总的全部来源，或重试加载练习信息后再搜索。','Clear search to read all available sources, or retry practice details before searching again.'):t('当前范围还没有面经关联的知识点。可以先按下方准备路径练习，或分享你的面经。','No concepts are linked by reports in this view yet. Start with a preparation path below, or share your experience.')}</p>{rows.length?<button type="button" className="button subtle" onClick={()=>setChoice('all')}>{t('显示全部状态','Show all learning states')}</button>:filters.q?<button type="button" className="button subtle" onClick={onClearSearch}>{t('清空搜索','Clear search')}</button>:emptyAction}</div>}
 </div>;
}
