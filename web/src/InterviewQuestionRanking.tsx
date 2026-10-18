import {useEffect,useId,useLayoutEffect,useRef,useState} from 'react';
import {ArrowRight,BookOpen,CheckCircle2,Clock3,RefreshCw} from 'lucide-react';
import {useI18n} from './i18n';
import OrganizationLogo from './OrganizationLogo';
import {interviewCatalog,type InterviewSummary} from './interviewModels';
import InterviewCompanySources from './InterviewCompanySources';
import InterviewSourcePreview from './InterviewSourcePreview';
import {interviewSourceHref} from './interviewSourcePreviewModel';
import {rolePreparationPaths} from './interviewPreparation';
import {interviewHref,type InterviewFilters,type InterviewStudyState} from './interviewNavigation';
import {interviewPracticeHref} from './interviewPracticeBatch';
import InterviewPracticeGroup,{InterviewGroupToggle,useInterviewPracticeGroup} from './InterviewPracticeGroup';
import {interviewConceptMatch,matchingInterviewCompanies,type InterviewMatchField} from './interviewSearch';
import PreparationGuide from './PreparationGuide';
import {interviewPracticeMatch} from './interviewPracticeCriteria';
import {useWorkspaceIntent} from './useWorkspaceIntent';
import {keepReadingTriggerVisible} from './readingViewport';
import {status,type Problem,type Saved} from './types';
import type {ReactNode} from 'react';
import './interview-ranking.css';

type PracticeFilter=InterviewStudyState;
type Props={filters:InterviewFilters;questions:InterviewSummary['questions'];problems:Problem[];saved:Record<string,Saved>;ready:boolean;error?:string;catalogError?:string;onRetryProgress?:()=>void;onRetryCatalog?:()=>void;onClearSearch:()=>void;emptyAction?:ReactNode;owner?:string;accountLoading?:boolean;sourceReturn?:{id:number;concept:string};onSourceReturnReady?:(id:number)=>void;onSourceReturnPending?:(id:number)=>void;onSourceReturnCancel?:(id:number)=>void};

export default function InterviewQuestionRanking({filters,questions,problems,saved,ready:loaded,error,catalogError,onRetryProgress,onRetryCatalog,onClearSearch,emptyAction,owner='guest',accountLoading=false,sourceReturn,onSourceReturnReady,onSourceReturnPending,onSourceReturnCancel}:Props){
 const {t,locale}=useI18n(),listId=useId(),countId=useId();
 const list=useRef<HTMLDivElement>(null),compactControl=useRef<HTMLButtonElement>(null),cancelledReturn=useRef<number>(undefined);
 const activeReturn=sourceReturn&&cancelledReturn.current!==sourceReturn.id?sourceReturn:undefined;
 const source=questions.length?'reports':'curated';
 const curated=source==='curated';
 const scope=interviewHref(filters);
 const [selection,setSelection]=useState<{scope:string;source:typeof source;value:PracticeFilter;limit:number;retained:number}>({scope,source,value:filters.study,limit:6,retained:6});
 const ready=loaded&&!error;
 const selected=selection.scope===scope&&selection.source===source?selection.value:filters.study;
 const choose=(value:PracticeFilter)=>{setSelection({scope,source,value,limit:6,retained:6});location.hash=interviewHref({...filters,study:value,queue:'',page:1}).slice(1);};
 const byId=new Map(problems.map(problem=>[problem.id,problem]));
 const paths=filters.concept?interviewCatalog.paths:rolePreparationPaths(filters.role);
 const curatedIds=[...new Set(paths.flatMap(path=>path.modules))].filter(id=>!filters.concept||id===filters.concept);
 const concepts:{id:string;evidence?:InterviewSummary['questions'][number]}[]=curated?curatedIds.map(id=>({id})):questions.map(evidence=>({id:evidence.problem_id,evidence}));
 const unavailable=concepts.filter(row=>!byId.has(row.id)).length;
 const title=(problem:Problem)=>locale==='en'?problem.translations?.en?.title||problem.title:problem.title;
 const query=filters.q.trim();
 const companyMatches=matchingInterviewCompanies(query);
 const matchLabel=(field:InterviewMatchField)=>({title:t('名称','Title'),module:t('模块','Module'),paper:t('论文','Paper'),goal:t('实现目标','Implementation goal'),company:t('来源公司','Source company')})[field];
 const searched=concepts.flatMap(({id,evidence},index)=>{
  const problem=byId.get(id);
  const matches=interviewConceptMatch(id,problem,evidence,query);
  if(!matches)return [];
  return [{id,evidence,problem,matches,rank:index+1,state:ready&&problem?status(problem,saved[problem.id]):undefined}];
 });
 const criteriaActive=!!filters.difficulty||!!filters.max_minutes;
 const rows=searched.filter(row=>interviewPracticeMatch(row.problem,filters)==='match');
 const unknownCriteria=searched.filter(row=>interviewPracticeMatch(row.problem,filters)==='unknown').length;
 const available=rows.filter(row=>row.problem);
 const progressResolved=ready&&available.length>0;
 const active=progressResolved?selected:'all';
 const limit=selection.scope===scope&&selection.source===source&&selection.value===selected?selection.limit:6;
 const counts={all:rows.length,todo:rows.filter(row=>row.state==='todo').length,started:rows.filter(row=>row.state==='started').length,done:rows.filter(row=>row.state==='done').length};
 const matching=rows.filter(row=>active==='all'||row.state===active);
 const returnIndex=activeReturn?matching.findIndex(row=>row.id===activeReturn.concept):-1;
 const shownLimit=Math.max(limit,returnIndex+1),shown=matching.slice(0,shownLimit);
 const moreCount=Math.min(6,matching.length-shown.length);
 const retainedLimit=selection.scope===scope&&selection.source===source&&selection.value===selected?Math.max(selection.retained,shownLimit):shownLimit;
 const retained=matching.slice(0,retainedLimit),expanded=shown.length>6;
 const intent=useWorkspaceIntent(JSON.stringify([owner,accountLoading,scope,source,active,matching.map(row=>row.id)]));
 const revealFocus=useRef<{target:string;claim:()=>boolean}>(undefined);
 function expand(count:number){
  if(count<=shown.length)return;
  revealFocus.current={target:`interview-ranked-${encodeURIComponent(matching[shown.length].id)}`,claim:intent.create()};
  setSelection({scope,source,value:selected,limit:count,retained:Math.max(retainedLimit,count)});
 }
 function compact(){
  if(!expanded)return;
  if(activeReturn&&cancelledReturn.current!==activeReturn.id){cancelledReturn.current=activeReturn.id;onSourceReturnCancel?.(activeReturn.id);}
  revealFocus.current={target:compactControl.current?.id||'',claim:intent.create()};
  setSelection({scope,source,value:selected,limit:6,retained:retainedLimit});
 }
 // Keep an origin beyond the compact first page mounted after the return is consumed.
 useLayoutEffect(()=>{if(returnIndex>=limit)setSelection({scope,source,value:selected,limit:returnIndex+1,retained:Math.max(retainedLimit,returnIndex+1)});},[returnIndex,limit,scope,source,selected,retainedLimit]);
 const returnReady=useRef(onSourceReturnReady);returnReady.current=onSourceReturnReady;
 useEffect(()=>{
  if(activeReturn&&returnIndex<0&&(curated||(!unavailable||!!problems.length||!!catalogError)&&(selected==='all'||ready||!!error)))returnReady.current?.(activeReturn.id);
 },[activeReturn?.id,activeReturn?.concept,returnIndex,curated,unavailable,problems.length,catalogError,selected,ready,error]);
 useLayoutEffect(()=>{
  const request=revealFocus.current;if(!request)return;revealFocus.current=undefined;
  const target=document.getElementById(request.target);
  if(target&&list.current?.contains(target)&&!target.closest('[hidden]')&&request.claim()){target.focus({preventScroll:true});keepReadingTriggerVisible(target);}
  else intent.cancel();
 },[selection.limit,shown.length,active,source]);
 const resume=ready&&active!=='done'?(matching.filter(row=>row.state==='started').sort((a,b)=>(saved[b.problem!.id]?.updated||'').localeCompare(saved[a.problem!.id]?.updated||''))[0]||matching.find(row=>row.state==='todo'))?.problem:undefined;
 const candidates=matching.flatMap(row=>row.problem?[row.problem]:[]);
 const group=useInterviewPracticeGroup({filters:{...filters,study:selected},source,problems,candidates,filterPending:selected!=='all'&&!ready,saved,ready,owner});
 const practice=(problem:Problem)=>group.href(problem)||interviewPracticeHref(problem,{...filters,study:selected},candidates);
 const stateLabel=(state:typeof rows[number]['state'])=>state==='done'?t('已通过','Passed'):state==='started'?t('进行中','In progress'):state==='todo'?t('未开始','Not started'):t('进度未就绪','Progress not ready');
 return <div ref={list} className="interview-question-workbench">
  {!!companyMatches.length&&<section className="interview-search-companies" aria-label={t('按搜索到的公司准备','Prepare for matching companies')}><span>{t('按公司准备','Prepare by company')}</span><div>{companyMatches.slice(0,4).map(company=><a key={company.id} href={interviewHref({...filters,company:company.id,q:'',queue:'',page:1})}><OrganizationLogo id={company.id} size={16}/>{locale==='en'?company.name_en:company.name}<ArrowRight size={12}/></a>)}{companyMatches.length>4&&<a href={interviewHref({...filters,view:'companies',company:'',concept:'',stage:'',saved:false,queue:'',page:1})}>{t(`查看全部 ${companyMatches.length} 家公司`,`View all ${companyMatches.length} companies`)}<ArrowRight size={12}/></a>}</div><small>{t('打开该公司的面经、官方资料与岗位准备题单。','Open the company’s experiences, official resources and role-based practice.')}</small></section>}
  {curated&&<div className="interview-curated-context"><span className="interview-panel-kicker"><BookOpen size={14}/>{t('编辑准备题单','CURATED PRACTICE')}</span><h3>{t('从这些知识点开始准备','Start with these practice concepts')}</h3><p>{t('当前范围尚无关联练习的面经样本。以下题单按岗位与知识体系编排；真实分享积累后，会显示对应知识点的社区提及统计。','No reports in this view link to a practice concept yet. These exercises follow the role curriculum. Real experiences will gradually build the community mention statistics.')}</p>{emptyAction}</div>}
  {unavailable>0&&<p className="interview-progress-status" role="status">{curated?t('部分练习信息尚未加载，搜索和进度汇总可能不完整。','Some practice details have not loaded. Search and progress totals may be incomplete.'):catalogError?t('部分练习信息暂不可用，搜索结果可能不完整；清除搜索后仍可阅读全部来源面经。','Some practice details are unavailable and search may be incomplete. Clear search to read all source reports.'):!problems.length?t('正在加载练习信息，来源面经仍可阅读。','Loading practice details. Source reports remain readable.'):t(`${unavailable} 个知识点暂未匹配到练习，仍保留它们的来源。`,`${unavailable} concepts have no available practice module. Their sources are kept here.`)} {catalogError&&onRetryCatalog&&<button type="button" className="text-button" onClick={onRetryCatalog}><RefreshCw size={12}/>{t('重试题库','Retry catalog')}</button>}</p>}
  {unknownCriteria>0&&<p className="interview-progress-status" role="status">{t(`${unknownCriteria} 个知识点缺少当前筛选所需的练习信息，暂未计入结果。清除难度与时长条件可继续查看其来源。`,`${unknownCriteria} concepts lack the metadata needed for these practice filters and are not included. Clear difficulty and duration to read their sources.`)}</p>}
  {!!rows.length&&<>
   <div className="interview-ranking-toolbar">
    <div><b>{curated?t('把准备变成可见进度','Make your preparation visible'):t('把高频变成准备进度','Turn concepts into preparation')}</b><p>{curated?t('按学习状态选择练习，自测、实现，再用测试验证。','Choose a learning state, explain your answer and verify it through code.'):t('筛选你的学习状态；榜单仍按社区分享者人数排序。','Filter your learning state; the ranking keeps its community contributor order.')}</p></div>
    {progressResolved&&!(query&&unavailable)&&!unknownCriteria&&<div className="interview-ranking-completion"><span>{t('已通过','Passed')} <b>{counts.done} / {rows.length}</b></span><progress aria-label={t('当前知识点练习进度','Practice progress for these concepts')} value={counts.done} max={rows.length}/></div>}
   </div>
   {!ready&&<p className="interview-progress-status" role="status">{error?curated?t('学习进度暂不可用，仍可自测和打开练习。','Learning progress is unavailable. Study prompts and practice remain available.'):t('学习进度暂不可用。仍可查看来源、自测和打开练习。','Learning progress is unavailable. Sources, study prompts and practice remain available.'):t('正在读取学习进度，暂不按状态筛选。','Loading learning progress before filtering by state.')} {error&&onRetryProgress&&<button type="button" className="text-button" onClick={onRetryProgress}>{t('重试学习进度','Retry learning progress')}</button>}</p>}
   <div className="interview-ranking-filters" role="group" aria-label={t('按学习状态筛选知识点','Filter concepts by learning state')}>{(['all','todo','started','done'] as const).map(value=><button key={value} type="button" disabled={value!=='all'&&!progressResolved} aria-pressed={selected===value} onClick={()=>choose(value)}><span>{value==='all'?t('全部知识点','All concepts'):stateLabel(value)}</span><small>{value==='all'||progressResolved?counts[value]:'—'}</small></button>)}</div>
   {resume&&<a className="interview-ranking-resume" href={practice(resume)}><span><Clock3 size={17}/><span><b>{status(resume,saved[resume.id])==='started'?t('继续当前范围的草稿','Resume a draft in this view'):t('开始当前范围的练习','Start practicing in this view')}</b><small>{title(resume)}</small></span></span><ArrowRight size={16}/></a>}
   <p id={countId} className="interview-ranking-result-count" role="status">{selected!=='all'&&!progressResolved?t('正在等待学习进度，筛选条件已保留。','Waiting for learning progress. Your filter is kept.'):t(`显示 ${shown.length} / ${matching.length} 个知识点`,`Showing ${shown.length} of ${matching.length} ${matching.length===1?'concept':'concepts'}`)}{selected!=='all'&&<button type="button" className="text-button" onClick={()=>choose('all')}>{t('显示全部状态','Show all learning states')}</button>}</p>
  </>}
  {(candidates.length>0||group.numbers.length>0||!!filters.queue||!!group.change)&&<InterviewPracticeGroup group={group} onRetryCatalog={onRetryCatalog}/>}
  {matching.length>6&&<div className="interview-ranking-display" role="group" aria-label={t('知识点列表显示','Concept list display')}><button type="button" aria-controls={listId} aria-describedby={countId} aria-disabled={shown.length===matching.length} onClick={()=>expand(matching.length)}>{t(`展开全部 ${matching.length} 个`,`Show all ${matching.length}`)}</button><button ref={compactControl} id={`${listId}-compact`} type="button" aria-controls={listId} aria-describedby={countId} aria-disabled={!expanded} onClick={compact}>{t('只显示前 6 个','Show first 6')}</button></div>}
  {shown.length?<div id={listId} className="interview-ranking-list">{retained.map(({id,evidence,problem,matches,rank,state},index)=>{
   const name=problem?title(problem):id;
   const sources=interviewSourceHref(filters,id);
   const heading=`interview-ranked-${encodeURIComponent(id)}`;
   return <article key={id} className="interview-ranking-row" aria-labelledby={heading} hidden={index>=shown.length}>
    <span className="interview-rank" aria-label={curated?t('编辑推荐练习','Curated exercise'):t(`社区排序第 ${rank} 位`,`Community rank ${rank}`)}>{curated?<BookOpen size={14}/>:String(rank).padStart(2,'0')}</span>
    <div className="interview-ranking-copy"><div className="interview-ranking-title">{problem?<a id={heading} href={practice(problem)}>{name}</a>:<b id={heading} tabIndex={-1}>{name}</b>}{problem&&<span className={`interview-ranking-state ${state||''}`}>{state==='done'?<CheckCircle2 size={12}/>:state==='started'?<Clock3 size={12}/>:null}{stateLabel(state)}</span>}</div>
     {!!matches.length&&<p className="interview-search-match"><span>{t('匹配：','Matched: ')}</span>{matches.map(matchLabel).join(' · ')}{matches.includes('paper')&&problem?.paper?.title&&<span className="interview-search-paper">{problem.paper.title}</span>}{evidence&&matches.includes('goal')&&problem&&<span className="interview-search-paper">{locale==='en'?problem.module.goal_en||problem.module.goal:problem.module.goal}</span>}</p>}
     {problem&&<p className="interview-ranking-module-meta">{[({E:t('简单','Easy'),M:t('中等','Medium'),H:t('困难','Hard')} as Record<string,string>)[problem.difficulty],problem.estimated_minutes?`${problem.estimated_minutes} min`:''].filter(Boolean).join(' · ')}</p>}
     {evidence?<InterviewCompanySources companies={evidence.companies} problemId={id} name={name} filters={filters}/>:problem&&<p className="interview-curated-goal">{locale==='en'?problem.module.goal_en||problem.module.goal:problem.module.goal}</p>}
    </div>
    <div className="interview-ranking-actions">{!evidence&&<small className="interview-curated-label">{t('编辑推荐','Curated practice')}</small>}{problem?<><a className="button subtle" href={practice(problem)}>{state==='done'?t('复习','Review'):state==='started'?t('继续练习','Resume practice'):state==='todo'?t('开始练习','Start practice'):t('打开练习','Open practice')}<ArrowRight size={13}/></a><InterviewGroupToggle problem={problem} name={name} group={group}/></>:<small>{t('练习暂不可用','Practice unavailable')}</small>}</div>
    {evidence&&<InterviewSourcePreview concept={id} name={name} reports={evidence.reports} contributors={evidence.contributors} sourceHref={sources} returnTo={scope} owner={owner} accountLoading={accountLoading} returnId={activeReturn?.concept===id?activeReturn.id:undefined} onReturnReady={onSourceReturnReady} onReturnPending={onSourceReturnPending} onReturnCancel={onSourceReturnCancel}/>}
    {problem&&<PreparationGuide problemId={problem.id} practiceHref={practice(problem)}/>}
   </article>;
  })}</div>:criteriaActive&&searched.length>0&&!rows.length?<div className="interview-empty"><BookOpen size={27}/><h3>{t('没有符合难度与时长的知识点','No concepts match this difficulty and duration')}</h3><p>{t('可以放宽本列表的练习条件，继续查看当前公司、搜索与学习状态下的知识点。','Relax the practice filters to keep browsing concepts within the current company, search and learning state.')}</p><a className="button subtle" href={interviewHref({...filters,difficulty:'',max_minutes:'',queue:'',page:1})}>{t('清除难度与时长','Clear difficulty and duration')}</a></div>:<div className="interview-empty"><BookOpen size={27}/><h3>{rows.length?t('当前状态下没有知识点','No concepts with this learning state'):filters.q&&unavailable?t('练习信息未齐，搜索暂不完整','Search is waiting for practice details'):filters.q?t('没有匹配的知识点','No matching concepts'):t('当前范围暂无编辑推荐练习','No curated exercises in this view')}</h3><p>{rows.length?curated?t('可以切换学习状态，继续准备其他知识点。','Switch learning states to prepare another concept.'):t('可以切换学习状态，继续阅读来源面经或准备其他知识点。','Switch learning states to read the source experiences or prepare another concept.'):filters.q&&unavailable?curated?t('清除搜索可浏览编辑题单，或重试加载练习信息后再搜索。','Clear search to browse the curated exercises, or retry practice details before searching again.'):t('清除搜索可阅读已汇总的全部来源，或重试加载练习信息后再搜索。','Clear search to read all available sources, or retry practice details before searching again.'):t('可以调整岗位或知识点筛选，或继续浏览下方的准备路径。','Adjust the role or concept filters, or browse the preparation paths below.')}</p>{rows.length?<button type="button" className="button subtle" onClick={()=>choose('all')}>{t('显示全部状态','Show all learning states')}</button>:filters.q?<button type="button" className="button subtle" onClick={onClearSearch}>{t('清空搜索','Clear search')}</button>:emptyAction}</div>}
  {matching.length>6&&<div className="interview-ranking-display interview-ranking-footer" role="group" aria-label={t('继续浏览知识点','Continue browsing concepts')}>{moreCount>0&&<button type="button" className="interview-ranking-more" aria-controls={listId} aria-describedby={countId} onClick={()=>expand(shown.length+6)}>{t(`再显示 ${moreCount} 个知识点`,`Show ${moreCount} more ${moreCount===1?'concept':'concepts'}`)}<ArrowRight size={14} aria-hidden="true"/></button>}<button type="button" aria-controls={listId} aria-describedby={countId} aria-disabled={!expanded} onClick={compact}>{t('只显示前 6 个','Show first 6')}</button></div>}
 </div>;
}
