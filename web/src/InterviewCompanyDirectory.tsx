import {controlIsActive} from './activeControls';
import {CompanyTargetButton,useCompanyTargetsDirectory} from './CompanyTargets';
import AppLink from './AppLink';
import InterviewCompanyComparison from './InterviewCompanyComparison';
import type {Problem,Saved} from './types';
import {useId,useRef,useLayoutEffect} from 'react';
import type {MouseEvent} from 'react';
import {companyPracticeProgress,type CompanyPracticeProgress} from './companyPracticeProgress';
import {ArrowRight,Flame,MessageCircle} from 'lucide-react';
import {useI18n} from './i18n';
import OrganizationLogo from './OrganizationLogo';
import {interviewCatalog,type InterviewSummary} from './interviewModels';
import {interviewHref,type InterviewFilters,type InterviewCompanyOrder} from './interviewNavigation';
import {companyResources} from './interviewPreparation';
import {matchesInterviewCompany} from './interviewSearch';
import {useWorkspaceIntent} from './useWorkspaceIntent';
import {revealReadingTarget} from './readingViewport';
import './interview-company-directory.css';

export default function InterviewCompanyDirectory({canAct,filters,summary,loading,error,onOrder,onClearSearch,problems,catalogError,owner='guest',onCompare,saved,progressReady,progressError,onRetryProgress,onRetryCatalog,onRetrySummary,communityAvailable=true}:{canAct?:()=>boolean;filters:InterviewFilters;summary?:InterviewSummary;loading:boolean;error?:string;onOrder:(order:InterviewCompanyOrder)=>void;onClearSearch:()=>void;problems?:Problem[];catalogError?:string;owner?:string;onCompare:(update:Partial<InterviewFilters>)=>void;saved?:Record<string,Saved>;progressReady?:boolean;progressError?:string;onRetryProgress?:()=>void;onRetryCatalog?:()=>void;onRetrySummary?:()=>void;communityAvailable?:boolean}){
 const {t,locale}=useI18n(),hint=useId(),hasCounts=communityAvailable&&!!summary,orderControl=useRef<HTMLSelectElement>(null),root=useRef<HTMLElement>(null),listId=useId();
 const targets=useCompanyTargetsDirectory(owner),targetToggle=useRef<HTMLInputElement>(null);
 const onlyTargets=!!targets?.only;
 const targetIds=new Set(targets?.state.snapshot?.items.map(row=>row.company_id)||[]);
 const targetUnconfirmed=onlyTargets&&(!targets?.state.snapshot||!!targets.state.error||targets.state.accountError);
 const intent=useWorkspaceIntent(JSON.stringify([owner,interviewHref(filters)]));
 const removing=useRef<{company:string;trigger:HTMLButtonElement;claim:()=>boolean}|undefined>(undefined);
 useLayoutEffect(()=>{
  if(!onlyTargets){removing.current=undefined;intent.cancel();return;}
  const pending=removing.current;if(!pending||targetIds.has(pending.company))return;
  removing.current=undefined;const allowed=pending.claim();
  if(allowed&&!targetUnconfirmed&&!pending.trigger.isConnected&&canUse()&&controlIsActive(targetToggle.current)){
   targetToggle.current!.focus({preventScroll:true});revealReadingTarget(targetToggle.current!,'nearest');
  }
 },[owner,onlyTargets,targets?.state.snapshot]);
 const preparation=companyPracticeProgress(communityAvailable&&!loading&&!error?summary:undefined,problems||[],saved||{},!!progressReady&&!progressError,!!catalogError);
 const name=(company:typeof interviewCatalog.companies[number])=>locale==='en'?company.name_en:company.name;
 const rows=interviewCatalog.companies.filter(company=>matchesInterviewCompany(company,filters.q)&&(!onlyTargets||!targetUnconfirmed&&targetIds.has(company.id))).map((company,index)=>({company,index,evidence:communityAvailable?summary?.companies.find(row=>row.company===company.id):undefined}));
 const requested=filters.company_order,waiting=(requested==='contributors'||requested==='reports')&&!hasCounts,active=waiting?'catalog':requested;
 rows.sort((a,b)=>{
  if(active==='contributors')return (b.evidence?.contributors||0)-(a.evidence?.contributors||0)||(b.evidence?.reports||0)-(a.evidence?.reports||0)||a.index-b.index;
  if(active==='reports')return (b.evidence?.reports||0)-(a.evidence?.reports||0)||(b.evidence?.contributors||0)-(a.evidence?.contributors||0)||a.index-b.index;
  return active==='name'?name(a.company).localeCompare(name(b.company),locale==='en'?'en':'zh-CN')||a.index-b.index:a.index-b.index;
 });
 const linked=rows.filter(row=>preparation?.has(row.company.id));
 const incomplete=linked.some(row=>preparation?.get(row.company.id)?.phase==='catalog');
 function canUse(trigger?:HTMLElement){return controlIsActive(root.current)&&canAct?.()!==false&&(!trigger||controlIsActive(trigger)&&!!root.current?.contains(trigger));}
 function guardLink(event:MouseEvent<HTMLAnchorElement>){if(!canUse(event.currentTarget))event.preventDefault();}
 function recover(event:MouseEvent<HTMLButtonElement>,callback?:()=>void){if(!canUse(event.currentTarget))return;if(document.activeElement===event.currentTarget)orderControl.current?.focus({preventScroll:true});callback?.();}
 function changeTargets(only:boolean){removing.current=undefined;intent.cancel();targets?.setOnly(only);}
 function showAll(event:MouseEvent<HTMLButtonElement>){if(!canUse(event.currentTarget))return;(targets?.state.snapshot?targetToggle.current:orderControl.current)?.focus({preventScroll:true});changeTargets(false);}
 function captureRemoval(event:MouseEvent<HTMLElement>){
  const ElementType=event.currentTarget.ownerDocument.defaultView?.Element;
  if(!onlyTargets||!canUse()||!ElementType||!(event.target instanceof ElementType))return;
  const trigger=event.target.closest<HTMLButtonElement>('button[data-company-target][aria-pressed="true"]');
  if(trigger&&trigger===document.activeElement&&trigger.getAttribute('aria-busy')!=='true')removing.current={company:trigger.dataset.companyTarget!,trigger,claim:intent.create()};
 }
 return <section ref={root} className="interview-company-directory" onClickCapture={captureRemoval} aria-label={t('公司准备索引','Company preparation directory')}>
  <div className="company-directory-toolbar"><p role="status" aria-atomic="true">{targetUnconfirmed?t('目标公司待确认','Target companies unconfirmed'):onlyTargets?t(`共 ${rows.length} 家目标公司`,`${rows.length} target ${rows.length===1?'company':'companies'}`):t(`共 ${rows.length} 家公司`,`${rows.length} ${rows.length===1?'company':'companies'}`)}</p><div className="company-directory-controls">{targets&&<label className="company-directory-target-toggle" data-selected={onlyTargets||undefined}><input ref={targetToggle} type="checkbox" checked={onlyTargets} disabled={!onlyTargets&&!targets.state.snapshot} aria-controls={listId} onChange={event=>{if(canUse(event.currentTarget))changeTargets(event.currentTarget.checked);}}/><span>{t('仅看目标公司','My target companies only')}</span><small>{targets.state.snapshot?targetIds.size:'—'}</small></label>}<label><span>{t('排序','Order')}</span><select ref={orderControl} aria-label={t('公司索引排序','Company directory order')} aria-describedby={hint} aria-controls={listId} value={requested} onChange={event=>{if(canUse(event.currentTarget))onOrder(event.target.value as InterviewCompanyOrder);}}><option value="catalog">{t('默认顺序','Default order')}</option><option value="contributors" disabled={!hasCounts}>{t('分享者最多','Most contributors')}</option><option value="reports" disabled={!hasCounts}>{t('面经最多','Most experiences')}</option><option value="name">{t('公司名称','Company name')}</option></select></label></div></div>
  <p id={hint} className="company-directory-hint">{!communityAvailable?waiting?t('分享统计排序会在账号应用中生效，当前按默认顺序展示；已保留选择。','Your report-count order is kept for the account app. This view uses the default order.'):t('浏览公司官方资料与准备路径；社区分享统计在账号应用中提供。','Browse official resources and preparation paths. Community counts are available in the account app.'):waiting?loading?t('正在等待面经统计，暂按默认顺序展示；已保留所选排序。','Waiting for report counts. Showing the default order until your chosen order is ready.'):t('面经统计暂不可用，暂按默认顺序展示；已保留所选排序。','Report counts are unavailable. Showing the default order and keeping your chosen order.'):t('分享统计沿用当前筛选；分享者与面经数量都相同时，沿用默认顺序。','Counts follow the current filters. Ties in both counts keep the default order.')}{!waiting&&error&&<> {t('统计暂不可用，仍可浏览公司与准备资料。','Counts are unavailable. Company preparation remains available.')}</>}</p>
  {communityAvailable&&rows.length>0&&summary&&!loading&&!error&&!preparation&&<div className="company-directory-recovery" role="status"><span>{t('暂时无法确认关联练习统计。','Linked practice counts could not be confirmed.')}</span>{onRetrySummary&&<button type="button" className="text-button" onClick={event=>recover(event,onRetrySummary)}>{t('重试面经统计','Retry report counts')}</button>}</div>}
  {linked.length>0&&<div className="company-directory-learning"><p>{t('个人练习进度 · 按当前面经范围的关联题目汇总','Your practice · Modules linked in this report view')}</p>{incomplete&&<div className="company-directory-recovery" role="status"><span>{catalogError?t('题库信息暂不可用，练习进度暂不汇总。','Catalog details are unavailable. Practice totals are not ready.'):t('部分关联练习信息未加载，练习进度暂不汇总。','Some linked modules have not loaded. Practice totals are not ready.')}</span>{onRetryCatalog&&(!!catalogError||!!problems?.length)&&<button type="button" className="text-button" onClick={event=>recover(event,onRetryCatalog)}>{t('重试题库','Retry catalog')}</button>}</div>}{(!progressReady||!!progressError)&&<div className="company-directory-recovery" role="status"><span>{progressError?t('学习进度暂不可用，仍可查看关联练习。','Learning progress is unavailable. Linked modules remain available.'):t('正在同步学习进度…','Syncing learning progress…')}</span>{progressError&&onRetryProgress&&<button type="button" className="text-button" onClick={event=>recover(event,onRetryProgress)}>{t('重试进度','Retry progress')}</button>}</div>}</div>}
  {communityAvailable&&<InterviewCompanyComparison key={owner} owner={owner} filters={filters} summary={summary} loading={loading} error={error} problems={problems} catalogError={catalogError} onChange={onCompare} saved={saved} progressReady={progressReady} progressError={progressError} onRetryProgress={onRetryProgress} onRetryCatalog={onRetryCatalog}/>}
  <div id={listId} className="company-directory-results" role="region" aria-label={t('公司准备结果','Company preparation results')} aria-busy={communityAvailable&&loading||onlyTargets&&!!targets?.state.loading}>{targetUnconfirmed?<div className="interview-empty company-directory-target-state" role="status"><h3>{targets?.state.loading?t('正在同步目标公司…','Syncing target companies…'):t('暂时无法确认目标公司','Target companies could not be confirmed')}</h3><p>{t('目标名单恢复后再展示公司结果，你也可以先浏览全部公司。','Company results will return when your targets are confirmed. You can browse all companies in the meantime.')}</p><div><button type="button" className="button subtle" onClick={event=>{if(canUse(event.currentTarget))targets?.open();}}>{t('查看目标公司状态','View target-company status')}</button><button type="button" className="text-button" onClick={showAll}>{t('查看全部公司','Show all companies')}</button></div></div>:rows.length?<div className="interview-company-grid">{rows.map(({company,evidence})=>{
   const title=name(company),resources=companyResources(company.id),concepts=communityAvailable&&summary?new Set(summary.questions.filter(row=>row.companies.some(source=>source.company===company.id&&source.reports>0&&source.contributors>0)).map(row=>row.problem_id)).size:undefined;
   const href=(view:InterviewFilters['view'])=>interviewHref({...filters,company:company.id,view,q:'',page:1,saved:false});
   const progress=preparation?.get(company.id),complete=progress?.phase==='ready'&&progress.done===progress.total;
   const practiceHref=interviewHref({...filters,company:company.id,view:'questions',q:'',page:1,saved:false,queue:'',study:complete?'done':'all'});
   const practiceLabel=!communityAvailable?t('准备知识点','Practice concepts'):progress?.phase==='ready'?complete?t('复习关联练习','Review practice'):progress.done+progress.started>0?t('继续准备','Continue practice'):t('开始准备','Start practice'):t('知识点与练习','Concepts & practice');
   return <article key={company.id} className="interview-company-entry" data-company={company.id}>
    <AppLink onClick={guardLink} href={href('companies')} className="interview-company-card"><span className="interview-company-mark" aria-hidden="true"><OrganizationLogo id={company.id} size={32}/></span><b>{title}</b><small>{!communityAvailable?t('查看官方资料与推荐路径','Explore official resources and curated paths'):evidence&&evidence.reports>0?t(`${evidence.reports} 篇面经 · ${evidence.contributors} 位分享者`,`${evidence.reports} ${evidence.reports===1?'report':'reports'} · ${evidence.contributors} ${evidence.contributors===1?'contributor':'contributors'}`):hasCounts?t('期待第一份分享','Awaiting the first experience'):t('查看面经与准备资料','Explore experiences and resources')}</small>{concepts!==undefined&&concepts>0&&<small className="company-directory-concepts">{t(`${concepts} 个关联知识点`,`${concepts} linked ${concepts===1?'concept':'concepts'}`)}</small>}<span className="interview-company-resource-count">{resources.length?t(`${resources.length} 份官方资料`,`${resources.length} official resources`):t('按岗位开始准备','Prepare by role')}</span><ArrowRight size={15} aria-hidden="true"/></AppLink>
    {progress&&<CompanyProgress value={progress} title={title} id={`${hint}-practice-${company.id}`}/>}<nav className="company-directory-actions" aria-label={t(`${title}的准备入口`,`Preparation entries for ${title}`)}>{communityAvailable&&<AppLink onClick={guardLink} data-company-view="reports" href={href('reports')}><MessageCircle size={14} aria-hidden="true"/>{t('面经','Experiences')}</AppLink>}<AppLink onClick={guardLink} data-company-view="questions" href={practiceHref} aria-describedby={progress?`${hint}-practice-${company.id}`:undefined}><Flame size={14} aria-hidden="true"/>{practiceLabel}</AppLink><CompanyTargetButton canAct={()=>canUse()} company={company.id} returnTo={href('companies')}/></nav>
   </article>;
  })}</div>:<div className="interview-empty"><h3>{onlyTargets?targetIds.size?filters.q?t('目标公司没有匹配结果','No matching target companies'):t('目标公司暂不在当前目录中','Saved targets are unavailable in this directory'):t('还没有目标公司','No target companies yet'):t('没有匹配的公司','No matching companies')}</h3><p>{onlyTargets&&!targetIds.size?t('先浏览全部公司，点击公司卡片的星标设为目标。','Browse all companies and use a card’s star to save a target.'):onlyTargets&&!filters.q?t('已保存的公司暂不可用，可以先浏览全部公司，或在目标公司中管理。','Your saved companies are unavailable. Browse all companies or manage your targets.'):t('可以清空公司搜索，保留当前岗位、时间与排序。','Clear the company search while keeping your role, period and order.')}</p>{(!onlyTargets||!!filters.q)&&<button type="button" className="button subtle" onClick={event=>{if(canUse(event.currentTarget))onClearSearch();}}>{t('清空公司搜索','Clear company search')}</button>}{onlyTargets&&<button type="button" className="text-button" onClick={showAll}>{t('查看全部公司','Show all companies')}</button>}</div>}</div>
 </section>;
}

function CompanyProgress({value,title,id}:{value:CompanyPracticeProgress;title:string;id:string}){
 const {t}=useI18n();
 return <div id={id} className="company-directory-practice"><div><span>{t('练习进度','Practice progress')}</span><b>{value.phase==='ready'?t(`已通过 ${value.done} / ${value.total}`,`Passed ${value.done} / ${value.total}`):'—'}</b></div>{value.phase==='ready'?<><progress aria-label={t(`${title}关联练习进度`,`Linked practice progress for ${title}`)} value={value.done} max={value.total}/>{value.started>0&&<small>{t(`${value.started} 个进行中`,`${value.started} in progress`)}</small>}</>:<small>{value.phase==='catalog'?t(`已载入 ${value.loaded} / ${value.total} 道关联练习`,`Loaded ${value.loaded} of ${value.total} linked modules`):t('等待学习进度','Waiting for learning progress')}</small>}</div>;
}
