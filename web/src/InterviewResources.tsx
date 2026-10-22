import ResourceSearchMatch from './ResourceSearchMatch';
import ResourceConceptPractice from './ResourceConceptPractice';
import type {Problem} from './types';
import {appLocation} from './appLocation';
import {controlIsActive} from './activeControls';
import {interviewHref,readInterviewRoute} from './interviewNavigation';
import AppLink from './AppLink';
import {useId,useLayoutEffect,useMemo,useRef,useState,type MouseEvent} from 'react';
import {ChevronDown,ExternalLink,Search,X} from 'lucide-react';
import {useI18n} from './i18n';
import OrganizationLogo from './OrganizationLogo';
import {interviewLabel} from './interviewModels';
import {companyResources,focusInterviewSection} from './interviewPreparation';
import {matchingInterviewResources,resourceBrowserQuery,resourceBrowserTerms,resourceSearchContexts,type ResourceKind} from './interviewResourceBrowsing';
import {useWorkspaceIntent} from './useWorkspaceIntent';
import {keepReadingTriggerVisible} from './readingViewport';
import {resourceLearningFields,resourceLearningContexts} from './resourceLearningSearch';
import {withLibraryContext} from './libraryNavigation';
import './interview-resource-browser.css';

const PAGE_SIZE=3;
const KIND_ORDER=['all','technical','guide','careers'] as const;
type ResourceProps={company?:string;problems?:Problem[];catalogError?:string;onRetryCatalog?:()=>void};
const noProblems:Problem[]=[];
export default function InterviewResources({company='',...props}:ResourceProps){return <ResourceBrowser key={company||'all'} company={company} {...props}/>;}
function ResourceBrowser({company,problems=noProblems,catalogError,onRetryCatalog}:ResourceProps&{company:string}){
 const {t,locale}=useI18n(),en=locale==='en',id=useId();
 const kindLabels={all:t('全部','All'),guide:t('官方指南','Guides'),careers:t('招聘入口','Careers'),technical:t('技术学习','Technical learning')};
 const cardLabels={guide:t('官方指南','Official guide'),careers:t('招聘入口','Careers entry'),technical:t('技术学习','Technical learning')};
 const root=useRef<HTMLElement>(null),finder=useRef<HTMLDetailsElement>(null),summary=useRef<HTMLElement>(null),allKind=useRef<HTMLButtonElement>(null),input=useRef<HTMLInputElement>(null),list=useRef<HTMLDivElement>(null),compactButton=useRef<HTMLButtonElement>(null),composing=useRef(false);
 const visit={},latestVisit=useRef(visit);latestVisit.current=visit;
 const [query,setQuery]=useState(''),[kind,setKind]=useState<ResourceKind>('all'),[display,setDisplay]=useState({scope:'',limit:PAGE_SIZE});
 const resources=useMemo(()=>companyResources(company),[company]);
 const learning=useMemo(()=>new Map(resources.map(resource=>[resource.id,resourceLearningFields(resource.id,problems,en)])),[resources,problems,en]);
 const learningText=useMemo(()=>new Map([...learning].map(([key,fields])=>[key,fields.map(field=>field.text)])),[learning]);
 const searched=matchingInterviewResources(resources,query,learningText),matches=searched.filter(resource=>kind==='all'||resource.kind===kind);
 const scope=JSON.stringify([company,kind,resourceBrowserTerms(query),matches.map(resource=>resource.id)]),limit=display.scope===scope?display.limit:PAGE_SIZE;
 const visible=matches.slice(0,limit),more=Math.min(PAGE_SIZE,matches.length-visible.length),expanded=visible.length>PAGE_SIZE;
 const active=Number(!!resourceBrowserQuery(query))+Number(kind!=='all'),intent=useWorkspaceIntent(scope),href=typeof window==='undefined'?'':window.location.href;
 const route=typeof window==='undefined'?undefined:readInterviewRoute(appLocation.hash),returnTo=route?.company===company?interviewHref(route):undefined;
 function currentView(){return latestVisit.current===visit&&controlIsActive(root.current)&&!root.current?.closest('details:not([open]),dialog:not([open])')&&window.location.href===href&&readInterviewRoute(appLocation.hash)?.company===company&&intent.isCurrent();}
 function currentControl(node:HTMLElement){return currentView()&&controlIsActive(node)&&root.current?.contains(node)&&!node.closest('details:not([open]),dialog:not([open])');}
 function currentFinder(){return currentView()&&finder.current?.open===true;}
 const reveal=useRef<{resource?:string;claim:()=>boolean}|undefined>(undefined);
 function expand(){if(!currentView()||!more)return;reveal.current={resource:matches[visible.length].id,claim:intent.create()};setDisplay({scope,limit:visible.length+more});}
 function compact(){if(!currentView()||!expanded)return;reveal.current={claim:intent.create()};setDisplay({scope,limit:PAGE_SIZE});}
 useLayoutEffect(()=>{
  const request=reveal.current;if(!request)return;reveal.current=undefined;
  const target=request.resource?Array.from(list.current?.querySelectorAll<HTMLAnchorElement>('a[data-resource-id]')||[]).find(link=>link.dataset.resourceId===request.resource):compactButton.current;
  if(target&&currentView()&&controlIsActive(target)&&root.current?.contains(target)&&request.claim()){target.focus({preventScroll:true});keepReadingTriggerVisible(target,100);}else intent.cancel();
 },[limit,scope]);
 function focusSummary(event:MouseEvent<HTMLButtonElement>){if(document.activeElement===event.currentTarget&&summary.current){summary.current.focus({preventScroll:true});keepReadingTriggerVisible(summary.current,100);}}
 function focusAll(event:MouseEvent<HTMLButtonElement>){const target=allKind.current;if(document.activeElement===event.currentTarget&&target){target.focus({preventScroll:true});if(currentView()&&document.activeElement===target)keepReadingTriggerVisible(target,100);}}
 function clearQuery(){if(!currentFinder())return;composing.current=false;setQuery('');input.current?.focus({preventScroll:true});}
 function closeFinder(){const node=finder.current;if(!currentFinder()||!node)return;if(node.contains(document.activeElement)&&summary.current){summary.current.focus({preventScroll:true});keepReadingTriggerVisible(summary.current,100);}node.open=false;}
 return <section ref={root} className="interview-resource-panel" aria-labelledby="interview-resources-heading">
  <span className="interview-panel-kicker">{t('从官方资料出发','START WITH OFFICIAL SOURCES')}</span>
  <h2 id="interview-resources-heading" tabIndex={-1}>{company?t(`${interviewLabel('companies',company,false)} 官方资料`,`${interviewLabel('companies',company,true)} resources`):t('官方准备与学习资料','Official preparation & learning')}</h2>
  <p className="interview-resource-context">{t('按面试指南、招聘入口和技术学习分类。技术资料帮助理解原理，不代表该公司的面试题或考查频率。','Browse interview guides, careers entries and technical learning. Learning resources explain concepts; they do not establish company interview questions or frequency.')}{company&&t(' 具体面试安排请与招聘团队确认。',' Confirm role-specific interview arrangements with the recruiting team.')}</p>
  {!!resources.length&&<>
   <div className="resource-browser-toolbar">
    <div className="resource-browser-kinds" role="group" aria-label={t('资料类型','Resource type')}>{KIND_ORDER.map(value=><button key={value} ref={value==='all'?allKind:undefined} data-resource-kind={value} type="button" aria-pressed={kind===value} aria-controls={`${id}-list`} onClick={event=>{if(currentControl(event.currentTarget))setKind(value);}}><span>{kindLabels[value]}</span><small>{value==='all'?searched.length:searched.filter(resource=>resource.kind===value).length}</small></button>)}</div>
    <details ref={finder} className="resource-browser-finder" onKeyDown={event=>{if(currentFinder()&&event.key==='Escape'&&!event.defaultPrevented&&!event.repeat&&!event.ctrlKey&&!event.metaKey&&!event.altKey&&!event.shiftKey&&!composing.current&&!event.nativeEvent.isComposing&&event.keyCode!==229){event.preventDefault();event.stopPropagation();closeFinder();}}}>
     <summary ref={summary}><Search size={16} aria-hidden="true"/><span>{t('搜索资料','Search resources')}</span>{!!resourceBrowserQuery(query)&&<small>{t('已搜索','Search active')}</small>}<ChevronDown size={15} aria-hidden="true"/></summary>
     <div className="resource-browser-filter-panel">
      <div className="resource-browser-search" role="search" aria-label={t('查找官方资料','Find official resources')}><Search size={15} aria-hidden="true"/><input ref={input} type="search" maxLength={200} value={query} aria-label={t('搜索官方资料','Search official resources')} aria-controls={`${id}-list`} aria-describedby={`${id}-help`} placeholder={t('公司、标题或关键词…','Company, title or keyword…')} onChange={event=>{if(currentFinder())setQuery(composing.current||(event.nativeEvent as InputEvent).isComposing?event.target.value:Array.from(event.target.value).slice(0,100).join(''));}} onCompositionStart={()=>{if(currentFinder())composing.current=true;}} onCompositionEnd={event=>{const current=currentFinder();composing.current=false;if(current)setQuery(Array.from(event.currentTarget.value).slice(0,100).join(''));}} onKeyDown={event=>{if(!currentFinder()||event.defaultPrevented)return;const ime=composing.current||event.nativeEvent.isComposing||event.keyCode===229;if(event.key==='Enter'){event.stopPropagation();if(!ime)event.preventDefault();return;}if(event.key==='Escape'&&!ime&&!event.repeat&&!event.ctrlKey&&!event.metaKey&&!event.altKey&&!event.shiftKey&&query){event.preventDefault();event.stopPropagation();clearQuery();}}}/>{!!query&&<button type="button" aria-label={t('清空资料搜索','Clear resource search')} aria-controls={`${id}-list`} onClick={clearQuery}><X size={16} aria-hidden="true"/></button>}</div>
      <p id={`${id}-help`}>{t('搜索已收录资料、平台整理的阅读自测和已加载的相关练习；不搜索私人草稿，也不改变面经或练习范围。最多100个字符。','Search maintained resources, curated reading prompts and loaded related exercises. Private drafts are excluded and your experience/practice scope stays unchanged. Up to 100 characters.')}</p>
     </div>
    </details>
    {active>0&&<button type="button" className="text-button" aria-controls={`${id}-list`} onClick={event=>{if(!currentControl(event.currentTarget))return;focusAll(event);if(!currentView())return;composing.current=false;setQuery('');setKind('all');setDisplay({scope:'',limit:PAGE_SIZE});}}>{t('重置资料筛选','Reset resource filters')}</button>}
   </div>
   <div ref={list} className="interview-resource-list" id={`${id}-list`}>{visible.map(resource=>{
    const contexts=resourceSearchContexts(resource,query,en),learningMatches=resourceLearningContexts(resource,query,learning.get(resource.id)||[]);
    return <article key={resource.id} aria-labelledby={`${id}-${resource.id}`}>
    <AppLink data-resource-id={resource.id} href={resource.url} target="_blank" rel="noreferrer">
     <span><OrganizationLogo id={resource.company} size={14}/><ResourceSearchMatch text={interviewLabel('companies',resource.company,en)} query={query}/><ExternalLink size={12}/></span>
     <b id={`${id}-${resource.id}`}><ResourceSearchMatch text={en?resource.name_en:resource.name} query={query}/></b>
     <small className="interview-resource-kind">{cardLabels[resource.kind as Exclude<ResourceKind,'all'>]}</small>
     <p><ResourceSearchMatch text={en?resource.description_en:resource.description} query={query}/></p>
     {contexts.map(context=><div key={context.key} className="resource-search-context"><small>{context.kind==='company'?t('匹配公司名称','Matched company name'):context.kind==='translation'?t('匹配另一种语言','Matched in the other language'):context.kind==='url'?t('匹配来源链接','Matched source address'):t('匹配资料编号','Matched resource ID')}</small><span lang={context.kind==='company'||context.kind==='translation'?(en?'zh-CN':'en'):undefined}><ResourceSearchMatch text={context.text} query={query}/></span></div>)}
     <small className="resource-browser-source"><ResourceSearchMatch text={new URL(resource.url).hostname.replace(/^www\./,'')} query={query}/><br/>{t('来源核对于','Source checked')} <time dateTime={resource.checked}>{resource.checked}</time></small>
    </AppLink>
    {learningMatches.length>0&&<div className="resource-learning-matches" aria-label={t('平台整理的匹配内容','Matching content curated by PaperLeet')}>
     {learningMatches.map(context=><div key={context.key} className="resource-learning-context" data-learning-kind={context.kind}>
      <small id={`${id}-${resource.id}-${context.key}`}>{context.kind==='exercise'?t('匹配相关练习 · 平台整理','Matched related exercise · PaperLeet'):context.kind==='question'?t('匹配阅读自测 · 平台整理','Matched reading self-check · PaperLeet'):t('匹配练习说明 · 平台整理','Matched practice note · PaperLeet')}</small>
      {context.problem&&returnTo?<AppLink data-learning-exercise={context.problem.id} aria-describedby={`${id}-${resource.id}-${context.key}`} href={withLibraryContext(context.problem.number,returnTo)} target="_blank" rel="noopener noreferrer" aria-label={t(`${context.problem.number}. ${context.problem.title}（新标签页打开）`,`${context.problem.number}. ${context.problem.translations?.en?.title||context.problem.title} (opens in a new tab)`)} onClick={event=>{if(!currentView()||!controlIsActive(event.currentTarget)||event.currentTarget.closest('details:not([open]),dialog:not([open])'))event.preventDefault();}}><span><ResourceSearchMatch text={context.text} query={query}/></span><ExternalLink size={13} aria-hidden="true"/></AppLink>:<p lang={context.lang}><ResourceSearchMatch text={context.text} query={query}/></p>}
     </div>)}
    </div>}
    <ResourceConceptPractice resource={resource.id} problems={problems} returnTo={returnTo} canAct={currentView} error={catalogError} onRetry={onRetryCatalog}/>
    </article>;
   })}</div>
   {!matches.length&&<div className="interview-resource-empty"><p>{t('当前资料范围没有匹配结果。可以分别清空搜索或显示全部类型。','No resources match this view. Clear the search or show all resource types.')}</p>{!!resourceBrowserQuery(query)&&<button type="button" className="text-button" aria-controls={`${id}-list`} onClick={event=>{if(!currentView())return;focusSummary(event);composing.current=false;setQuery('');}}>{t('清空资料关键词','Clear resource keywords')}</button>}{kind!=='all'&&<button type="button" className="text-button" aria-controls={`${id}-list`} onClick={event=>{if(!currentControl(event.currentTarget))return;focusAll(event);if(currentView())setKind('all');}}>{t('显示全部资料类型','Show all resource types')}</button>}</div>}
   <div className="resource-browser-actions"><span id={`${id}-count`} role="status" aria-live="polite" aria-atomic="true">{t(`显示 ${visible.length} / ${matches.length} 份资料${active?` · 共收录 ${resources.length} 份`:''}`,`Showing ${visible.length} of ${matches.length} ${matches.length===1?'resource':'resources'}${active?` · ${resources.length} maintained in this scope`:''}`)}</span>{matches.length>PAGE_SIZE&&<div role="group" aria-label={t('继续浏览官方资料','Continue browsing official resources')}>{more>0&&<button type="button" className="resource-browser-more" aria-controls={`${id}-list`} aria-describedby={`${id}-count`} onClick={expand}>{t(`再显示 ${more} 份资料`,`Show ${more} more ${more===1?'resource':'resources'}`)}</button>}<button type="button" ref={compactButton} aria-controls={`${id}-list`} aria-describedby={`${id}-count`} aria-disabled={!expanded} onClick={compact}>{t('收起资料列表','Show fewer resources')}</button></div>}</div>
  </>}
  {!resources.length&&<div className="interview-resource-empty"><p>{t('暂未收录该公司的官方准备资料。可以先按岗位练习，再结合社区面经复盘。','No official preparation resource is listed for this company yet. Start with role-based practice and community experiences.')}</p><button type="button" className="text-button" onClick={()=>{if(currentView())focusInterviewSection('interview-preparation-heading');}}>{t('查看准备路径','Explore preparation paths')}</button></div>}
 </section>;
}
