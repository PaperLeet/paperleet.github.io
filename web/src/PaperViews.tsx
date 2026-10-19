import {appLocation} from './appLocation';
import AppLink from './AppLink';
import { matchingPaperModules, paperProgress, selectLibraryPractice } from './libraryPractice';
import { useEffect, useId, useRef, useState } from "react";
import {
  ArrowLeft, ArrowRight, BookOpen, CheckCircle2, ChevronDown,
  ChevronRight, Code2, Search, Library, Shuffle, SlidersHorizontal, X,
} from "lucide-react";
import { useI18n } from "./i18n";
import { BilingualText } from "./BilingualText";
import { status } from "./types";
import type { Category, Paper, Problem, Saved } from "./types";
import { paperCollections, papersInCollection } from "./paperCollections";
import { eligibleUnattemptedProblems, pickRandomUnattemptedProblem } from "./randomProblem";
import { recommendPractice } from "./practice-recommendation";
import { deferredFeature,FeatureRetry } from './deferredFeature';
import "./practice-journey.css";
import { StateIcon } from "./PaperState";
import { emptyLibraryFilters, libraryHref, libraryTags, sameLibraryTag, normalizeLibraryFilters, readLibraryRoute, sortLibraryPapers, withLibraryContext, type LibraryFilters } from './libraryNavigation';

import type {LibraryCompanyScope} from './libraryCompanyScope';
import {CompanyTools} from './LibraryCompanyTools';
type Practice = { problems: Problem[]; saved: Record<string, Saved> };
export { paperModules, matchingPaperModules, paperProgress } from './libraryPractice';

// Collection discovery loads independently of search and the first usable list.
const LibraryCollections = deferredFeature<import('./LibraryCollections').Props>(()=>import('./LibraryCollections'), ['精选合集','featured collections'], {loading:()=><div className="library-collections-placeholder" aria-hidden="true"/>});
const ResultControls = deferredFeature<import('./LibraryResultControls').Props>(()=>import('./LibraryResultControls'), ['列表显示','list display'], {loading:()=>null});
const ActiveFilters = deferredFeature<import('react').ComponentProps<typeof import('./LibraryResultControls').LibraryActiveFilters>>(()=>import('./LibraryResultControls').then(module=>({default:module.LibraryActiveFilters})),['当前筛选','active filters'],{loading:()=>null});
const TableHeaders=deferredFeature<import('./LibraryResultControls').LibraryHeaderProps>(()=>import('./LibraryResultControls').then(module=>({default:module.LibraryTableHeaders})),['',''],{loading:props=><LibraryHeadersFallback {...props}/>,failure:props=><LibraryHeadersFallback {...props}/>});
function LibraryHeadersFallback({moduleView}:import('./LibraryResultControls').LibraryHeaderProps){const {t}=useI18n();return <><span>{t('状态','Status')}</span><span>{moduleView?t('练习','Module'):t('论文','Paper')}</span><span className="paper-year">{moduleView?t('用时','Time'):t('年份','Year')}</span><span>{moduleView?t('难度','Difficulty'):t('已通过 / 模块','Solved / Modules')}</span><span/></>;}
const LibraryOverview = deferredFeature(() => import('./LibraryOverview'), ['练习概览','practice overview'], {loading:()=>null});
const LibraryPagination = deferredFeature<import('./LibraryPagination').Props>(()=>import('./LibraryPagination'), ['题库分页','library pages'], {loading:props=><LibraryPageFallback {...props}/>,failure:(props,retry)=><><LibraryPageFallback {...props}/><FeatureRetry retry={retry}/></>});

function LibraryPageFallback({page,size,total,compact,onPage}:import('./LibraryPagination').Props){
 const {t}=useI18n(),pages=Math.max(1,Math.ceil(total/size));
 if(compact||!total)return null;
 return <nav className="library-page-fallback" aria-label={t('题库分页','Library pages')}><button type="button" className="text-button" disabled={page<=1||page>pages} onClick={()=>onPage(page-1)}>{t('上一页','Previous page')}</button><span>{page} / {pages}</span><button type="button" className="text-button" disabled={page>=pages} onClick={()=>onPage(page+1)}>{t('下一页','Next page')}</button></nav>;
}

export function PaperLibrary({ papers, categories, problems, saved, collectionId, progressReady = true, route = '', onFilters, companyScope }: Practice & { papers: Paper[]; categories: Category[]; collectionId?: string; progressReady?: boolean; route?: string; onFilters?: (href: string,paging?:boolean) => void; companyScope?:LibraryCompanyScope }) {
  const { t, locale } = useI18n();
  const searchInput = useRef<HTMLInputElement>(null);
  const libraryMain = useRef<HTMLDivElement>(null);
  const resultsHeading=useRef<HTMLDivElement>(null),listId=useId();
  const [filtersOpen, setFiltersOpen] = useState(false);
  const [density, setDensity] = useState<'comfortable' | 'compact'>(() => {
    try { return localStorage.getItem('paperleet.library.density.v1') === 'compact' ? 'compact' : 'comfortable'; }
    catch { return 'comfortable'; }
  });
  function changeDensity(value: 'comfortable' | 'compact') {
    setDensity(value);
    try { localStorage.setItem('paperleet.library.density.v1', value); } catch { /* Usable without storage. */ }
  }
  useEffect(() => {
    const focusSearch = (event: KeyboardEvent) => {
      if (event.key !== "/" || event.ctrlKey || event.metaKey || event.altKey || (event.target as HTMLElement | null)?.closest('input, textarea, select, [contenteditable="true"], [role="dialog"], dialog[open]') || document.querySelector('dialog[open]')) return;
      event.preventDefault(); searchInput.current?.focus(); libraryMain.current?.scrollIntoView({ block: "start" });
    };
    window.addEventListener("keydown", focusSearch);
    return () => window.removeEventListener("keydown", focusSearch);
  }, []);
  const practice = { problems, saved };
  const selectedCollection = paperCollections.find((collection) => collection.id === collectionId);
  const course = selectedCollection?.kind === "course";
  const listedPapers = papersInCollection(papers, selectedCollection);
  const filters = normalizeLibraryFilters(readLibraryRoute(route)?.filters || emptyLibraryFilters(), listedPapers);
  const {query,category,tag,difficulty:diff,state,sort = 'default'} = filters;
  const advancedFilterCount = [tag, diff, state].filter(value => value !== 'all').length;
  const company=filters.company||'',companyDays=filters.company_days||'365',moduleView=course||!!company||filters.view==='modules',directModules=filters.view==='modules'&&!course&&!company;
  const companyData=companyScope?.data?.scope===companyScope?.scope?companyScope?.data:undefined;
  const companyReady=!company||companyData?.phase==='ready';
  const waitingForStatus = state !== 'all' && !progressReady || !companyReady;
  const viewHref = libraryHref(selectedCollection?.id, filters);
  const editedHref = useRef<string | undefined>(undefined);
  function change(patch: Partial<LibraryFilters>,paging=false) {
    const next = normalizeLibraryFilters({...filters,page:undefined,...patch},listedPapers);
    const href = libraryHref(selectedCollection?.id,next);
    editedHref.current = href;
    onFilters?.(href,paging);
  }
  const listedModuleIds = new Set(listedPapers.flatMap((paper) => paper.module_ids));
  const listedModules = problems.filter((problem) => listedModuleIds.has(problem.id));
  const selectedCategory=categories.find(item=>item.id===category);
  const visibleCategories = categories.filter((item) => listedPapers.some((paper) => paper.category === item.id));
  const tags = libraryTags(listedPapers, category);
  const selectedTag = tag === 'all' ? 'all' : tags.find(name => sameLibraryTag(name, tag)) ?? tag;
  const overviewModules=company?listedModules.filter(problem=>companyData?.ids.has(problem.id)):listedModules;
  const solvedModules = overviewModules.filter((p) => status(p, saved[p.id]) === "done").length;
  const selection = selectLibraryPractice(papers, problems, saved, { collectionId: selectedCollection?.id, filters },companyReady?companyData?.ids:undefined,locale);
  const filtered = sortLibraryPapers(selection.papers, sort);
  const filteredModules = directModules||company&&sort==='default'?selection.problems:filtered.flatMap(paper => selection.problems.filter(problem => paper.module_ids.includes(problem.id)));
  const rowIds = (moduleView ? filteredModules : filtered).map(row => row.id);
  const rowKey = rowIds.join(',');
  const page=filters.page||1,size=filters.page_size||20,pageCount=Math.max(1,Math.ceil(rowIds.length/size)),offset=(page-1)*size,outOfRange=page>pageCount;
  function currentView(){
    if(!libraryMain.current?.isConnected||libraryMain.current.closest('[hidden],[aria-hidden="true"],[inert]'))return false;
    const current=readLibraryRoute(appLocation.hash)||readLibraryRoute(appLocation.hash.split('?')[0]);
    return !!current&&libraryHref(current.collectionId,normalizeLibraryFilters(current.filters,listedPapers))===viewHref;
  }
  function changeSort(sort:import('./libraryNavigation').LibrarySort){
    if(!currentView())return;
    change({sort});
  }
  function changeView(value:'papers'|'modules'){
    if(!currentView()||(filters.view||'papers')===value)return;
    change({view:value==='modules'?'modules':undefined},true);
  }
  function paginate(next:number,nextSize= size){
    if(waitingForStatus||!currentView()||!Number.isSafeInteger(next)||next<1||next>Math.max(1,Math.ceil(rowIds.length/nextSize))||![20,50,100].includes(nextSize)||next===page&&nextSize===size)return;
    resultsHeading.current?.focus({preventScroll:true});resultsHeading.current?.scrollIntoView({block:'start',behavior:'instant'});
    change({page:next>1?next:undefined,page_size:nextSize===20?undefined:nextSize},true);
  }
  const pagination={page,size,total:rowIds.length,modules:!!moduleView,listId,onPage:(page:number)=>paginate(page),onSize:(size:import('./libraryNavigation').LibraryPageSize)=>paginate(outOfRange?1:Math.floor(offset/size)+1,size)};
  useEffect(() => {
    if (waitingForStatus) return;
    let disposed = false, stop: (() => void) | undefined;
    const restore = editedHref.current !== viewHref;
    editedHref.current = undefined;
    const origin={y:window.scrollY,active:document.activeElement};
    void import('./libraryPosition').then(({watchLibraryPosition}) => {
      if (!disposed) stop = watchLibraryPosition(viewHref, restore, origin);
    }).catch(() => { /* Paging and return links also work without position storage. */ });
    return () => { disposed = true; stop?.(); };
  }, [viewHref, waitingForStatus, rowKey]);
  const unattempted = eligibleUnattemptedProblems(filteredModules, saved);
  function randomProblem() {
    if (!progressReady || !companyReady) return;
    const next = pickRandomUnattemptedProblem(filteredModules, saved);
    if (next) appLocation.hash = withLibraryContext(next.number,viewHref).slice(1);
  }
  function clear() {
    if(!currentView())return;
    if(libraryMain.current?.querySelector('.library-result-actions button:focus,.no-results button:focus'))searchInput.current?.focus({preventScroll:true});
    change({...emptyLibraryFilters(), sort,company:undefined,company_days:undefined});
  }
  const hasFilters = !!company || !!query || diff !== "all" || state !== "all" || tag !== "all" || category !== "all";
  const nextInCollection = selectedCollection && progressReady && companyReady ? recommendPractice(filteredModules, saved, course) || filteredModules[0] : undefined;
  const nextState = nextInCollection && status(nextInCollection, saved[nextInCollection.id]);
  return <main className={`home paper-library${selectedCollection ? " collection-page" : ""}`}>
    {selectedCollection ? <header className="collection-page-header">
      <AppLink className="back-link" href="#"><ArrowLeft size={15} aria-hidden="true" />{t("全部论文", "All papers")}</AppLink>
      <div className="collection-page-title-row">
        <div className="collection-page-identity"><span className={`collection-page-mark collection-${selectedCollection.id}`} aria-hidden="true"><BookOpen size={25} /></span><div><h1>{t(selectedCollection.name, selectedCollection.name_en)}</h1><div className="collection-page-meta"><span>{course ? t("课程", "Course") : `${listedPapers.length} ${t("篇论文", "papers")}`}</span><span>{listedModules.length} {t("道题", "modules")}</span></div></div></div>
        {nextInCollection && <AppLink className="button collection-practice" href={withLibraryContext(nextInCollection.number, viewHref)}><Code2 size={16} aria-hidden="true" />{nextState === "done" ? t("回顾练习", course ? "Review course" : "Review collection") : nextState === "started" ? t("继续练习", "Resume practice") : t("开始练习", "Start practice")}<ArrowRight size={15} aria-hidden="true" /></AppLink>}
      </div>
    </header> : <>
    <LibraryCollections papers={papers} problems={problems} saved={saved} progressReady={progressReady}/>
    </>}
    <div className="library-layout">
    <div className="library-main" ref={libraryMain} id="paper-library" data-density={density}>
    <header className="library-heading">
      <div className="library-title">{selectedCollection ? <h2>{course ? t("课程练习", "Course modules") : t("论文", "Papers")}</h2> : <h1><BilingualText zh="论文题库" en="Papers" /> <span>{listedPapers.length}</span></h1>}</div>
      <AppLink className="text-button" href="#/paper-registry">{t("论文库","Paper registry")}</AppLink>
      <AppLink className="text-button" href="#/favorites">{t("收藏题目","Saved problems")}</AppLink>
      <div className="library-progress"><CheckCircle2 size={16} /><BilingualText zh="已通过模块" en="Modules solved" /><b>{companyReady?solvedModules:'—'} / {companyReady?overviewModules.length:'—'}</b></div>
    </header>
    {!course && <div className="paper-categories" aria-label={t("论文分类", "Paper categories")}>
      <button aria-pressed={category === "all"} onClick={() => { change({category:"all",tag:"all"}); }}><Library size={16} /><BilingualText zh="全部" en="All" /><span>{listedPapers.length}</span></button>
      {visibleCategories.map((item) => {
        const label = [item.name, item.name_en];
        return <button key={item.id} className={`category-${item.id}`} aria-pressed={category === item.id} onClick={() => { change({category:item.id,tag:"all"}); }}><i aria-hidden="true" /><BilingualText zh={label[0]} en={label[1]} /><span>{listedPapers.filter((paper) => paper.category === item.id).length}</span></button>;
      })}
    </div>}
    <section className="problem-list" aria-label={course ? t("课程练习", "Course modules") : t("论文列表", "Paper list")}>
      <div className="library-toolbar">
        <div className="search"><Search size={17} aria-hidden="true" /><input ref={searchInput} type="search" maxLength={200} aria-label={t("搜索论文或模块", "Search papers or modules")} placeholder={directModules?t("题目编号、标题、作者或 arXiv…","Exercise number, title, author or arXiv…"):t("标题、作者、arXiv 或模块…", "Title, author, arXiv or module…")} value={query} onChange={(e) => change({query:e.target.value})} onKeyDown={(e) => { if (e.key === "Escape"&&!e.repeat&&!e.ctrlKey&&!e.metaKey&&!e.altKey&&!e.shiftKey&&!e.nativeEvent.isComposing&&e.nativeEvent.keyCode!==229) {e.preventDefault();e.stopPropagation();change({query:''});} }} />{query ? <button type="button" className="search-clear" aria-label={t("清空搜索", "Clear search")} onClick={() => { change({query:""}); searchInput.current?.focus(); }}><X size={15} /></button> : <kbd aria-hidden="true">/</kbd>}</div>
        <button type="button" className="library-filter-toggle" aria-controls="library-advanced-filters" aria-expanded={filtersOpen} onClick={() => setFiltersOpen(open => !open)}><SlidersHorizontal size={16} aria-hidden="true" />{t("筛选", "Filters")}{advancedFilterCount > 0 && <span>{advancedFilterCount}</span>}<ChevronDown size={13} aria-hidden="true" /></button>
        <div className="filter-bar" id="library-advanced-filters" data-expanded={filtersOpen}>
          <div className="select-wrap" data-active={selectedTag!=="all"||undefined}><select aria-label={t("主题筛选", "Filter by topic")} value={selectedTag} onChange={(e) => change({tag:e.target.value})}>
            <option value="all">{t("主题", "Topic")}</option>{tags.map((name) => <option key={name}>{name}</option>)}
          </select><ChevronDown size={12} aria-hidden="true" /></div>
          <div className="select-wrap" data-active={diff!=="all"||undefined}><select aria-label={t("模块难度筛选", "Filter module difficulty")} value={diff} onChange={(e) => change({difficulty:e.target.value})}>
            <option value="all">{t("难度", "Difficulty")}</option><option value="E">{t("简单", "Easy")}</option><option value="M">{t("中等", "Medium")}</option><option value="H">{t("困难", "Hard")}</option>
          </select><ChevronDown size={12} aria-hidden="true" /></div>
          <div className="select-wrap" data-active={state!=="all"||undefined}><select aria-label={moduleView ? t("练习状态筛选", "Filter module status") : t("论文状态筛选", "Filter paper status")} disabled={!progressReady} value={state} onChange={(e) => change({state:e.target.value})}>
            <option value="all">{t("状态", "Status")}</option><option value="todo">{t("未开始", "Not started")}</option><option value="started">{t("进行中", "In progress")}</option><option value="done">{t("已完成", "Solved")}</option>
          </select><ChevronDown size={12} aria-hidden="true" /></div>
        </div>
        <button type="button" className="random-problem" aria-label={t('随机一题', 'Random problem')} onClick={randomProblem} disabled={!progressReady || !companyReady || !unattempted.length} title={!companyReady?t("公司练习范围尚未确认","Company practice scope is unconfirmed"):!progressReady ? t("学习进度尚未就绪", "Practice progress is not ready") : unattempted.length ? t(`从当前范围内 ${unattempted.length} 道未开始的题目中随机选择`, `Pick from ${unattempted.length} unattempted problems in this view`) : t("当前范围内没有未开始的题目", "No unattempted problems in this view")}><Shuffle size={17} aria-hidden="true" /><span>{t("随机一题", "Random problem")}</span></button>
      </div>
      <CompanyTools company={company} days={companyDays} onChange={(company,days)=>change({company:company||undefined,company_days:company?days:undefined})}/>
      {company&&companyScope&&<CompanyTools company={company} days={companyDays} returnTo={viewHref} {...companyScope}/>}
      <div className="library-results"><div ref={resultsHeading} className="library-result-summary" tabIndex={-1} role="group" aria-label={t('题库列表结果','Library list results')}><span role="status" aria-atomic="true">{waitingForStatus ? !companyReady?t('公司练习范围尚未确认…','Company practice scope is unconfirmed…'):t('正在核对学习进度…','Checking your learning progress…') : <>{!moduleView && <>{filtered.length} {t("篇论文", filtered.length === 1 ? "paper" : "papers")} · </>}{filteredModules.length} {t("道题", filteredModules.length === 1 ? "module" : "modules")}{!!rowIds.length&&!outOfRange&&<span className="library-visible-range"> · {t(`显示 ${offset+1}–${Math.min(offset+size,rowIds.length)}`,`Showing ${offset+1}–${Math.min(offset+size,rowIds.length)}`)}</span>}{progressReady && filtered.length > 0 && !unattempted.length && <span className="random-empty"> · {t("当前范围暂无未开始的题目", "No unattempted problems in this view")}</span>}</>}</span></div>{!waitingForStatus&&<LibraryPagination {...pagination} compact/>}<ResultControls hasFilters={hasFilters} moduleView={moduleView} onView={!course&&!company?changeView:undefined} company={company} course={course} sort={sort} density={density} onClear={clear} onSort={changeSort} onDensity={changeDensity}/></div>
      {hasFilters&&<ActiveFilters filters={filters} categoryLabel={selectedCategory?t(selectedCategory.name,selectedCategory.name_en):category} onChange={change}/>}
      <div className="table-head paper-table-grid"><TableHeaders moduleView={moduleView} company={company} course={course} sort={sort} listId={listId} onSort={changeSort}/></div>
      <div className="problem-rows" id={listId} aria-busy={waitingForStatus}>
        {!waitingForStatus && (moduleView ? filteredModules.slice(offset,offset+size).map(module => <div className="paper-result-group" key={module.id}>
          <AppLink className="problem-row paper-table-grid" data-module-id={module.id} href={withLibraryContext(module.number,viewHref)}>
            <StateIcon state={status(module,saved[module.id])} />
            <div className="problem-info paper-info"><strong><span>{module.number}.</span>{locale === "en" ? module.translations?.en?.title || module.title : module.title}</strong>{directModules&&<span className="library-module-source">{papers.find(paper=>paper.module_ids.includes(module.id))?.short_title}</span>}<small>{t(module.module.goal,module.module.goal_en)}</small></div>
            <span className="paper-year">{module.estimated_minutes || 25} min</span>
            <span className={`difficulty ${module.difficulty}`}>{({E:t("简单","Easy"),M:t("中等","Medium"),H:t("困难","Hard")} as Record<string,string>)[module.difficulty]}</span>
            <ChevronRight size={16} aria-hidden="true" />
          </AppLink>
          <CompanyTools problem={module} route={route} returnTo={viewHref} companyScope={company?companyScope:undefined}/>
        </div>) : filtered.slice(offset,offset+size).map((paper) => {
          const progress = paperProgress(paper, practice);
          const matching = matchingPaperModules(paper,problems,query,diff);
          const showMatches = !!query.trim() || diff !== "all";
          return <div className="paper-result-group" key={paper.id}><AppLink className="problem-row paper-table-grid" data-paper-id={paper.id} key={paper.id} href={withLibraryContext(`papers/${paper.id}`,viewHref)}>
            <StateIcon state={progress.state} />
            <div className="problem-info paper-info"><div className="paper-title-line"><strong><span>{paper.number}.</span>{paper.short_title}</strong><span className="paper-title-tags">{paper.tags.map(tag => <span key={tag}>{tag}</span>)}</span></div><small title={paper.title}>{paper.title}</small></div>
            <span className="paper-year">{paper.kind === "course" ? t("专题", "Course") : paper.year || "—"}</span>
            <div className={`paper-row-progress ${progress.state}`}><b>{progress.completed}</b><span> / {progress.total}</span></div>
            <ChevronRight size={16} aria-hidden="true" />
          </AppLink>{showMatches && <div className="matching-modules"><span>{t("匹配模块", "Matching modules")}</span>{matching.map(module=><AppLink key={module.id} href={withLibraryContext(module.number,viewHref)}><Code2 size={13} aria-hidden="true" />{module.number}. {locale === "en" ? module.translations?.en?.title || module.title : module.title}<ArrowRight size={12} aria-hidden="true" /></AppLink>)}</div>}</div>;
        }))}
      </div>
      {!waitingForStatus&&outOfRange&&!!rowIds.length&&<div className="no-results library-page-recovery" role="status"><h3>{t('这一页已没有结果','This page is no longer available')}</h3><p>{t(`当前筛选共 ${pageCount} 页，可以返回有效页继续浏览。`,`This view has ${pageCount} pages. Choose an available page to keep browsing.`)}</p><div><button type="button" className="button subtle" onClick={()=>paginate(pageCount)}>{t('返回最后一页','Open last page')}</button><button type="button" className="text-button" onClick={()=>paginate(1)}>{t('回到第一页','Back to first page')}</button></div></div>}
      {!waitingForStatus&&<LibraryPagination {...pagination}/>}
      {!waitingForStatus && !filtered.length && <div className="no-results"><span className="empty-state-icon"><Search size={25} aria-hidden="true" /></span><h3>{moduleView ? t("没有符合条件的练习", "No matching modules") : t("没有符合条件的论文", "No matching papers")}</h3><p>{t("试试其他关键词，或减少筛选条件。", "Try another keyword or remove a filter to broaden your search.")}</p><button className="button subtle" onClick={clear}>{t("清除筛选", "Clear filters")}<ArrowRight size={14} aria-hidden="true" /></button></div>}
    </section>
    </div>
    <LibraryOverview dailyProblems={problems} companyScope={company?companyScope:undefined} onCompanyFilter={(company:string)=>change({company,company_days:undefined})} problems={overviewModules} practiceProblems={waitingForStatus?[]:filteredModules} scopeReady={!waitingForStatus} returnTo={viewHref} saved={saved} ready={progressReady&&companyReady} difficulty={diff} onDifficulty={difficulty => change({difficulty})} />
    </div>
  </main>;
}
