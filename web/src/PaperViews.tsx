import { paperModules, matchingPaperModules, paperProgress, selectLibraryPractice } from './libraryPractice';
import { useEffect, useRef, useState } from "react";
import {
  ArrowLeft, ArrowRight, BookOpen, CheckCircle2, ChevronDown,
  ChevronLeft, ChevronRight, Circle, ExternalLink, Search, Library, Shuffle, SlidersHorizontal, List, Rows3, X,
} from "lucide-react";
import { useI18n } from "./i18n";
import { BilingualText } from "./BilingualText";
import { status } from "./types";
import type { Category, Paper, Problem, Saved } from "./types";
import { paperCollections, papersInCollection } from "./paperCollections";
import { eligibleUnattemptedProblems, pickRandomUnattemptedProblem } from "./randomProblem";
import { deferredFeature } from "./deferredFeature";
import LibraryOverview from "./LibraryOverview";
const CollectionArtwork = deferredFeature(() => import("./CollectionArtwork"), ["合集插图", "collection artwork"], {loading:()=>null,failure:()=>null});
import { getProblemStats, ProblemStatsSummary } from "./ProblemStats";
import type { ProblemStatsMap } from "./ProblemStats";
import { PracticeStart } from "./PracticeJourney";

type Practice = { problems: Problem[]; saved: Record<string, Saved> };
export { paperModules, matchingPaperModules, paperProgress } from './libraryPractice';

function StateIcon({ state }: { state: string }) {
  const { t } = useI18n();
  const label = state === "done" ? t("已完成", "Solved") : state === "started" ? t("进行中", "In progress") : t("未开始", "Not started");
  return <span className={`row-state ${state}`} role="img" aria-label={label} title={label}>
    {state === "done" ? <CheckCircle2 size={18} /> : state === "started" ? <span className="in-progress-dot" /> : <Circle size={17} />}
  </span>;
}

function ProgressBar({ completed, total }: { completed: number; total: number }) {
  return <div className="paper-progress-bar" aria-hidden="true"><span style={{ width: `${total ? completed / total * 100 : 0}%` }} /></div>;
}

const categoryLabels: Record<string, [string, string]> = {
  "reinforcement-learning": ["强化学习", "Reinforcement Learning"],
  "post-training": ["后训练与微调", "Post-training"],
  "generative-models": ["生成模型", "Generative models"],
  architecture: ["架构与推理", "Architecture"],
  optimization: ["优化算法", "Optimization"],
  representation: ["表征学习", "Representation"],
  "retrieval": ["检索与推荐", "Retrieval & Recommendation"],
  "world-models": ["世界模型", "World Models"],
};

export function PaperLibrary({ papers, categories, problems, saved, collectionId, progressReady = true }: Practice & { papers: Paper[]; categories: Category[]; collectionId?: string; progressReady?: boolean }) {
  const { t } = useI18n();
  const searchInput = useRef<HTMLInputElement>(null);
  const libraryMain = useRef<HTMLDivElement>(null);
  const [filtersOpen, setFiltersOpen] = useState(false);
  const [density, setDensity] = useState<'comfortable' | 'compact'>(() => {
    try { return localStorage.getItem('paperleet.library.density.v1') === 'compact' ? 'compact' : 'comfortable'; }
    catch { return 'comfortable'; }
  });
  function changeDensity(value: 'comfortable' | 'compact') {
    setDensity(value);
    try { localStorage.setItem('paperleet.library.density.v1', value); } catch { /* Preferences remain usable without storage. */ }
  }
  function browse() { libraryMain.current?.scrollIntoView({block:'start',behavior:window.matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth'}); }
  useEffect(() => {
    const focusSearch = (event: KeyboardEvent) => {
      if (event.key !== '/' || event.ctrlKey || event.metaKey || event.altKey || (event.target as HTMLElement | null)?.closest('input, textarea, select, [contenteditable="true"], [role="dialog"], dialog[open]')) return;
      event.preventDefault(); searchInput.current?.focus(); libraryMain.current?.scrollIntoView({block:'start'});
    };
    window.addEventListener('keydown',focusSearch);
    return () => window.removeEventListener('keydown',focusSearch);
  },[]);
  const [query, setQuery] = useState("");
  const [diff, setDiff] = useState("all");
  const [runtime, setRuntime] = useState("all");
  const [state, setState] = useState("all");
  const [tag, setTag] = useState("all");
  const [category, setCategory] = useState("all");
  const practice = { problems, saved };
  const selectedCollection = paperCollections.find((collection) => collection.id === collectionId);
  const listedPapers = papersInCollection(papers, selectedCollection);
  const filters = normalizeLibraryFilters(readLibraryRoute(route)?.filters || emptyLibraryFilters(), listedPapers);
  const {query,category,tag,difficulty:diff,runtime,state} = filters;
  const waitingForStatus = state !== 'all' && !progressReady;
  const viewHref = libraryHref(selectedCollection?.id, filters);
  const editedHref = useRef<string | undefined>(undefined);
  function change(patch: Partial<LibraryFilters>) {
    const next = normalizeLibraryFilters({...filters,...patch},listedPapers);
    const href = libraryHref(selectedCollection?.id,next);
    editedHref.current = href;
    onFilters?.(href);
  }
  useEffect(() => {
    if (waitingForStatus) return;
    let disposed = false, stop: (() => void) | undefined;
    const restore = editedHref.current !== viewHref;
    editedHref.current = undefined;
    void import('./libraryPosition').then(({watchLibraryPosition}) => {
      if (!disposed) stop = watchLibraryPosition(viewHref,restore);
    }).catch(() => { /* Return links and URL filters also work without scroll recovery. */ });
    return () => { disposed = true; stop?.(); };
  },[viewHref,waitingForStatus]);
  const collectionRail = useRef<HTMLElement>(null);
  const [railEdges, setRailEdges] = useState({ previous: false, next: false });
  useEffect(() => {
    const rail = collectionRail.current;
    if (!rail) return;
    const reveal = () => {
      const card = rail.querySelector('[aria-current="page"]');
      if (!card) return;
      const bounds = rail.getBoundingClientRect(), selected = card.getBoundingClientRect();
      if (selected.left < bounds.left) rail.scrollLeft -= bounds.left - selected.left;
      else if (selected.right > bounds.right) rail.scrollLeft += selected.right - bounds.right;
    };
    reveal();
    const measure = () => setRailEdges({ previous: rail.scrollLeft > 2, next: rail.scrollLeft + rail.clientWidth < rail.scrollWidth - 2 });
    measure();
    rail.addEventListener("scroll", measure, { passive: true });
    const observer = new ResizeObserver(() => {
      reveal();
      measure();
    });
    observer.observe(rail);
    return () => { rail.removeEventListener("scroll", measure); observer.disconnect(); };
  }, [selectedCollection?.id]);
  const listedModuleIds = new Set(listedPapers.flatMap((paper) => paper.module_ids));
  const listedModules = problems.filter((problem) => listedModuleIds.has(problem.id));
  const visibleCategories = categories.filter((item) => listedPapers.some((paper) => paper.category === item.id));
  const tags = [...new Set(listedPapers.filter((paper) => category === "all" || paper.category === category).flatMap((paper) => paper.tags))];
  const solvedModules = listedModules.filter((p) => status(p, saved[p.id]) === "done").length;
  const selection = selectLibraryPractice(papers, problems, saved, { collectionId: selectedCollection?.id, filters });
  const filtered = selection.papers;
  const filteredModules = filtered.flatMap(paper => selection.problems.filter(problem => paper.module_ids.includes(problem.id)));
  const unattempted = eligibleUnattemptedProblems(filteredModules, saved);
  function randomProblem() {
    if (!progressReady) return;
    const next = pickRandomUnattemptedProblem(filteredModules, saved);
    if (next) location.hash = withLibraryContext(next.number,viewHref).slice(1);
  }
  function scrollCollections(direction: number) {
    const rail = collectionRail.current;
    if (rail) rail.scrollBy({ left: direction * rail.clientWidth });
  }
  function clear() { change(emptyLibraryFilters()); }
  const hasFilters = !!query || diff !== "all" || state !== "all" || tag !== "all" || category !== "all" || runtime !== "all";
  return <main className="home paper-library">
    <PracticeStart problems={listedModules} saved={saved} ready={progressReady} onBrowse={browse} />
    <section className="featured-collections" aria-label={t("精选合集", "Featured collections")}>
      <div className="collections-heading">
        <h2>{t("精选合集", "Featured collections")}</h2>
        <div className="collection-controls">
          <button aria-label={t("上一组合集", "Previous collections")} disabled={!railEdges.previous} onClick={() => scrollCollections(-1)}><ChevronLeft size={18} /></button>
          <button aria-label={t("下一组合集", "Next collections")} disabled={!railEdges.next} onClick={() => scrollCollections(1)}><ChevronRight size={18} /></button>
        </div>
      </div>
      <nav ref={collectionRail} className="collection-rail" aria-label={t("精选合集导航", "Featured collection navigation")}>
        {paperCollections.map((collection) => {
          const members = papersInCollection(papers, collection);
          const count = new Set(members.flatMap((paper) => paper.module_ids)).size;
          const label = t(collection.name, collection.name_en);
          const selected = selectedCollection?.id === collection.id;
          return <a key={collection.id} className={`collection-card collection-${collection.id}`} href={`#/collections/${collection.id}`} aria-current={selected ? "page" : undefined} aria-label={`${label} · ${members.length} ${t("篇论文", "papers")} · ${count} ${t("道题", "problems")}`}>
            <div className="collection-copy">
              <span className="collection-count">{members.length} {t("篇论文", "papers")}<span> / </span>{count} {t("道题", "problems")}</span>
              <h3>{label}</h3>
              <p>{t(collection.description, collection.description_en)}</p>
            </div>
            <CollectionArtwork kind={collection.id} />
            <span className="collection-open" aria-hidden="true">{selected ? <CheckCircle2 size={20} /> : <ArrowRight size={20} />}</span>
          </a>;
        })}
      </nav>
    </section>
    <div className="library-layout">
    <div className="library-main" id="paper-library" ref={libraryMain} data-density={density}>
    <header className="library-heading">
      <div className="library-title"><h1><BilingualText zh={selectedCollection?.name || "论文题库"} en={selectedCollection?.name_en || "Papers"} /> <span>{listedPapers.length}</span></h1>{selectedCollection && <a className="clear-collection" href="#" onClick={clear} aria-label={t("返回全部论文", "Back to all papers")} title={t("返回全部论文", "Back to all papers")}><X size={17} /></a>}</div>
      <div className="library-progress"><CheckCircle2 size={16} /><BilingualText zh="已通过模块" en="Modules solved" /><b>{solvedModules} / {listedModules.length}</b></div>
    </header>
    <div className="paper-categories" aria-label={t("论文分类", "Paper categories")}>
      <button aria-pressed={category === "all"} onClick={() => { setCategory("all"); setTag("all"); }}><Library size={16} /><BilingualText zh="全部" en="All" /><span>{listedPapers.length}</span></button>
      {visibleCategories.map((item) => {
        const label = categoryLabels[item.id] || [item.name, item.name_en];
        return <button key={item.id} className={`category-${item.id}`} aria-pressed={category === item.id} onClick={() => { setCategory(item.id); setTag("all"); }}><i aria-hidden="true" /><BilingualText zh={label[0]} en={label[1]} /><span>{listedPapers.filter((paper) => paper.category === item.id).length}</span></button>;
      })}
    </div>
    <section className="problem-list" aria-label={t("论文列表", "Paper list")}>
      <div className="library-toolbar">
        <div className="search"><Search size={17} aria-hidden="true" /><input ref={searchInput} type="search" maxLength={200} aria-label={t("搜索论文或模块", "Search papers or modules")} placeholder={t("搜索论文或模块…", "Search papers or modules…")} value={query} onChange={(e) => setQuery(e.target.value)} onKeyDown={(e) => { if (e.key === "Escape") setQuery(""); }} />{query ? <button className="search-clear" type="button" aria-label={t("清空搜索","Clear search")} onClick={()=>{setQuery("");searchInput.current?.focus();}}><X size={15}/></button> : <kbd aria-hidden="true">/</kbd>}</div>
        <button className="library-filter-toggle" aria-controls="library-advanced-filters" aria-expanded={filtersOpen} onClick={()=>setFiltersOpen(value=>!value)}><SlidersHorizontal size={16} aria-hidden="true"/>{t('筛选','Filters')}<ChevronDown size={13} aria-hidden="true"/></button>
        <div className="filter-bar" id="library-advanced-filters" data-expanded={filtersOpen}>
          <div className="select-wrap"><select aria-label={t("主题筛选", "Filter by topic")} value={tag} onChange={(e) => setTag(e.target.value)}>
            <option value="all">{t("主题", "Topic")}</option>{tags.map((name) => <option key={name}>{name}</option>)}
          </select><ChevronDown size={12} aria-hidden="true" /></div>
          <div className="select-wrap"><select aria-label={t("模块难度筛选", "Filter module difficulty")} value={diff} onChange={(e) => change({difficulty:e.target.value})}>
            <option value="all">{t("难度", "Difficulty")}</option><option value="E">{t("简单", "Easy")}</option><option value="M">{t("中等", "Medium")}</option><option value="H">{t("困难", "Hard")}</option>
          </select><ChevronDown size={12} aria-hidden="true" /></div>
          <div className="select-wrap"><select aria-label={t("判题位置筛选", "Filter by judge location")} value={runtime} onChange={(e) => change({runtime:e.target.value})}>
            <option value="all">{t("运行位置", "Run location")}</option><option value="browser">{t("浏览器可运行", "Runs in browser")}</option><option value="server">{t("服务端判题", "Server judging")}</option>
          </select><ChevronDown size={12} aria-hidden="true" /></div>
          <div className="select-wrap"><select aria-label={t("论文状态筛选", "Filter paper status")} disabled={!progressReady} value={state} onChange={(e) => change({state:e.target.value})}>
            <option value="all">{t("状态", "Status")}</option><option value="todo">{t("未开始", "Not started")}</option><option value="started">{t("进行中", "In progress")}</option><option value="done">{t("已完成", "Solved")}</option>
          </select><ChevronDown size={12} aria-hidden="true" /></div>
        </div>
        <button className="random-problem" aria-label={t("随机一题","Random problem")} onClick={randomProblem} disabled={!progressReady || !unattempted.length} title={!progressReady ? t("学习进度尚未就绪", "Practice progress is not ready") : unattempted.length ? t(`从当前范围内 ${unattempted.length} 道未开始的题目中随机选择`, `Pick from ${unattempted.length} unattempted problems in this view`) : t("当前范围内没有未开始的题目", "No unattempted problems in this view")}><Shuffle size={17} aria-hidden="true"/><span>{t("随机一题","Random problem")}</span></button>
      </div>
      <div className="library-results"><span role="status">{filtered.length} {t("篇论文", filtered.length === 1 ? "paper" : "papers")} · {filteredModules.length} {t("道题", "modules")}{progressReady && !unattempted.length && <span className="random-empty"> · {t("当前范围暂无未开始的题目", "No unattempted problems in this view")}</span>}</span><div className="library-result-actions">{hasFilters && <button className="text-button" onClick={clear}>{t("重置筛选", "Reset filters")}</button>}<div className="library-density" role="group" aria-label={t('列表密度','List density')}><button aria-label={t('舒适列表','Comfortable list')} title={t('舒适列表','Comfortable list')} aria-pressed={density==='comfortable'} onClick={()=>changeDensity('comfortable')}><Rows3 size={15}/></button><button aria-label={t('紧凑列表','Compact list')} title={t('紧凑列表','Compact list')} aria-pressed={density==='compact'} onClick={()=>changeDensity('compact')}><List size={15}/></button></div></div></div>
      <div className="table-head paper-table-grid"><span>{t("状态", "Status")}</span><span>{t("论文", "Paper")}</span><span className="paper-year">{t("年份", "Year")}</span><span>{t("已通过 / 模块", "Solved / Modules")}</span><span /></div>
      <div className="problem-rows">
        {!waitingForStatus && filtered.map((paper) => {
          const progress = paperProgress(paper, practice);
          return <div className="paper-result-group" key={paper.id}><a className="problem-row paper-table-grid" data-paper-id={paper.id} href={`#/papers/${paper.id}`}>
            <StateIcon state={progress.state} />
            <div className="problem-info paper-info"><strong><span>{paper.number}.</span>{paper.short_title}</strong><small title={paper.title}>{paper.title}</small></div>
            <span className="paper-year">{paper.year || "—"}</span>
            <div className={`paper-row-progress ${progress.state}`}><b>{progress.completed}</b><span> / {progress.total}</span></div>
            <ChevronRight size={16} aria-hidden="true" />
          </a>{!!query.trim() && <div className="matching-modules"><span>{t('匹配模块','Matching modules')}</span>{matchingPaperModules(paper,problems,query,diff,runtime).map(module=><a key={module.id} href={`#/${module.number}`}>{module.number}. {t(module.title,module.translations?.en?.title || module.title)}<ArrowRight size={12} aria-hidden="true"/></a>)}</div>}</div>;
        })}
      </div>
      {!waitingForStatus && !filtered.length && <div className="no-results"><Search size={25} /><p>{t("没有符合条件的论文", "No matching papers")}</p><button className="text-button" onClick={clear}>{t("清除筛选", "Clear filters")}</button></div>}
    </section>
    </div>
    <LibraryOverview problems={listedModules} saved={saved} ready={progressReady} difficulty={diff} onDifficulty={setDiff} onBrowse={browse}/>
    </div>
  </main>;
}

export function PaperDetail({ paper, problems, saved, stats }: Practice & { paper: Paper; stats?: ProblemStatsMap }) {
  const { t, locale } = useI18n();
  const modules = paperModules(paper, problems);
  const progress = paperProgress(paper, { problems, saved });
  const next = modules.find((p) => !p.module.optional && status(p, saved[p.id]) !== "done") || modules[0];
  const authors = Array.isArray(paper.authors) ? paper.authors.join(", ") : paper.authors;
  return <main className="home paper-detail">
    <a className="back-link paper-library-link" href={returnTo || '#'}><ArrowLeft size={15} />{returnTo && returnTo !== '#' ? t('返回题库结果','Back to library results') : t("全部论文", "All papers")}</a>
    <div className="paper-detail-header">
      <div><h1><span>{paper.number}.</span> {paper.short_title}</h1><p className="paper-full-title">{paper.title}</p><div className="paper-detail-meta">{paper.venue && <span>{paper.venue}</span>}{paper.year && !paper.venue?.includes(String(paper.year)) && <span>{paper.year}</span>}{paper.tags.map((tag) => <span className="paper-topic" key={tag}>{tag}</span>)}</div></div>
      <a className="button subtle" href={paper.url} target="_blank" rel="noreferrer"><BookOpen size={15} />{t("阅读论文", "Read paper")}<ExternalLink size={13} /></a>
    </div>
    <div className="home-columns">
      <div className="main-column">
        <section className="paper-overview"><p>{t(paper.summary, paper.summary_en)}</p><details className="paper-scope-details"><summary>{t("练习范围", "Practice scope")}</summary><p>{t(paper.scope, paper.scope_en)}</p></details></section>
        <section className="paper-module-list">
          <div className="list-title"><h2>{t("核心模块", "Core modules")}</h2><span>{progress.completed} / {progress.total} {t("已通过", "solved")}</span></div>
          {modules.map((problem) => {
            const title = locale === "en" ? problem.translations?.en?.title || problem.title : problem.title;
            const state = status(problem, saved[problem.id]);
            return <a href={withLibraryContext(problem.number,returnTo)} className="paper-module-card" key={problem.id}>
              <div className="module-card-top"><StateIcon state={state} /><h3><span>{problem.number}.</span> {title}</h3><ChevronRight size={17} /></div>
              <p>{t(problem.module.goal, problem.module.goal_en)}</p>
              <div className="module-card-tags"><span className={`difficulty ${problem.difficulty}`}>{({ E: "Easy", M: "Medium", H: "Hard" } as Record<string, string>)[problem.difficulty] || problem.difficulty}</span><span>{problem.estimated_minutes || 25} min</span><span>{problem.module.optional ? t("可选拓展", "Optional extension") : t("核心模块", "Core module")}</span>{state === "done" && <span className="accepted">{t("已通过", "Solved")}</span>}</div>
              <div className="module-functions"><span>{t("需要实现", "Implement")}</span>{problem.module.required_functions.map((name) => <code key={name}>{name}</code>)}</div>
              {!!problem.module.provided_functions.length && <div className="module-functions provided-functions"><span>{t("已提供", "Provided")}</span>{problem.module.provided_functions.map((name) => <code key={name}>{name}</code>)}</div>}
              <ProblemStatsSummary stats={getProblemStats(stats, problem.id)} />
            </a>;
          })}
        </section>
      </div>
      <aside className="home-sidebar">
        <section className="paper-completion-card"><div className="aside-title"><h3>{t("论文进度", "Paper progress")}</h3><StateIcon state={progress.state} /></div><div className="paper-completion-count"><b>{progress.completed}</b><span>/ {progress.total}</span></div><p>{t("核心模块已通过", "core modules solved")}</p><ProgressBar {...progress} />{progress.state === "done" && <div className="core-complete"><CheckCircle2 size={16} />{t("本篇核心模块已完成", "Selected core modules complete")}</div>}{next && <a className="button" href={withLibraryContext(next.number,returnTo)}>{progress.state === "done" ? t("回顾模块", "Review modules") : progress.state === "started" ? t("继续实现", "Continue practice") : t("开始实现", "Start implementing")}<ArrowRight size={15} /></a>}</section>
        <section className="paper-source-card"><div className="aside-title"><h3>{t("论文信息", "Paper details")}</h3><BookOpen size={16} /></div><p>{authors}</p><a className="text-button" href={paper.url} target="_blank" rel="noreferrer">{paper.arxiv ? `arXiv:${paper.arxiv}` : t("原始论文", "Original paper")}<ExternalLink size={13} /></a></section>
      </aside>
    </div>
  </main>;
}
