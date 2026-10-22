import AppLink from './AppLink';
import { ArrowLeft, ArrowRight, BookOpen, CheckCircle2, Clock3, Code2, ExternalLink } from "lucide-react";
import { ProblemFavoriteButton } from "./deferredPages";
import { useI18n } from "./i18n";
import { status } from "./types";
import type { Paper, Problem, Saved } from "./types";
import { paperModules, paperProgress } from "./libraryPractice";
import { recommendPractice } from "./practice-recommendation";
import { readLibraryRoute,discussionPracticeReturn,withLibraryContext } from "./libraryNavigation";
import { getProblemStats, ProblemStatsSummary } from "./ProblemStats";
import type { ProblemStatsMap } from "./ProblemStats";
import { ModuleJudgeBadge } from "./JudgeCapability";
import { StateIcon } from "./PaperState";
import { hostedAppAvailable } from "./deployment";
import "./lc-paper.css";

type Practice = { problems: Problem[]; saved: Record<string, Saved> };

export function PaperDetail({ paper, problems, practiceProblems = problems, saved, stats, returnTo, progressReady = true,scopeReady=true }: Practice & { paper: Paper; practiceProblems?: Problem[]; stats?: ProblemStatsMap; returnTo?: string; progressReady?: boolean;scopeReady?:boolean }) {
  const { t, locale } = useI18n();
  const modules = paperModules(paper, problems);
  const progress = paperProgress(paper, { problems, saved });
  const scopedModules = modules.filter(module => practiceProblems.some(candidate => candidate.id === module.id));
  const company=!!readLibraryRoute(returnTo||'')?.filters.company;
  const visibleModules=company?scopedModules:modules;
  const next = progressReady && scopeReady ? recommendPractice(scopedModules,saved) || scopedModules[0] : undefined;
  const nextState = next && status(next,saved[next.id]);
  const completedCourse = paper.kind === "course" && progressReady && progress.state === "done";
  const continueWith = completedCourse && !company ? recommendPractice(problems.filter(problem => !paper.module_ids.includes(problem.id)), saved) : undefined;
  const primary = continueWith || next;
  const primaryStarted = !!primary && status(primary,saved[primary.id]) === "started";
  const primaryLabel = continueWith
    ? primaryStarted ? t("继续练习", "Resume practice") : t("继续学习", "Continue learning")
    : nextState === "done" ? t("回顾", "Review") : primaryStarted ? t("继续", "Continue") : t("开始", "Start");
  const estimatedMinutes = modules.filter(module=>!module.module.optional).reduce((sum,module)=>sum+(module.estimated_minutes || 25),0);
  const nextTitle = primary && (locale === "en" ? primary.translations?.en?.title || primary.title : primary.title);
  const authors = Array.isArray(paper.authors) ? paper.authors.join(", ") : paper.authors;
  const level = (value: string) => ({ E: t("简单", "Easy"), M: t("中等", "Medium"), H: t("困难", "Hard") } as Record<string, string>)[value] || value;
  // Never show a fabricated 0 while private progress is still loading.
  const solved = progressReady ? progress.completed : "—";
  const percent = progressReady && progress.total ? Math.round(progress.completed / progress.total * 100) : 0;
  const core = modules.filter(module => !module.module.optional);
  const levels = ["E", "M", "H"].map(value => {
    const group = core.filter(module => module.difficulty === value);
    return { value, total: group.length, done: group.filter(module => status(module, saved[module.id]) === "done").length };
  }).filter(row => row.total);
  return <main className="home paper-detail">
    <AppLink className="back-link paper-library-link" href={returnTo || '#'}><ArrowLeft size={15} />{discussionPracticeReturn(returnTo||'')?t('返回来源帖子','Back to source post'):returnTo?.startsWith('#/paper-registry')?t('返回论文库','Back to paper registry'):returnTo && returnTo !== '#' ? t('返回题库结果','Back to library results') : t("全部论文", "All papers")}</AppLink>
    <div className="paper-detail-header">
      <div><h1><span>{paper.number}.</span> {paper.short_title}</h1><p className="paper-full-title">{paper.title}</p><div className="paper-detail-meta">{paper.venue && <span>{paper.venue}</span>}{paper.year && !paper.venue?.includes(String(paper.year)) && <span>{paper.year}</span>}{paper.tags.map((tag) => <span className="paper-topic" key={tag}>{tag}</span>)}</div></div>
      <div className="paper-detail-actions">{primary ? <div className="paper-primary-action"><AppLink className="button" href={withLibraryContext(primary.number,continueWith&&!discussionPracticeReturn(returnTo||'') ? '#' : returnTo)}><Code2 size={15} />{primaryLabel}<ArrowRight size={15} /></AppLink><small className="paper-action-target">{continueWith && <>{t("其他论文", "Another paper")} · </>}{nextTitle}</small></div> : <small role="status">{!scopeReady?t("公司练习范围尚未确认","Company practice scope is unconfirmed"):progressReady ? t("当前题库筛选中没有匹配模块", "No modules match this library view") : t("正在加载学习进度…", "Loading progress…")}</small>}<AppLink className="button subtle" href={paper.url} target="_blank" rel="noreferrer"><BookOpen size={15} />{paper.kind === "course" ? t("阅读参考资料", "Read references") : t("阅读论文", "Read paper")}<ExternalLink size={13} /></AppLink></div>
    </div>
    <div className="paper-at-a-glance" aria-label={t("练习概览", "Practice at a glance")}><div><Code2 size={17}/><span>{t("核心模块","Core modules")}<b>{progress.total} {t("个",progress.total === 1 ? "module" : "modules")}</b></span></div><div><Clock3 size={17}/><span>{t("预计练习","Estimated practice")}<b>{estimatedMinutes} min</b></span></div><div><CheckCircle2 size={17}/><span>{t("学习进度","Your progress")}<b>{solved} / {progress.total}</b></span></div></div>
    <div className="home-columns">
      <div className="main-column">
        <section className="paper-overview"><p>{t(paper.summary, paper.summary_en)}</p><details className="paper-scope-details"><summary>{t("练习范围", "Practice scope")}</summary><p>{t(paper.scope, paper.scope_en)}</p></details></section>
        <section className="paper-module-list">
          <div className="list-title"><h2>{t("核心模块", "Core modules")}</h2><span>{solved} / {progress.total} {t("已通过", "solved")}</span></div>
          <div className="paper-module-rows" role="list" data-favorites={hostedAppAvailable || undefined}>
          {visibleModules.map((problem) => {
            const title = locale === "en" ? problem.translations?.en?.title || problem.title : problem.title;
            const state = status(problem, saved[problem.id]);
            const isNext = progressReady && state !== "done" && next?.id === problem.id;
            const moduleStats = getProblemStats(stats, problem.id);
            return <div className="paper-module-shell" role="listitem" key={problem.id}><AppLink href={withLibraryContext(problem.number,returnTo)} className="paper-module-card" data-next={isNext || undefined} data-state={state}>
              <div className="module-card-top"><StateIcon state={state} /><div className="module-card-title"><h3><span>{problem.number}.</span> {title}</h3>{isNext && <span className="module-next-label">{state === "started" ? t("接着上次继续", "Pick up where you left off") : t("下一步推荐", "Up next")}<ArrowRight size={12}/></span>}</div><div className="module-card-tags"><span className="module-time"><Clock3 size={13} aria-hidden="true" />{problem.estimated_minutes || 25} min</span><span className={`difficulty ${problem.difficulty}`}>{level(problem.difficulty)}</span></div></div>
              <p>{t(problem.module.goal, problem.module.goal_en)}</p>
              <div className="module-card-code">
                <div className="module-functions"><span>{t("需要实现", "Implement")}</span>{problem.module.required_functions.map((name) => <code key={name}>{name}</code>)}</div>
                {!!problem.module.provided_functions.length && <div className="module-functions provided-functions"><span>{t("已提供", "Provided")}</span>{problem.module.provided_functions.map((name) => <code key={name}>{name}</code>)}</div>}
                {problem.module.optional && <span className="module-optional">{t("可选拓展", "Optional extension")}</span>}
                <ModuleJudgeBadge problem={problem} />
                {moduleStats && <ProblemStatsSummary stats={moduleStats} />}
              </div>
            </AppLink><div className="module-favorite-slot"><ProblemFavoriteButton problemId={problem.id} returnTo={withLibraryContext('papers/'+paper.id,returnTo)}/></div></div>;
          })}
          </div>
        </section>
      </div>
      <aside className="home-sidebar">
        <section className="paper-completion-card" aria-busy={!progressReady || undefined}>
          <div className="aside-title"><h3>{paper.kind === "course" ? t("专题进度", "Course progress") : t("论文进度", "Paper progress")}</h3>{progress.state === "done" && <StateIcon state="done" />}</div>
          <div className="paper-progress-ring" role="img" aria-label={progressReady ? t(`${progress.completed} / ${progress.total} 核心模块已通过，${percent}%`, `${progress.completed} of ${progress.total} core modules solved, ${percent}%`) : t("正在加载进度", "Loading progress")} style={{ background: `conic-gradient(var(--green) ${percent}%, var(--surface3) 0)` }}>
            <div><b>{solved}<span> / {progress.total}</span></b><small>{t("已通过", "Solved")}</small></div>
          </div>
          {levels.length > 0 && <ul className="paper-difficulty-list" aria-label={t("按难度统计", "By difficulty")}>
            {levels.map(row => <li key={row.value} className={`paper-difficulty-${row.value}`}><span>{level(row.value)}</span><b>{progressReady ? row.done : "—"}<small> / {row.total}</small></b><span className="paper-difficulty-track" aria-hidden="true"><span style={{ width: `${progressReady ? row.done / row.total * 100 : 0}%` }} /></span></li>)}
          </ul>}
          {progress.state === "done" && <div className="core-complete"><CheckCircle2 size={16} />{t("本篇核心模块已完成", "Selected core modules complete")}</div>}
          {next && <AppLink className="button subtle" href={withLibraryContext(next.number,returnTo)}>{nextState === "done" ? t("回顾", "Review") : nextState === "started" ? t("继续", "Continue") : t("开始", "Start")}<ArrowRight size={15} /></AppLink>}
          {completedCourse && <AppLink className="course-explore text-button" href="#">{t("探索全部论文", "Explore all papers")}<ArrowRight size={14} aria-hidden="true" /></AppLink>}
        </section>
        <section className="paper-source-card"><div className="aside-title"><h3>{paper.kind === "course" ? t("参考资料", "References") : t("论文信息", "Paper details")}</h3><BookOpen size={16} /></div><p>{authors}</p><AppLink className="text-button" href={paper.url} target="_blank" rel="noreferrer">{paper.arxiv ? `arXiv:${paper.arxiv}` : paper.kind === "course" ? t("参考资料", "References") : t("原始论文", "Original paper")}<ExternalLink size={13} /></AppLink></section>
      </aside>
    </div>
  </main>;
}

export default PaperDetail;
