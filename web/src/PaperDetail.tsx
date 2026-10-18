import { ArrowLeft, ArrowRight, BookOpen, CheckCircle2, ChevronRight, Clock3, Code2, ExternalLink } from "lucide-react";
import { ProblemFavoriteButton } from "./deferredPages";
import { useI18n } from "./i18n";
import { status } from "./types";
import type { Paper, Problem, Saved } from "./types";
import { paperModules, paperProgress } from "./libraryPractice";
import { recommendPractice } from "./practice-recommendation";
import { discussionPracticeReturn,withLibraryContext } from "./libraryNavigation";
import { getProblemStats, ProblemStatsSummary } from "./ProblemStats";
import type { ProblemStatsMap } from "./ProblemStats";
import { ModuleJudgeBadge } from "./JudgeCapability";
import { StateIcon, ProgressBar } from "./PaperState";

type Practice = { problems: Problem[]; saved: Record<string, Saved> };

export function PaperDetail({ paper, problems, practiceProblems = problems, saved, stats, returnTo, progressReady = true }: Practice & { paper: Paper; practiceProblems?: Problem[]; stats?: ProblemStatsMap; returnTo?: string; progressReady?: boolean }) {
  const { t, locale } = useI18n();
  const modules = paperModules(paper, problems);
  const progress = paperProgress(paper, { problems, saved });
  const scopedModules = modules.filter(module => practiceProblems.some(candidate => candidate.id === module.id));
  const next = progressReady ? recommendPractice(scopedModules,saved) || scopedModules[0] : undefined;
  const nextState = next && status(next,saved[next.id]);
  const completedCourse = paper.kind === "course" && progressReady && progress.state === "done";
  const continueWith = completedCourse ? recommendPractice(problems.filter(problem => !paper.module_ids.includes(problem.id)), saved) : undefined;
  const primary = continueWith || next;
  const primaryStarted = !!primary && status(primary,saved[primary.id]) === "started";
  const primaryLabel = continueWith
    ? primaryStarted ? t("继续练习", "Resume practice") : t("继续学习", "Continue learning")
    : nextState === "done" ? t("回顾实现", "Review implementation") : primaryStarted ? t("继续实现", "Continue practice") : t("开始实现", "Start implementing");
  const estimatedMinutes = modules.filter(module=>!module.module.optional).reduce((sum,module)=>sum+(module.estimated_minutes || 25),0);
  const nextTitle = primary && (locale === "en" ? primary.translations?.en?.title || primary.title : primary.title);
  const authors = Array.isArray(paper.authors) ? paper.authors.join(", ") : paper.authors;
  return <main className="home paper-detail">
    <a className="back-link paper-library-link" href={returnTo || '#'}><ArrowLeft size={15} />{discussionPracticeReturn(returnTo||'')?t('返回来源帖子','Back to source post'):returnTo?.startsWith('#/paper-registry')?t('返回论文库','Back to paper registry'):returnTo && returnTo !== '#' ? t('返回题库结果','Back to library results') : t("全部论文", "All papers")}</a>
    <div className="paper-detail-header">
      <div><h1><span>{paper.number}.</span> {paper.short_title}</h1><p className="paper-full-title">{paper.title}</p><div className="paper-detail-meta">{paper.venue && <span>{paper.venue}</span>}{paper.year && !paper.venue?.includes(String(paper.year)) && <span>{paper.year}</span>}{paper.tags.map((tag) => <span className="paper-topic" key={tag}>{tag}</span>)}</div></div>
      <div className="paper-detail-actions">{primary ? <div className="paper-primary-action"><a className="button" href={withLibraryContext(primary.number,continueWith&&!discussionPracticeReturn(returnTo||'') ? '#' : returnTo)}><Code2 size={15} />{primaryLabel}<ArrowRight size={15} /></a><small className="paper-action-target">{continueWith && <>{t("其他论文", "Another paper")} · </>}{nextTitle}</small></div> : <small role="status">{progressReady ? t("当前题库筛选中没有匹配模块", "No modules match this library view") : t("正在加载学习进度…", "Loading progress…")}</small>}<a className="button subtle" href={paper.url} target="_blank" rel="noreferrer"><BookOpen size={15} />{paper.kind === "course" ? t("阅读参考资料", "Read references") : t("阅读论文", "Read paper")}<ExternalLink size={13} /></a></div>
    </div>
    <div className="paper-at-a-glance" aria-label={t("练习概览", "Practice at a glance")}><div><Code2 size={17}/><span>{t("核心模块","Core modules")}<b>{progress.total} {t("个",progress.total === 1 ? "module" : "modules")}</b></span></div><div><Clock3 size={17}/><span>{t("预计练习","Estimated practice")}<b>{estimatedMinutes} min</b></span></div><div><CheckCircle2 size={17}/><span>{t("学习进度","Your progress")}<b>{progressReady ? progress.completed : "—"} / {progress.total}</b></span></div></div>
    <div className="home-columns">
      <div className="main-column">
        <section className="paper-overview"><p>{t(paper.summary, paper.summary_en)}</p><details className="paper-scope-details"><summary>{t("练习范围", "Practice scope")}</summary><p>{t(paper.scope, paper.scope_en)}</p></details></section>
        <section className="paper-module-list">
          <div className="list-title"><h2>{t("核心模块", "Core modules")}</h2><span>{progress.completed} / {progress.total} {t("已通过", "solved")}</span></div>
          {modules.map((problem) => {
            const title = locale === "en" ? problem.translations?.en?.title || problem.title : problem.title;
            const state = status(problem, saved[problem.id]);
            const isNext = progressReady && state !== "done" && next?.id === problem.id;
            return <div className="paper-module-shell" key={problem.id}><a href={withLibraryContext(problem.number,returnTo)} className="paper-module-card" data-next={isNext || undefined}>
              {isNext && <span className="module-next-label">{state === "started" ? t("接着上次继续", "Pick up where you left off") : t("下一步推荐", "Up next")}<ArrowRight size={12}/></span>}
              <div className="module-card-top"><StateIcon state={state} /><h3><span>{problem.number}.</span> {title}</h3><ChevronRight size={17} /></div>
              <p>{t(problem.module.goal, problem.module.goal_en)}</p>
              <div className="module-card-tags"><span className={`difficulty ${problem.difficulty}`}>{({ E: t("简单", "Easy"), M: t("中等", "Medium"), H: t("困难", "Hard") } as Record<string, string>)[problem.difficulty] || problem.difficulty}</span><span>{problem.estimated_minutes || 25} min</span><ModuleJudgeBadge problem={problem} /><span>{problem.module.optional ? t("可选拓展", "Optional extension") : t("核心模块", "Core module")}</span>{state === "done" && <span className="accepted">{t("已通过", "Solved")}</span>}</div>
              <div className="module-functions"><span>{t("需要实现", "Implement")}</span>{problem.module.required_functions.map((name) => <code key={name}>{name}</code>)}</div>
              {!!problem.module.provided_functions.length && <div className="module-functions provided-functions"><span>{t("已提供", "Provided")}</span>{problem.module.provided_functions.map((name) => <code key={name}>{name}</code>)}</div>}
              <ProblemStatsSummary stats={getProblemStats(stats, problem.id)} />
            </a><div className="module-favorite-slot"><ProblemFavoriteButton problemId={problem.id} returnTo={withLibraryContext('papers/'+paper.id,returnTo)}/></div></div>;
          })}
        </section>
      </div>
      <aside className="home-sidebar">
        <section className="paper-completion-card"><div className="aside-title"><h3>{paper.kind === "course" ? t("专题进度", "Course progress") : t("论文进度", "Paper progress")}</h3><StateIcon state={progress.state} /></div><div className="paper-completion-count"><b>{progress.completed}</b><span>/ {progress.total}</span></div><p>{t("核心模块已通过", "core modules solved")}</p><ProgressBar {...progress} />{progress.state === "done" && <div className="core-complete"><CheckCircle2 size={16} />{t("本篇核心模块已完成", "Selected core modules complete")}</div>}{next && <a className="button" href={withLibraryContext(next.number,returnTo)}>{nextState === "done" ? t("回顾模块", "Review modules") : nextState === "started" ? t("继续实现", "Continue practice") : t("开始实现", "Start implementing")}<ArrowRight size={15} /></a>}{completedCourse && <a className="course-explore text-button" href="#">{t("探索全部论文", "Explore all papers")}<ArrowRight size={14} aria-hidden="true" /></a>}</section>
        <section className="paper-source-card"><div className="aside-title"><h3>{paper.kind === "course" ? t("参考资料", "References") : t("论文信息", "Paper details")}</h3><BookOpen size={16} /></div><p>{authors}</p><a className="text-button" href={paper.url} target="_blank" rel="noreferrer">{paper.arxiv ? `arXiv:${paper.arxiv}` : paper.kind === "course" ? t("参考资料", "References") : t("原始论文", "Original paper")}<ExternalLink size={13} /></a></section>
      </aside>
    </div>
  </main>;
}

export default PaperDetail;
