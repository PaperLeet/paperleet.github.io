import {BookOpen, CheckCircle2, ExternalLink, FileCode2} from 'lucide-react';
import type {Problem} from './types';
import type {ProblemStatistics} from './ProblemStats';
import {ProblemStats} from './ProblemStatsView';
import {ProblemCompanyTags} from './deferredPages';
import {withLibraryContext} from './libraryNavigation';
import {useI18n} from './i18n';

export type ProblemOverviewProps = {
  problem: Problem;
  prerequisite?: Pick<Problem, 'number'>;
  returnTo?: string;
  course: boolean;
  notesUnlocked: boolean;
  stats?: ProblemStatistics;
  statsLoading: boolean;
  onCode: () => void;
};

/** Public reading metadata; the statement and editor load independently in App. */
export default function ProblemOverview({problem,prerequisite,returnTo,course,notesUnlocked,stats,statsLoading,onCode}:ProblemOverviewProps) {
  const {t}=useI18n();
  return <>
    <div className="problem-heading">
      <div className="heading-kicker">
        <FileCode2 size={18} />
        <span>
          {t("模块", "MODULE")} {problem.number}
        </span>
        {notesUnlocked && (
          <span className="accepted">
            <CheckCircle2 size={14} />
            {t("已通过", "Solved")}
          </span>
        )}
      </div>
      <h1>
        {problem.number}. {problem.title}
      </h1>
      <ProblemCompanyTags problemId={problem.id} number={problem.number} onCode={onCode}/>
      <div className="heading-tags">
        <span className={`difficulty ${problem.difficulty}`}>
          {({ E: t("简单", "Easy"), M: t("中等", "Medium"), H: t("困难", "Hard") } as Record<string,string>)[problem.difficulty] || problem.difficulty}
        </span>
        <span>PyTorch</span>
        <span>{problem.estimated_minutes || 25} min</span>
        {problem.parent && (
          <a
            href={prerequisite ? withLibraryContext(prerequisite.number,returnTo) : `#/${problem.parent}`}
          >
            {t("前置题", "Prerequisite")}:{" "}
            {prerequisite?.number || "—"}
          </a>
        )}
      </div>
      <ProblemStats stats={stats} loading={statsLoading} />
    </div>
    <a
      className="paper-card"
      href={
        problem.paper?.url ||
        `https://arxiv.org/abs/${problem.paper?.arxiv}`
      }
      target="_blank"
      rel="noreferrer"
    >
      <BookOpen size={19} />
      <div>
        <small>
          {course ? t("参考资料", "REFERENCES") : t("阅读论文", "READ THE PAPER")}{" "}
          <span>{problem.paper?.venue}</span>
        </small>
        <strong>{problem.paper?.title}</strong>
        <p>
          {Array.isArray(problem.paper?.authors)
            ? problem.paper.authors.join(", ")
            : problem.paper?.authors}
        </p>
      </div>
      <ExternalLink size={14} />
    </a>
    {problem.module && (
      <section className="module-scope-card">
        <h3>{t("本模块的目标", "Your implementation goal")}</h3>
        <p>{t(problem.module.goal, problem.module.goal_en)}</p>
        <p className="module-scope-detail">
          {t(problem.module.scope, problem.module.scope_en)}
        </p>
        <div className="module-functions">
          <span>{t("需要实现", "Implement")}</span>
          {problem.module.required_functions.map((name) => (
            <code key={name}>{name}</code>
          ))}
        </div>
        {!!problem.module.provided_functions.length && (
          <div className="module-functions provided-functions">
            <span>{t("已提供", "Provided")}</span>
            {problem.module.provided_functions.map((name) => (
              <code key={name}>{name}</code>
            ))}
          </div>
        )}
      </section>
    )}
  </>;
}
