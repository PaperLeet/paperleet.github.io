import AppLink from './AppLink';
import { ArrowRight, BookOpen, CheckCircle2, Clock3, Code2, FlaskConical } from "lucide-react";
import { useI18n } from "./i18n";
import { useAccount } from "./social";
import { status } from "./types";
import type { Problem, Saved } from "./types";
import { recommendPractice } from "./practice-recommendation";
import "./practice-journey.css";
import { withLibraryContext } from './libraryNavigation';

export function PracticeStart({ problems, saved, ready, returnTo, onBrowse }: {
  problems: Problem[];
  saved: Record<string, Saved>;
  ready: boolean;
  returnTo?: string;
  onBrowse?: () => void;
}) {
  const { t, locale } = useI18n();
  const { health, user } = useAccount();
  const core = problems.filter((p) => !p.module.optional);
  const solved = core.filter((p) => status(p, saved[p.id]) === "done").length;
  const attempted = core.some((p) => status(p, saved[p.id]) !== "todo");
  const next = ready ? recommendPractice(problems, saved) : undefined;
  const resuming = next && status(next, saved[next.id]) === "started";
  const title = next && (locale === "en" ? next.translations?.en?.title || next.title : next.title);
  return <section className="practice-start" data-returning={ready && attempted || undefined} aria-label={t("你的下一步", "Your next step")}>
    <div className="practice-start-copy">
      <h2>{!ready ? t("正在加载学习进度…", "Loading your progress…") : !core.length ? t("当前范围暂无核心模块", "No core modules in this view") : !next ? t("本范围核心模块已完成", "Core modules complete in this view") : attempted ? t("接着上次，继续前进", "Keep your learning going") : t("把论文里的想法，写成代码。", "Turn paper ideas into working code.")}</h2>
      {!attempted && <ol className="journey-steps">
        <li><BookOpen size={15} aria-hidden="true" />{t("读目标和公式", "Read the goal")}</li>
        <li><Code2 size={15} aria-hidden="true" />{t("补全核心函数", "Implement the core")}</li>
        <li><FlaskConical size={15} aria-hidden="true" />{t("运行测试与实验", "Test and explore")}</li>
      </ol>}
      {onBrowse && <div className="journey-links"><button className="button subtle" onClick={onBrowse}>{t("浏览论文题库","Explore papers")}<ArrowRight size={14}/></button></div>}
    </div>
    <div className="practice-recommendation" aria-live="polite" aria-busy={!ready}>
      {next ? <>
        <span className="journey-kicker">{resuming ? t("继续未完成的练习", "Resume your unfinished module") : attempted ? t("下一道推荐练习", "Your next recommended module") : t("推荐第一题", "Recommended first module")}</span>
        <h3>{title}</h3>
        <p>{t(next.module.goal, next.module.goal_en)}</p>
        <div className="journey-meta"><span>{({ E: t("简单", "Easy"), M: t("中等", "Medium"), H: t("困难", "Hard") } as Record<string, string>)[next.difficulty]}</span><span><Clock3 size={13} aria-hidden="true" />{next.estimated_minutes || 25} min</span></div>
        <AppLink className="button" href={withLibraryContext(next.number,returnTo)}>{resuming ? t("继续实现", "Resume practice") : t("开始这道题", "Start this module")}<ArrowRight size={15} aria-hidden="true" /></AppLink>
        <small>{next.browser_judge ? (health?.mode === "hosted" ? t("可直接在浏览器运行；登录后可由服务端确认账号成绩。", "Run in your browser now; sign in for server-verified account progress.") : t("浏览器 CPU 判题 · 无需安装或 GPU", "Browser CPU judging · no setup or GPU needed")) : health?.mode === "hosted" && !user ? t("可先阅读题面；登录内测账号后运行测试。", "Read first; sign in with your beta account to run tests.") : health?.mode === "local" ? t("本地 CPU 判题 · 草稿自动保存在此浏览器", "Local CPU judging · drafts saved in this browser") : t("题目只要求实现所列核心函数。", "Only the listed core functions are required.")}</small>
      </> : ready ? <>{core.length ? <CheckCircle2 size={25} aria-hidden="true" /> : <BookOpen size={25} aria-hidden="true" />}<h3>{core.length ? t("做得不错，回顾一下收获", "Take a moment to review") : t("调整筛选，开始练习", "Adjust your filters to begin")}</h3><p>{core.length ? t("可以重访实验和坑点，或切换合集探索其他方向。", "Revisit experiments and notes, or choose another collection to explore.") : t("试试其他关键词、减少筛选条件，或切换合集。", "Try another query, remove a filter, or choose a different collection.")}</p></> : <p role="status">{t("进度就绪后推荐下一道题。", "Your next module will appear when progress is ready.")}</p>}
      {ready && <div className="journey-completion"><div className="journey-completion-track" aria-hidden="true"><span style={{width:`${core.length ? solved / core.length * 100 : 0}%`}} /></div><span className="journey-progress">{solved} / {core.length} {t("核心模块已通过", "core modules solved")}</span></div>}
    </div>
  </section>;
}

export {PracticeNextStep} from './PracticeNextStep';
