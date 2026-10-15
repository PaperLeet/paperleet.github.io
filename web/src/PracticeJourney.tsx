import { ArrowRight, BookOpen, CheckCircle2, Clock3, Code2, FlaskConical } from "lucide-react";
import { useI18n } from "./i18n";
import { useAccount } from "./social";
import { status } from "./types";
import type { Problem, Saved } from "./types";
import { recommendPractice } from "./practice-recommendation";
import { BilingualText } from "./BilingualText";
import "./practice-journey.css";

export function PracticeStart({ problems, saved, ready, onBrowse }: {
  problems: Problem[];
  saved: Record<string, Saved>;
  ready: boolean;
  onBrowse?: () => void;
}) {
  const { t } = useI18n();
  const { health, user } = useAccount();
  const core = problems.filter((p) => !p.module.optional);
  const solved = core.filter((p) => status(p, saved[p.id]) === "done").length;
  const attempted = core.some((p) => status(p, saved[p.id]) !== "todo");
  const next = ready ? recommendPractice(problems, saved) : undefined;
  const resuming = next && status(next, saved[next.id]) === "started";
  const text = (zh: string, en: string) => <BilingualText zh={zh} en={en} />;
  return <section className="practice-start" data-returning={ready && attempted || undefined} aria-label={t("你的下一步", "Your next step")}>
    <div className="practice-start-copy">
      <span className="journey-kicker">{text("读懂论文 · 写出核心算法", "Read the paper. Implement the idea.")}</span>
      <h2>{!ready ? text("正在加载学习进度…", "Loading your progress…") : !next ? text("本范围核心模块已完成", "Core modules complete in this view") : attempted ? text("接着上次，继续前进", "Keep your learning going") : text("从第一道算法题开始", "Start with your first algorithm")}</h2>
      <p>{text("只实现论文的关键函数，配套代码已提供。用测试验证，再到实验台观察算法行为。", "Implement the key functions with supporting code provided. Check your work with tests, then explore how it behaves in the lab.")}</p>
      <ol className="journey-steps">
        <li><BookOpen size={15} aria-hidden="true" />{text("读目标和公式", "Read the goal")}</li>
        <li><Code2 size={15} aria-hidden="true" />{text("补全核心函数", "Implement the core")}</li>
        <li><FlaskConical size={15} aria-hidden="true" />{text("运行测试与实验", "Test and explore")}</li>
      </ol>
      {onBrowse && <div className="journey-links"><button className="button subtle" onClick={onBrowse}>{t('浏览论文题库','Explore papers')}<ArrowRight size={14}/></button></div>}
    </div>
    <div className="practice-recommendation" aria-live="polite" aria-busy={!ready}>
      {next ? <>
        <span className="journey-kicker">{resuming ? text("继续未完成的练习", "Resume your unfinished module") : attempted ? text("下一道推荐练习", "Your next recommended module") : text("推荐第一题", "Recommended first module")}</span>
        <h3><BilingualText zh={next.title} en={next.translations?.en?.title || next.title} /></h3>
        <p><BilingualText zh={next.module.goal} en={next.module.goal_en} /></p>
        <div className="journey-meta"><span>{({ E: text("简单", "Easy"), M: text("中等", "Medium"), H: text("困难", "Hard") } as Record<string, ReturnType<typeof text>>)[next.difficulty]}</span><span><Clock3 size={13} aria-hidden="true" />{next.estimated_minutes || 25} min</span></div>
        <a className="button" href={`#/${next.number}`}>{resuming ? text("继续实现", "Resume practice") : text("开始这道题", "Start this module")}<ArrowRight size={15} aria-hidden="true" /></a>
        <small>{next.browser_judge ? (health?.mode === "hosted" ? text("可直接在浏览器运行；登录后可由服务端确认账号成绩。", "Run in your browser now; sign in for server-verified account progress.") : text("浏览器 CPU 判题 · 无需安装或 GPU", "Browser CPU judging · no setup or GPU needed")) : health?.mode === "hosted" && !user ? text("可先阅读题面；登录内测账号后运行测试。", "Read first; sign in with your beta account to run tests.") : health?.mode === "local" ? text("本地 CPU 判题 · 未登录的代码仅在当前页面保留", "Local CPU judging · guest code stays only on this page") : text("题目只要求实现所列核心函数。", "Only the listed core functions are required.")}</small>
      </> : ready ? <><CheckCircle2 size={25} aria-hidden="true" /><h3>{text("做得不错，回顾一下收获", "Take a moment to review")}</h3><p>{text("可以重访实验和坑点，或切换合集探索其他方向。", "Revisit experiments and notes, or choose another collection to explore.")}</p></> : <p role="status">{text("进度就绪后推荐下一道题。", "Your next module will appear when progress is ready.")}</p>}
      {ready && <span className="journey-progress">{solved} / {core.length} {text("核心模块已通过", "core modules solved")}</span>}
    </div>
  </section>;
}

export function PracticeNextStep({ next, samePaper, onNotes, onLab }: {
  next?: Problem;
  samePaper: boolean;
  onNotes: () => void;
  onLab: () => void;
}) {
  const { locale, t } = useI18n();
  return <section className="practice-next-step" aria-label={t("通过后的下一步", "After passing")}>
    <strong><CheckCircle2 size={16} aria-hidden="true" />{t("核心实现已通过，接下来…", "Core implementation passed. Next…")}</strong>
    <div className="journey-review-actions"><button className="text-button" onClick={onNotes}>{t("复盘常见坑点", "Review common pitfalls")}</button><button className="text-button" onClick={onLab}>{t("查看实验结果", "Explore the experiment")}</button></div>
    {next ? <a className="journey-next-link" href={`#/${next.number}`}><span><small>{samePaper ? t("继续本篇论文", "Continue this paper") : t("推荐下一题", "Try another module")}</small><b>{locale === "en" ? next.translations?.en?.title || next.title : next.title}</b></span><ArrowRight size={18} aria-hidden="true" /></a> : <a className="journey-next-link" href="#">{t("核心题库已完成 · 返回题库回顾", "Core catalog complete · return to review")}<ArrowRight size={18} aria-hidden="true" /></a>}
  </section>;
}
