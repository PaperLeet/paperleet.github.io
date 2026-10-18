import AppLink from './AppLink';
import {ArrowRight,CheckCircle2} from 'lucide-react';
import {useI18n} from './i18n';
import {discussionPracticeReturn,withLibraryContext} from './libraryNavigation';
import type {Problem} from './types';

export function PracticeNextStep({ next, samePaper, onNotes, onLab, onSolution, returnTo, scopeLabel, completionLabel }: {
  next?: Problem;
  samePaper: boolean;
  scopeLabel?:string;
  completionLabel?:string;
  onNotes: () => void;
  onLab: () => void;
  onSolution?:()=>void;
  returnTo?: string;
}) {
  const { locale, t } = useI18n();
  return <section className="practice-next-step" aria-label={t("通过后的下一步", "After passing")}>
    <strong><CheckCircle2 size={16} aria-hidden="true" />{t("核心实现已通过，接下来…", "Core implementation passed. Next…")}</strong>
    <div className="journey-review-actions"><button className="text-button" onClick={onNotes}>{t("复盘常见坑点", "Review common pitfalls")}</button><button className="text-button" onClick={onLab}>{t("查看实验结果", "Explore the experiment")}</button>{onSolution&&<button type="button" className="text-button" onClick={onSolution}>{t('分享解题思路','Share your approach')}</button>}</div>
    {next ? <AppLink className="journey-next-link" href={withLibraryContext(next.number,returnTo)}><span><small>{scopeLabel || (samePaper ? t("继续本篇论文", "Continue this paper") : t("推荐下一题", "Try another module"))}</small><b>{locale === "en" ? next.translations?.en?.title || next.title : next.title}</b></span><ArrowRight size={18} aria-hidden="true" /></AppLink> : <AppLink className="journey-next-link" href={returnTo || '#'}>{completionLabel || (discussionPracticeReturn(returnTo||'')?t('返回来源帖子，继续交流','Back to source post to discuss'):returnTo && returnTo !== "#" ? t("本范围核心模块已完成 · 返回回顾", "Core modules complete in this view · return to review") : t("核心题库已完成 · 返回题库回顾", "Core catalog complete · return to review"))}<ArrowRight size={18} aria-hidden="true" /></AppLink>}
  </section>;
}
