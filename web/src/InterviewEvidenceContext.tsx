import {Info,Users} from 'lucide-react';
import {useI18n} from './i18n';
import type {InterviewSummary} from './interviewModels';

export default function InterviewEvidenceContext({summary}:{summary:InterviewSummary}){
 const {t}=useI18n();
 return <aside className="interview-evidence-context" aria-label={t('样本与统计口径','Sample and counting method')}>
  <div><Users size={15}/><b>{t(`${summary.contributors} 位分享者 · ${summary.reports} 篇公开面经`,`${summary.contributors} ${summary.contributors===1?'contributor':'contributors'} · ${summary.reports} public ${summary.reports===1?'report':'reports'}`)}</b></div>
  <p>{t('按当前公司、岗位、轮次、知识点和发布时间汇总。收藏只影响你的阅读列表，不改变公开统计。','Counts follow the selected company, role, stage, concept and publication period. Saving reports changes your reading list, never the public ranking.')}</p>
  <p><Info size={13}/>{t('这是社区样本中的知识点提及，不代表实际考察概率或官方题库。可打开每一项的来源面经，核对具体语境。','These are concept mentions in community reports, not interview probabilities or an official question bank. Open the source experiences to check their context.')}</p>
 </aside>;
}
