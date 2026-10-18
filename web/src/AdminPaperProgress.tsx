import {useId} from 'react';
import {useI18n} from './i18n';
import type {Paper} from './AdminParts';

export default function AdminPaperProgress({paper}:{paper:Paper}){
 const {t}=useI18n(),id=useId();
 const steps=[t('论文初审','Paper intake'),t('生成与验证','Generation & validation'),t('候选版本审核','Candidate review'),t('正式发布','Publication')];
 const stage=paper.status==='pending_review'?0:['suitability','generation','validation'].includes(paper.status)?1:['pool','ready'].includes(paper.status)?2:paper.status==='published'?3:undefined;
 const approved=paper.maintainer_review?.approved===true;
 const message=paper.status==='pending_review'?t('核对论文与出题方向。初审通过后，才会进入适配评估与生成。','Review the paper and proposed exercise. Intake approval allows suitability assessment and generation.')
  :stage===1?t('处理正在进行。可以查看投稿进度；当前不能重复批准生成。','Processing is underway. Open the submission to follow progress; generation cannot be approved again while it is running.')
  :stage===2?approved?t(`当前候选 v${paper.version} 已通过管理员审核。正式发布仍需满足验证、社区评审和发布流程。`,`Candidate v${paper.version} has administrator approval. Validation, community review and the publication process still apply.`):t(`请核对候选 v${paper.version} 的题面、实现要求和验证结果。批准仅适用于这个版本。`,`Review candidate v${paper.version}, its exercise requirements and validation results. Approval applies only to this version.`)
  :stage===3?t('这篇投稿已发布，可查看投稿记录；此处不再修改审核决定。','This submission is published. Its record remains available; review decisions can no longer be changed here.')
  :['changes','rejected'].includes(paper.status)?t('先核对审核意见和当前内容。重新批准会回到适配评估与生成流程，旧候选版本不会直接发布。','Review the feedback and current content. Approving again restarts assessment and generation; it does not publish an earlier candidate.')
  :t('暂时无法识别审核阶段，请刷新后核对投稿状态。','The review stage is unavailable. Refresh and check the submission status.');
 return <section className="admin-paper-progress" aria-labelledby={id}>
  <h3 id={id}>{t('审核进度与下一步','Review progress & next step')}</h3>
  <ol aria-label={t('论文审核阶段','Paper review stages')}>{steps.map((label,index)=><li key={index} aria-current={stage===index?'step':undefined}><span aria-hidden="true">{index+1}</span>{label}</li>)}</ol>
  <p>{message}</p>
 </section>;
}
