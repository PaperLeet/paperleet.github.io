import {useI18n} from './i18n';
import {readInterviewRoute} from './interviewNavigation';
import {preparationProgress} from './interviewPreparation';
import {interviewPracticeBatch} from './interviewPracticeBatch';
import {PracticeNextStep} from './PracticeNextStep';
import type {Problem,Saved} from './types';

export default function InterviewPracticeNextStep({current,problems,saved,next,returnTo,onNotes,onLab,course}:{current:Problem;problems:Problem[];saved:Record<string,Saved>;next?:Problem;returnTo?:string;onNotes:()=>void;onLab:()=>void;course?:boolean}){
 const {t,locale}=useI18n();
 const track=returnTo?readInterviewRoute(returnTo)?.track:undefined;
 const progress=track?preparationProgress(track,problems,saved,current):undefined;
 const scoped=progress?.modules.some(problem=>problem.id===current.id)?progress:undefined;
 const batch=interviewPracticeBatch(returnTo,problems,saved,current);
 const candidate=batch?batch.next:scoped?scoped.next:next;
 return <PracticeNextStep next={candidate} samePaper={!!candidate&&candidate.paper_id===current.paper_id} returnTo={returnTo} onNotes={onNotes} onLab={onLab}
  scopeLabel={batch?t('继续当前准备批次','Continue this preparation batch'):scoped?t(`继续「${scoped.path.name}」`,`Continue ${locale==='en'?scoped.path.name_en:scoped.path.name}`):course&&candidate?.paper_id===current.paper_id?t("继续本专题","Continue this course"):undefined}
  completionLabel={batch?(!batch.currentIncluded?t('这组练习已调整 · 返回准备列表','This practice batch has changed · return to preparation'):batch.missing?t('部分练习暂不可用 · 返回准备列表','Some batch modules are unavailable · return to preparation'):t('本组练习已完成 · 返回准备列表','Practice batch complete · return to preparation')):scoped?(scoped.missing?t('部分路径练习暂不可用 · 返回准备页','Some path modules are unavailable · return to preparation'):t('准备路径已完成 · 返回复盘','Preparation path complete · return to review')):undefined}/>;
}
