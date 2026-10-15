import {useI18n} from './i18n';
import {readInterviewRoute} from './interviewNavigation';
import {preparationProgress} from './interviewPreparation';
import {PracticeNextStep} from './PracticeNextStep';
import type {Problem,Saved} from './types';

export default function InterviewPracticeNextStep({current,problems,saved,next,returnTo,onNotes,onLab}:{current:Problem;problems:Problem[];saved:Record<string,Saved>;next?:Problem;returnTo?:string;onNotes:()=>void;onLab:()=>void}){
 const {t,locale}=useI18n();
 const track=returnTo?readInterviewRoute(returnTo)?.track:undefined;
 const progress=track?preparationProgress(track,problems,saved,current):undefined;
 const scoped=progress?.modules.some(problem=>problem.id===current.id)?progress:undefined;
 const candidate=scoped?scoped.next:next;
 return <PracticeNextStep next={candidate} samePaper={!!candidate&&candidate.paper_id===current.paper_id} returnTo={returnTo} onNotes={onNotes} onLab={onLab}
  scopeLabel={scoped?t(`继续「${scoped.path.name}」`,`Continue ${locale==='en'?scoped.path.name_en:scoped.path.name}`):undefined}
  completionLabel={scoped?(scoped.missing?t('部分路径练习暂不可用 · 返回准备页','Some path modules are unavailable · return to preparation'):t('准备路径已完成 · 返回复盘','Preparation path complete · return to review')):undefined}/>;
}
