import {useI18n} from './i18n';
import {readInterviewRoute} from './interviewNavigation';
import {preparationProgress} from './interviewPreparation';
import {interviewPracticeBatch} from './interviewPracticeBatch';
import {PracticeNextStep} from './PracticeNextStep';
import {readLibraryRoute,withLibraryContext} from './libraryNavigation';
import {status} from './types';
import type {Problem,Saved} from './types';

export default function InterviewPracticeNextStep({current,problems,saved,next,returnTo,onNotes,onLab,onSolution,course,scopeReady=true}:{current:Problem;problems:Problem[];saved:Record<string,Saved>;next?:Problem;returnTo?:string;onNotes:()=>void;onLab:()=>void;onSolution?:()=>void;course?:boolean;scopeReady?:boolean}){
 const {t,locale}=useI18n();
 const company=!!readLibraryRoute(returnTo||'')?.filters.company;
 const track=returnTo?readInterviewRoute(returnTo)?.track:undefined;
 const progress=track?preparationProgress(track,problems,saved,current):undefined;
 const scoped=progress?.modules.some(problem=>problem.id===current.id)?progress:undefined;
 const batch=interviewPracticeBatch(returnTo,problems,saved,current);
 const candidate=batch?batch.next:scoped?scoped.next:next;
 const courseModules=course?problems.filter(problem=>problem.paper_id===current.paper_id&&!problem.module.optional):[];
 const completedCourse=!company&&!candidate&&!batch&&!scoped&&courseModules.length>0&&courseModules.every(problem=>status(problem,saved[problem.id])==='done');
 return <PracticeNextStep next={scopeReady?candidate:undefined} samePaper={!!candidate&&candidate.paper_id===current.paper_id} returnTo={completedCourse?withLibraryContext(`papers/${current.paper_id}`,returnTo):returnTo} onNotes={onNotes} onLab={onLab} onSolution={onSolution}
  scopeLabel={company?t('继续当前公司练习','Continue current company practice'):batch?t('继续当前准备批次','Continue this preparation batch'):scoped?t(`继续「${scoped.path.name}」`,`Continue ${locale==='en'?scoped.path.name_en:scoped.path.name}`):course&&candidate?.paper_id===current.paper_id?t("继续本专题","Continue this course"):undefined}
  completionLabel={!scopeReady?t('公司练习范围尚未确认 · 返回核对','Company scope is unconfirmed · return to review'):company?(!problems.length?t('公司范围暂无匹配练习 · 调整筛选','No exercises match this company scope · adjust filters'):!problems.some(problem=>problem.id===current.id)?t('练习范围已调整 · 返回公司练习','Practice scope changed · return to company practice'):t('本范围核心练习已完成 · 返回回顾','Core exercises complete in this scope · return to review')):completedCourse?t('专题已完成 · 查看下一步','Course complete · choose your next step'):batch?(!batch.currentIncluded?t('这组练习已调整 · 返回准备列表','This practice batch has changed · return to preparation'):batch.missing?t('部分练习暂不可用 · 返回准备列表','Some batch modules are unavailable · return to preparation'):t('本组练习已完成 · 返回准备列表','Practice batch complete · return to preparation')):scoped?(scoped.missing?t('部分路径练习暂不可用 · 返回准备页','Some path modules are unavailable · return to preparation'):t('准备路径已完成 · 返回复盘','Preparation path complete · return to review')):undefined}/>;
}
