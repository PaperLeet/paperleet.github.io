import {useState} from 'react';
import {BookOpen} from 'lucide-react';
import {useI18n} from './i18n';
import PreparationGuide from './PreparationGuide';
import {hasPreparationGuide} from './preparationGuides';
import {interviewCatalog} from './interviewModels';
import {interviewHref,type InterviewFilters} from './interviewNavigation';
import {rolePreparationPaths} from './interviewPreparation';
import {withLibraryContext} from './libraryNavigation';
import type {Problem} from './types';

export default function PreparationHandbook({filters,problems,catalogError,onRetryCatalog}:{filters:InterviewFilters;problems:Problem[];catalogError?:string;onRetryCatalog?:()=>void}){
 const {t,locale}=useI18n();
 const [choice,setChoice]=useState('');
 const selectedPath=interviewCatalog.paths.find(path=>path.id===filters.track);
 const paths=selectedPath?[selectedPath]:rolePreparationPaths(filters.role),ids=paths.flatMap(path=>path.modules);
 const byId=new Map(problems.map(problem=>[problem.id,problem]));
 const candidates=selectedPath&&!filters.concept?selectedPath.modules.flatMap(id=>byId.has(id)?[byId.get(id)!]:[]):problems;
 const offered=candidates.filter(problem=>hasPreparationGuide(problem.id)&&(filters.concept?problem.id===filters.concept:ids.includes(problem.id)));
 const selected=offered.find(problem=>problem.id===choice)||offered[0];
 if(filters.concept&&!hasPreparationGuide(filters.concept))return null;
 const title=(problem:Problem)=>locale==='en'?problem.translations?.en?.title||problem.title:problem.title;
 const track=selected?(paths.find(path=>path.modules.includes(selected.id))||interviewCatalog.paths.find(path=>path.modules.includes(selected.id)))?.id||'':'';
 const returnTo=selected?interviewHref({...filters,view:'reports',concept:selected.id,track,page:1,q:''}):undefined;
 return <section className="preparation-handbook" id="preparation-handbook" aria-labelledby="preparation-handbook-heading">
  <div className="interview-section-title"><span><BookOpen size={18}/><h2 id="preparation-handbook-heading" tabIndex={-1}>{t('面试自测手册','Interview study handbook')}</h2></span><small>{offered.length?t(`${offered.length} 个知识点 · 每个 3 个自测角度`,`${offered.length} concepts · 3 prompts each`):t('基于站内练习整理','Based on PaperLeet exercises')}</small></div>
  <p className="preparation-handbook-intro">{t('把公式讲清楚，把实现推出来，再用边界情况检验理解。题目约定和教学范围以对应题面为准。','Explain the formula, work through implementation and test your understanding with edge cases. The linked exercise defines its conventions and teaching scope.')}</p>
  {selected?<><label className="preparation-guide-selector"><span>{t('选择自测知识点','Choose a study concept')}</span><select value={selected.id} onChange={event=>setChoice(event.target.value)}>{offered.map(problem=><option key={problem.id} value={problem.id}>{title(problem)}</option>)}</select></label>{catalogError&&<p className="preparation-guide-catalog-notice" role="status">{t('正在显示已加载的练习。','Showing the exercises already loaded.')}<button type="button" className="text-button" onClick={onRetryCatalog}>{t('重试题库','Retry catalog')}</button></p>}<PreparationGuide key={filters.concept||'handbook'} problemId={selected.id} practiceHref={withLibraryContext(selected.number,returnTo)}/></>:<div className="preparation-guide-catalog-notice" role="status"><p>{catalogError?t('练习目录暂时无法读取，自测手册会在目录恢复后显示。','The exercise catalog is unavailable. Study prompts will appear when it recovers.'):problems.length?t('当前范围暂无可用的自测知识点。','No study concepts are available in this view.'):t('正在加载自测知识点…','Loading study concepts…')}</p>{catalogError&&<button type="button" className="text-button" onClick={onRetryCatalog}>{t('重试题库','Retry catalog')}</button>}</div>}
 </section>;
}
