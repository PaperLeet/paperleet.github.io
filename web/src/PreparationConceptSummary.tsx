import {useId} from 'react';
import {ArrowRight,Code2} from 'lucide-react';
import AppLink from './AppLink';
import {useI18n} from './i18n';
import type {Problem} from './types';
import './preparation-concept-summary.css';

/** Describe the selected public exercise without assuming grades or company evidence. */
export default function PreparationConceptSummary({problem,practiceHref,experiencesHref}:{problem:Problem;practiceHref:string;experiencesHref?:string}){
 const {t,locale}=useI18n(),heading=useId();
 const title=locale==='en'?problem.translations?.en?.title||problem.title:problem.title;
 const goal=locale==='en'?problem.module?.goal_en||problem.module?.goal:problem.module?.goal;
 const difficulty=({E:t('简单','Easy'),M:t('中等','Medium'),H:t('困难','Hard')} as Record<string,string>)[problem.difficulty];
 const names=[...new Set(problem.module?.required_functions||[])];
 const minutes=typeof problem.estimated_minutes==='number'&&Number.isFinite(problem.estimated_minutes)&&problem.estimated_minutes>0?problem.estimated_minutes:undefined;
 return <section className="preparation-selected-concept" aria-labelledby={heading}>
  <div className="preparation-selected-copy">
   <div className="preparation-selected-meta"><span>{t('当前自测知识点','Selected study concept')}</span>{Number.isSafeInteger(problem.number)&&problem.number>0&&<span>#{problem.number}</span>}{difficulty&&<span data-difficulty={problem.difficulty}>{difficulty}</span>}{minutes!==undefined&&<span>{t(`预计 ${minutes} 分钟`,`Estimated ${minutes} min`)}</span>}</div>
   <h3 id={heading}>{title}</h3>{goal&&<p className="preparation-selected-goal">{goal}</p>}
   {(names.length>0||problem.paper?.title)&&<dl>{names.length>0&&<div><dt>{t('待实现函数','Functions to implement')}</dt><dd className="preparation-selected-functions">{names.map(name=><code key={name}>{name}</code>)}</dd></div>}{problem.paper?.title&&<div><dt>{t('学习材料','Study source')}</dt><dd>{problem.paper.title}</dd></div>}</dl>}
  </div>
  <div className="preparation-selected-actions" role="group" aria-label={t(`${title}的下一步`,`Next steps for ${title}`)}>
   <AppLink className="button subtle preparation-selected-practice" href={practiceHref}><Code2 size={16} aria-hidden="true"/>{t('打开练习','Open practice')}<ArrowRight size={14} aria-hidden="true"/></AppLink>
   {experiencesHref&&<AppLink className="preparation-selected-experiences" href={experiencesHref}>{t('浏览相关面经','Browse related experiences')}<ArrowRight size={14} aria-hidden="true"/></AppLink>}
  </div>
 </section>;
}
