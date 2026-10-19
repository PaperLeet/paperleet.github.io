import AppLink from './AppLink';
import {useI18n} from './i18n';
import {topicTitle,topicHref} from './communityTopics';
import {solutionsHref} from './communityNavigation';
import {interviewLabel} from './interviewModels';
import type {HighlightContext as Context} from './communityHighlightsModel';

export default function HighlightContext({context,newTab=false}:{context?:Context;newTab?:boolean}){
 const {t,locale}=useI18n();
 if(!context)return null;
 const kind=context.kind,label=kind==='interview'?t('面经','Interview experience'):kind==='solution'?t('题解','Solution'):kind==='question'?t('题目讨论','Problem discussion'):t('讨论','Discussion');
 return <div className="community-highlight-context"><span className="community-highlight-kind">{label}</span>
  {kind==='interview'?<span>{(['companies','roles','stages'] as const).map((group,index)=>interviewLabel(group,[context.company,context.role,context.stage][index],locale==='en')).join(' · ')}</span>:
   kind==='question'||kind==='solution'?context.problem.retired?<span>{topicTitle(context.problem,locale==='en')} · {t('已归档','Archived')}</span>:<AppLink href={kind==='solution'?solutionsHref(context.problem.id):topicHref(context.problem.id)} target={newTab?'_blank':undefined} rel={newTab?'noopener noreferrer':undefined}>{topicTitle(context.problem,locale==='en')}{newTab?' ↗':''}</AppLink>:null}
 </div>;
}
