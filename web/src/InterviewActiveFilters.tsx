import {X} from 'lucide-react';
import {useI18n} from './i18n';
import OrganizationLogo from './OrganizationLogo';
import {interviewCatalog,interviewLabel} from './interviewModels';
import {emptyInterviewFilters,type InterviewFilters} from './interviewNavigation';
import type {Problem} from './types';

export default function InterviewActiveFilters({filters,problems,onChange}:{filters:InterviewFilters;problems:Problem[];onChange:(change:Partial<InterviewFilters>)=>void}){
 const {t,locale}=useI18n(),en=locale==='en';
 const chips:{key:string;label:string;change:Partial<InterviewFilters>}[]=[];
 for(const key of ['company','role','stage'] as const)if(filters[key])chips.push({key,label:interviewLabel(({company:'companies',role:'roles',stage:'stages'} as const)[key],filters[key],en),change:{[key]:''}});
 if(filters.outcome)chips.push({key:'outcome',label:t(`结果：${interviewLabel('outcomes',filters.outcome,false)}`,`Outcome: ${interviewLabel('outcomes',filters.outcome,true)}`),change:{outcome:''}});
 if(filters.q)chips.push({key:'q',label:`“${filters.q}”`,change:{q:''}});
 if(filters.days!=='365')chips.push({key:'days',label:filters.days==='all'?t('全部时间','All time'):t(`近 ${filters.days} 天`,`Past ${filters.days} days`),change:{days:'365'}});
 if(filters.concept){const problem=problems.find(problem=>problem.id===filters.concept);chips.push({key:'concept',label:problem?(en?problem.translations?.en?.title||problem.title:problem.title):filters.concept,change:{concept:''}});}
 if(filters.view==='questions'&&filters.study!=='all')chips.push({key:'study',label:({todo:t('未开始','Not started'),started:t('进行中','In progress'),done:t('已通过','Passed')})[filters.study],change:{study:'all'}});
 if(filters.view==='questions'&&filters.difficulty)chips.push({key:'difficulty',label:t('难度：','Difficulty: ')+({E:t('简单','Easy'),M:t('中等','Medium'),H:t('困难','Hard')})[filters.difficulty],change:{difficulty:''}});
 if(filters.view==='questions'&&filters.max_minutes)chips.push({key:'max_minutes',label:t(`单题 ≤ ${filters.max_minutes} 分钟`,`Up to ${filters.max_minutes} min per exercise`),change:{max_minutes:''}});
 if(filters.location)chips.push({key:'location',label:t(`地点包含：${filters.location}`,`Location contains: ${filters.location}`),change:{location:''}});
 if(filters.month_from)chips.push({key:'month_from',label:t(`面试自 ${filters.month_from}`,`Interview from ${filters.month_from}`),change:{month_from:''}});
 if(filters.month_to)chips.push({key:'month_to',label:t(`面试至 ${filters.month_to}`,`Interview through ${filters.month_to}`),change:{month_to:''}});
 const path=interviewCatalog.paths.find(path=>path.id===filters.track);
 if(path)chips.push({key:'track',label:t(`准备路径：${path.name}`,`Path: ${path.name_en}`),change:{track:''}});
 if(!chips.length)return null;
 return <div className="interview-active-filters" role="group" aria-label={t('已选筛选条件','Active filters')}>
  {chips.map(chip=><button type="button" key={chip.key} title={chip.label} aria-label={t(`移除筛选：${chip.label}`,`Remove filter: ${chip.label}`)} onClick={()=>onChange(chip.change)}>{chip.key==='company'&&<OrganizationLogo id={filters.company} size={14}/>}<span>{chip.label}</span><X size={12} aria-hidden="true"/></button>)}
  <button type="button" className="text-button" onClick={()=>onChange({...emptyInterviewFilters(),view:filters.view,saved:filters.saved})}>{t('清除筛选','Clear filters')}</button>
 </div>;
}
