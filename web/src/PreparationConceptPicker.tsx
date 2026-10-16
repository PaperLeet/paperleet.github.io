import {useId,useMemo,useState} from 'react';
import {useI18n} from './i18n';
import CommunitySearch from './CommunitySearch';
import {preparationConceptQuery,searchPreparationConcepts} from './preparationConceptSearch';
import type {Problem} from './types';
import './preparation-concept-picker.css';

export default function PreparationConceptPicker({problems,selected,onSelect}:{problems:Problem[];selected:Problem;onSelect:(id:string)=>void}){
 const {t,locale}=useI18n(),statusId=useId();
 const [query,setQuery]=useState('');
 const matches=useMemo(()=>searchPreparationConcepts(problems,query),[problems,query]);
 const held=!!query&&!matches.some(problem=>problem.id===selected.id);
 const title=(problem:Problem)=>locale==='en'?problem.translations?.en?.title||problem.title:problem.title;
 const option=(problem:Problem)=><option key={problem.id} value={problem.id}>{title(problem)}</option>;
 return <div className="preparation-concept-picker">
  {(problems.length>1||!!query)&&<>
   <CommunitySearch applied={query} onApply={value=>setQuery(preparationConceptQuery(value))} label={t('搜索自测知识点','Search study concepts')} placeholder={t('题名、编号或函数名…','Title, number or function…')}/>
   <p id={statusId} className="preparation-concept-status" role="status">{query?t(`匹配 ${matches.length} / ${problems.length} 个知识点。`,`${matches.length} of ${problems.length} concepts match.`):t('输入中英文题名、题号或函数名，按 Enter 筛选。','Filter by Chinese/English title, number or function, then press Enter.')}{held&&' '+t('当前提纲已保留，选择其他知识点后切换。','Your current guide stays open until you choose another concept.')}{query&&!matches.length&&' '+t('可清空搜索或换个关键词。','Clear the search or try another term.')}</p>
  </>}
  <label className="preparation-guide-selector"><span>{t('选择自测知识点','Choose a study concept')}</span><select value={selected.id} disabled={!matches.length} aria-describedby={problems.length>1||query?statusId:undefined} onChange={event=>{if(matches.some(problem=>problem.id===event.target.value))onSelect(event.target.value);}}>
   {held?<><optgroup label={t('当前阅读','Currently reading')}>{option(selected)}</optgroup>{matches.length>0&&<optgroup label={t('匹配的知识点','Matching concepts')}>{matches.map(option)}</optgroup>}</>:matches.map(option)}
  </select></label>
 </div>;
}
