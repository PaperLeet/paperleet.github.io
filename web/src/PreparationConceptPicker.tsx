import {useId,useMemo,useState} from 'react';
import {ArrowLeft,ArrowRight} from 'lucide-react';
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
 const index=matches.findIndex(problem=>problem.id===selected.id);
 const previous=index>0?matches[index-1]:undefined,next=index<0?matches[0]:matches[index+1];
 const title=(problem:Problem)=>locale==='en'?problem.translations?.en?.title||problem.title:problem.title;
 const describe=(problem:Problem)=>`${problem.number}. ${title(problem)}`;
 const option=(problem:Problem)=><option key={problem.id} value={problem.id}>{describe(problem)}</option>;
 return <div className="preparation-concept-picker">
  {(problems.length>1||!!query)&&<>
   <CommunitySearch applied={query} onApply={value=>setQuery(preparationConceptQuery(value))} label={t('搜索自测知识点','Search study concepts')} placeholder={t('题名、编号、函数或实现目标…','Title, number, function or goal…')}/>
   <p id={statusId} className="preparation-concept-status" role="status">{query?t(`匹配 ${matches.length} / ${problems.length} 个知识点。`,`${matches.length} of ${problems.length} concepts match.`):t('输入题名、题号、函数、实现目标或论文名，按 Enter 筛选。','Filter by title, number, function, implementation goal or paper, then press Enter.')}{held&&' '+t('当前提纲已保留，选择其他知识点后切换。','Your current guide is kept until you choose another concept.')}{query&&!matches.length&&' '+t('可清空搜索或换个关键词。','Clear the search or try another term.')}</p>
  </>}
  <label className="preparation-guide-selector"><span>{t('选择自测知识点','Choose a study concept')}</span><select value={selected.id} disabled={!matches.length} aria-describedby={problems.length>1||query?statusId:undefined} onChange={event=>{if(matches.some(problem=>problem.id===event.target.value))onSelect(event.target.value);}}>
   {held?<><optgroup label={t('当前阅读','Currently reading')}>{option(selected)}</optgroup>{matches.length>0&&<optgroup label={t('匹配的知识点','Matching concepts')}>{matches.map(option)}</optgroup>}</>:matches.map(option)}
  </select></label>
  {(problems.length>1||!!query)&&<div className="preparation-concept-navigation" role="group" aria-label={t('知识点导航','Study concept navigation')}>
   <button type="button" aria-disabled={!previous} aria-label={previous?t(`上一知识点：${describe(previous)}`,`Previous concept: ${describe(previous)}`):t('没有上一个知识点','No previous concept')} title={previous?describe(previous):t('没有上一个知识点','No previous concept')} onClick={()=>{if(previous)onSelect(previous.id);}}><ArrowLeft size={15} aria-hidden="true"/>{t('上一个','Previous')}</button>
   <span className="preparation-concept-position" role="status">{!matches.length?t('无匹配项','No matches'):index<0?t(`${matches.length} 个匹配项`,`${matches.length} ${matches.length===1?'match':'matches'}`):`${index+1} / ${matches.length}`}</span>
   <button type="button" aria-disabled={!next} aria-label={next?index<0?t(`查看首个匹配项：${describe(next)}`,`Read first match: ${describe(next)}`):t(`下一知识点：${describe(next)}`,`Next concept: ${describe(next)}`):t('没有下一个知识点','No next concept')} title={next?describe(next):t('没有下一个知识点','No next concept')} onClick={()=>{if(next)onSelect(next.id);}}>{index<0?t('首个匹配项','First match'):t('下一个','Next')}<ArrowRight size={15} aria-hidden="true"/></button>
  </div>}
 </div>;
}
