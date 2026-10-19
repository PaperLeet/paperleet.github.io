import {useId,useRef,useState} from 'react';
import {ChevronDown,SlidersHorizontal} from 'lucide-react';
import {useI18n} from './i18n';
import {interviewCatalog} from './interviewModels';
import {hostedEdition} from './deployment';
import type {InterviewFilters} from './interviewNavigation';
import InterviewContextFilters from './InterviewContextFilters';
import './interview-browse-filters.css';

type Props={filters:InterviewFilters;loading:boolean;outcomeAvailable:boolean;contextAvailable:boolean;availabilityNotice:boolean;searchHelp:string;outcomeHelp:string;onChange:(value:Partial<InterviewFilters>)=>void};

/** Collapse presentation only: applied filters and unfinished location edits stay intact. */
export default function InterviewBrowseFilters({filters,loading,outcomeAvailable,contextAvailable,availabilityNotice,searchHelp,outcomeHelp,onChange}:Props){
 const {t,locale}=useI18n(),id=useId(),panel=useRef<HTMLDetailsElement>(null),summary=useRef<HTMLElement>(null);
 const [draft,setDraft]=useState(false);
 const reports=filters.view==='reports',questions=filters.view==='questions';
 const advancedCount=[reports&&filters.stage,hostedEdition&&filters.outcome,questions&&filters.difficulty,questions&&filters.max_minutes,hostedEdition&&filters.location,hostedEdition&&filters.month_from,hostedEdition&&filters.month_to].filter(Boolean).length;
 const [open,setOpen]=useState(()=>advancedCount>0);
 const extra=hostedEdition||questions;
 function close(){
  if(!panel.current?.isConnected)return;
  if(panel.current.contains(document.activeElement))summary.current?.focus();
  setOpen(false);
 }
 const options=(group:'companies'|'roles'|'stages'|'outcomes')=>interviewCatalog[group].map(row=><option key={row.id} value={row.id}>{locale==='en'?row.name_en:row.name}</option>);
 return <>
  <div className="interview-selects interview-primary-filters">
   <label><span>{t('公司','Company')}</span><select aria-label={t('全部公司','All companies')} value={filters.company} onChange={event=>onChange({company:event.target.value,...(filters.view==='companies'?{q:''}:{})})}><option value="">{t('全部公司','All companies')}</option>{options('companies')}</select></label>
   <label><span>{t('岗位方向','Role')}</span><select aria-label={t('全部岗位','All roles')} value={filters.role} onChange={event=>onChange({role:event.target.value})}><option value="">{t('全部岗位','All roles')}</option>{options('roles')}</select></label>
   {hostedEdition&&<label><span>{t('发布时间','Published')}</span><select aria-label={t('面经发布时间范围','Report publication period')} value={filters.days} onChange={event=>onChange({days:event.target.value as InterviewFilters['days']})}><option value="365">{t('近一年','Past year')}</option><option value="90">{t('近三个月','Past 90 days')}</option><option value="30">{t('近一个月','Past 30 days')}</option><option value="all">{t('全部时间','All time')}</option></select></label>}
  </div>
  {extra&&<details ref={panel} className="interview-more-filters" open={open} onToggle={event=>setOpen(event.currentTarget.open)}>
   <summary ref={summary} aria-controls={id}><SlidersHorizontal size={16} aria-hidden="true"/><span>{t('更多筛选','More filters')}</span>{advancedCount>0&&<small>{t(`${advancedCount} 项已应用`,`${advancedCount} applied`)}</small>}{draft&&hostedEdition&&<small className="interview-filter-draft">{t('有未应用输入','Unapplied edits')}</small>}<ChevronDown size={16} aria-hidden="true"/></summary>
   <div id={id} className="interview-more-filter-content">
    <div className="interview-selects interview-secondary-filters">
     {reports&&<label><span>{t('面试轮次','Interview stage')}</span><select aria-label={t('面试轮次','Interview stage')} value={filters.stage} onChange={event=>onChange({stage:event.target.value})}><option value="">{t('全部轮次','All stages')}</option>{options('stages')}</select></label>}
     {hostedEdition&&(outcomeAvailable||!!filters.outcome)&&<label><span>{t('面试结果','Interview outcome')}</span><select aria-label={t('面试结果','Interview outcome')} aria-describedby={outcomeHelp} value={filters.outcome} disabled={loading||!outcomeAvailable} onChange={event=>onChange({outcome:event.target.value})}><option value="">{t('全部结果','All outcomes')}</option>{options('outcomes')}</select></label>}
     {questions&&<><label><span>{t('练习难度','Practice difficulty')}</span><select aria-label={t('练习难度','Practice difficulty')} aria-describedby={searchHelp} value={filters.difficulty} onChange={event=>onChange({difficulty:event.target.value as InterviewFilters['difficulty']})}><option value="">{t('练习难度不限','Any practice difficulty')}</option><option value="E">{t('简单','Easy')}</option><option value="M">{t('中等','Medium')}</option><option value="H">{t('困难','Hard')}</option></select></label><label><span>{t('单题预计时长','Estimated time per exercise')}</span><select aria-label={t('单题预计时长','Estimated time per exercise')} aria-describedby={searchHelp} value={filters.max_minutes} onChange={event=>onChange({max_minutes:event.target.value as InterviewFilters['max_minutes']})}><option value="">{t('单题时长不限','Any exercise duration')}</option>{(['15','30','45','60'] as const).map(value=><option key={value} value={value}>{t(`单题 ≤ ${value} 分钟`,`Up to ${value} min per exercise`)}</option>)}</select></label></>}
    </div>
    {hostedEdition&&<><p id={outcomeHelp} className={filters.outcome?'interview-search-help':'sr-only'}>{t('结果由分享者填写，筛选同时作用于面经和公开统计。','Outcomes are self-reported. This filter applies to experiences and public counts.')}</p><InterviewContextFilters embedded context={filters} available={contextAvailable} loading={loading} availabilityNotice={availabilityNotice} onApply={onChange} onDraftChange={setDraft}/></>}
    <footer><span>{t('收起不会清除筛选或未应用的输入。','Collapsing keeps filters and unapplied edits.')}</span><button type="button" className="text-button" onClick={close}>{t('收起筛选','Collapse filters')}</button></footer>
   </div>
  </details>}
 </>;
}
