import {Info,SlidersHorizontal} from 'lucide-react';
import {useI18n} from './i18n';
import type {InterviewFilterSupportState} from './interviewFilterSupportModel';

export default function InterviewFilterSupport({state,onClear}:{state:InterviewFilterSupportState;onClear:()=>void}){
 const {t}=useI18n();
 if(state.kind==='ready'||state.kind==='readonly')return null;
 const pending=state.kind==='pending',both=state.outcome&&state.context;
 const message=pending?both?t('正在确认结果、地点与月份筛选支持，已保留你的选择。','Checking outcome, location and month support. Your choices are kept.'):state.context?t('正在确认地点与面试月份筛选支持，已保留你的选择。','Checking location and interview-month support. Your choices are kept.'):t('正在确认结果筛选支持，已保留你的选择。','Checking outcome filter support. Your choice is kept.'):both?t('面试结果、地点与月份筛选暂不可用，已保留所选条件。','Outcome, location and month filtering is unavailable. Your choices are kept.'):state.context?t('当前服务暂不支持地点与面试月份筛选，已保留所选条件。','This service does not support location and interview-month filters yet. Your choices are kept.'):t('当前服务暂不支持按面试结果筛选，已保留所选条件。','Outcome filtering is unavailable on this service. Your chosen filter is kept.');
 return <section className="interview-filter-support" role="status" aria-atomic="true" aria-busy={pending}>
  {pending?<span className="spinner" aria-hidden="true"/>:<Info size={19} aria-hidden="true"/>}<div><strong>{pending?t('正在确认筛选','Checking filters'):t('部分筛选暂不可用','Some filters are unavailable')}</strong><p>{message}{!pending&&' '+t('清除不支持的条件后，可继续浏览其他范围。','Clear the unsupported filters to continue browsing.')}</p></div>
  {!pending&&<button type="button" className="button subtle" onClick={onClear}><SlidersHorizontal size={15} aria-hidden="true"/>{t('清除不支持的筛选','Clear unsupported filters')}</button>}
 </section>;
}
