import {useId,useRef,useState} from 'react';
import {CalendarDays,ChevronDown,MapPin} from 'lucide-react';
import {useI18n} from './i18n';
import {emptyInterviewContext,hasInterviewContext,interviewContextError,type InterviewContext} from './interviewContext';
import './interview-context-filters.css';

type State={scope:string;values:InterviewContext;open:boolean;error?:ReturnType<typeof interviewContextError>};
export default function InterviewContextFilters({context,available,loading,onApply}:{context:InterviewContext;available:boolean;loading:boolean;onApply:(value:InterviewContext)=>void}){
 const {t}=useI18n(),id=useId(),composing=useRef(false),applied={location:context.location,month_from:context.month_from,month_to:context.month_to},scope=JSON.stringify(applied);
 const [state,setState]=useState<State>(()=>({scope,values:applied,open:hasInterviewContext(applied)}));
 let current=state;
 if(state.scope!==scope){current={scope,values:applied,open:state.open||hasInterviewContext(applied)};setState(current);}
 const draft=current.values,changed=JSON.stringify(draft)!==scope;
 const error=current.error==='location'?t('地点最多 80 个字符，不能含控制字符。','Use up to 80 characters without control characters for location.'):current.error==='month'?t('月份须在 1900-01 至 2099-12 之间。','Use a month from 1900-01 through 2099-12.'):current.error==='order'?t('结束月份不能早于开始月份。','The end month cannot be before the start month.'):'';
 const update=(key:keyof InterviewContext,value:string)=>setState(previous=>({...previous,values:{...previous.values,[key]:value},error:undefined}));
 return <details className="interview-context-filters" open={current.open} onToggle={event=>{const open=event.currentTarget.open;setState(previous=>previous.open===open?previous:{...previous,open});}}>
  <summary><MapPin size={15} aria-hidden="true"/><span>{t('地点与面试月份','Location and interview month')}</span>{hasInterviewContext(applied)&&<small>{t('已筛选','Filtered')}</small>}<ChevronDown size={15} aria-hidden="true"/></summary>
  <form onSubmit={event=>{event.preventDefault();if(composing.current||!available||loading)return;const values={...draft,location:draft.location.trim()},error=interviewContextError(values);if(error){setState(previous=>({...previous,error}));return;}setState(previous=>({...previous,values,error:undefined}));onApply(values);}}>
   <p id={id+'-help'}>{t('筛选分享者填写的地点和实际面试月份，同时作用于面经列表、高频统计与公司对比。与上方“发布时间范围”分别生效；限制月份时，不包含未填写月份的面经。','Filter the author-provided location and actual interview month across experiences, frequent concepts and company comparisons. Publication time is a separate filter. A month range excludes experiences without a stated month.')}</p>
   <div className="interview-context-fields"><label htmlFor={id+'-location'}>{t('地点包含','Location contains')}<input id={id+'-location'} type="search" maxLength={160} value={draft.location} aria-describedby={id+'-help'} placeholder={t('例如 Remote 或上海','e.g. Remote or London')} onCompositionStart={()=>{composing.current=true;}} onCompositionEnd={()=>{composing.current=false;}} onKeyDown={event=>{if(event.key==='Enter'&&(event.nativeEvent.isComposing||event.nativeEvent.keyCode===229||composing.current))event.preventDefault();}} onChange={event=>update('location',event.target.value)}/></label>
    <label htmlFor={id+'-from'}><span><CalendarDays size={14} aria-hidden="true"/>{t('开始月份','From month')}</span><input id={id+'-from'} type="month" min="1900-01" max="2099-12" value={draft.month_from} aria-describedby={id+'-help'} onChange={event=>update('month_from',event.target.value)}/></label>
    <label htmlFor={id+'-to'}>{t('结束月份','Through month')}<input id={id+'-to'} type="month" min="1900-01" max="2099-12" value={draft.month_to} aria-describedby={id+'-help'} onChange={event=>update('month_to',event.target.value)}/></label></div>
   {error&&<p className="interview-context-error" role="alert">{error}</p>}
   <div className="interview-context-actions"><button type="submit" className="button subtle" disabled={!available||loading}>{t('应用地点与月份','Apply location and months')}</button>{changed&&<button type="button" className="text-button" onClick={()=>setState(previous=>({...previous,values:applied,error:undefined}))}>{t('恢复当前条件','Restore applied values')}</button>}{hasInterviewContext(applied)&&<button type="button" className="text-button" onClick={()=>{setState(previous=>({...previous,values:emptyInterviewContext(),error:undefined}));onApply(emptyInterviewContext());}}>{t('清除地点与月份','Clear location and months')}</button>}</div>
   <p className="interview-context-status" role="status">{loading?t('正在确认筛选支持，已保留输入。','Checking filter support; your input is kept.'):!available?t('当前服务暂不支持地点与月份筛选，可以继续使用其他条件。','This service does not support location and month filters yet. Other filters remain available.'):changed?t('这些输入尚未应用；点击应用后更新列表与统计。','These edits are not applied yet. Apply to update the list and counts.'):hasInterviewContext(applied)?t('列表与统计使用上方已应用的条件。','The list and counts use the applied values above.'):t('留空表示不限地点或月份。','Leave a field empty for any location or month.')}</p>
  </form>
 </details>;
}
