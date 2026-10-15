import {useState} from 'react';
import {Check,Code2,Search,X} from 'lucide-react';
import {useI18n} from './i18n';
import {interviewCatalog,type InterviewDetails} from './interviewModels';
import type {Problem} from './types';
import './interviews.css';
export type InterviewModule=Pick<Problem,'id'|'number'|'title'|'translations'>;
export default function InterviewFields({value,onChange,problems=[],disabled=false}:{value:InterviewDetails;onChange:(value:InterviewDetails)=>void;problems?:InterviewModule[];disabled?:boolean}){
  const {t,locale}=useI18n();const [query,setQuery]=useState('');
  const edit=(change:Partial<InterviewDetails>)=>onChange({...value,...change});
  const matches=problems.filter(p=>`${p.number} ${p.title} ${p.translations?.en?.title||''}`.toLowerCase().includes(query.trim().toLowerCase())).slice(0,query?12:6);
  return <fieldset className="interview-fields" disabled={disabled}>
    <legend>{t('面试信息','Interview details')}</legend>
    <div className="interview-field-grid">
      {([['company','companies',t('公司','Company')],['role','roles',t('岗位方向','Role')],['stage','stages',t('面试轮次','Stage')],['outcome','outcomes',t('面试结果','Outcome')]] as const).map(([field,group,label])=><label key={field}>{label}<select required value={value[field]} onChange={event=>edit({[field]:event.target.value})}>{interviewCatalog[group].map(row=><option key={row.id} value={row.id}>{locale==='en'?row.name_en:row.name}</option>)}</select></label>)}
      <label>{t('面试月份（可选）','Interview month (optional)')}<input type="month" min="1900-01" max={new Date().toISOString().slice(0,7)} value={value.interview_month} onChange={event=>edit({interview_month:event.target.value})}/></label>
      <label>{t('城市 / 远程（可选）','City / remote (optional)')}<input maxLength={80} value={value.location} placeholder={t('如：北京 / 远程','e.g. Beijing / remote')} onChange={event=>edit({location:event.target.value})}/></label>
    </div>
    <div className="interview-question-selector"><b><Code2 size={15}/>{t('考察知识点','Interview concepts')}<small>{value.questions.length} / 12</small></b><p>{t('关联本轮实际考察的知识点；站内练习不代表公司的原始面试题。','Link concepts actually discussed in this interview. These exercises are not claimed to be original company questions.')}</p>
      {!!value.questions.length&&<div className="interview-selected-questions">{value.questions.map(id=>{const p=problems.find(p=>p.id===id);return <button key={id} type="button" onClick={()=>edit({questions:value.questions.filter(q=>q!==id)})}>{p?`${p.number}. ${locale==='en'?p.translations?.en?.title||p.title:p.title}`:id}<X size={12}/><span className="sr-only">{t('移除关联','Remove link')}</span></button>;})}</div>}
      {problems.length>0?<><label className="interview-module-search"><Search size={15}/><input aria-label={t('搜索关联练习','Search related modules')} value={query} maxLength={100} onChange={event=>setQuery(event.target.value)} placeholder={t('搜索 Attention、LoRA、BPR…','Search Attention, LoRA, BPR…')}/></label><div className="interview-module-options">{matches.map(p=><button key={p.id} type="button" aria-pressed={value.questions.includes(p.id)} disabled={!value.questions.includes(p.id)&&value.questions.length>=12} onClick={()=>edit({questions:value.questions.includes(p.id)?value.questions.filter(q=>q!==p.id):[...value.questions,p.id].sort()})}>{value.questions.includes(p.id)?<Check size={14}/>:<Code2 size={14}/>}<span>{p.number}. {locale==='en'?p.translations?.en?.title||p.title:p.title}</span></button>)}</div>{!matches.length&&<p role="status">{t('没有找到匹配的练习','No matching modules')}</p>}</>:<p>{t('题库加载后可以添加关联；现有关联会保留。','Add links once the catalog loads. Existing links are preserved.')}</p>}
    </div>
  </fieldset>;
}
