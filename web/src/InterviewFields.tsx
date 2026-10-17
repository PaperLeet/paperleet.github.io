import {useId,useLayoutEffect,useRef,useState} from 'react';
import {Check,Code2,X} from 'lucide-react';
import {useI18n} from './i18n';
import {interviewCatalog,type InterviewDetails} from './interviewModels';
import type {Problem} from './types';
import RelatedModuleSearch from './RelatedModuleSearch';
import {searchRelatedModules} from './relatedModuleMatching';
import './interviews.css';
export type InterviewModule=Pick<Problem,'id'|'number'|'title'|'translations'>;
export default function InterviewFields({value,onChange,problems=[],disabled=false}:{value:InterviewDetails;onChange:(value:InterviewDetails)=>void;problems?:InterviewModule[];disabled?:boolean}){
  const {t,locale}=useI18n();const [query,setQuery]=useState('');
  const [reveal,setReveal]=useState({query:'',limit:6});
  const root=useRef<HTMLFieldSetElement>(null),search=useRef<HTMLInputElement>(null),heading=useRef<HTMLElement>(null),optionsId=useId();
  const focus=useRef<{opener:HTMLElement;href:string;query:string;kind:'remove'|'more';ids:string[]}>(undefined);
  const edit=(change:Partial<InterviewDetails>)=>onChange({...value,...change});
  const matches=searchRelatedModules(problems,query),initialLimit=query.trim()?12:6,limit=reveal.query===query?Math.max(initialLimit,reveal.limit):initialLimit,shown=matches.slice(0,limit);
  const title=(p:InterviewModule)=>`${p.number}. ${locale==='en'?p.translations?.en?.title||p.title:p.title}`;
  useLayoutEffect(()=>{
    const intent=focus.current;if(!intent)return;focus.current=undefined;
    if(disabled||intent.href!==location.href||intent.query!==query)return;
    if(document.activeElement!==intent.opener&&!(document.activeElement===document.body&&!intent.opener.isConnected))return;
    const controls=Array.from(root.current?.querySelectorAll<HTMLButtonElement>(intent.kind==='remove'?'[data-selected-module]':'[data-module-id]')||[]);
    const target=intent.ids.map(id=>controls.find(node=>(intent.kind==='remove'?node.dataset.selectedModule:node.dataset.moduleId)===id&&!node.disabled)).find(Boolean);
    (target||intent.opener.isConnected&&intent.opener||search.current||heading.current)?.focus({preventScroll:true});
  },[value.questions,query,reveal,disabled,problems]);
  function remove(id:string,opener:HTMLElement){
    const index=value.questions.indexOf(id);
    focus.current={opener,href:location.href,query,kind:'remove',ids:[value.questions[index+1],value.questions[index-1]].filter(Boolean)};
    edit({questions:value.questions.filter(question=>question!==id)});
  }
  return <fieldset ref={root} className="interview-fields" disabled={disabled}>
    <legend>{t('面试信息','Interview details')}</legend>
    <div className="interview-field-grid">
      {([['company','companies',t('公司','Company')],['role','roles',t('岗位方向','Role')],['stage','stages',t('面试轮次','Stage')],['outcome','outcomes',t('面试结果','Outcome')]] as const).map(([field,group,label])=><label key={field}>{label}<select required value={value[field]} onChange={event=>edit({[field]:event.target.value})}>{interviewCatalog[group].map(row=><option key={row.id} value={row.id}>{locale==='en'?row.name_en:row.name}</option>)}</select></label>)}
      <label>{t('面试月份（可选）','Interview month (optional)')}<input type="month" min="1900-01" max={new Date().toISOString().slice(0,7)} value={value.interview_month} onChange={event=>edit({interview_month:event.target.value})}/></label>
      <label>{t('城市 / 远程（可选）','City / remote (optional)')}<input maxLength={80} value={value.location} placeholder={t('如：北京 / 远程','e.g. Beijing / remote')} onChange={event=>edit({location:event.target.value})}/></label>
    </div>
    <div className="interview-question-selector"><b ref={heading} tabIndex={-1}><Code2 size={15}/>{t('考察知识点','Interview concepts')}<small>{value.questions.length} / 12</small></b><p>{t('关联本轮实际考察的知识点；站内练习不代表公司的原始面试题。','Link concepts actually discussed in this interview. These exercises are not claimed to be original company questions.')}</p>
      {!!value.questions.length&&<div className="interview-selected-questions">{value.questions.map(id=>{const p=problems.find(p=>p.id===id),name=p?title(p):id;return <button key={id} type="button" data-selected-module={id} aria-label={t(`移除${name}的关联`,`Remove link to ${name}`)} onClick={event=>remove(id,event.currentTarget)}>{name}<X size={12} aria-hidden="true"/></button>;})}</div>}
      {problems.length>0?<>
        <RelatedModuleSearch value={query} onChange={setQuery} label={t('搜索关联练习','Search related modules')} disabled={disabled} inputRef={search}/>
        <div id={optionsId} className="interview-module-options">{shown.map(p=><button key={p.id} type="button" data-module-id={p.id} aria-pressed={value.questions.includes(p.id)} disabled={!value.questions.includes(p.id)&&value.questions.length>=12} onClick={()=>edit({questions:value.questions.includes(p.id)?value.questions.filter(q=>q!==p.id):[...value.questions,p.id].sort()})}>{value.questions.includes(p.id)?<Check size={14}/>:<Code2 size={14}/>}<span>{title(p)}</span></button>)}</div>
        {!matches.length&&<p>{t('没有找到匹配的练习，试试题号、名称或模块 ID。','No matching modules. Try a number, title or module ID.')}</p>}
        <div className="interview-module-results"><span role="status">{t(`显示 ${shown.length} / ${matches.length} 道练习`,`Showing ${shown.length} of ${matches.length} modules`)}</span>{shown.length<matches.length&&<button type="button" className="text-button" aria-controls={optionsId} onClick={event=>{
          focus.current={opener:event.currentTarget,href:location.href,query,kind:'more',ids:matches.slice(shown.length,shown.length+6).map(p=>p.id)};
          setReveal({query,limit:shown.length+6});
        }}>{t(`再显示 ${Math.min(6,matches.length-shown.length)} 道`,`Show ${Math.min(6,matches.length-shown.length)} more`)}</button>}</div>
      </>:<p>{t('题库加载后可以添加关联；现有关联会保留。','Add links once the catalog loads. Existing links are preserved.')}</p>}
    </div>
  </fieldset>;
}
