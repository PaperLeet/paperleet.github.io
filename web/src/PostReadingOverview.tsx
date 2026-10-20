import {useMemo} from 'react';
import {Clock3,Code2,ListTree,Sigma,Table2} from 'lucide-react';
import {useI18n} from './i18n';
import {postReadingOverview} from './postReadingStats';
import './post-reading-overview.css';

export default function PostReadingOverview({body,detailed=false}:{body:string;detailed?:boolean}){
 const {t}=useI18n(),overview=useMemo(()=>postReadingOverview(body),[body]);
 if(!overview)return null;
 const {minutes,sections,code,math,tables}=overview;
 const items=[
  ...(minutes?[{key:'time',icon:Clock3,label:t(`正文约 ${minutes} 分钟`,`Text: about ${minutes} min`)}]:[]),
  ...(sections?[{key:'sections',icon:ListTree,label:t(`${sections} 个小节`,`${sections} ${sections===1?'section':'sections'}`)}]:[]),
  ...(detailed&&code?[{key:'code',icon:Code2,label:t(`${code} 段代码`,`${code} code ${code===1?'block':'blocks'}`)}]:[]),
  ...(detailed&&math?[{key:'math',icon:Sigma,label:t(`${math} 处公式`,`${math} ${math===1?'formula':'formulas'}`)}]:[]),
  ...(detailed&&tables?[{key:'tables',icon:Table2,label:t(`${tables} 个表格`,`${tables} ${tables===1?'table':'tables'}`)}]:[]),
 ];
 if(!items.length)return null;
 const hint=t('阅读时间按正文长度粗略估算；代码、公式与深入理解需另留时间。','Reading time is a rough text-length estimate. Allow extra time for code, formulas and deeper study.');
 return <div className={'post-reading-overview'+(detailed?' is-detailed':'')}>
  <ul aria-label={t('阅读概览','Reading overview')}>{items.map(({key,icon:Icon,label})=><li key={key} data-reading-metric={key} title={key==='time'?hint:undefined}><Icon size={14} aria-hidden="true"/>{label}</li>)}</ul>
  {detailed&&minutes&&<p>{hint}</p>}
 </div>;
}
