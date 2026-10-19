import {useState} from 'react';
import AppLink from './AppLink';
import {useI18n} from './i18n';
import {interviewCatalog} from './interviewModels';
import data from './data/external-company-practice.json';
import type {Problem} from './types';
import './external-company-practice.css';

const companies=[...new Set(data.items.map(row=>row.company))].sort();
export function companyPracticeMatches(source:string,company:string):boolean {
  if(!company)return true;
  if(company==='google')return source==='Google';
  if(company==='amazon')return source==='Amazon'||source==='AWS';
  if(company==='bytedance')return source==='ByteDance';
  const known=interviewCatalog.companies.find(row=>row.id===company);
  if(company==='other')return !interviewCatalog.companies.some(row=>row.id!=='other'&&companyPracticeMatches(source,row.id));
  return known?.name_en.toLowerCase()===source.toLowerCase();
}
export default function ExternalCompanyPractice({company='',query='',problems}:{company?:string;query?:string;problems:Problem[]}) {
  const {t,locale}=useI18n(),[selected,setSelected]=useState(''),[search,setSearch]=useState(''),[limit,setLimit]=useState(10);
  const terms=(query+' '+search).toLowerCase().trim().split(/\s+/).filter(Boolean);
  const rows=data.items.filter(row=>companyPracticeMatches(row.company,company)&&(!selected||row.company===selected)&&terms.every(term=>(row.title+' '+row.company).toLowerCase().includes(term)));
  return <details className="external-company-practice">
    <summary>{t('外部公司场景练习','External company practice')} · TrenTorch · {rows.length} {t('题','problems')}</summary>
    <p>{t('来源平台明确说明：这些是受公司业务场景启发的原创练习，不声称是真实面试题。此处收录公开索引与原题链接，只按公司和关键词筛选，不计入真实面经、公司提及次数或面试考频。','The source describes these as original exercises inspired by company scenarios, not actual interview questions. This directory filters by company and keywords, links to the originals and does not contribute to experience counts, company mentions or interview frequency.')}</p>
    <div className="external-practice-controls"><label>{t('来源公司标签','Source company tag')}<select aria-label={t('来源公司标签','Source company tag')} value={selected} onChange={event=>{setSelected(event.target.value);setLimit(10);}}><option value="">{t('全部来源公司','All source companies')}</option>{companies.filter(name=>companyPracticeMatches(name,company)).map(name=><option key={name}>{name}</option>)}</select></label><label>{t('搜索外部练习','Search external practice')}<input type="search" maxLength={200} value={search} onChange={event=>{setSearch(event.target.value);setLimit(10);}} placeholder={t('题目或公司名称','Problem or company name')}/></label></div>
    <p className="external-practice-source"><AppLink href={data.source_url} target="_blank" rel="noreferrer">{t('查看来源及说明','View source and attribution')}</AppLink> · {t('核对日期','Reviewed')} {data.reviewed_on}</p>
    {!rows.length?<p role="status">{t('当前筛选没有外部公司练习。','No external company exercises match these filters.')}</p>:<ul>{rows.slice(0,limit).map(row=><li key={row.id}><div><AppLink href={row.url} target="_blank" rel="noreferrer">{row.title}</AppLink><small>{row.company} · {row.difficulty==='E'?t('简单','Easy'):t('中等','Medium')} · {t('公司场景练习','Company-inspired practice')}</small></div><div className="external-practice-related">{row.related_modules.map(id=>{const module=problems.find(problem=>problem.id===id);return module?<AppLink key={id} href={'#/'+module.number}>{t('站内相关练习：','Related practice: ')}{locale==='en'?module.translations?.en?.title||module.title:module.title}</AppLink>:null;})}</div></li>)}</ul>}
    {limit<rows.length&&<button type="button" className="text-button" onClick={()=>setLimit(value=>value+10)}>{t(`再显示 ${Math.min(10,rows.length-limit)} 题`,`Show ${Math.min(10,rows.length-limit)} more`)}</button>}
  </details>;
}
