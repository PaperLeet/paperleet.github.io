import {useState} from 'react';
import {useI18n} from './i18n';
import {useCommunityTaxonomy} from './useCommunityTaxonomy';
import './community-taxonomy.css';

export function TaxonomyStatus(){
 const {t}=useI18n(),taxonomy=useCommunityTaxonomy();
 if(!taxonomy.supported||!taxonomy.error&&taxonomy.ready)return null;
 return <p className="community-taxonomy-status" role="status">{taxonomy.error?t('版位与标签暂时无法读取，现有内容仍可阅读。','Boards and tags could not load. Existing content remains readable.'):t('正在读取社区版位与标签…','Loading community boards and tags…')}{taxonomy.error&&<button type="button" className="text-button" disabled={taxonomy.loading} onClick={()=>void taxonomy.refresh()}>{t('重试配置','Retry configuration')}</button>}</p>;
}
export default function CommunityTagPicker({value,onChange,disabled=false}:{value:string[];onChange:(value:string[])=>void;disabled?:boolean}){
 const {t}=useI18n(),taxonomy=useCommunityTaxonomy(),[query,setQuery]=useState('');
 if(!taxonomy.supported)return value.length?<p role="status">{t('此服务暂不支持保存标签，当前选择仍保留在草稿。','This service cannot save tags. Your selection is kept in the draft.')}</p>:null;
 const rows=taxonomy.data.tags.filter(row=>(row.enabled||value.includes(row.id))&&(!query||[row.zh,row.en,row.id].some(text=>text.toLowerCase().includes(query.toLowerCase()))));
 return <fieldset className="community-tag-picker" disabled={disabled||!taxonomy.ready}><legend>{t('标签（可选，最多 5 个）','Tags (optional, up to 5)')}</legend><TaxonomyStatus/>{taxonomy.data.tags.length>12&&<input type="search" aria-label={t('查找社区标签','Find community tags')} value={query} onChange={event=>setQuery(event.target.value)} placeholder={t('搜索标签','Search tags')}/>}
  <div>{rows.map(row=><label key={row.id}><input type="checkbox" checked={value.includes(row.id)} disabled={!value.includes(row.id)&&(!row.enabled||value.length>=5)} onChange={()=>onChange(value.includes(row.id)?value.filter(id=>id!==row.id):[...value,row.id])}/><span>{t(row.zh,row.en)}{!row.enabled&&<small> · {t('已停用','Paused')}</small>}</span></label>)}</div>
  {value.filter(id=>!taxonomy.data.tags.some(row=>row.id===id)).map(id=><label key={id}><input type="checkbox" checked onChange={()=>onChange(value.filter(item=>item!==id))}/><span>{id} · {t('暂不可用','Unavailable')}</span></label>)}
  {!rows.length&&<p>{query?t('没有匹配的标签。','No tags match this search.'):t('暂时没有可选标签。','No tags are available yet.')}</p>}
 </fieldset>;
}
