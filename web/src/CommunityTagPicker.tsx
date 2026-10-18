import {useId,useRef,useState} from 'react';
import {Search,X} from 'lucide-react';
import {useI18n} from './i18n';
import {useCommunityTaxonomy} from './useCommunityTaxonomy';
import {matchingCommunityTags} from './communityTagBrowsing';
import './community-taxonomy.css';

export function TaxonomyStatus({writing=false,onRetryStart}:{writing?:boolean;onRetryStart?:()=>void}){
 const {t}=useI18n(),taxonomy=useCommunityTaxonomy();
 if(!taxonomy.supported||!taxonomy.error&&taxonomy.ready)return null;
 return <p className="community-taxonomy-status" role="status">{taxonomy.error?writing?t('版位与标签配置暂不可用，正文与已选标签已保留。','Board and tag settings could not load. Your writing and selected tags are kept.'):t('版位与标签暂时无法读取，现有内容仍可阅读。','Boards and tags could not load. Existing content remains readable.'):t('正在读取社区版位与标签…','Loading community boards and tags…')}{taxonomy.error&&<button type="button" className="text-button" disabled={taxonomy.loading} onClick={event=>{if(onRetryStart)onRetryStart();else event.currentTarget.closest<HTMLElement>('.community-feed-toolbar')?.focus({preventScroll:true});void taxonomy.refresh();}}>{t('重试配置','Retry configuration')}</button>}</p>;
}
type ExistingTag={id:string;zh:string;en:string};
export default function CommunityTagPicker({value,onChange,disabled=false,existingTags=[],mode='create'}:{value:string[];onChange:(value:string[])=>void;disabled?:boolean;existingTags?:readonly ExistingTag[];mode?:'create'|'edit'}){
 const {t}=useI18n(),taxonomy=useCommunityTaxonomy(),[query,setQuery]=useState(''),fieldset=useRef<HTMLFieldSetElement>(null),search=useRef<HTMLInputElement>(null),chips=useRef(new Map<string,HTMLButtonElement>()),latest=useRef(value),composing=useRef(false),hint=useId();
 latest.current=value;
 if(!taxonomy.supported&&!value.length)return null;
 const rows=matchingCommunityTags(taxonomy.data.tags,query,value),existing=new Set(existingTags.map(tag=>tag.id));
 const tagName=(id:string)=>{const tag=taxonomy.data.tags.find(row=>row.id===id)||existingTags.find(row=>row.id===id);return tag?t(tag.zh,tag.en):id;};
 function change(next:string[]){latest.current=next;onChange(next);}
 function remove(id:string,opener?:HTMLElement){
  if(disabled)return;const current=latest.current,index=current.indexOf(id);if(index<0)return;
  if(opener&&document.activeElement===opener){const target=chips.current.get(current[index+1]||current[index-1])||search.current||(!taxonomy.supported?fieldset.current?.closest('form')?.querySelector<HTMLElement>('textarea'):undefined)||fieldset.current;target?.focus({preventScroll:true});}
  change(current.filter(item=>item!==id));
 }
 function toggle(id:string){
  if(disabled)return;const current=latest.current;if(current.includes(id)){remove(id);return;}
  if(!taxonomy.supported||!taxonomy.ready||current.length>=5||!taxonomy.data.tags.some(tag=>tag.id===id&&tag.enabled))return;
  change([...current,id]);
 }
 function clearSearch(){setQuery('');search.current?.focus({preventScroll:true});}
 return <div className="community-taxonomy-field">
  <TaxonomyStatus writing onRetryStart={()=>fieldset.current?.focus({preventScroll:true})}/>
  {!taxonomy.supported&&mode!=='edit'&&<p className="community-taxonomy-status" role="status">{t('此服务暂不支持保存标签，当前选择仍保留在草稿。','This service cannot save tags. Your selection is kept in the draft.')}</p>}
  <fieldset ref={fieldset} className="community-tag-picker" disabled={disabled} tabIndex={-1} aria-describedby={hint}>
   <legend>{t('标签（可选，最多 5 个）','Tags (optional, up to 5)')}</legend>
   <p className="community-tag-picker-count" id={hint} role="status">{t(`已选 ${value.length} / 5`,`${value.length} / 5 selected`)}{value.length>=5&&<span>{t(' · 移除一个后可继续选择',' · Remove one to choose another')}</span>}</p>
   {value.length>0&&<div className="community-selected-tags" role="group" aria-label={t('已选社区标签','Selected community tags')}>
    {value.map(id=>{const tag=taxonomy.data.tags.find(row=>row.id===id),state=!taxonomy.supported||!taxonomy.ready?'pending':!tag?'unavailable':tag.enabled?'active':existing.has(id)?'retained':'paused';return <div className="community-selected-tag" key={id} data-state={state}>
     <button type="button" ref={node=>{if(node)chips.current.set(id,node);else chips.current.delete(id);}} aria-label={t('移除标签：','Remove tag: ')+tagName(id)} onClick={event=>remove(id,event.currentTarget)}><span>#{tagName(id)}</span><X size={14} aria-hidden="true"/></button>
     {state!=='active'&&<small>{state==='pending'?t('配置待确认','Settings unconfirmed'):state==='unavailable'?t('暂不可用，请重试配置或移除','Unavailable; retry settings or remove'):state==='retained'?t('原帖历史标签，可保留或移除','Historic tag on this post; keep or remove'):mode==='edit'?t('已停用，不能新增到此帖','Paused; cannot add to this post'):t('已停用，新帖请移除','Paused; remove for a new post')}</small>}
    </div>;})}
   </div>}
   {taxonomy.supported&&<>
    {(taxonomy.data.tags.length>12||query)&&<div className="community-tag-picker-search"><Search size={16} aria-hidden="true"/><input ref={search} type="search" maxLength={100} aria-label={t('查找社区标签','Find community tags')} value={query} onChange={event=>setQuery(event.target.value)} onCompositionStart={()=>{composing.current=true;}} onCompositionEnd={()=>{composing.current=false;}} onKeyDown={event=>{if(event.key==='Escape'&&query&&!composing.current&&!event.nativeEvent.isComposing&&event.keyCode!==229){event.preventDefault();event.stopPropagation();clearSearch();}}} placeholder={t('搜索名称、标识或说明','Search names, identifiers or descriptions')}/>{query&&<button type="button" aria-label={t('清空标签选择搜索','Clear tag picker search')} onClick={clearSearch}><X size={15} aria-hidden="true"/></button>}</div>}
    <div className="community-tag-picker-options">{rows.map(tag=>{const description=hint+'-'+tag.id;return <div className="community-tag-picker-option" key={tag.id} data-tag-choice={tag.id}>
     <label><input type="checkbox" checked={value.includes(tag.id)} disabled={!value.includes(tag.id)&&(!taxonomy.ready||!tag.enabled||value.length>=5)} aria-describedby={tag.description?description:undefined} onChange={()=>toggle(tag.id)}/><span>{t(tag.zh,tag.en)}{!tag.enabled&&<small> · {t('已停用','Paused')}</small>}</span></label>
     {tag.description&&<small id={description}>{tag.description}</small>}
    </div>;})}</div>
    {!rows.length&&<p className="community-tag-picker-empty">{query?t('没有匹配的标签，已选项仍保留在上方。','No tags match. Selected tags remain above.'):taxonomy.ready?t('暂时没有可选标签。','No tags are available yet.'):t('配置读取后可继续选择，当前已选项可以移除。','More choices appear after settings load. You can remove current selections.')}{query&&<button type="button" className="text-button" onClick={clearSearch}>{t('清空搜索','Clear search')}</button>}</p>}
   </>}
  </fieldset>
 </div>;
}
