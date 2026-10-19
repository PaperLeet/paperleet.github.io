import {useId,useRef} from 'react';
import {ChevronDown} from 'lucide-react';
import {useI18n} from './i18n';
import {useCommunityTaxonomy} from './useCommunityTaxonomy';
import {useCommunityTagChoices} from './useCommunityTagChoices';
import CommunityTagChoiceControls from './CommunityTagChoiceControls';
import CommunityTagSearch from './CommunityTagSearch';
import {keepReadingTriggerVisible} from './readingViewport';
import './community-tag-browser.css';

export default function CommunityTagBrowser({value,onChange}:{value?:string;onChange:(id?:string)=>void}){
 const {t}=useI18n(),taxonomy=useCommunityTaxonomy(),listId=useId(),statusId=useId(),details=useRef<HTMLDetailsElement>(null),summary=useRef<HTMLElement>(null),search=useRef<HTMLInputElement>(null),composing=useRef(false);
 const choices=useCommunityTagChoices(taxonomy.data.tags,value,taxonomy.supported);
 const {query,setQuery}=choices;
 if(!taxonomy.supported||!taxonomy.data.tags.length&&!value)return null;
 const selected=taxonomy.data.tags.find(tag=>tag.id===value),available=taxonomy.data.tags.filter(tag=>tag.enabled).length;
 function close(){if(!details.current?.isConnected)return;if(details.current.contains(document.activeElement)&&summary.current){summary.current.focus({preventScroll:true});keepReadingTriggerVisible(summary.current,100);}details.current.open=false;}
 function clearSearch(){setQuery('');search.current?.focus({preventScroll:true});}
 return <details className="community-tag-browser" ref={details} onCompositionStart={()=>{composing.current=true;}} onCompositionEnd={()=>{composing.current=false;}} onKeyDown={event=>{
  if(event.key!=='Escape'||event.defaultPrevented||event.repeat||event.ctrlKey||event.metaKey||event.altKey||event.shiftKey||composing.current||event.nativeEvent.isComposing||event.keyCode===229)return;
  event.preventDefault();event.stopPropagation();close();
 }}>
  <summary ref={summary} aria-label={t('浏览标签：','Browse tags: ')+(selected?t(selected.zh,selected.en):value||t('全部标签','All tags'))}><span className="community-tag-mark" aria-hidden="true">#</span><span>{selected?t(selected.zh,selected.en):value||t('浏览标签','Browse tags')}</span>{!value&&<small>{available}</small>}<ChevronDown size={15} aria-hidden="true"/></summary>
  <div className="community-tag-browser-panel">
   <header><div><strong>{t('按标签发现讨论与题解','Discover discussions and solutions by tag')}</strong><p>{t('与当前版位、搜索和排序组合筛选。','Combine a tag with your current board, search and order.')}</p></div><button type="button" className="text-button" onClick={close}>{t('收起','Collapse')}</button></header>
   <CommunityTagSearch value={query} onChange={setQuery} inputRef={search} label={t('搜索浏览标签','Search tags to browse')} clearLabel={t('清空标签搜索','Clear tag search')} placeholder={t('名称、标识或说明…','Name, identifier or description…')} className="community-tag-browser-search" onEmptyEscape={close} statusId={statusId}/>
   <button type="button" className="community-tag-all" aria-pressed={!value} onClick={()=>{if(value)onChange();}}>{t('全部标签','All tags')}</button>
   <div id={listId} ref={choices.root} className="community-tag-options" role="group" aria-label={t('可浏览的社区标签','Community tags to browse')}>
    {choices.visible.map(tag=><button type="button" key={tag.id} data-tag-id={tag.id} data-tag-option={tag.id} aria-pressed={value===tag.id} onClick={()=>{if(value!==tag.id)onChange(tag.id);}}><span>#{t(tag.zh,tag.en)}</span>{tag.description&&<small>{tag.description}</small>}{!tag.enabled&&<small className="community-tag-paused">{t('已停用 · 可阅读历史帖子','Paused · historical posts remain readable')}</small>}</button>)}
   </div>
   <CommunityTagChoiceControls shown={choices.visible.length} total={choices.rows.length} more={choices.more} expanded={choices.expanded} listId={listId} statusId={statusId} compactRef={choices.compactControl} onMore={choices.expand} onCompact={choices.compact}/>
   {!choices.rows.length&&<p className="community-tag-browser-empty" role="status">{query?t('没有匹配的标签，试试其他名称或说明。','No tags match. Try another name or description.'):t('当前没有可浏览的标签。','No tags are available to browse.')}{query&&<button type="button" className="text-button" onClick={clearSearch}>{t('清空搜索','Clear search')}</button>}</p>}
  </div>
 </details>;
}

export function CommunityTagContext({value,onClear}:{value?:string;onClear:()=>void}){
 const {t}=useI18n(),taxonomy=useCommunityTaxonomy();
 if(!value||!taxonomy.supported)return null;
 const selected=taxonomy.data.tags.find(tag=>tag.id===value);
 if(selected&&!selected.description&&selected.enabled)return null;
 return <aside className="community-tag-context" aria-label={t('当前标签说明','Current tag details')}>
  <span className="community-tag-mark" aria-hidden="true">#</span><div><strong>{selected?t(selected.zh,selected.en):value}</strong>{selected?.description&&<p>{selected.description}</p>}{selected&&!selected.enabled&&<p>{t('这个标签已停用，历史帖子仍可阅读；新帖请选择其他开放标签。','This tag is paused. Historical posts remain readable; choose an active tag for a new post.')}</p>}{!selected&&<><p role="status">{!taxonomy.ready||taxonomy.error?t('暂时无法确认此标签的配置，筛选已保留。','The tag settings could not be confirmed. Your filter is kept.'):t('未找到这个标签。可以更新标签配置，或移除此筛选继续浏览。','This tag was not found. Refresh its settings or remove this filter to keep browsing.')}</p><div><button type="button" className="text-button" disabled={taxonomy.loading} onClick={()=>void taxonomy.refresh()}>{t('更新标签配置','Refresh tag settings')}</button><button type="button" className="text-button" onClick={onClear}>{t('移除此标签筛选','Remove this tag filter')}</button></div></>}</div>
 </aside>;
}
