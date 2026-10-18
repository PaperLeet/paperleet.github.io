import AppLink from './AppLink';
import {useContext,useRef,useState,useSyncExternalStore} from 'react';
import {Search,X} from 'lucide-react';
import {useI18n} from './i18n';
import {adminHref,adminListView} from './adminLinks';
import {AdminDraftContext,AdminDraftRow} from './AdminDrafts';
import {useAdminRead,ReadStatus,TermForm,type Term} from './AdminParts';
import {matchingAdminTerms,taxonomyBrowse,taxonomyPageSize,validAdminTaxonomy,type TaxonomyBrowse} from './adminTaxonomyBrowsing';
import type {AdminListProps} from './AdminLists';

export default function AdminTaxonomy({owner,view,epoch,run,blocked,onChange}:AdminListProps){
 const {t,locale}=useI18n(),kind=view.section as 'boards'|'tags',{store}=useContext(AdminDraftContext)!;
 const [browse,setBrowse]=useState(()=>taxonomyBrowse(store.readTaxonomyBrowse(kind))),current=useRef(browse),search=useRef<HTMLInputElement>(null),more=useRef<HTMLButtonElement>(null),results=useRef<HTMLDivElement>(null),composing=useRef(false);
 const [,disclosures]=useState(0);
 useSyncExternalStore(listener=>store.subscribe('*',listener),()=>store.taxonomyRetentionSnapshot(kind),()=>store.taxonomyRetentionSnapshot(kind));
 current.current=browse;
 const result=useAdminRead<{boards:Term[];tags:Term[]}>('admin/taxonomy',owner,epoch,validAdminTaxonomy),rows=result.data?.[kind]||[],matches=matchingAdminTerms(rows,browse),shown=new Set(matches.slice(0,browse.limit).map(row=>row.id));
 const newIdentifier=store.newTaxonomyIdentifier(kind);
 const retained=new Set(rows.filter(row=>!shown.has(row.id)&&(row.id===view.item||row.id===newIdentifier||store.read(kind+':'+row.id)||store.isOpen(kind+':'+row.id))).map(row=>row.id));
 const active=rows.filter(row=>row.enabled).length,changed=!!browse.query||browse.status!=='all',remaining=Math.max(0,matches.length-browse.limit);
 function change(next:Partial<TaxonomyBrowse>){const value=taxonomyBrowse({...current.current,...next});current.current=value;store.setTaxonomyBrowse(kind,value);setBrowse(value);}
 function clear(all=false){search.current?.focus({preventScroll:true});change({query:'',limit:20,...(all?{status:'all'}as const:{})});}
 function reveal(limit:number){if(limit>=matches.length)more.current?.focus({preventScroll:true});change({limit});}
 function showMore(){if(remaining){if(current.current.limit>=matches.length)return;reveal(Math.min(matches.length,current.current.limit+taxonomyPageSize));}else{results.current?.focus({preventScroll:true});change({limit:20});}}
 return <>
  <p className="admin-intro">{t('中英文名称用于前台展示。标识创建后不可修改；通过停用保留历史内容，数字较小的排在前面。','Bilingual names appear in the community. Identifiers are permanent. Pause items to preserve history; lower positions appear first.')}</p>
  <div className="admin-taxonomy-toolbar">
   <form role="search" className="admin-taxonomy-search" onSubmit={event=>event.preventDefault()}>
    <label htmlFor={'admin-'+kind+'-search'}>{t(kind==='boards'?'查找版位':'查找标签',kind==='boards'?'Find boards':'Find tags')}</label>
    <div><Search size={18} aria-hidden="true"/><input id={'admin-'+kind+'-search'} ref={search} type="search" maxLength={100} value={browse.query} placeholder={t('搜索名称、标识或说明','Search names, identifiers or descriptions')} onChange={event=>change({query:event.target.value,limit:20})} onCompositionStart={()=>{composing.current=true;}} onCompositionEnd={()=>{composing.current=false;}} onKeyDown={event=>{if(event.key==='Escape'&&browse.query&&!composing.current&&!event.nativeEvent.isComposing&&event.keyCode!==229&&!event.ctrlKey&&!event.metaKey&&!event.altKey){event.preventDefault();event.stopPropagation();clear();}}}/>{browse.query&&<button type="button" aria-label={t('清空配置搜索','Clear settings search')} onClick={()=>clear()}><X size={17} aria-hidden="true"/></button>}</div>
   </form>
   <label className="admin-taxonomy-status-filter">{t('使用状态','Availability')}<select value={browse.status} onChange={event=>change({status:event.target.value as TaxonomyBrowse['status'],limit:20})}><option value="all">{t('全部状态','Any state')}</option><option value="active">{t('已启用','Active')}</option><option value="paused">{t('已停用','Paused')}</option></select></label>
   <button type="button" className="button" disabled={blocked||result.loading||!!result.error} onClick={()=>onChange({...view,item:undefined,create:!view.create})}>{view.create?t('收起新建','Close new item'):t('新建','Create')}</button>
  </div>
  <ReadStatus {...result} retry={()=>{search.current?.focus({preventScroll:true});result.retry();}}/>
  {view.create&&<TermForm row={{id:'',zh:'',en:'',description:'',enabled:true,position:100,revision:0,posts:0}} kind={kind} run={run} blocked={blocked||result.loading||!!result.error} existingIds={rows.map(row=>row.id)} creating/>}
  {result.data&&<div ref={results} tabIndex={-1} className="admin-taxonomy-results" role="group" aria-label={t('配置浏览结果','Settings browsing results')}>
   <p role="status">{t(`匹配 ${matches.length} / ${rows.length} 项 · 已显示 ${shown.size} 项`,`${matches.length} of ${rows.length} match · ${shown.size} shown`)}{retained.size>0&&<span>{t(` · 另保留 ${retained.size} 项核对`,` · ${retained.size} more kept for review`)}</span>}</p>
   <p>{t(`已启用 ${active} · 已停用 ${rows.length-active}`,`${active} active · ${rows.length-active} paused`)}{changed&&<button type="button" className="text-button" onClick={()=>clear(true)}>{t('清除筛选','Clear filters')}</button>}</p>
   <small>{t('已展开、已编辑、链接打开或与新建标识相同的项目会保留显示；筛选不会更改配置。','Items you opened, edited, linked to or matched by a new identifier stay visible. Filters do not change settings.')}</small>
  </div>}
  {result.data&&view.item&&!rows.some(row=>row.id===view.item)&&<p role="status">{t('此配置项已不存在，其他草稿仍保留。','This setting is unavailable. Other drafts are kept.')} <AppLink href={adminHref(adminListView(view))}>{t('返回完整列表','Back to the full list')}</AppLink></p>}
  {result.data&&!matches.length&&<div className="admin-taxonomy-empty" role="status"><p>{changed?retained.size?t('没有匹配的配置项。保留的核对项仍显示在下面。','No settings match. Items kept for review remain below.'):t('没有匹配的配置项。试试其他关键词或清除筛选。','No settings match. Try other keywords or clear the filters.'):t('这里还没有配置项，可以先新建一个。','There are no settings here yet. Create the first item.')}</p>{changed&&<button type="button" className="text-button" onClick={()=>clear(true)}>{t('清除搜索与状态筛选','Clear search and state filters')}</button>}</div>}
  <div className="admin-items admin-taxonomy-items">{rows.map(row=><AdminDraftRow className="admin-card" key={row.id} draftKey={kind+':'+row.id} forceOpen={row.id===view.item} hidden={!shown.has(row.id)&&!retained.has(row.id)} onOpenChange={()=>disclosures(value=>value+1)}>
   <summary data-admin-term={row.id}>
    <span className="admin-taxonomy-identity"><b>{t(row.zh,row.en)}</b>{row.zh!==row.en&&<small>{locale==='en'?row.zh:row.en}</small>}<code>{row.id}</code>{row.description&&<small className="admin-taxonomy-description">{row.description}</small>}</span>
    <span className="admin-taxonomy-meta"><span className="admin-taxonomy-badge" data-state={row.enabled?'active':'paused'}>{row.enabled?t('已启用','Active'):t('已停用','Paused')}</span><span>{t(`排序 ${row.position}`,`Position ${row.position}`)} · {row.posts} {t('篇帖子','posts')}</span>{retained.has(row.id)&&<small className="admin-taxonomy-retained">{row.id===view.item?t('链接打开项','Opened from link'):row.id===newIdentifier?t('与新建标识相同','Matches new identifier'):t('保留核对项','Kept for review')}</small>}</span>
   </summary>
   <TermForm row={row} kind={kind} run={run} blocked={blocked||result.loading||!!result.error}/>
  </AdminDraftRow>)}</div>
  {result.data&&matches.length>taxonomyPageSize&&<div className="admin-taxonomy-more"><button ref={more} type="button" className="button subtle" onClick={showMore}>{remaining?t(`再显示 ${Math.min(remaining,taxonomyPageSize)} 项`,`Show ${Math.min(remaining,taxonomyPageSize)} more`):t('收起为前 20 项','Show the first 20')}</button>{remaining>taxonomyPageSize&&<button type="button" className="text-button" onClick={()=>reveal(matches.length)}>{t(`显示全部 ${matches.length} 项`,`Show all ${matches.length}`)}</button>}</div>}
 </>;
}
