import AppLink from './AppLink';
import {useLayoutEffect,useMemo,useRef,useState} from 'react';
import {ArrowLeft,ArrowUpRight,BookOpen,Search,X} from 'lucide-react';
import {useI18n} from './i18n';
import type {Category,Paper,Problem} from './types';
import {searchPaperRegistry} from './paperRegistrySearch';
import {readPaperRegistryRoute,paperRegistryHref,sortLibraryPapers,withLibraryContext,type PaperRegistryView} from './libraryNavigation';
import {watchRegistryPosition,type RegistryPositionNotice} from './registryPosition';
import './paper-registry.css';

export const registryPageSize=20;
export default function PaperRegistry({papers,problems,categories=[],route='paper-registry',error,onRetry,onFilters}:{papers:Paper[];problems:Problem[];categories?:Category[];route?:string;error?:string;onRetry:()=>void;onFilters?:(href:string)=>void}){
 const root=useRef<HTMLElement>(null),position=useRef<ReturnType<typeof watchRegistryPosition>>(undefined),edited=useRef<string>(undefined),[readingPosition,setReadingPosition]=useState<RegistryPositionNotice&{href:string}>();
 const {t,locale}=useI18n(),search=useRef<HTMLInputElement>(null),results=useRef<HTMLDivElement>(null),composing=useRef(false),[notice,setNotice]=useState<{view:string;text:string}>();
 const view=readPaperRegistryRoute(route),query=view?.query||'',category=view?.category||'all',sort=view?.sort||'default',page=view?.page||1;
 const entries=useMemo(()=>sortLibraryPapers(searchPaperRegistry(papers,problems,query).filter(paper=>category==='all'||paper.category===category),sort),[papers,problems,query,category,sort]);
 const published=useMemo(()=>papers.filter(paper=>paper.kind!=='course'),[papers]),byId=useMemo(()=>new Map(problems.map(problem=>[problem.id,problem])),[problems]);
 const groups=useMemo(()=>[...new Set(published.map(paper=>paper.category))].map(id=>({id,count:published.filter(paper=>paper.category===id).length,category:categories.find(row=>row.id===id)})),[published,categories]);
 const label=(id:string)=>{const item=categories.find(row=>row.id===id);return item?t(item.name,item.name_en):id;};
 const href=view?paperRegistryHref(view):'#/paper-registry',last=Math.max(1,Math.ceil(entries.length/registryPageSize)),offset=(page-1)*registryPageSize,visible=entries.slice(offset,offset+registryPageSize),changed=!!query||category!=='all'||sort!=='default';
 const validView=!!view;
 useLayoutEffect(()=>{
  if(!validView||!root.current)return;
  const controller=watchRegistryPosition(root.current,href,edited.current!==href,value=>setReadingPosition(value?{...value,href}:undefined));edited.current=undefined;position.current=controller;
  return()=>{controller.stop();if(position.current===controller)position.current=undefined;};
 },[href,validView]);
 useLayoutEffect(()=>{position.current?.ready(!error&&papers.length>0);},[href,validView,error,papers,problems]);
 useLayoutEffect(()=>{position.current?.commit(readingPosition?.href===href?readingPosition.request:undefined);},[href,validView,readingPosition]);
 function change(patch:Partial<PaperRegistryView>){if(!view)return;try{const next=paperRegistryHref({...view,...patch});edited.current=next;setNotice(undefined);onFilters?.(next);}catch{setNotice({view:href,text:t('搜索最多 200 个字符，请移除控制字符后重试。','Use up to 200 characters without control characters.')});}}
 function clear(patch:Partial<PaperRegistryView>={query:'',category:'all',sort:'default',page:1}){search.current?.focus({preventScroll:true});change(patch);}
 function turn(next:number){results.current?.focus({preventScroll:true});results.current?.scrollIntoView({block:'start'});change({page:next});}
 return <main className="social-page paper-registry-page" ref={root} data-registry-view={href}>
  <AppLink className="text-button" href="#"><ArrowLeft size={16}/>{t('返回题库','Back to practice')}</AppLink>
  <header className="paper-registry-heading"><div><span className="paper-registry-eyebrow">{t('PaperLeet · 论文目录','PaperLeet · Paper directory')}</span><h1>{t('论文库','Paper registry')}</h1><p>{t('按来源查找已收录论文，了解练习覆盖范围，投稿前先查重。','Find published papers and their exercises before proposing something new.')}</p></div><AppLink className="button subtle" href="#/submit">{t('提交论文','Submit paper')}<ArrowUpRight size={16}/></AppLink></header>
  {!view?<div className="paper-registry-empty" role="alert"><p>{t('这份论文库筛选链接无效。','This paper registry filter link is invalid.')}</p><AppLink className="button subtle" href="#/paper-registry">{t('查看完整论文库','Open the full paper registry')}</AppLink></div>:<>
   <div className="paper-registry-toolbar">
    <form role="search" className="paper-registry-search" onSubmit={event=>event.preventDefault()}><Search size={19} aria-hidden="true"/><input ref={search} type="search" value={query} maxLength={200} onChange={event=>change({query:event.target.value,page:1})} onCompositionStart={()=>{composing.current=true;}} onCompositionEnd={()=>{composing.current=false;}} onKeyDown={event=>{if(event.key==='Escape'&&query&&!composing.current&&!event.nativeEvent.isComposing&&event.keyCode!==229&&!event.ctrlKey&&!event.metaKey&&!event.altKey){event.preventDefault();event.stopPropagation();clear({query:'',page:1});}}} aria-label={t('搜索已收录论文','Search published papers')} placeholder={t('论文、作者、题号或 arXiv 链接…','Paper, author, exercise number or arXiv link…')}/>{query&&<button type="button" aria-label={t('清空论文搜索','Clear paper search')} onClick={()=>clear({query:'',page:1})}><X size={17} aria-hidden="true"/></button>}</form>
    <label>{t('论文分类','Paper category')}<select value={category} onChange={event=>change({category:event.target.value,page:1})}><option value="all">{t(`全部分类 · ${published.length}`,`All categories · ${published.length}`)}</option>{groups.map(group=><option key={group.id} value={group.id}>{label(group.id)} · {group.count}</option>)}{category!=='all'&&!groups.some(group=>group.id===category)&&<option value={category}>{t('暂不可用：','Unavailable: ')+label(category)}</option>}</select></label>
    <label>{t('排序','Sort')}<select value={sort} onChange={event=>change({sort:event.target.value as PaperRegistryView['sort'],page:1})}><option value="default">{t('收录编号','Catalog number')}</option><option value="newest">{t('最新发表','Newest first')}</option><option value="oldest">{t('最早发表','Oldest first')}</option><option value="title">{t('简称 A–Z','Short name A–Z')}</option></select></label>
   </div>
   {notice?.view===href&&<p className="paper-registry-notice" role="alert">{notice.text}</p>}
   {changed&&<div className="paper-registry-applied" role="group" aria-label={t('论文库筛选条件','Paper registry filters')}>
    {query&&<button type="button" aria-label={t('移除论文搜索：','Remove paper search: ')+query} onClick={()=>clear({query:'',page:1})}><span>{query}</span><X size={14} aria-hidden="true"/></button>}
    {category!=='all'&&<button type="button" aria-label={t('移除论文分类：','Remove paper category: ')+label(category)} onClick={()=>clear({category:'all',page:1})}><span>{label(category)}</span><X size={14} aria-hidden="true"/></button>}
    {sort!=='default'&&<button type="button" onClick={()=>clear({sort:'default',page:1})}><span>{t('已更改排序','Custom order')}</span><X size={14} aria-hidden="true"/></button>}
    <button type="button" className="text-button" onClick={()=>clear()}>{t('重置筛选','Reset filters')}</button>
   </div>}
   {!error&&papers.length>0&&readingPosition?.href===href&&<div className="paper-registry-position" role="status" data-registry-position-request={readingPosition.request}><BookOpen size={18} aria-hidden="true"/><span>{readingPosition.kind==='restored'?t('已回到上次浏览的论文位置。','Back at your previous paper position.'):readingPosition.kind==='missing'?t('目录已有更新，上次浏览的论文不在当前页。','The directory changed. Your previous paper is not on this page.'):t('论文库已加载，可以回到上次浏览的位置。','The directory is ready. Return to your previous paper position.')}</span><div className="paper-registry-position-actions">{readingPosition.kind==='available'&&<button type="button" className="text-button" onClick={event=>{event.currentTarget.focus({preventScroll:true});position.current?.restore(event.currentTarget);}}>{t('回到上次位置','Return to previous position')}</button>}{readingPosition.kind==='missing'&&<AppLink href={paperRegistryHref({...view,query:readingPosition.paper,category:'all',page:1})}>{t('查找来源论文','Find the previous paper')}</AppLink>}<button type="button" className="paper-registry-position-dismiss" aria-label={t('关闭位置提示','Dismiss position notice')} onClick={event=>{event.currentTarget.focus({preventScroll:true});position.current?.dismiss(event.currentTarget,event.detail===0);}}><X size={16} aria-hidden="true"/></button></div></div>}
   {error?<div className="paper-registry-empty" role="alert"><p>{t('论文库暂时不可用，当前筛选已保留。','The paper registry is unavailable. Your filters are kept.')}</p><button className="button subtle" type="button" onClick={()=>{search.current?.focus({preventScroll:true});onRetry();}}>{t('重试','Retry')}</button></div>:!papers.length?<p role="status">{t('正在加载论文库…','Loading papers…')}</p>:<>
    <div className="paper-registry-count" ref={results} tabIndex={-1} role="group" aria-label={t('论文库结果','Paper registry results')}><span role="status">{t(`${entries.length} / ${published.length} 篇论文`,`${entries.length} of ${published.length} papers`)}{visible.length>0&&t(` · 第 ${offset+1}–${offset+visible.length} 项`,` · ${offset+1}–${offset+visible.length} shown`)}</span><span>{t(`第 ${page} / ${last} 页`,`Page ${page} of ${last}`)}</span></div>
    {!entries.length?<div className="paper-registry-empty"><BookOpen size={28} aria-hidden="true"/><h2>{t('没有匹配的已收录论文','No published papers match')}</h2><p>{t('试试论文简称、作者或题号，也可以清除筛选。投稿时仍会检查现有候选题中的重复论文。','Try a short name, author or exercise number, or clear the filters. Submission also checks existing proposals.')}</p>{changed&&<button type="button" className="button subtle" onClick={()=>clear()}>{t('清除筛选','Clear filters')}</button>}<AppLink href="#/submit">{t('去提交论文','Submit a paper')}</AppLink></div>:!visible.length?<div className="paper-registry-empty" role="status"><p>{t('当前页已无内容，筛选条件仍保留。','This page is now empty. Your filters are kept.')}</p><button className="button subtle" type="button" onClick={()=>turn(Math.min(501,last))}>{t('返回最后一页','Go to the last page')}</button></div>:<ol className="paper-registry-list" start={offset+1}>{visible.map(paper=><li key={paper.id} data-paper-registry-id={paper.id} data-registry-resumed={readingPosition?.href===href&&readingPosition.kind==='restored'&&readingPosition.paper===paper.id?true:undefined}>
     <div className="paper-registry-meta"><span>#{paper.number} · {paper.short_title}</span><span>{[paper.year,paper.venue].filter(Boolean).join(' · ')}</span></div>
     <h2><AppLink href={withLibraryContext(`papers/${paper.id}`,href)}>{paper.title}</AppLink></h2><p className="paper-registry-authors">{Array.isArray(paper.authors)?paper.authors.join(', '):paper.authors}</p>
     {(paper.summary||paper.summary_en)&&<p className="paper-registry-summary">{locale==='en'?paper.summary_en||paper.summary:paper.summary||paper.summary_en}</p>}
     <div className="paper-registry-links"><AppLink className="paper-registry-source" href={paper.url} target="_blank" rel="noopener noreferrer">{paper.arxiv?`arXiv:${paper.arxiv}`:t('原文','Source')}<ArrowUpRight size={15} aria-hidden="true"/></AppLink>{paper.module_ids.flatMap(id=>{const problem=byId.get(id);return problem?[<AppLink key={id} href={withLibraryContext(problem.number,href)}>#{problem.number} {locale==='en'?problem.translations?.en?.title||problem.title:problem.title}</AppLink>]:[];})}{!paper.module_ids.some(id=>byId.has(id))&&<span className="paper-registry-missing">{t('练习信息暂不可用','Exercise information is unavailable')}</span>}</div>
    </li>)}</ol>}
    {entries.length>registryPageSize&&<nav className="paper-registry-pager" aria-label={t('论文库翻页','Paper registry pages')}><button type="button" disabled={page<=1} onClick={()=>turn(page-1)}>{t('上一页','Previous')}</button><span>{t(`第 ${page} / ${last} 页`,`Page ${page} of ${last}`)}</span><button type="button" disabled={page>=last||page>=501} onClick={()=>turn(page+1)}>{t('下一页','Next')}</button></nav>}
   </>}
  </>}
 </main>;
}
